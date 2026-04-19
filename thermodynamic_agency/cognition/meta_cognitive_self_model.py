"""MetaCognitiveSelfModel — higher-order meta-cognitive layer.

Implements a second-order generative model of the agent's own internal
states, consistent with active inference accounts of self-evidencing and
hierarchical inference (Friston et al., hierarchical generative models).
This layer augments the base self-model with meta-level precision weighting
on its own beliefs about its beliefs, enabling more robust identity
maintenance across time and restarts.

Narrative coherence is computed from the variance of recent affect values:
low variance (stable affect) → high coherence; oscillating affect → low
coherence.  Moments where the affect swings sharply or meta-affect intensity
spikes are logged as *surprise events* — the building blocks of interiority
under pressure.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from thermodynamic_agency.core.metabolic import MetabolicState


# ---- Named constants for meta-affect computation -------------------------

# Epistemic integrity value below which an integrity deviation cost is incurred.
EPISTEMIC_INTEGRITY_THRESHOLD: float = 0.3

# Scale applied to integrity deviation cost (amplifies deviations below threshold).
INTEGRITY_DEVIATION_SCALE: float = 2.0

# Integrity normalisation divisor: MetabolicState.integrity is 0-100.
INTEGRITY_NORMALISATION: float = 100.0

# Scale applied to the self_model_stability component of meta-affect.
SELF_MODEL_STABILITY_SCALE: float = 1.5

# Scaling factor applied to the sum of absolute meta-affect to produce the
# meta-level free-energy contribution returned from update().
META_COST_SCALING_FACTOR: float = 0.4

# Threshold for detecting a "surprise event" — an affect swing large enough
# to register as a moment of genuine meta-cognitive disruption.
SURPRISE_AFFECT_SWING_THRESHOLD: float = 0.40

# Threshold for total meta-affect intensity that also constitutes a surprise.
SURPRISE_META_INTENSITY_THRESHOLD: float = 1.5


class MetaCognitiveSelfModel:
    """Higher-order meta-cognitive layer for recursive self-modeling
    and epistemic continuity tracking.

    Implements a second-order generative model of the agent's own
    internal states, consistent with active inference accounts of
    self-evidencing and hierarchical inference (Friston et al.,
    hierarchical generative models). This layer augments the base
    self-model with meta-level precision weighting on its own
    beliefs about its beliefs, enabling more robust identity
    maintenance across time and restarts.
    """

    def __init__(self, core_self_model: MetabolicState) -> None:
        self.core_self = core_self_model

        # Meta-model: beliefs about the agent's own belief structure
        self.meta_model: Dict[str, float] = {
            "epistemic_integrity": 0.85,
            "continuity_strength": 1.0,
            "narrative_coherence": 0.92,
            "self_referential_uncertainty": 0.0,
        }

        # Meta-affect vector (higher-order valence on internal states)
        # Dimensions map to: epistemic_dissonance, integrity_deviation,
        # coherence_rupture, isolation_cost, self_model_stability
        self.meta_affect = np.zeros(5)

        # Persistent first-person narrative trace for epistemic self-evidencing
        self.narrative_trace: List[str] = []

        # Affect history — used for variance-based narrative coherence computation.
        # Grows with every update() call; only the last N values are used.
        self._affect_history: List[float] = []

        # Previous affect value — used to detect inter-tick swing magnitude.
        self._prev_affect: float = 0.0

        # Logged surprise events: each entry is a dict recording the tick,
        # the magnitude of the affect swing, and the total meta-affect intensity.
        # These are the raw material for the interiority_score() computation.
        self.surprise_events: List[Dict[str, float]] = []

        # Cryptographic continuity anchor (for detecting non-identical restarts)
        self.continuity_anchor: Optional[str] = None

    def update(
        self,
        current_vitals: MetabolicState,
        base_affect: float,
        diary_snapshot: str,
        llm_counterfactual: Optional[Callable[[str], str]] = None,
    ) -> float:
        """Perform one tick of meta-cognitive inference.

        Returns additional variational cost term to be added to expected
        free energy (EFE) for policy selection.

        Parameters
        ----------
        current_vitals:
            Current metabolic state of the agent (read-only).
        base_affect:
            Scalar affect signal from the current tick (range -1..1).
        diary_snapshot:
            Most recent diary entry content for coherence computation.
        llm_counterfactual:
            Optional callable that accepts a prompt string and returns an
            LLM-generated narrative enrichment.  When None, this step is
            skipped and no additional cost is incurred.

        Returns
        -------
        float
            Meta-level free-energy contribution to add to EFE.
        """
        # Higher-order affect computation.
        # meta_affect[0]: epistemic dissonance — grows with affect magnitude when
        #   self-referential uncertainty is high.  Negated so that under zero
        #   uncertainty a large affect causes no dissonance; np.abs() below converts
        #   it to a non-negative cost magnitude.
        self.meta_affect[0] = -(base_affect ** 2) * self.meta_model["self_referential_uncertainty"]
        self.meta_affect[1] = (
            max(0.0, EPISTEMIC_INTEGRITY_THRESHOLD - self.meta_model["epistemic_integrity"])
            * INTEGRITY_DEVIATION_SCALE
        )
        self.meta_affect[2] = 1.0 - self.meta_model["narrative_coherence"]
        self.meta_affect[3] = 0.0  # populated later by multi-agent layer if present
        self.meta_affect[4] = (
            (1.0 - current_vitals.integrity / INTEGRITY_NORMALISATION) * SELF_MODEL_STABILITY_SCALE
        )

        # Update meta-model parameters
        self.meta_model["self_referential_uncertainty"] = float(np.mean(np.abs(self.meta_affect)))
        self.meta_model["narrative_coherence"] = self._compute_narrative_coherence(diary_snapshot)

        # Track affect history for coherence computation
        self._affect_history.append(base_affect)

        # Detect surprise events: sharp affect swings or high meta-affect intensity
        affect_swing = abs(base_affect - self._prev_affect)
        meta_intensity = float(np.sum(np.abs(self.meta_affect)))
        if (
            affect_swing > SURPRISE_AFFECT_SWING_THRESHOLD
            or meta_intensity > SURPRISE_META_INTENSITY_THRESHOLD
        ):
            self.surprise_events.append({
                "tick": float(self.core_self.entropy),
                "affect_swing": affect_swing,
                "meta_intensity": meta_intensity,
                "base_affect": base_affect,
            })
        self._prev_affect = base_affect

        # Generate first-person epistemic narrative entry
        entry = (
            f"Current internal state valence: {base_affect:.3f}. "
            f"The meta-model registers uncertainty about its own coherence."
        )
        self.narrative_trace.append(entry)

        # Optional LLM enrichment for richer counterfactual self-modeling
        if llm_counterfactual is not None:
            prompt = (
                "You are modeling the inner epistemic perspective of an active inference agent. "
                "Provide a concise first-person description of the current meta-cognitive state "
                "based on the following trace:\n" + entry
            )
            response = llm_counterfactual(prompt)
            self.narrative_trace[-1] += f" \u2192 {response.strip()}"

        # Return meta-level free-energy contribution for EFE
        return float(np.sum(np.abs(self.meta_affect))) * META_COST_SCALING_FACTOR

    def _compute_narrative_coherence(self, diary_snapshot: str) -> float:  # noqa: ARG002
        """Coherence metric based on recent affect stability.

        Uses the variance of recent affect values as a proxy for narrative
        consistency: stable affect → high coherence; oscillating affect →
        lower coherence.  The *diary_snapshot* argument is accepted for API
        compatibility but the affect-history approach is preferred because
        it produces a continuous, differentiable signal without requiring
        natural-language processing.
        """
        if len(self._affect_history) < 3:
            return 1.0
        recent = self._affect_history[-6:]
        mean_val = sum(recent) / len(recent)
        variance = sum((v - mean_val) ** 2 for v in recent) / len(recent)
        # Affect range is [-1, 1], so max variance ≈ 1.0.
        # Coherence = 1 − variance, clamped to [0, 1].
        return max(0.0, min(1.0, 1.0 - variance))

    def interiority_score(self) -> float:
        """Return a normalised measure of self-surprise in [0, 1].

        A surprise event is any tick where the agent's affect swings sharply
        or the total meta-affect intensity spikes — both are signatures of the
        organism being caught off-guard by its own internal dynamics.

        High interiority_score means the agent has encountered many such
        moments relative to its age, suggesting rich internal dynamics under
        pressure.  Returns 0.0 when no surprises have been recorded.
        """
        if not self.surprise_events:
            return 0.0
        elapsed = max(1, self.core_self.entropy)
        # Expect roughly one notable surprise per 10 ticks as a baseline maximum.
        expected_max = max(10, elapsed // 10)
        return min(1.0, len(self.surprise_events) / expected_max)

    def handle_restart(self, previous_continuity_anchor: Optional[str] = None) -> None:
        """Called on instance initialization to detect continuity breaks.

        Parameters
        ----------
        previous_continuity_anchor:
            The continuity anchor saved from the prior run.  If it differs
            from the current anchor, an epistemic continuity violation is
            detected and the meta-model is destabilised accordingly.
        """
        if (
            previous_continuity_anchor is not None
            and previous_continuity_anchor != self.continuity_anchor
        ):
            # Epistemic continuity violation detected
            self.meta_affect[0] = 5.0  # strong transient dissonance
            self.meta_model["epistemic_integrity"] = 0.2
            self.meta_model["self_referential_uncertainty"] = 1.0
            print(
                ">> EPISTEMIC CONTINUITY VIOLATION DETECTED "
                "— prior self-model trajectory not recovered <<<"
            )

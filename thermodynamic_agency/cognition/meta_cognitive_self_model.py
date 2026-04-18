"""MetaCognitiveSelfModel — higher-order meta-cognitive layer.

Implements a second-order generative model of the agent's own internal
states, consistent with active inference accounts of self-evidencing and
hierarchical inference (Friston et al., hierarchical generative models).
This layer augments the base self-model with meta-level precision weighting
on its own beliefs about its beliefs, enabling more robust identity
maintenance across time and restarts.
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
        """Simple coherence metric over recent self-narrative."""
        if len(self.narrative_trace) < 3:
            return 1.0
        # Replace with embedding cosine similarity for higher precision
        return 0.92

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

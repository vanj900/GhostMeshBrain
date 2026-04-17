"""g_factor — PCA-based g-factor measurement for GhostMesh.

After running a cognitive battery over *N* episodes (recommended ≥ 20), the
resulting *N × 6* score matrix is passed to :func:`measure_g`.  The function
runs a single-component PCA and returns:

* **g_scores** — the first principal component score for each agent/episode.
* **variance_explained** — fraction of total variance captured by the first
  PC.  Values above 0.35 (35 %) indicate that a common underlying factor is
  emerging.
* **loadings** — the six task loadings on the first PC.  Large positive
  loadings indicate tasks that contribute most to g.

Implementation notes
--------------------
The PCA uses only the Python standard library (``math``).  Steps:

1. **Standardise** each column (zero mean, unit variance) so tasks with
   different score ranges contribute equally.
2. **Covariance matrix** — computed as ``Cᵀ × C / (n − 1)`` where *C* is the
   standardised matrix.  The resulting matrix is 6 × 6.
3. **Power iteration** — finds the dominant eigenvector (first PC loadings)
   without a full eigendecomposition.  Converges in ≤ 200 iterations for the
   sizes encountered here.
4. **Variance explained** — dominant eigenvalue / trace(covariance matrix).
   Since each standardised column has variance ≈ 1, the trace equals the
   number of tasks (6), and variance_explained ∈ (0, 1].

Usage
-----
    from thermodynamic_agency.evaluation import measure_g, GFactorResult

    scores = [
        [0.8, 0.7, 0.6, 0.9, 0.5, 0.7],  # episode 1
        [0.4, 0.3, 0.5, 0.4, 0.3, 0.5],  # episode 2
        # … more episodes …
    ]
    result = measure_g(scores)
    print(f"G-factor explains {result.variance_explained * 100:.1f}% of variance")
    print(f"Task loadings: {dict(zip(result.task_names, result.loadings))}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from thermodynamic_agency.evaluation.cognitive_battery import TASK_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GFactorResult:
    """G-factor measurement result from a batch of cognitive battery runs.

    Attributes
    ----------
    g_scores:
        First PC score for each episode (length = number of evaluated episodes).
        Higher values indicate a more broadly capable agent on that episode.
    variance_explained:
        Fraction of total variance captured by the first principal component.
        Values ≥ 0.35 indicate meaningful emergent general intelligence.
    loadings:
        First PC loadings for the six tasks (same order as ``TASK_NAMES``).
        Positive loadings mean the task correlates with the g-factor.
    task_names:
        The six task names in canonical order.
    n_episodes:
        Number of episodes used to compute g.
    """

    g_scores: list[float]
    variance_explained: float
    loadings: list[float]
    task_names: list[str] = field(default_factory=lambda: list(TASK_NAMES))
    n_episodes: int = 0

    def __post_init__(self) -> None:
        if self.n_episodes == 0:
            self.n_episodes = len(self.g_scores)

    @property
    def variance_explained_pct(self) -> float:
        """Variance explained as a percentage (0–100)."""
        return self.variance_explained * 100.0

    def is_significant(self, threshold: float = 0.35) -> bool:
        """Return True if g explains at least *threshold* of the variance."""
        return self.variance_explained >= threshold

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"G-factor (N={self.n_episodes} episodes)",
            f"  Variance explained by PC1: {self.variance_explained_pct:.1f}%"
            + (" ✓ significant" if self.is_significant() else " (not yet significant)"),
            "  Task loadings:",
        ]
        for name, loading in zip(self.task_names, self.loadings):
            bar = "█" * int(abs(loading) * 20)
            sign = "+" if loading >= 0 else "-"
            lines.append(f"    {name:20s} {sign}{abs(loading):.3f} {bar}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def measure_g(scores: Sequence[Sequence[float]]) -> GFactorResult:
    """Compute the g-factor from a batch of cognitive battery score vectors.

    Parameters
    ----------
    scores:
        A sequence of score vectors.  Each vector must have exactly 6 elements
        (one per task) in the order defined by ``TASK_NAMES``.  At least 3
        vectors are required; ≥ 20 is recommended for a stable estimate.

    Returns
    -------
    GFactorResult
        G-factor result with scores, variance explained, and task loadings.

    Raises
    ------
    ValueError
        If fewer than 3 score vectors are provided, or if any vector does not
        have exactly 6 elements.
    """
    n_rows = len(scores)
    if n_rows < 3:
        raise ValueError(
            f"measure_g() requires at least 3 score vectors; got {n_rows}."
        )

    n_tasks = 6
    for i, row in enumerate(scores):
        if len(row) != n_tasks:
            raise ValueError(
                f"Score vector at index {i} has {len(row)} elements; "
                f"expected {n_tasks}."
            )

    # ── Step 1: standardise (z-score per column) ──────────────────────────
    std_matrix = _standardise(scores, n_rows, n_tasks)

    # ── Step 2: covariance matrix (n_tasks × n_tasks) ─────────────────────
    cov = _covariance(std_matrix, n_rows, n_tasks)

    # ── Step 3: dominant eigenvector via power iteration ──────────────────
    loadings, dominant_eigenvalue = _power_iteration(cov, n_tasks)

    # ── Step 4: variance explained ────────────────────────────────────────
    trace = sum(cov[j][j] for j in range(n_tasks))
    variance_explained = dominant_eigenvalue / trace if trace > 1e-12 else 0.0
    variance_explained = min(1.0, max(0.0, variance_explained))

    # ── Step 5: project each row onto the first PC ────────────────────────
    g_scores: list[float] = []
    for row in std_matrix:
        score = sum(row[j] * loadings[j] for j in range(n_tasks))
        g_scores.append(score)

    return GFactorResult(
        g_scores=g_scores,
        variance_explained=variance_explained,
        loadings=loadings,
        task_names=list(TASK_NAMES),
        n_episodes=n_rows,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python linear algebra helpers
# ─────────────────────────────────────────────────────────────────────────────

def _standardise(
    scores: Sequence[Sequence[float]],
    n_rows: int,
    n_cols: int,
) -> list[list[float]]:
    """Return a z-scored copy of *scores* (mean 0, std 1 per column)."""
    # Column means
    means = [
        sum(scores[i][j] for i in range(n_rows)) / n_rows
        for j in range(n_cols)
    ]
    # Column standard deviations (ddof=1)
    stds: list[float] = []
    for j in range(n_cols):
        var = sum((scores[i][j] - means[j]) ** 2 for i in range(n_rows))
        var /= max(1, n_rows - 1)
        stds.append(math.sqrt(var) if var > 0.0 else 1.0)

    return [
        [(scores[i][j] - means[j]) / stds[j] for j in range(n_cols)]
        for i in range(n_rows)
    ]


def _covariance(
    std_matrix: list[list[float]],
    n_rows: int,
    n_cols: int,
) -> list[list[float]]:
    """Return the (n_cols × n_cols) sample covariance matrix of *std_matrix*."""
    denom = max(1, n_rows - 1)
    cov = [[0.0] * n_cols for _ in range(n_cols)]
    for j in range(n_cols):
        for k in range(j, n_cols):
            c = sum(std_matrix[i][j] * std_matrix[i][k] for i in range(n_rows))
            c /= denom
            cov[j][k] = c
            cov[k][j] = c
    return cov


def _power_iteration(
    matrix: list[list[float]],
    n: int,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> tuple[list[float], float]:
    """Find the dominant eigenvector and eigenvalue via power iteration.

    Parameters
    ----------
    matrix:
        A symmetric (n × n) matrix.
    n:
        Dimension of the matrix.
    max_iter:
        Maximum number of iterations.
    tol:
        Convergence tolerance (L2 norm of the step change in the eigenvector).

    Returns
    -------
    (eigenvector, eigenvalue):
        The normalised dominant eigenvector and its Rayleigh quotient.
    """
    # Initialise with a uniform vector
    b = [1.0 / math.sqrt(n)] * n

    for _ in range(max_iter):
        # Matrix–vector product: Mb
        mb = [
            sum(matrix[i][j] * b[j] for j in range(n))
            for i in range(n)
        ]
        # Normalise
        norm = math.sqrt(sum(x * x for x in mb))
        if norm < 1e-12:
            break
        b_new = [x / norm for x in mb]

        # Check convergence
        diff = math.sqrt(sum((b_new[i] - b[i]) ** 2 for i in range(n)))
        b = b_new
        if diff < tol:
            break

    # Rayleigh quotient: bᵀ M b
    mb = [sum(matrix[i][j] * b[j] for j in range(n)) for i in range(n)]
    eigenvalue = sum(b[i] * mb[i] for i in range(n))

    return b, eigenvalue

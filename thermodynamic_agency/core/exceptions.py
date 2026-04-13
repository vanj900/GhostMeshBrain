"""Death exceptions — the hard limits of the thermodynamic organism.

These are deliberately *not* caught inside the metabolic loop; they propagate
to the top-level pulse runner which decides whether to restart or terminate.
"""


class GhostDeathException(RuntimeError):
    """Base class for all fatal metabolic failures."""

    def __init__(self, message: str = "", state: dict | None = None) -> None:
        super().__init__(message)
        self.state = state or {}


class EnergyDeathException(GhostDeathException):
    """Energy (E) reached zero — the organism ran out of computational glucose.

    Recovery: external resource injection or cold restart.
    """


class ThermalDeathException(GhostDeathException):
    """Heat (T) reached critical threshold — context congestion / thermal overload.

    Recovery: forced Janitor pass to cool context, then restart.
    """


class MemoryCollapseException(GhostDeathException):
    """Integrity (M) collapsed below survivable threshold.

    Recovery: Surgeon pass + ethical reboot from saved priors.
    """


class EntropyDeathException(GhostDeathException):
    """Stability (S) fell to zero — entropic dissolution.

    Recovery: full cold restart from stable snapshot.
    """

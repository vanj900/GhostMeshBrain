"""Package __main__ — enables `python -m thermodynamic_agency`."""

from thermodynamic_agency.pulse import GhostMesh

if __name__ == "__main__":
    mesh = GhostMesh()
    mesh.run()

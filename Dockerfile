FROM python:3.11-slim

LABEL org.opencontainers.image.title="GhostMesh"
LABEL org.opencontainers.image.description="A bio-digital thermodynamic organism — sovereign, mortal, self-governing."

# /dev/shm is available by default in Docker with --shm-size (or at the
# default 64 MB).  All ephemeral state lives there.
WORKDIR /ghost

# Install the package (no optional deps by default)
COPY pyproject.toml README.md ./
COPY thermodynamic_agency/ ./thermodynamic_agency/
COPY scripts/ ./scripts/
COPY examples/ ./examples/

RUN pip install --no-cache-dir -e .

# Default env — override with -e flags or --env-file
ENV GHOST_PULSE=5 \
    GHOST_COMPUTE_LOAD=1.0 \
    GHOST_HUD=1 \
    GHOST_STATE_FILE=/dev/shm/ghost_metabolic.json \
    GHOST_DIARY_PATH=/dev/shm/ghost_diary.db \
    GHOST_ENV_EVENTS=1

# Run the Python pulse loop by default (no bash dependency required)
CMD ["python", "-m", "thermodynamic_agency"]

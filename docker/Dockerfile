FROM python:3.12-slim

# curl: the price client shells out to it for the Yahoo chart endpoint --
# a live finding (data/prices.py's module docstring), not a style choice:
# httpx and curl get different results (429 vs 200) for the byte-identical
# request, most likely TLS/client fingerprinting on Yahoo's side.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# Full extras: dashboard (Streamlit) + taming (ipca) alongside the core
# dependencies, so one image serves the CLI, the dashboard, and the
# statistical-taming pipeline without a second build.
RUN uv pip install --system -e ".[dashboard,taming]"

# DuckDB file and raw/cache pulls are meant to be mounted, not baked into
# the image -- see build guide Section 8 ("mounted as a volume, so the
# pipeline runs identically on any machine").
VOLUME ["/app/data"]

EXPOSE 8501

ENTRYPOINT ["factorzoo"]
CMD ["status"]

# To run the dashboard instead of the CLI:
#   docker run -p 8501:8501 -v $(pwd)/data:/app/data --entrypoint streamlit \
#     factorzoo run src/factorzoo/dashboard/app.py --server.address 0.0.0.0

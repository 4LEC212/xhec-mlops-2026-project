FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

WORKDIR /app_home

# Install the dependencies first, so this layer stays cached when only the code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY bin ./bin
COPY src/web_service ./src/web_service

ENV PATH="/app_home/.venv/bin:$PATH"
# The browser reaches the Prefect UI through the host port 4200 (mapped to 4201 in the container),
# so the UI must also call the Prefect API through this port
ENV PREFECT_SERVER_UI_API_URL="http://localhost:4200/api"

EXPOSE 8001 4201

CMD ["bash", "bin/run_services.sh"]

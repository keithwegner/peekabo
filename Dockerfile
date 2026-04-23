FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install .

COPY configs ./configs
COPY examples ./examples

RUN mkdir -p /app/runs /app/examples/captures \
    && chown -R app:app /app

USER app

ENTRYPOINT ["peekaboo"]
CMD ["--help"]

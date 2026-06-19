FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

ARG QUARTO_VERSION=1.6.42

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git-lfs \
        jq \
        make \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && curl -L -o /tmp/quarto.deb "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb" \
    && apt-get install -y /tmp/quarto.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/quarto.deb

ENV PATH="/home/vscode/.local/bin:${PATH}"

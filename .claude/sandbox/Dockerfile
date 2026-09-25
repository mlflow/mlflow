# syntax=docker/dockerfile:1

# Toolchain image for running agents in CI, under gVisor.
# Built and published by `.github/workflows/sandbox-image.yml`.
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ARG NODE_VERSION=24.19.0
# https://nodejs.org/dist/v24.19.0/SHASUMS256.txt
ARG NODE_SHA256=f625d97cd707df4ff96254916fbc5ff014f09c09effe5a1e0ca8f6d41a8789d4
# Matches `.github/actions/setup-node`.
ARG NPM_VERSION=12.0.1
# Launcher only; `mlflow/server/js/.yarnrc.yml` picks the real Yarn via `yarnPath`.
ARG YARN_VERSION=1.22.22
ARG GH_VERSION=2.98.0
# gh_2.98.0_checksums.txt on the release.
ARG GH_SHA256=3b8ac6b30336802fc1a858d7c084e11cdf24ac1a761ca90b68022d7d729208de
# Upstream publishes no checksum manifest, so recompute the digest by hand on a
# version bump. https://github.com/vercel-labs/agent-browser/releases
ARG AGENT_BROWSER_VERSION=0.34.0
ARG AGENT_BROWSER_SHA256=69eadf5d8d6003a06a5cd2f914ebb261c7754fe1335a9190122c334e91909789

# The uv `.github/actions/setup-python` pins.
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu carries no fonts, so Chrome falls back to a face with no usable metrics
# and screenshots come out with mangled spacing. Liberation supplies the
# Arial/Helvetica/Times metrics; the emoji face avoids tofu boxes.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
       fontconfig \
       fonts-dejavu-core \
       fonts-liberation \
       fonts-noto-color-emoji \
       git \
       jq \
       sudo \
  && rm -rf /var/lib/apt/lists/*

# `.tar.gz` rather than `.tar.xz` to avoid pulling in xz-utils for one download.
RUN curl -fsSL --retry 3 "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" -o /tmp/node.tar.gz \
  && echo "${NODE_SHA256}  /tmp/node.tar.gz" | sha256sum -c - \
  && tar -xzf /tmp/node.tar.gz -C /usr/local --strip-components=1 --no-same-owner \
  && rm /tmp/node.tar.gz \
  && npm install -g "npm@${NPM_VERSION}" "yarn@${YARN_VERSION}"

RUN curl -fsSL --retry 3 "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" -o /tmp/gh.tar.gz \
  && echo "${GH_SHA256}  /tmp/gh.tar.gz" | sha256sum -c - \
  && tar -xzf /tmp/gh.tar.gz -C /usr/local --strip-components=1 --no-same-owner "gh_${GH_VERSION}_linux_amd64/bin/gh" \
  && rm /tmp/gh.tar.gz

# uid 1001 is `runner` on GitHub-hosted runners, so bind mounts stay writable
# without a recursive chmod. sudo lets an agent install a package to reproduce a
# report; root here is still only root inside the sandbox.
RUN useradd --create-home --uid 1001 --shell /bin/bash agent \
  && echo 'agent ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/agent \
  && chmod 0440 /etc/sudoers.d/agent

USER agent
ENV HOME=/home/agent
ENV PATH=/home/agent/.local/bin:$PATH

# Mirrors `.github/actions/setup-python`.
ENV UV_INDEX_STRATEGY=unsafe-first-match \
    UV_NO_PROGRESS=1 \
    UV_EXCLUDE_NEWER=P7D

# Not npm: its allowScripts policy blocks the postinstall that puts the binary in
# place. Unpinned, like `.claude/scripts/install-claude.sh`.
RUN mkdir -p "$HOME/.local/bin" \
  && CLAUDE_RELEASES=https://downloads.claude.ai/claude-code-releases \
  && CLAUDE_VERSION="$(curl -fsSL --retry 3 "$CLAUDE_RELEASES/latest")" \
  && CLAUDE_SHA256="$(curl -fsSL --retry 3 "$CLAUDE_RELEASES/$CLAUDE_VERSION/manifest.json" | jq -r '.platforms["linux-x64"].checksum')" \
  && curl -fsSL --retry 3 "$CLAUDE_RELEASES/$CLAUDE_VERSION/linux-x64/claude" -o "$HOME/.local/bin/claude" \
  && echo "$CLAUDE_SHA256  $HOME/.local/bin/claude" | sha256sum -c - \
  && chmod +x "$HOME/.local/bin/claude"

# Under the agent's home so `agent-browser install` can chmod its own binary.
# `--with-deps` adds the shared libraries Chrome links against.
RUN curl -fsSL --retry 3 "https://github.com/vercel-labs/agent-browser/releases/download/v${AGENT_BROWSER_VERSION}/agent-browser-linux-x64" -o "$HOME/.local/bin/agent-browser" \
  && echo "${AGENT_BROWSER_SHA256}  $HOME/.local/bin/agent-browser" | sha256sum -c - \
  && chmod +x "$HOME/.local/bin/agent-browser" \
  && agent-browser install --with-deps

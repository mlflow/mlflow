# Assistant sandbox (experimental)

This package runs code the MLflow Assistant would otherwise run on the server host —
the `Bash` tool it issues, and the coding-agent CLI providers (Claude Code and Codex) —
inside a locked-down Docker container instead. It is **off by default** and
**experimental**: the flags and behavior may change or be removed.

## What it does

When enabled, two execution paths move off the host and into a disposable container:

- The assistant's `Bash` tool (used by the API-based providers) runs each command in a
  container via a run-to-completion primitive (`run_in_sandbox`).
- The coding-agent CLI providers run their vendor CLI (`claude` for Claude Code, `codex` for
  Codex) in a container, streaming its output back to the browser (`start_sandbox_process`).

Each container is hardened: all Linux capabilities dropped, `no-new-privileges`, memory /
CPU / PID limits, and it runs as the server's own user so files it writes into a mounted
workspace are not left owned by root. Containers are labeled and cleaned up on server
startup if a previous run left them behind.

## Enabling it

The sandbox turns on automatically when the assistant is put into remote/multi-user mode and a
`docker` executable is available:

```bash
export MLFLOW_ENABLE_REMOTE_ASSISTANT=true
```

In that mode the `Bash` tool and the coding-agent CLI providers run in a container, for local
users too. Remote clients can use only the MLflow AI Gateway provider: the coding-agent CLI
providers (Claude Code and Codex) stay local-only, even in the sandbox. Left unset (the
default), the assistant is localhost-only and runs that work in a host subprocess, exactly as
before. Requirements: a POSIX host with a reachable Docker daemon — the server enables the sandbox
when a `docker` executable is on PATH, and an unreachable daemon surfaces when a turn starts.

To override that automatic behavior, set `MLFLOW_ENABLE_ASSISTANT_SANDBOX` explicitly — `true`
forces the sandbox on (even locally; a turn then fails at container start if Docker is missing) and
`false` opts out of it (local users' work then runs on the host, and remote clients cannot use
the assistant):

```bash
export MLFLOW_ENABLE_ASSISTANT_SANDBOX=false  # opt out; run on the host
```

Relevant environment variables:

| Variable                             | Purpose                                                                                                                                        | Default                           |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `MLFLOW_ENABLE_REMOTE_ASSISTANT`     | Put the assistant in remote/multi-user mode; when a `docker` executable is present, its work runs in the sandbox.                              | `false`                           |
| `MLFLOW_ENABLE_ASSISTANT_SANDBOX`    | Override the automatic decision: `true` forces the sandbox on, `false` forces it off. Unset derives it from remote mode + Docker.              | unset                             |
| `MLFLOW_SANDBOX_DOCKER_IMAGE`        | Image for the `Bash` sandbox (needs Python + MLflow). Auto-built if missing (see below).                                                       | `mlflow-sandbox:latest`           |
| `MLFLOW_ASSISTANT_SANDBOX_CLI_IMAGE` | Image for the coding-agent CLI sandbox — Claude Code and/or Codex (needs the CLI(s) + their runtime). **Operator-provided; never auto-built.** | `mlflow-assistant-sandbox:latest` |
| `MLFLOW_SANDBOX_EGRESS_PROXY`        | Optional proxy to steer container egress through (see below).                                                                                  | unset                             |

## Let the sandbox reach the server

Commands in the sandbox call the tracking server at the address the browser used to open the MLflow
UI. A loopback address (`localhost` or `127.0.0.1`) is rewritten to `host.docker.internal`, which
the server's default allowed hosts (its protection against DNS rebinding) do not include. So if you
open the UI at `localhost`, every `mlflow` command in the sandbox fails with
`403 Invalid Host header - possible DNS rebinding attack detected`, and the assistant cannot read
any MLflow data. Add `host.docker.internal` to the allowed hosts. `--allowed-hosts` (or
`MLFLOW_SERVER_ALLOWED_HOSTS`) replaces the default list, so also keep the defaults that your users
need: the loopback names, private IP addresses (`10.*`, `172.16.*` to `172.31.*`, `192.168.*`), and
the server's own name. For example:

```bash
mlflow server --host 0.0.0.0 \
  --allowed-hosts "localhost,localhost:*,127.0.0.1,127.0.0.1:*,host.docker.internal,host.docker.internal:*,10.*,192.168.*,mlflow.example.com,mlflow.example.com:*"
```

On Linux, `host.docker.internal` is the address of the Docker bridge, not the host's loopback
address, so a server that listens only on `127.0.0.1` (the default `--host`) cannot be reached from
the sandbox. Listen on an address the bridge can reach, for example `--host 0.0.0.0`.

When Docker runs in a virtual machine, as it does on macOS, that machine must also be able to see
the server's temporary folder (`TMPDIR`): the CLI sandbox passes its input to the container
through a folder there. Docker Desktop shares it by default. Colima, for example, shares only your
home folder by default, so set `TMPDIR` to a folder under your home folder before you start the
server.

## The `Bash` sandbox image

If `MLFLOW_SANDBOX_DOCKER_IMAGE` is not present locally, the server builds a minimal one when a
turn first needs it: a `python:<version>-slim` base image from Docker Hub with the server's Python
version, and MLflow installed to match the server. MLflow comes from `MLFLOW_HOME` if it is set,
and otherwise from the server's source checkout (a development build), from GitHub (a development
build without a checkout), or from the package index (a released version). This build does not
use `PIP_INDEX_URL` or `PIP_EXTRA_INDEX_URL`. If the server cannot reach Docker Hub or the public
package index, build an image with the server's Python and MLflow versions yourself, and set
`MLFLOW_SANDBOX_DOCKER_IMAGE` to it. The image is not rebuilt when MLflow changes: delete it to
rebuild it.

## Building the CLI image

The CLI sandbox image must contain the CLI(s) for the provider(s) you enable, their Node
runtime, and MLflow (so the CLI's own tools can call the `mlflow` CLI). It is intentionally
not auto-built — you control what goes in it. Install only the CLI(s) you use. A reference
starting point:

```dockerfile
FROM node:20-slim

# Python + MLflow so the CLI's tools can run `mlflow` commands inside the sandbox.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --break-system-packages mlflow

# The provider CLIs (provide the `claude` / `codex` binaries on PATH). Install whichever
# provider(s) you enable; both can coexist in one image.
RUN npm install -g @anthropic-ai/claude-code @openai/codex

# Mountpoint for the CLI's HOME. At runtime the server bind-mounts a private, server-owned
# directory here (holding --resume state), so this only needs to exist.
RUN mkdir -p /home/sandbox
```

Build and point the server at it:

```bash
docker build -t mlflow-assistant-sandbox:latest -f Dockerfile .
export MLFLOW_ASSISTANT_SANDBOX_CLI_IMAGE=mlflow-assistant-sandbox:latest
```

Pin the CLI package(s) to a known version for reproducibility. Provider credentials — for
Claude Code (`ANTHROPIC_API_KEY`, Bedrock/Vertex, or a proxy) and for Codex (`OPENAI_API_KEY`,
`OPENAI_BASE_URL`) — are read from the server's environment and forwarded into the container;
other host secrets are not.

## Egress

Sandbox containers run on a dedicated bridge network, which separates them from containers on
other Docker networks, and can still reach the tracking server on the host via
`host.docker.internal`. This does not limit what a sandbox can reach: every session's sandbox
shares this network, so sandboxes can reach each other, and through `host.docker.internal` a
sandbox can reach services that listen on the host's network interfaces, including ports that
other containers publish. When the sandbox is enabled, the server logs a warning at startup
that says so.

If that dedicated network cannot be created or accessed (a restricted Docker setup), the sandbox
logs a warning and falls back to Docker's default `bridge` network so it still runs — but on the
shared default bridge that isolation from other containers is reduced.

Set `MLFLOW_SANDBOX_EGRESS_PROXY` to steer HTTP(S) egress through an allowlisting proxy (only the
self-host bypass list — `host.docker.internal` and loopback — is excluded, so a co-located
tracking server stays reachable; a remote tracking host must be allowlisted in the proxy itself):

```bash
export MLFLOW_SANDBOX_EGRESS_PROXY=http://proxy.internal:3128
```

> **This is not a hard egress boundary.** The proxy env vars only affect clients that honor
> them, and the container still has network access — code that opens a raw socket, or a
> runtime that ignores the proxy env (for example Node's `fetch`, the runtime behind the
> `claude` and `codex` CLIs), can still reach other hosts, including the cloud metadata
> endpoint. For a
> real boundary, pair this with host-level firewall rules (or a Docker network with no route
> to the destinations you want to block). Do not treat the proxy alone as isolation.

## Cleanup

`$HOME` directories for CLI sessions live under the server's session directory and are reaped
when untouched for a day. Orphaned containers from a previous server run are removed on the
next startup (matched by a per-boot label, so a running server never removes another worker's
live container).

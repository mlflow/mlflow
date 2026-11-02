// Forwards Claude Code's OTLP traces, logs, and metrics to the Databricks OTel ingest endpoint, adding the
// bearer token and Unity Catalog table headers so the sandboxed agent never sees the token. Only
// the fixed routes are served, so the agent cannot use the token for any other API or table.
import http from "node:http";
import https from "node:https";

const MAX_BODY_BYTES = 8 * 1024 * 1024;

let token = "";
for await (const chunk of process.stdin) {
  token += chunk.toString("utf8");
  if (token.length > 16_384) throw new Error("Telemetry bearer is too large");
}
token = token.trimEnd();
if (!token) throw new Error("Telemetry bearer is missing");
delete process.env.TELEMETRY_TOKEN;

const host = new URL(process.env.DATABRICKS_HOST);
if (host.protocol !== "https:" || host.username || host.password || host.search || host.hash) {
  throw new Error("Invalid Databricks host");
}
// <catalog>.<schema>.<table_prefix> of a Unity Catalog trace location.
const location = process.env.TRACE_LOCATION ?? "";
if (!/^[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$/.test(location)) {
  throw new Error("Invalid trace location");
}

const routes = {
  "/v1/traces": { path: "/api/2.0/otel/v1/traces", table: `${location}_otel_spans` },
  "/v1/logs": { path: "/api/2.0/otel/v1/logs", table: `${location}_otel_logs` },
  "/v1/metrics": { path: "/api/2.0/otel/v1/metrics", table: `${location}_otel_metrics` },
};

function handle(request, response) {
  let local;
  try {
    local = new URL(request.url ?? "/", "http://localhost");
  } catch {
    response.writeHead(400).end();
    return;
  }
  if (local.pathname === "/healthz") {
    response.writeHead(200).end();
    return;
  }
  const route = routes[local.pathname];
  if (!route || request.method !== "POST") {
    response.writeHead(404).end();
    return;
  }

  const chunks = [];
  let size = 0;
  request.on("data", (chunk) => {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) {
      response.writeHead(413).end();
      request.destroy();
      return;
    }
    chunks.push(chunk);
  });
  request.on("end", () => {
    if (response.headersSent) return;
    const body = Buffer.concat(chunks);
    const upstream = new URL(route.path, host);
    const outgoing = https.request(
      upstream,
      {
        method: "POST",
        headers: {
          authorization: "Bearer " + token,
          "content-type": request.headers["content-type"] ?? "application/x-protobuf",
          "content-length": body.length,
          "x-databricks-uc-table-name": route.table,
        },
      },
      (incoming) => {
        process.stdout.write(
          `${JSON.stringify({ path: local.pathname, bytes: body.length, status: incoming.statusCode })}\n`,
        );
        response.writeHead(incoming.statusCode ?? 502, {
          "content-type": incoming.headers["content-type"] ?? "application/x-protobuf",
        });
        incoming.pipe(response);
      },
    );
    outgoing.setTimeout(60_000, () => outgoing.destroy(new Error("Telemetry upstream timeout")));
    outgoing.on("error", (err) => {
      process.stdout.write(`${JSON.stringify({ path: local.pathname, error: err.message })}\n`);
      if (response.headersSent) response.destroy();
      else response.writeHead(502).end();
    });
    outgoing.end(body);
  });
}

// TCP for exporters that go through the sandbox's HTTP proxy, and a Unix socket for those that
// connect directly: srt allows Unix sockets, and a relay in the sandbox forwards to it.
http.createServer(handle).listen(8081, "127.0.0.1");
if (process.env.TELEMETRY_SOCKET) {
  http.createServer(handle).listen(process.env.TELEMETRY_SOCKET);
}

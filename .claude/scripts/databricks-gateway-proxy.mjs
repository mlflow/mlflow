// Databricks counterpart to anthropic-proxy.mjs. Stdin carries three lines: the
// OAuth client ID, its secret, then a per-run client key callers must present
// as x-api-key. All three live only in this process's heap — never in
// /proc/<pid>/environ or argv. Upstream is a workspace AI Gateway, so this
// exchanges the client ID and secret for a short-lived bearer and refreshes it
// as it expires; no Anthropic credential exists in the chain.
//
// NOTE for reviewers and agents: in CI this proxy serves the Claude CLI's own
// API traffic on 127.0.0.1:8082. Do not kill, restart, or port-squat the
// running instance (e.g. `pkill -f databricks-gateway-proxy`) — that severs the
// review's own API connection. To experiment, run a copy with PORT=<other port>.
import http from "node:http";
import https from "node:https";
import { text } from "node:stream/consumers";

const PORT = Number(process.env.PORT ?? 8082);
// Parsed, not string-trimmed: Node's `host` option is a hostname and would take
// "example.com:443" as one.
const UPSTREAM = process.env.DATABRICKS_HOST ? new URL(process.env.DATABRICKS_HOST) : null;
// The CLI's base URL is `<host>/ai-gateway/anthropic`, so it sends `/v1/messages`
// and the prefix has to be restored here.
const UPSTREAM_PREFIX = "/ai-gateway/anthropic";
// All a review needs. Not a control on an attacker holding the secret, who can
// ask for any scope the principal is entitled to.
const SCOPE = "ai-gateway";
const REFRESH_SKEW_MS = 60_000;

const [CLIENT_ID, CLIENT_SECRET, CLIENT_KEY] = (await text(process.stdin))
  .split("\n")
  .map((s) => s.trim());
if (!CLIENT_ID || !CLIENT_SECRET || !CLIENT_KEY) {
  console.error(
    "databricks-gateway-proxy: expected client id, client secret, and client key on stdin"
  );
  process.exit(1);
}
if (!UPSTREAM) {
  console.error("databricks-gateway-proxy: DATABRICKS_HOST is required");
  process.exit(1);
}

const upstreamTarget = { hostname: UPSTREAM.hostname, port: UPSTREAM.port || 443 };

let cached = null; // { token, expiresAt }
let pending = null;

function mintToken() {
  const body = new URLSearchParams({ grant_type: "client_credentials", scope: SCOPE }).toString();
  const auth = Buffer.from(`${CLIENT_ID}:${CLIENT_SECRET}`).toString("base64");
  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        ...upstreamTarget,
        path: "/oidc/v1/token",
        method: "POST",
        headers: {
          authorization: `Basic ${auth}`,
          "content-type": "application/x-www-form-urlencoded",
          "content-length": Buffer.byteLength(body),
        },
      },
      async (res) => {
        const raw = await text(res);
        if (res.statusCode !== 200) {
          // The body carries the OAuth error code, the only way to tell a bad
          // secret from a missing workspace assignment.
          reject(new Error(`token endpoint returned ${res.statusCode}: ${raw}`));
          return;
        }
        let parsed;
        try {
          parsed = JSON.parse(raw);
        } catch {
          reject(new Error("token endpoint returned a non-JSON body"));
          return;
        }
        if (!parsed.access_token) {
          reject(new Error("token endpoint returned no access_token"));
          return;
        }
        // Lifetime is Databricks' to change, so log it rather than assume it.
        console.log(
          `databricks-gateway-proxy: minted gateway token, expires_in ${parsed.expires_in}s`
        );
        resolve({
          token: parsed.access_token,
          expiresAt: Date.now() + Number(parsed.expires_in ?? 3600) * 1000,
        });
      }
    );
    req.on("error", reject);
    req.end(body);
  });
}

function getToken() {
  if (cached && Date.now() < cached.expiresAt - REFRESH_SKEW_MS) {
    return Promise.resolve(cached.token);
  }
  // Collapse concurrent misses onto one exchange; Claude issues many parallel
  // requests and each would otherwise mint its own token.
  pending ??= mintToken()
    .then((fresh) => {
      cached = fresh;
      return fresh.token;
    })
    .finally(() => {
      pending = null;
    });
  return pending;
}

const server = http.createServer((req, res) => {
  // Log the path only: the log is dumped into the public workflow log, so
  // keep query strings out of it.
  console.log(`databricks-gateway-proxy: ${req.method} ${req.url.split("?")[0]}`);
  if (req.url === "/healthz" || req.url === "/api/hello") {
    res.writeHead(200).end("ok");
    return;
  }
  if (req.headers["x-api-key"] !== CLIENT_KEY) {
    res.writeHead(401).end("unauthorized");
    return;
  }
  if (!["GET", "POST"].includes(req.method) || !req.url.startsWith("/v1/")) {
    res.writeHead(403).end("forbidden");
    return;
  }
  // `req` stays paused until it is piped, so awaiting the token here buffers the
  // request body rather than dropping it.
  getToken().then(
    (token) => {
      const headers = { ...req.headers, host: UPSTREAM.host, authorization: `Bearer ${token}` };
      delete headers["x-api-key"];
      const upstream = https.request(
        { ...upstreamTarget, path: `${UPSTREAM_PREFIX}${req.url}`, method: req.method, headers },
        (up) => {
          // Node decodes chunked framing on the upstream response and re-frames
          // what we write, so drop hop-by-hop headers instead of echoing them.
          const { "transfer-encoding": _te, connection: _conn, ...respHeaders } = up.headers;
          res.writeHead(up.statusCode, respHeaders);
          up.pipe(res); // pipe to preserve SSE streaming
        }
      );
      upstream.on("error", (err) => {
        if (!res.headersSent) {
          res.writeHead(502);
        }
        res.end(`databricks-gateway-proxy: upstream error: ${err.message}`);
      });
      req.pipe(upstream);
    },
    (err) => {
      res.writeHead(502).end(`databricks-gateway-proxy: token exchange failed: ${err.message}`);
    }
  );
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`databricks-gateway-proxy: listening on 127.0.0.1:${PORT}`);
});

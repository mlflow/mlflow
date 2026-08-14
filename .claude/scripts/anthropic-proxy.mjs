// Forwarding proxy that holds the Anthropic API key so the Claude CLI process
// (and anything it spawns) never sees it. Stdin carries two lines: the real
// API key, then a per-run client key callers must present as x-api-key. Both
// live only in this process's heap — never in /proc/<pid>/environ or argv.
import http from "node:http";
import https from "node:https";
import { text } from "node:stream/consumers";

const PORT = 8082;

const [KEY, CLIENT_KEY] = (await text(process.stdin)).split("\n").map((s) => s.trim());
if (!KEY || !CLIENT_KEY) {
  console.error("anthropic-proxy: expected API key and client key on stdin");
  process.exit(1);
}

const server = http.createServer((req, res) => {
  console.log(`anthropic-proxy: ${req.method} ${req.url}`);
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
  const headers = { ...req.headers, host: "api.anthropic.com", "x-api-key": KEY };
  delete headers.authorization;
  const upstream = https.request(
    { host: "api.anthropic.com", path: req.url, method: req.method, headers },
    (up) => {
      res.writeHead(up.statusCode, up.headers);
      up.pipe(res); // pipe to preserve SSE streaming
    }
  );
  upstream.on("error", (err) => {
    if (!res.headersSent) {
      res.writeHead(502);
    }
    res.end(`anthropic-proxy: upstream error: ${err.message}`);
  });
  req.pipe(upstream);
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`anthropic-proxy: listening on 127.0.0.1:${PORT}`);
});

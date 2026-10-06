import http from "node:http";
import https from "node:https";

let token = "";
for await (const chunk of process.stdin) {
  token += chunk.toString("utf8");
  if (token.length > 16_384) throw new Error("Gateway bearer is too large");
}
token = token.trimEnd();
if (!token) throw new Error("Gateway bearer is missing");
delete process.env.GATEWAY_TOKEN;

const base = new URL(process.env.GATEWAY_URL);
if (base.protocol !== "https:" || base.username || base.password || base.search || base.hash) {
  throw new Error("Invalid gateway URL");
}

function forwardedHeaders(headers) {
  const result = { ...headers };
  for (const name of String(headers.connection ?? "").split(",")) {
    delete result[name.trim().toLowerCase()];
  }
  for (const name of [
    "connection",
    "proxy-connection",
    "keep-alive",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
  ]) {
    delete result[name];
  }
  return result;
}

const server = http.createServer((request, response) => {
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
  if (!local.pathname.startsWith("/anthropic/")) {
    response.writeHead(404).end();
    return;
  }

  const upstream = new URL(base);
  upstream.pathname = `${base.pathname.replace(/\/+$/, "")}${local.pathname}`;
  upstream.search = local.search;
  const headers = forwardedHeaders(request.headers);
  headers.host = upstream.host;
  headers.authorization = "Bearer " + token;

  const outgoing = https.request(upstream, { method: request.method, headers }, (incoming) => {
    response.writeHead(incoming.statusCode ?? 502, forwardedHeaders(incoming.headers));
    incoming.pipe(response);
  });
  outgoing.setTimeout(600_000, () => outgoing.destroy(new Error("Gateway timeout")));
  outgoing.on("error", () => {
    if (response.headersSent) response.destroy();
    else response.writeHead(502).end();
  });
  request.on("aborted", () => outgoing.destroy());
  response.on("close", () => {
    if (!response.writableFinished) outgoing.destroy();
  });
  request.pipe(outgoing);
});
server.requestTimeout = 0;
server.listen(8080, "127.0.0.1");

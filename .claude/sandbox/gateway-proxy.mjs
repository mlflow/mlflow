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

function accessTimestamp(date) {
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const pad = (value) => String(value).padStart(2, "0");
  const time = [date.getUTCHours(), date.getUTCMinutes(), date.getUTCSeconds()].map(pad).join(":");
  return `${pad(date.getUTCDate())}/${months[date.getUTCMonth()]}/${date.getUTCFullYear()}:${time} +0000`;
}

function quotedLogField(value) {
  return JSON.stringify(String(value ?? "-"));
}

const server = http.createServer((request, response) => {
  let bodyBytes = 0;
  let logged = false;
  const logAccess = () => {
    if (logged) return;
    logged = true;
    const status = response.writableFinished ? response.statusCode : 499;
    const requestLine = `${request.method} ${request.url} HTTP/${request.httpVersion}`;
    const address = request.socket.remoteAddress ?? "-";
    const fields = [
      `${address} - - [${accessTimestamp(new Date())}]`,
      quotedLogField(requestLine),
      status,
      bodyBytes,
      quotedLogField(request.headers.referer),
      quotedLogField(request.headers["user-agent"]),
    ];
    process.stdout.write(`${fields.join(" ")}\n`);
  };
  response.on("finish", logAccess);
  response.on("close", logAccess);

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
    incoming.on("data", (chunk) => {
      bodyBytes += chunk.length;
    });
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

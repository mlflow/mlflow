import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import http from "node:http";
import https from "node:https";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const proxyPath = fileURLToPath(new URL("./otel-proxy.mjs", import.meta.url));
const repositoryRoot = path.resolve(path.dirname(proxyPath), "../..");
const upstreamToken = "upstream-test-token";
const placeholderToken = "placeholder-test-token";
const localAuthorization = ["Bearer", placeholderToken].join(" ");
const upstreamAuthorization = ["Bearer", upstreamToken].join(" ");
const tablePrefix = "catalog.schema.mlflow";
const routes = {
  "/v1/traces": {
    path: "/api/2.0/otel/v1/traces",
    table: `${tablePrefix}_otel_spans`,
  },
  "/v1/logs": {
    path: "/api/2.0/otel/v1/logs",
    table: `${tablePrefix}_otel_logs`,
  },
  "/v1/metrics": {
    path: "/api/2.0/otel/v1/metrics",
    table: `${tablePrefix}_otel_metrics`,
  },
};

function requestProxy({ socketPath, method = "POST", pathname, headers = {}, body }) {
  return new Promise((resolve, reject) => {
    let responseResult;
    const request = http.request(
      { host: "127.0.0.1", port: 8081, socketPath, method, path: pathname, headers },
      (response) => {
        const chunks = [];
        responseResult = {
            status: response.statusCode,
            headers: response.headers,
            body: "",
          };
        response.on("data", (chunk) => chunks.push(chunk));
        response.on("end", () => {
          responseResult.body = Buffer.concat(chunks).toString();
          resolve(responseResult);
        });
      },
    );
    request.on("error", (error) => {
      if (responseResult) resolve(responseResult);
      else reject(error);
    });
    if (body) request.write(body);
    request.end();
  });
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function waitForProxy(child, stderr) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (child.exitCode !== null) {
      throw new Error(`Telemetry proxy exited during startup: ${stderr()}`);
    }
    try {
      const response = await requestProxy({ method: "GET", pathname: "/healthz" });
      if (response.status === 200) return;
    } catch {
      await delay(25);
    }
  }
  throw new Error(`Telemetry proxy did not become ready: ${stderr()}`);
}

async function stopProcess(child) {
  if (child.exitCode !== null) return;
  child.kill();
  await once(child, "exit");
}

test("telemetry proxy authenticates local callers and forwards fixed OTLP routes", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "mlflow-otel-proxy-"));
  const keyPath = path.join(directory, "key.pem");
  const certPath = path.join(directory, "cert.pem");
  const socketPath = path.join(directory, "otel.sock");
  execFileSync(
    "openssl",
    [
      "req",
      "-x509",
      "-newkey",
      "rsa:2048",
      "-nodes",
      "-keyout",
      keyPath,
      "-out",
      certPath,
      "-subj",
      "/CN=127.0.0.1",
      "-addext",
      "subjectAltName=IP:127.0.0.1",
      "-days",
      "1",
    ],
    { stdio: "ignore" },
  );
  const upstreamCalls = [];
  const upstream = https.createServer(
    { key: await readFile(keyPath), cert: await readFile(certPath) },
    (request, response) => {
      const chunks = [];
      request.on("data", (chunk) => chunks.push(chunk));
      request.on("end", () => {
        upstreamCalls.push({
          path: request.url,
          headers: request.headers,
          body: Buffer.concat(chunks),
        });
        response.writeHead(200, { "content-type": "application/x-protobuf" }).end("ok");
      });
    },
  );
  await new Promise((resolve) => upstream.listen(0, "127.0.0.1", resolve));
  const port = upstream.address().port;

  let stdout = "";
  let stderr = "";
  const proxy = spawn(process.execPath, [proxyPath], {
    env: {
      ...process.env,
      DATABRICKS_HOST: `https://127.0.0.1:${port}`,
      TRACE_LOCATION: tablePrefix,
      TELEMETRY_SOCKET: socketPath,
      NODE_EXTRA_CA_CERTS: certPath,
    },
    stdio: ["pipe", "pipe", "pipe"],
  });
  proxy.stdout.setEncoding("utf8").on("data", (chunk) => (stdout += chunk));
  proxy.stderr.setEncoding("utf8").on("data", (chunk) => (stderr += chunk));
  proxy.stdin.end(`${upstreamToken}\n${placeholderToken}\n`);

  try {
    await waitForProxy(proxy, () => stderr);
    await t.test("health checks do not require authorization on either listener", async () => {
      for (const target of [{}, { socketPath }]) {
        assert.equal((await requestProxy({ ...target, method: "GET", pathname: "/healthz" })).status, 200);
      }
    });

    await t.test("missing and incorrect authorization are rejected without forwarding", async () => {
      for (const target of [{}, { socketPath }]) {
        for (const headers of [{}, { authorization: ["Bearer", "wrong"].join(" ") }]) {
          const response = await requestProxy({ ...target, pathname: "/v1/traces", headers });
          assert.equal(response.status, 401);
          assert.equal(response.headers["www-authenticate"], "Bearer");
        }
      }
      assert.equal(upstreamCalls.length, 0);
    });

    await t.test("valid authorization forwards every signal over both listeners", async () => {
      for (const target of [{}, { socketPath }]) {
        for (const [pathname, route] of Object.entries(routes)) {
          const body = Buffer.from(`payload for ${pathname}`);
          const response = await requestProxy({
            ...target,
            pathname,
            headers: {
              authorization: localAuthorization,
              "content-type": "application/x-protobuf",
            },
            body,
          });
          assert.equal(response.status, 200);
          assert.equal(response.body, "ok");
          const forwarded = upstreamCalls.at(-1);
          assert.equal(forwarded.path, route.path);
          assert.equal(forwarded.headers.authorization, upstreamAuthorization);
          assert.equal(forwarded.headers["x-databricks-uc-table-name"], route.table);
          assert.equal(forwarded.headers["content-type"], "application/x-protobuf");
          assert.deepEqual(forwarded.body, body);
        }
      }
      assert.equal(upstreamCalls.length, 6);
    });

    await t.test("only POST requests to fixed routes are forwarded", async () => {
      const headers = { authorization: localAuthorization };
      assert.equal(
        (await requestProxy({ method: "GET", pathname: "/v1/traces", headers })).status,
        404,
      );
      assert.equal(
        (await requestProxy({ pathname: "/v1/other", headers })).status,
        404,
      );
      assert.equal(upstreamCalls.length, 6);
    });

    await t.test("request bodies larger than the limit are not forwarded", async () => {
      const response = await requestProxy({
        pathname: "/v1/traces",
        headers: {
          authorization: localAuthorization,
          "content-length": String(8 * 1024 * 1024 + 1),
        },
        body: Buffer.alloc(8 * 1024 * 1024 + 1),
      });
      assert.equal(response.status, 413);
      assert.equal(upstreamCalls.length, 6);
    });

    assert.equal(stdout.includes(upstreamToken), false);
    assert.equal(stdout.includes(placeholderToken), false);
  } finally {
    await stopProcess(proxy);
    await new Promise((resolve) => upstream.close(resolve));
    await rm(directory, { recursive: true, force: true });
  }
});

test("telemetry proxy rejects malformed stdin credentials", async () => {
  const cases = [
    ["", "Telemetry bearer is missing"],
    [`${upstreamToken}\n`, "Proxy bearer is missing"],
    [`${upstreamToken}\n${placeholderToken}\nextra\n`, "Unexpected telemetry proxy credentials"],
    ["x".repeat(16_385), "Telemetry proxy credentials are too large"],
  ];
  for (const [credentials, expectedError] of cases) {
    const child = spawn(process.execPath, [proxyPath], {
      env: process.env,
      stdio: ["pipe", "ignore", "pipe"],
    });
    let stderr = "";
    child.stderr.setEncoding("utf8").on("data", (chunk) => (stderr += chunk));
    child.stdin.end(credentials);
    const [exitCode] = await once(child, "exit");
    assert.notEqual(exitCode, 0);
    assert.match(stderr, new RegExp(expectedError));
  }
});

test("triage and nightly configure the common OTLP header only for the agent step", async () => {
  const header = "OTEL_EXPORTER_OTLP_HEADERS: ${{ steps.telemetry.outcome == 'success' && format('Authorization={0} {1}', 'Bearer', steps.telemetry-proxy.outputs.placeholder-token) || '' }}";
  for (const workflow of ["triage.yml", "nightly.yml"]) {
    const contents = await readFile(
      path.join(repositoryRoot, ".github", "workflows", workflow),
      "utf8",
    );
    assert.equal(contents.split(header).length - 1, 1);
    assert.match(contents, /OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: http:\/\/localhost:8081\/v1\/traces/);
    assert.match(contents, /OTEL_EXPORTER_OTLP_LOGS_ENDPOINT: http:\/\/localhost:8081\/v1\/logs/);
    assert.match(contents, /OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: http:\/\/localhost:8081\/v1\/metrics/);
  }
  const configuredHeader = (enabled, token) =>
    enabled ? `Authorization=${["Bearer", token].join(" ")}` : "";
  assert.equal(
    configuredHeader(true, placeholderToken),
    ["Authorization=", ["Bearer", placeholderToken].join(" ")].join(""),
  );
  assert.equal(configuredHeader(false, placeholderToken), "");
});

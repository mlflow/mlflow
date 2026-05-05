const { createProxyMiddleware } = require('http-proxy-middleware');

// eslint-disable-next-line
module.exports = function (app) {
  // The MLflow Gunicorn server is running on port 5000, so we should redirect server requests
  // (eg /ajax-api) to that port.
  // Exception: If the caller has specified an MLFLOW_PROXY, we instead forward server requests
  // there.
  // eslint-disable-next-line no-undef
  const proxyTarget = process.env.MLFLOW_PROXY || 'http://localhost:5000/';
  // eslint-disable-next-line no-undef
  const proxyStaticTarget = process.env.MLFLOW_STATIC_PROXY || proxyTarget;
  app.use(
    createProxyMiddleware('/ajax-api', {
      target: proxyTarget,
      changeOrigin: true,
    }),
  );
  // The legacy per-resource permission CRUD endpoints (e.g.
  // ``/api/2.0/mlflow/experiments/permissions/create``) are exposed under
  // ``/api`` rather than ``/ajax-api``. Without this proxy line, the dev
  // server returns its SPA fallback (200 + index.html) for those POSTs and
  // the frontend silently treats the grant as successful even though the
  // backend never saw the request.
  app.use(
    createProxyMiddleware('/api', {
      target: proxyTarget,
      changeOrigin: true,
    }),
  );
  app.use(
    createProxyMiddleware('/graphql', {
      target: proxyTarget,
      changeOrigin: true,
    }),
  );
  app.use(
    createProxyMiddleware('/get-artifact', {
      target: proxyStaticTarget,
      ws: true,
      changeOrigin: true,
    }),
  );
  app.use(
    createProxyMiddleware('/model-versions/get-artifact', {
      target: proxyStaticTarget,
      ws: true,
      changeOrigin: true,
    }),
  );
  app.use(
    createProxyMiddleware('/gateway', {
      target: proxyTarget,
      changeOrigin: true,
    }),
  );
};

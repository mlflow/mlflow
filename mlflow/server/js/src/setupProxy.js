const { createProxyMiddleware } = require('http-proxy-middleware');

// eslint-disable-next-line
module.exports = function (app) {
  // The MLflow Gunicorn server is running on port 5000, so we should redirect server requests
  // (eg /ajax-api) to that port.
  // Exception: If the caller has specified an MLFLOW_PROXY, we instead forward server requests
  // there.
  const proxyTarget = process.env.MLFLOW_PROXY || 'http://localhost:5000/';
  app.use(
    createProxyMiddleware('/ajax-api', {
      target: proxyTarget,
      changeOrigin: true,
    }),
  );
  app.use(
    createProxyMiddleware('/get-artifact', {
      target: proxyTarget,
      ws: true,
      changeOrigin: true,
    }),
  );
  app.use(
    createProxyMiddleware('/get-model-version-artifact', {
      target: proxyTarget,
      ws: true,
      changeOrigin: true,
    }),
  );
};

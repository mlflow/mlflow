const { createProxyMiddleware } = require('http-proxy-middleware');

// eslint-disable-next-line
module.exports = function(app) {
  // The MLflow Gunicorn server is running on port 5000, so we should redirect server requests
  // (eg /ajax-api) to that port.
  // Exception: If the caller has specified an MLFLOW_PROXY, we instead forward server requests
  // there.
  const proxyTarget = process.env.MLFLOW_PROXY || 'http://localhost:5000/';
  const proxyStaticTarget = process.env.MLFLOW_STATIC_PROXY || proxyTarget;
  const iamTarget = process.env.IAM_PROXY || 'http://localhost:5002/';
  app.use(
    createProxyMiddleware('/user-service/v1/entities-mapping/v1/iam-api', {
      target: iamTarget,
      changeOrigin: true,
      pathRewrite: {
        '^/user-service/v1/entities-mapping/v1/iam-api': '/',
      },
    }),
  );
  app.use(
    createProxyMiddleware('/user-service/v1/entities-mapping/v1/ajax-api', {
      target: proxyTarget,
      changeOrigin: true,
    }),
  );
  app.use(
createProxyMiddleware('/user-service/v1/entities-mapping/v1/get-artifact', {
      target: proxyStaticTarget,
      ws: true,
      changeOrigin: true,
    }),
  );
  app.use(
createProxyMiddleware('/user-service/v1/entities-mapping/v1/model-versions/get-artifact', {
      target: proxyStaticTarget,
      ws: true,
      changeOrigin: true,
    }),
  );
};

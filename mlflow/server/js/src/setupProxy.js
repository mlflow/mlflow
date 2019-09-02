const proxy = require('http-proxy-middleware');

function setupProxyFn(app) {
  app.use(proxy('/ajax-api', { target: 'http://localhost:5000' }));
  app.use(proxy('/get-artifact', { target: 'http://localhost:5000', ws: true }));
}

module.exports = setupProxyFn;

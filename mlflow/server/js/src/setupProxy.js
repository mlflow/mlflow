const proxy = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(proxy('/ajax-api', { target: 'http://localhost:5000/' }));
  app.use(proxy('/get-artifact', { target: 'http://localhost:5000/', ws: true }));
};
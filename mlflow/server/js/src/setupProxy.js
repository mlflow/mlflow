import proxy from 'http-proxy-middleware';

module.exports = (app) => {
    // Configured as described in https://stackoverflow.com/a/52620241
    // and https://www.npmjs.com/package/http-proxy-middleware#tldr
    app.use(proxy('/ajax-api',
        { target: 'http://localhost:5000' }
    ));
    app.use(proxy('/get-artifact',
        { target: 'http://localhost:5000', ws: true }
    ));
};

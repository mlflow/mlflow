class TraefikMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):

        script_name = environ.get("HTTP_X_FORWARDED_PREFIX", "")
        if script_name:
            environ["SCRIPT_NAME"] = script_name

        scheme = environ.get("HTTP_X_FORWARDED_PROTO", "")
        if scheme:
            environ["wsgi.url_scheme"] = scheme

        return self.app(environ, start_response)

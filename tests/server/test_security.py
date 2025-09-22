"""
Tests for MLflow server security middleware.
"""

import os
import pytest
from unittest import mock

from flask import Flask, Response
from werkzeug.test import Client

from mlflow.server.security import SecurityMiddleware, init_security_middleware


class TestSecurityMiddleware:
    def test_default_allowed_hosts(self):
        middleware = SecurityMiddleware()
        assert "localhost" in middleware.allowed_hosts
        assert "127.0.0.1" in middleware.allowed_hosts
        assert "[::1]" in middleware.allowed_hosts
        assert "localhost:5000" in middleware.allowed_hosts
        assert "127.0.0.1:5000" in middleware.allowed_hosts

    def test_custom_allowed_hosts(self):
        middleware = SecurityMiddleware(allowed_hosts=["example.com", "app.example.com"])
        assert "example.com" in middleware.allowed_hosts
        assert "app.example.com" in middleware.allowed_hosts
        assert "localhost" not in middleware.allowed_hosts

    def test_dns_rebinding_protection(self):
        app = Flask(__name__)
        middleware = SecurityMiddleware(
            allowed_hosts=["localhost", "127.0.0.1"],
            enable_host_validation=True,
        )

        @app.route("/test")
        def test_endpoint():
            return "OK"

        @app.before_request
        def before_request():
            from flask import request

            response = middleware.process_request(request)
            if response:
                return response

        client = Client(app)

        # Test with valid localhost host header
        response = client.get("/test", headers={"Host": "localhost"})
        assert response.status_code == 200

        # Test with valid IP host header
        response = client.get("/test", headers={"Host": "127.0.0.1"})
        assert response.status_code == 200

        # Test with malicious external host header (DNS rebinding attack)
        response = client.get("/test", headers={"Host": "evil.attacker.com"})
        assert response.status_code == 403
        assert b"Invalid Host header" in response.data

        # Test with missing Host header
        # Werkzeug test client will add localhost if HTTP_HOST is empty,
        # so we need to test differently
        with app.test_request_context("/test"):
            from flask import request

            # Create a request without Host header
            result = middleware.process_request(request)
            # Since werkzeug always adds a host, this test would pass
            # We'd need a real HTTP request without Host header to test this

    def test_cors_protection(self):
        app = Flask(__name__)
        middleware = SecurityMiddleware(
            allowed_origins=["http://localhost:3000", "https://app.example.com"]
        )

        @app.route("/api/test", methods=["POST"])
        def test_endpoint():
            return "OK"

        @app.before_request
        def before_request():
            from flask import request

            response = middleware.process_request(request)
            if response:
                return response

        @app.after_request
        def after_request(response):
            from flask import request

            return middleware.process_response(response, request)

        client = Client(app)

        # Test POST request with allowed origin
        response = client.post("/api/test", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

        # Test POST request with disallowed origin
        response = client.post("/api/test", headers={"Origin": "http://evil.com"})
        assert response.status_code == 403
        assert b"Cross-origin request blocked" in response.data

        # Test POST request without Origin header (same-origin)
        response = client.post("/api/test")
        assert response.status_code == 200

        # Test GET request (should not validate origin)
        response = client.get("/api/test", headers={"Origin": "http://evil.com"})
        assert response.status_code in [200, 405]  # 405 if GET not allowed

    def test_insecure_cors_mode(self):
        app = Flask(__name__)
        middleware = SecurityMiddleware(allow_insecure_cors=True)

        @app.route("/api/test", methods=["POST"])
        def test_endpoint():
            return "OK"

        @app.before_request
        def before_request():
            from flask import request

            response = middleware.process_request(request)
            if response:
                return response

        @app.after_request
        def after_request(response):
            from flask import request

            return middleware.process_response(response, request)

        client = Client(app)

        # Test that any origin is allowed in insecure mode
        response = client.post("/api/test", headers={"Origin": "http://evil.com"})
        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "*"

    def test_preflight_options_request(self):
        app = Flask(__name__)
        middleware = SecurityMiddleware(allowed_origins=["http://localhost:3000"])

        @app.route("/api/test", methods=["POST", "OPTIONS"])
        def test_endpoint():
            return "OK"

        @app.before_request
        def before_request():
            from flask import request

            response = middleware.process_request(request)
            if response:
                return response

        client = Client(app)

        # Test OPTIONS preflight with allowed origin
        response = client.options("/api/test", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 204
        assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
        assert "Access-Control-Allow-Methods" in response.headers

        # Test OPTIONS preflight with disallowed origin
        response = client.options("/api/test", headers={"Origin": "http://evil.com"})
        assert response.status_code == 200  # Flask default OPTIONS handler

    def test_security_headers(self):
        app = Flask(__name__)
        middleware = SecurityMiddleware()

        @app.route("/test")
        def test_endpoint():
            return "OK"

        @app.after_request
        def after_request(response):
            from flask import request

            return middleware.process_response(response, request)

        client = Client(app)

        response = client.get("/test")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"

    def test_health_endpoint_bypass(self):
        app = Flask(__name__)
        middleware = SecurityMiddleware(allowed_hosts=["localhost"])

        @app.route("/health")
        def health():
            return "OK"

        @app.before_request
        def before_request():
            from flask import request

            response = middleware.process_request(request)
            if response:
                return response

        client = Client(app)

        # Health endpoint should work even with invalid host header
        response = client.get("/health", headers={"Host": "evil.com"})
        assert response.status_code == 200

    def test_private_ip_detection(self):
        middleware = SecurityMiddleware()

        # Test private IPs
        assert middleware._is_private_ip("192.168.1.1")
        assert middleware._is_private_ip("10.0.0.1")
        assert middleware._is_private_ip("172.16.0.1")
        assert middleware._is_private_ip("127.0.0.1")
        assert middleware._is_private_ip("localhost")
        assert middleware._is_private_ip("::1")

        # Test with ports
        assert middleware._is_private_ip("192.168.1.1:8080")
        assert middleware._is_private_ip("[::1]:8080")

    @mock.patch.dict(os.environ, {"MLFLOW_CORS_ALLOWED_ORIGINS": "http://app1.com,http://app2.com"})
    def test_environment_variable_configuration(self):
        app = Flask(__name__)
        middleware = init_security_middleware(app)

        assert "http://app1.com" in middleware.allowed_origins
        assert "http://app2.com" in middleware.allowed_origins

    @mock.patch.dict(os.environ, {"MLFLOW_ALLOW_INSECURE_CORS": "true"})
    def test_insecure_mode_from_env(self):
        app = Flask(__name__)
        middleware = init_security_middleware(app)

        assert middleware.allow_insecure_cors is True

    @mock.patch.dict(os.environ, {"MLFLOW_HOST_HEADER_VALIDATION": "false"})
    def test_disabled_host_validation_from_env(self):
        app = Flask(__name__)
        middleware = init_security_middleware(app)

        assert middleware.enable_host_validation is False

import mlflow.models.docker_utils as U

import os
import xml.etree.ElementTree as ET
import multiprocessing
import traceback
import pytest


@pytest.mark.parametrize(
    ("kwargs"),
    [
        (
            dict(
                http_proxy="http://foo:8080",
                http_proxy_protocol="http",
                http_proxy_host="foo",
                http_proxy_port="8080",
            ),
        ),
        (
            dict(
                http_proxy="http://foo:8080",
                no_proxy="qux",
                http_proxy_protocol="http",
                http_proxy_host="foo",
                http_proxy_port="8080",
                non_proxy_hosts="qux",
            ),
        ),
        (
            dict(
                http_proxy="http://foo:8080",
                no_proxy="qux,quux",
                http_proxy_protocol="http",
                http_proxy_host="foo",
                http_proxy_port="8080",
                non_proxy_hosts="qux|quux",
            ),
        ),
        (
            dict(
                http_proxy="http://bar@foo:8080",
                http_proxy_protocol="http",
                http_proxy_host="foo",
                http_proxy_port="8080",
                http_proxy_user="bar",
            ),
        ),
        (
            dict(
                http_proxy="http://bar:baz@foo:8080",
                http_proxy_protocol="http",
                http_proxy_host="foo",
                http_proxy_port="8080",
                http_proxy_user="bar",
                http_proxy_password="baz",
            ),
        ),
        (
            dict(
                https_proxy="http://foo:8080",
                https_proxy_protocol="http",
                https_proxy_host="foo",
                https_proxy_port="8080",
            ),
        ),
        (
            dict(
                https_proxy="http://foo:8080",
                no_proxy="qux",
                https_proxy_protocol="http",
                https_proxy_host="foo",
                https_proxy_port="8080",
                non_proxy_hosts="qux",
            ),
        ),
        (
            dict(
                https_proxy="http://foo:8080",
                no_proxy="qux,quux",
                https_proxy_protocol="http",
                https_proxy_host="foo",
                https_proxy_port="8080",
                non_proxy_hosts="qux|quux",
            ),
        ),
        (
            dict(
                https_proxy="http://bar@foo:8080",
                https_proxy_protocol="http",
                https_proxy_host="foo",
                https_proxy_port="8080",
                https_proxy_user="bar",
            ),
        ),
        (
            dict(
                https_proxy="http://bar:baz@foo:8080",
                https_proxy_protocol="http",
                https_proxy_host="foo",
                https_proxy_port="8080",
                https_proxy_user="bar",
                https_proxy_password="baz",
            ),
        ),
        (
            dict(
                HTTP_PROXY="http://foo:8080",
                http_proxy_protocol="http",
                http_proxy_host="foo",
                http_proxy_port="8080",
            ),
        ),
        (
            dict(
                HTTP_PROXY="http://foo:8080",
                NO_PROXY="qux",
                http_proxy_protocol="http",
                http_proxy_host="foo",
                http_proxy_port="8080",
                non_proxy_hosts="qux",
            ),
        ),
        (
            dict(
                HTTP_PROXY="http://foo:8080",
                NO_PROXY="qux,quux",
                http_proxy_protocol="http",
                http_proxy_host="foo",
                http_proxy_port="8080",
                non_proxy_hosts="qux|quux",
            ),
        ),
        (
            dict(
                HTTP_PROXY="http://bar@foo:8080",
                http_proxy_protocol="http",
                http_proxy_host="foo",
                http_proxy_port="8080",
                http_proxy_user="bar",
            ),
        ),
        (
            dict(
                HTTP_PROXY="http://bar:baz@foo:8080",
                http_proxy_protocol="http",
                http_proxy_host="foo",
                http_proxy_port="8080",
                http_proxy_user="bar",
                http_proxy_password="baz",
            ),
        ),
        (
            dict(
                HTTPS_PROXY="http://foo:8080",
                https_proxy_protocol="http",
                https_proxy_host="foo",
                https_proxy_port="8080",
            ),
        ),
        (
            dict(
                HTTPS_PROXY="http://foo:8080",
                NO_PROXY="qux",
                https_proxy_protocol="http",
                https_proxy_host="foo",
                https_proxy_port="8080",
                non_proxy_hosts="qux",
            ),
        ),
        (
            dict(
                HTTPS_PROXY="http://foo:8080",
                NO_PROXY="qux,quux",
                https_proxy_protocol="http",
                https_proxy_host="foo",
                https_proxy_port="8080",
                non_proxy_hosts="qux|quux",
            ),
        ),
        (
            dict(
                HTTPS_PROXY="http://bar@foo:8080",
                https_proxy_protocol="http",
                https_proxy_host="foo",
                https_proxy_port="8080",
                https_proxy_user="bar",
            ),
        ),
        (
            dict(
                HTTPS_PROXY="http://bar:baz@foo:8080",
                https_proxy_protocol="http",
                https_proxy_host="foo",
                https_proxy_port="8080",
                https_proxy_user="bar",
                https_proxy_password="baz",
            ),
        ),
    ],
)
def test_get_maven_settings(kwargs):
    # Multiprocessing is used here so we can set env vars,
    # without risking breaking the actual environment,
    # and lets us test the logic of using both cases of variables
    q = multiprocessing.Queue()

    http_proxy_protocol = kwargs[0].get("http_proxy_protocol")
    http_proxy_host = kwargs[0].get("http_proxy_host")
    http_proxy_port = kwargs[0].get("http_proxy_port")
    http_proxy_user = kwargs[0].get("http_proxy_user")
    http_proxy_password = kwargs[0].get("http_proxy_password")
    https_proxy_protocol = kwargs[0].get("https_proxy_protocol")
    https_proxy_host = kwargs[0].get("https_proxy_host")
    https_proxy_port = kwargs[0].get("https_proxy_port")
    https_proxy_user = kwargs[0].get("https_proxy_user")
    https_proxy_password = kwargs[0].get("https_proxy_password")
    non_proxy_hosts = kwargs[0].get("non_proxy_hosts")

    def get(q, kwargs):
        http_proxy = kwargs[0].get("http_proxy")
        https_proxy = kwargs[0].get("https_proxy")
        no_proxy = kwargs[0].get("no_proxy")
        HTTP_PROXY = kwargs[0].get("HTTP_PROXY")
        HTTPS_PROXY = kwargs[0].get("HTTPS_PROXY")
        NO_PROXY = kwargs[0].get("NO_PROXY")

        def env(k, v):
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = v

        try:
            env("http_proxy", http_proxy)
            env("https_proxy", https_proxy)
            env("no_proxy", no_proxy)
            env("HTTP_PROXY", HTTP_PROXY)
            env("HTTPS_PROXY", HTTPS_PROXY)
            env("NO_PROXY", NO_PROXY)
            s = U.get_maven_settings()
        except:  # pylint: disable=bare-except
            q.put((False, traceback.format_exc()))
            return
        q.put((True, s))

    def assert_proxy(
        settings,
        proxy_id,
        expected_protocol,
        expected_host,
        expected_port,
        expected_user,
        expected_password,
        expected_non_proxy_hosts,
    ):
        proxy_node = list(settings.findall(f"./proxies/proxy/id[.='{proxy_id}']/.."))[0]

        protocol_node = list(proxy_node.findall("./protocol"))[0]
        host_node = list(proxy_node.findall("./host"))[0]
        port_node = list(proxy_node.findall("./port"))[0]

        protocol = protocol_node.text
        host = host_node.text
        port = port_node.text

        assert protocol == expected_protocol
        assert host == expected_host
        assert port == expected_port
        if expected_user is not None:
            user_node = list(proxy_node.findall("./username"))[0]
            user = user_node.text
            assert user == expected_user
        if expected_password is not None:
            password_node = list(proxy_node.findall("./password"))[0]
            password = password_node.text
            assert password == expected_password
        if expected_non_proxy_hosts is not None:
            non_proxy_hosts_node = list(proxy_node.findall("./nonProxyHosts"))[0]
            non_proxy_hosts = non_proxy_hosts_node.text
            assert non_proxy_hosts == expected_non_proxy_hosts

    p = multiprocessing.Process(target=get, args=(q, kwargs))
    p.start()
    p.join()
    assert not q.empty()
    success, result = q.get()
    # If success, result is the XML string, otherwise, it is the exception stack trace
    assert success, result
    settings_str = result
    settings = ET.fromstring(settings_str)
    if http_proxy_host is None and https_proxy_host is None:
        assert len(list(settings.iter("proxies"))) == 0
        return

    if http_proxy_host is not None:
        assert_proxy(
            settings,
            "http_proxy",
            http_proxy_protocol,
            http_proxy_host,
            http_proxy_port,
            http_proxy_user,
            http_proxy_password,
            non_proxy_hosts,
        )
    if https_proxy_host is not None:
        assert_proxy(
            settings,
            "https_proxy",
            https_proxy_protocol,
            https_proxy_host,
            https_proxy_port,
            https_proxy_user,
            https_proxy_password,
            non_proxy_hosts,
        )

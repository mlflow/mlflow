import contextlib
import socket
import subprocess
import sys
import time

import requests
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


def get_safe_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def server(port):
    with subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--directory", "build/html"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    ) as prc:
        try:
            for _ in range(5):
                try:
                    if requests.get(f"http://localhost:{port}").ok:
                        break
                except requests.exceptions.ConnectionError:
                    time.sleep(0.5)
            else:
                raise RuntimeError("Server did not start")

            yield
        finally:
            prc.terminate()


def main():
    port = get_safe_port()

    class Crawler(CrawlSpider):
        name = "broken-links"
        allowed_domains = ["localhost"]
        start_urls = [f"http://localhost:{port}/"]
        handle_httpstatus_list = [404]
        rules = (Rule(LinkExtractor(), callback="parse_item", follow=True),)
        links = set()

        def parse_item(self, response):
            if response.status == 404:
                self.links.add(
                    (
                        response.url,
                        response.request.headers.get("Referer", None).decode("utf-8"),
                    )
                )

    with server(port):
        process = CrawlerProcess(settings={"LOG_LEVEL": "ERROR"})
        process.crawl(Crawler)
        process.start()

    if Crawler.links:
        print("Broken links found:")
        for link, referer in Crawler.links:
            print(f"{link} in {referer}")
        sys.exit(1)


if __name__ == "__main__":
    main()

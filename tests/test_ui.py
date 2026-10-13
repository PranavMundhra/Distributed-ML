from html.parser import HTMLParser

from test_gateway import gateway_client


class LocalAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        path = attributes.get("src") if tag == "script" else attributes.get("href") if tag == "link" else None
        if path and path.startswith("/static/"):
            self.paths.add(path)


def test_gateway_serves_dashboard_and_referenced_assets_without_worker():
    def handler(request):
        raise AssertionError("Serving the UI must not require a live worker")

    with gateway_client(handler) as (client, _, _):
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        parser = LocalAssets()
        parser.feed(response.text)
        assert any(path.endswith(".js") for path in parser.paths)
        assert any(path.endswith(".css") for path in parser.paths)
        for path in parser.paths:
            asset = client.get(path)
            assert asset.status_code == 200
            assert asset.content
        assert client.get("/static/missing.js").status_code == 404

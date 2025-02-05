import os
from pathlib import Path

import requests


def fetch_changed_files(pr: str) -> list[Path]:
    pr_num = pr.rsplit("/", 1)[-1]
    url = f"https://api.github.com/repos/mlflow/mlflow/pulls/{pr_num}/files"
    per_page = 100
    changed_files = []
    for page in range(1, 100):
        r = requests.get(url, params={"page": page, "per_page": per_page})
        r.raise_for_status()
        files = r.json()
        changed_files.extend(f["filename"] for f in files)
        if len(files) < per_page:
            return [Path(f) for f in changed_files]


def main() -> None:
    pr = os.environ.get("CIRCLE_PULL_REQUEST")
    if pr is None:
        return

    BUILD_DIR = Path("_build/")
    DOCS_DIR = Path("docs/docs/")
    changed_pages: list[Path] = []
    for f in fetch_changed_files(pr):
        if f.suffix == ".mdx":
            path = (
                f.parent / "index.html"
                if f.name == "index.mdx"
                else f.parent / f.stem / "index.html"
            )
            changed_pages.append(path.relative_to(DOCS_DIR))

    links = "".join(f'<li><a href="{p}"><h2>{p}</h2></a></li>' for p in changed_pages)
    diff_html = f"""
<h1>Changed Pages</h1>
<ul>{links}</ul>
"""
    BUILD_DIR.mkdir(exist_ok=True)
    BUILD_DIR.joinpath("diff.html").write_text(diff_html)


if __name__ == "__main__":
    main()

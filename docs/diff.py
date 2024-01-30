import pathlib
import re
import subprocess


def main():
    SOURCE_REGEX = re.compile(r"<!-- source: (.+) -->")
    BUILD_DIR = pathlib.Path("build/html")
    changed_files = set(
        subprocess.run(
            ["git", "diff", "--name-only", "master"],
            text=True,
            capture_output=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )
    print(changed_files)
    changed_pages = set()
    for p in BUILD_DIR.rglob("**/*.html"):
        if m := SOURCE_REGEX.search(p.read_text()):
            source = m.group(1)
            if source in changed_files:
                changed_pages.add(p.relative_to(BUILD_DIR))
    print(changed_pages)

    links = "".join(f'<li><a href="{p}">{p}</a></li>' for p in changed_pages)
    diff_html = f"""
<h1>Changed Pages</h1>
<ul>{links}</ul>
"""
    BUILD_DIR.joinpath("diff.html").write_text(diff_html)


if __name__ == "__main__":
    main()

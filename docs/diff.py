import difflib
import pathlib
import shutil
import subprocess
import tempfile
import os


def main():
    with tempfile.TemporaryDirectory() as d:
        # master
        subprocess.run(["git", "checkout", "origin/master"], check=True)
        subprocess.run(["make", "rsthtml"], check=True)
        now = next(pathlib.Path(".").glob("build/html/**/*.html")).stat().st_mtime
        shutil.copytree("build/html", pathlib.Path(d) / "build/html")

        # pr
        pr_number = os.environ["CIRCLE_PR_NUMBER"]
        subprocess.run(["git", "fetch", "origin", f"refs/pull/{pr_number}/merge:pr"], check=True)
        subprocess.run(["git", "checkout", "pr"], check=True)
        subprocess.run(["git", "diff", "pr", "origin/master"], check=True)
        subprocess.run(["make", "rsthtml"], check=True)

        # diff
        hrefs = []
        for p in pathlib.Path(".").glob("build/html/**/*.html"):
            if p.stat().st_mtime > now:
                diff = list(
                    difflib.unified_diff(
                        p.read_text().splitlines(keepends=True),
                        (pathlib.Path(d) / p).read_text().splitlines(keepends=True),
                    )
                )
                if diff:
                    print("".join(diff))
                    hrefs.append(p.relative_to("build/html"))

        # generate html file for each rst file
        links = "\n".join(f"<a href='{href}'>{href}</a>" for href in hrefs)
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document</title>
</head>
<body>
  {links}
</body>
</html>
"""
        pathlib.Path("build/html/diff.html").write_text(html)


if __name__ == "__main__":
    main()

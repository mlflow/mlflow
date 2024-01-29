import difflib
import pathlib
import shutil
import subprocess
import tempfile


def main():
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["make", "rsthtml"], check=True)
        now = next(pathlib.Path(".").glob("build/html/**/*.html")).stat().st_mtime
        shutil.copytree("build/html", pathlib.Path(d) / "build/html")
        subprocess.run(["git", "checkout", "origin/master"], check=True)
        subprocess.run(["make", "rsthtml"], check=True)
        for p in pathlib.Path(".").glob("build/html/**/*.html"):
            if p.stat().st_mtime > now:
                diff = list(
                    difflib.unified_diff(
                        p.read_text().splitlines(keepends=True),
                        (pathlib.Path(d) / p).read_text().splitlines(keepends=True),
                    )
                )
                if diff:
                    print(p)


if __name__ == "__main__":
    main()

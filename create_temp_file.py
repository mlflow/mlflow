import os
import tempfile

for name in ["COM6", "foo"]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, name)

            with open(path, "w") as f:
                f.write("bar")
        print("succeeded")
    except Exception:
        print("failed")

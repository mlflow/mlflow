import socket

import pytest

SKILL_MD = "---\nname: demo\ndescription: Demo skill\n---\n# Demo\n"
BIG_FILE = b"\0" * 5000


@pytest.fixture
def skill_tree(tmp_path):
    root = tmp_path / "source"
    demo = root / "skills" / "demo"
    demo.mkdir(parents=True)
    (demo / "SKILL.md").write_text(SKILL_MD)
    (demo / "scripts").mkdir()
    (demo / "scripts" / "run.py").write_text("print('hi')\n")
    (root / "README.md").write_text("top-level readme\n")
    (root / "big.bin").write_bytes(BIG_FILE)
    return root


@pytest.fixture
def closed_port():
    # A port that was just bound and released, so connecting to it is refused immediately.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

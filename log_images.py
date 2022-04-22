import tempfile
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

import mlflow


class TemporaryDirectory(tempfile.TemporaryDirectory):
    def __enter__(self):
        return Path(super().__enter__())


url1 = "https://upload.wikimedia.org/wikipedia/commons/c/cc/ESC_large_ISS022_ISS022-E-11387-edit_01.JPG"
tall_image = "https://images.unsplash.com/photo-1587410131477-f01b22c59e1c?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8dGFsbCUyMHRvd2VyfGVufDB8fDB8fA%3D%3D&w=1000&q=80"
gif_url = "https://media.wired.jp/photos/61ce989a6e712a10a3de7e60/master/w_1600,c_limit/4c8da6532cd53ab03956208dd5ad4e24.gif"


def fetch_image(url):
    headers = {"User-Agent": "My User Agent 1.0", "From": "youremail@domain.com"}
    response = requests.get(url, headers=headers)
    return Image.open(BytesIO(response.content))


with TemporaryDirectory() as tmpdir:
    img1 = fetch_image(url1)
    for idx, ratio in enumerate([2.0, 1.0, 0.5, 0.2, 0.1]):
        w, h = img1.size
        width = int(w * ratio)
        height = int(h * ratio)
        resized = img1.resize((width, height), Image.LANCZOS)
        resized.save(tmpdir / f"{idx}_w_{width}_h_{height}_r_{ratio}.jpg")

    with open("test.gif", "wb") as f:
        f.write(requests.get(gif_url).content)

    with open("tall.jpeg", "wb") as f:
        f.write(requests.get(tall_image).content)

    with mlflow.start_run():
        mlflow.log_artifacts(tmpdir, artifact_path="images")
        mlflow.log_artifact("test.gif", artifact_path="images")
        mlflow.log_artifact("tall.jpeg", artifact_path="images")

import argparse
import math
import os
import requests


class MultipartUploadRestApi:
    def __init__(self, host, port, artifact_path, local_file):
        self.base_url = f"http://{host}:{port}/api/2.0/mlflow-artifacts"
        self.artifact_path = artifact_path
        self.local_file = local_file

    def create_multipart_upload(self, num_parts):
        r = requests.post(
            f"{self.base_url}/mpu/create/{self.artifact_path}",
            json={
                "path": self.local_file,
                "num_parts": num_parts,
            }
        )
        r.raise_for_status()
        create = r.json()
        return create["upload_id"], create["credentials"]

    def complete_multipart_upload(self, upload_id, parts):
        r = requests.post(
            f"{self.base_url}/mpu/complete/{self.artifact_path}",
            json={
                "path": self.local_file,
                "upload_id": upload_id,
                "parts": parts,
            }
        )
        r.raise_for_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--artifact-path", type=str)
    parser.add_argument("--local-file", type=str)
    args = parser.parse_args()

    file_size = os.path.getsize(args.local_file)
    chunk_size = 5 * 1024 * 1024 + 1  # S3 requires part size >= 5 MiB
    num_parts = math.ceil(file_size / chunk_size)

    api = MultipartUploadRestApi(
        args.host,
        args.port,
        args.artifact_path,
        args.local_file,
    )
    upload_id, credentials = api.create_multipart_upload(num_parts)
    print("Created multipart upload")
    print("upload_id:", upload_id)
    print("credentials:", credentials)

    parts = []
    for i, credential in enumerate(credentials):
        print(f"Uploading part #{i + 1}")
        with open(args.local_file, "rb") as f:
            f.seek(i * chunk_size)
            chunk = f.read(chunk_size)

        response = requests.put(credential["url"], data=chunk)
        response.raise_for_status()
        parts.append({
            "part_number": credential["part_number"],
            "etag": response.headers["ETag"],
        })

    print("Uploaded all parts:", parts)

    api.complete_multipart_upload(upload_id, parts)
    print("Multipart upload completed")

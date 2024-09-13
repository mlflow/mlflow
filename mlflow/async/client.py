"""
python mlflow/async/client.py
"""
import contextlib
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from pydantic import BaseModel


class Tag(BaseModel):
    key: str
    value: str


class Experiment(BaseModel):
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str
    last_update_time: int
    creation_time: int
    tags: List[Tag]


class RunInfo(BaseModel):
    run_id: str
    run_uuid: str
    experiment_id: str
    run_name: str
    status: str
    start_time: int
    end_time: int
    artifact_uri: str
    lifecycle_stage: str


class RunData(BaseModel):
    tags: List[Tag]


class Run(BaseModel):
    info: RunInfo
    data: RunData


class Header(BaseModel):
    name: str
    value: str


class CredentialInfo(BaseModel):
    run_id: str
    path: str
    signed_uri: str
    headers: List[Header]
    type: str


class AsyncMlflowClient:
    def __init__(self, host: str, token: str):
        self.host = host
        self.token = token
        self.session: aiohttp.ClientSession = None
        self.api_prefix = "/api/2.0/mlflow"

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def _request(self, method: str, path: str, headers: Optional[Dict] = None, **kwargs):
        print(f"Request: {method} {path}")
        async with self.session.request(
            method,
            f"{self.host}{self.api_prefix}{path}",
            headers={"Authorization": f"Bearer {self.token}", **(headers or {})},
            **kwargs,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def create_experiment(self, name: str) -> Experiment:
        resp = await self._request("POST", "/experiments/create", json={"name": name})
        return Experiment(**resp)

    async def get_or_create_experiment(self, name: str) -> Experiment:
        try:
            return await self.create_experiment(name)
        except aiohttp.ClientResponseError as e:
            if e.status == 400:
                return await self.get_experiment_by_name(name)
            raise

    async def get_experiment_by_name(self, name: str) -> Experiment:
        resp = await self._request(
            "GET", "/experiments/get-by-name", params={"experiment_name": name}
        )
        return Experiment(**resp["experiment"])

    async def create_run(
        self,
        experiment_id: str,
        user_id: Optional[str] = None,
        start_time: Optional[int] = None,
        tags: Optional[List[Tag]] = None,
        run_name: Optional[str] = None,
    ):
        tags = tags or []
        start_time = start_time or int(time.time() * 1000)
        resp = await self._request(
            "POST",
            "/runs/create",
            json={
                "experiment_id": experiment_id,
                "user_id": user_id,
                "start_time": start_time,
                "tags": tags,
                "run_name": run_name,
            },
        )
        return Run(**resp["run"])

    async def end_run(self, run_id: str, end_time: Optional[int] = None):
        end_time = end_time or int(time.time() * 1000)
        await self._request(
            "POST",
            "/runs/update",
            json={"run_id": run_id, "end_time": end_time},
        )

    async def get_run(self, run_id: str) -> Run:
        resp = await self._request("GET", "/runs/get", params={"run_id": run_id})
        return Run(**resp["run"])

    @contextlib.asynccontextmanager
    async def start_run(self, experiment_id: str, run_name: Optional[str] = None) -> Run:
        run = await self.create_run(experiment_id, run_name=run_name)
        try:
            yield run
        finally:
            await self.end_run(run.info.run_id)

    async def log_batch(
        self,
        run_id: str,
        metrics: Optional[List[Dict]] = None,
        params: Optional[List[Dict]] = None,
        tags: Optional[List[Dict]] = None,
    ) -> None:
        await self._request(
            "POST",
            "/runs/log-batch",
            json={
                "run_id": run_id,
                "metrics": metrics or [],
                "params": params or [],
                "tags": tags or [],
            },
        )

    async def log_metrics(
        self,
        run_id: str,
        metrics: Dict,
        step: Optional[int] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        step = step or 0
        timestamp = timestamp or int(time.time() * 1000)
        await self.log_batch(
            run_id,
            metrics=[
                {
                    "key": key,
                    "value": value,
                    "timestamp": timestamp,
                    "step": step,
                }
                for key, value in metrics.items()
            ],
        )

    async def log_params(
        self,
        run_id: str,
        params: Dict,
    ) -> None:
        await self.log_batch(
            run_id,
            params=[
                {
                    "key": key,
                    "value": value,
                }
                for key, value in params.items()
            ],
        )

    async def set_tags(
        self,
        run_id: str,
        tags: Dict,
    ) -> None:
        await self.log_batch(
            run_id,
            tags=[
                {
                    "key": key,
                    "value": value,
                }
                for key, value in tags.items()
            ],
        )

    async def _upload_artifact(self, credential_info: CredentialInfo, local_path: Path) -> None:
        resp = await self.session.put(
            credential_info.signed_uri,
            headers={h.name: h.value for h in credential_info.headers},
            data=local_path.read_bytes(),
        )
        resp.raise_for_status()

    async def log_artifacts(
        self, run_id: str, local_dir: str, artifact_path: Optional[str] = None
    ) -> None:
        local_dir = Path(local_dir).resolve()
        local_paths = [p.resolve() for p in local_dir.rglob("*") if p.is_file()]
        rel_paths = [p.relative_to(local_dir) for p in local_paths]
        if artifact_path:
            rel_paths = [f"{artifact_path}/{p}" for p in rel_paths]

        resp = await self._request(
            "POST",
            "/artifacts/credentials-for-write",
            json={"run_id": run_id, "path": rel_paths},
        )
        credential_infos = [CredentialInfo(**ci) for ci in resp["credential_infos"]]

        jobs = []
        for path, credential_info in zip(local_paths, credential_infos):
            jobs.append(self._upload_artifact(credential_info, local_path=path))

        await asyncio.gather(*jobs)

    async def log_artifact(self, run_id: str, local_path: str, artifact_path: Optional[str] = None):
        local_path = Path(local_path)
        rel_path = local_path.relative_to(local_path.parent)
        if artifact_path:
            rel_path = f"{artifact_path}/{rel_path}"

        resp = await self._request(
            "POST",
            "/artifacts/credentials-for-write",
            json={"run_id": run_id, "path": [str(rel_path)]},
        )
        credential_info = CredentialInfo(**resp["credential_infos"][0])
        await self._upload_artifact(credential_info, local_path)

    async def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            splits = artifact_file.rsplit("/", 1)
            if len(splits) == 2:
                artifact_path, artifact_name = splits
            else:
                artifact_path = None
                artifact_name = splits[0]

            tmp_file = Path(tmpdir).joinpath(artifact_name)
            tmp_file.write_text(text)

            await self.log_artifact(run_id, tmp_file, artifact_path)

    async def search_experiments(
        self,
        view_type: str = "ACTIVE_ONLY",
        max_results: int = 100,
        filter_string: Optional[str] = None,
    ) -> List[Run]:
        resp = await self._request(
            "POST",
            "/experiments/search",
            params={
                "view_type": view_type,
                "max_results": max_results,
                "filter": filter_string or "",
            },
        )
        return [Experiment(**run) for run in resp.get("experiments", [])]


@contextlib.contextmanager
def timer():
    start = time.time()
    try:
        yield
    finally:
        print(f"Took {time.time() - start:.2f} seconds")


def load_host_and_token():
    host = None
    token = None
    for l in (Path.home() / ".databrickscfg").read_text().splitlines():
        if l.startswith("host"):
            host = l.split("=")[1].strip()
        if l.startswith("token"):
            token = l.split("=")[1].strip()
    assert host is not None
    assert token is not None
    return host, token


async def main():
    host, token = load_host_and_token()
    name = "/Users/harutaka.kawamura@databricks.com/async-mlflow"
    async with AsyncMlflowClient(host, token) as client:
        experiment = await client.get_or_create_experiment(name)
        async with client.start_run(experiment.experiment_id) as run:
            with timer():
                # Parallelize logging operations
                await asyncio.gather(
                    client.log_params(run.info.run_id, {"param1": "value1"}),
                    client.log_metrics(run.info.run_id, {"accuracy": 0.9}),
                    client.set_tags(run.info.run_id, {"tag1": "value1"}),
                    client.log_artifact(run.info.run_id, __file__),
                    client.log_artifacts(run.info.run_id, Path(__file__).parent, "dir"),
                    client.log_text(run.info.run_id, "Hello, world!", "test.txt"),
                )

        run = await client.get_run(run.info.run_id)

        with timer():
            # Search experiments with different conditions
            experiments = sum(
                await asyncio.gather(
                    client.search_experiments(max_results=3),
                    client.search_experiments(max_results=2),
                    client.search_experiments(max_results=1),
                ),
                [],
            )
            print(len(experiments))

        with timer():
            exps1 = await client.search_experiments(max_results=3)
            exps2 = await client.search_experiments(max_results=2)
            exps3 = await client.search_experiments(max_results=1)
            experiments = exps1 + exps2 + exps3
            print(len(experiments))

    # Do the same thing with mlflow
    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(name)
    with mlflow.start_run() as run:
        with timer():
            mlflow.log_params({"param1": "value1"})
            mlflow.log_metrics({"accuracy": 0.9})
            mlflow.set_tags({"tag1": "value1"})
            mlflow.log_artifact(__file__)
            mlflow.log_artifacts("mlflow/async", "dir")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

import platform
import shutil
import subprocess
import tempfile
import textwrap
import urllib.request
import zipfile
from pathlib import Path
from typing import Literal

SYSTEM = platform.system()
MACHINE = platform.machine()
CACHE_DIR = Path(".cache/protobuf_cache")
REPO_ROOT = Path(".")  # Use repo root as base for proto includes


def gen_protos(
    proto_dir: Path,
    proto_files: list[Path],
    lang: Literal["python", "java"],
    protoc_bin: Path,
    protoc_include_paths: list[Path],
    out_dir: Path,
) -> None:
    assert lang in ["python", "java"]
    out_dir.mkdir(parents=True, exist_ok=True)

    include_args = []
    for include_path in protoc_include_paths:
        include_args.append(f"-I={include_path}")

    subprocess.check_call(
        [
            protoc_bin,
            *include_args,
            f"-I={proto_dir}",
            f"--{lang}_out={out_dir}",
            *proto_files,  # proto_files now contain full paths from proto_dir
        ]
    )


def gen_stub_files(
    proto_dir: Path,
    proto_files: list[Path],
    protoc_bin: Path,
    protoc_include_paths: list[Path],
    out_dir: Path,
) -> None:
    include_args = []
    for include_path in protoc_include_paths:
        include_args.append(f"-I={include_path}")

    subprocess.check_call(
        [
            protoc_bin,
            *include_args,
            f"-I={proto_dir}",
            f"--pyi_out={out_dir}",
            *proto_files,  # proto_files now contain full paths from proto_dir
        ]
    )


def apply_python_gencode_replacement(file_path: Path) -> None:
    content = file_path.read_text()

    for old, new in python_gencode_replacements:
        content = content.replace(old, new)

    file_path.write_text(content, encoding="UTF-8")


def _get_python_output_path(proto_file_path: Path) -> Path:
    return proto_file_path.parent / (proto_file_path.stem + "_pb2.py")


def to_paths(*args: str) -> list[Path]:
    return list(map(Path, args))


basic_proto_files = to_paths(
    "mlflow/protos/databricks.proto",
    "mlflow/protos/service.proto",
    "mlflow/protos/model_registry.proto",
    "mlflow/protos/databricks_artifacts.proto",
    "mlflow/protos/mlflow_artifacts.proto",
    "mlflow/protos/internal.proto",
    "mlflow/protos/scalapb/scalapb.proto",
    "mlflow/protos/assessments.proto",
    "mlflow/protos/datasets.proto",
    "mlflow/protos/webhooks.proto",
)
uc_proto_files = to_paths(
    "mlflow/protos/databricks_managed_catalog_messages.proto",
    "mlflow/protos/databricks_managed_catalog_service.proto",
    "mlflow/protos/databricks_uc_registry_messages.proto",
    "mlflow/protos/databricks_uc_registry_service.proto",
    "mlflow/protos/databricks_filesystem_service.proto",
    "mlflow/protos/unity_catalog_oss_messages.proto",
    "mlflow/protos/unity_catalog_oss_service.proto",
    "mlflow/protos/unity_catalog_prompt_messages.proto",
    "mlflow/protos/unity_catalog_prompt_service.proto",
)
tracing_proto_files = to_paths("mlflow/protos/databricks_tracing.proto")
facet_proto_files = to_paths("mlflow/protos/facet_feature_statistics.proto")
python_proto_files = basic_proto_files + uc_proto_files + facet_proto_files + tracing_proto_files
test_proto_files = to_paths("tests/protos/test_message.proto")


python_gencode_replacements = [
    # Replace absolute imports with relative imports within mlflow.protos package
    (
        "from mlflow.protos.scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2",
        "from .scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2",
    ),
    (
        "from mlflow.protos import databricks_pb2 as databricks__pb2",
        "from . import databricks_pb2 as databricks__pb2",
    ),
    (
        "from mlflow.protos import databricks_uc_registry_messages_pb2 as databricks__uc__registry__messages__pb2",  # noqa: E501
        "from . import databricks_uc_registry_messages_pb2 as databricks_uc_registry_messages_pb2",
    ),
    (
        "from mlflow.protos import databricks_managed_catalog_messages_pb2 as databricks__managed__catalog__"  # noqa: E501
        "messages__pb2",
        "from . import databricks_managed_catalog_messages_pb2 as databricks_managed_"
        "catalog_messages_pb2",
    ),
    (
        "from mlflow.protos import unity_catalog_oss_messages_pb2 as unity__catalog__oss__messages__pb2",  # noqa: E501
        "from . import unity_catalog_oss_messages_pb2 as unity_catalog_oss_messages_pb2",
    ),
    (
        "from mlflow.protos import unity_catalog_prompt_messages_pb2 as unity__catalog__prompt__messages__pb2",  # noqa: E501
        "from . import unity_catalog_prompt_messages_pb2 as unity_catalog_prompt_messages_pb2",
    ),
    (
        "from mlflow.protos import service_pb2 as service__pb2",
        "from . import service_pb2 as service__pb2",
    ),
    (
        "from mlflow.protos import assessments_pb2 as assessments__pb2",
        "from . import assessments_pb2 as assessments__pb2",
    ),
    (
        "from mlflow.protos import datasets_pb2 as datasets__pb2",
        "from . import datasets_pb2 as datasets__pb2",
    ),
    (
        "from mlflow.protos import webhooks_pb2 as webhooks__pb2",
        "from . import webhooks_pb2 as webhooks__pb2",
    ),
]


def gen_python_protos(protoc_bin: Path, protoc_include_paths: list[Path], out_dir: Path) -> None:
    # Generate Python code for both MLflow and test protos using repo root as -I directory
    all_proto_files = python_proto_files + test_proto_files
    gen_protos(
        REPO_ROOT,
        all_proto_files,
        "python",
        protoc_bin,
        protoc_include_paths,
        out_dir,
    )

    for proto_file in python_proto_files:
        apply_python_gencode_replacement(out_dir / _get_python_output_path(proto_file))


def download_file(url: str, output_path: Path) -> None:
    urllib.request.urlretrieve(url, output_path)


def download_opentelemetry_protos(version: str = "v1.7.0") -> Path:
    """
    Download OpenTelemetry proto files from GitHub.
    Returns the path to the opentelemetry-proto directory.
    """
    otel_proto_dir = CACHE_DIR / f"opentelemetry-proto-{version}"

    if not otel_proto_dir.exists():
        print(f"Downloading OpenTelemetry proto files {version}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "otel-proto.zip"
            download_file(
                f"https://github.com/open-telemetry/opentelemetry-proto/archive/refs/tags/{version}.zip",
                zip_path,
            )
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            # Move the extracted directory to cache
            extracted_dir = Path(tmpdir) / f"opentelemetry-proto-{version[1:]}"  # Remove 'v' prefix
            shutil.move(str(extracted_dir), str(otel_proto_dir))

    return otel_proto_dir


def download_and_extract_protoc(version: Literal["3.19.4", "26.0"]) -> tuple[Path, Path]:
    """
    Download and extract specific version protoc tool for Linux systems,
    return extracted protoc executable file path and include path.
    """
    assert SYSTEM == "Linux", "This script only supports Linux systems."
    assert MACHINE in ["x86_64", "aarch64"], (
        "This script only supports x86_64 or aarch64 CPU architectures."
    )

    cpu_type = "x86_64" if MACHINE == "x86_64" else "aarch_64"
    protoc_zip_filename = f"protoc-{version}-linux-{cpu_type}.zip"

    downloaded_protoc_bin = CACHE_DIR / f"protoc-{version}" / "bin" / "protoc"
    downloaded_protoc_include_path = CACHE_DIR / f"protoc-{version}" / "include"
    if not (downloaded_protoc_bin.is_file() and downloaded_protoc_include_path.is_dir()):
        with tempfile.TemporaryDirectory() as t:
            zip_path = Path(t) / protoc_zip_filename
            download_file(
                f"https://github.com/protocolbuffers/protobuf/releases/download/v{version}/{protoc_zip_filename}",
                zip_path,
            )
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(CACHE_DIR / f"protoc-{version}")

        # Make protoc executable
        downloaded_protoc_bin.chmod(0o755)
    return downloaded_protoc_bin, downloaded_protoc_include_path


def generate_final_python_gencode(
    gencode3194_path: Path, gencode5260_path: Path, out_path: Path
) -> None:
    gencode3194 = gencode3194_path.read_text()
    gencode5260 = gencode5260_path.read_text()

    merged_code = f"""
import google.protobuf
from packaging.version import Version
if Version(google.protobuf.__version__).major >= 5:
{textwrap.indent(gencode5260, "  ")}
else:
{textwrap.indent(gencode3194, "  ")}
"""
    out_path.write_text(merged_code, encoding="UTF-8")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_gencode_dir:
        temp_gencode_path = Path(temp_gencode_dir)
        proto3194_out = temp_gencode_path / "3.19.4"
        proto5260_out = temp_gencode_path / "26.0"
        proto3194_out.mkdir(exist_ok=True)
        proto5260_out.mkdir(exist_ok=True)

        protoc3194, protoc3194_include = download_and_extract_protoc("3.19.4")
        protoc5260, protoc5260_include = download_and_extract_protoc("26.0")

        # Download OpenTelemetry proto files
        otel_proto_dir = download_opentelemetry_protos()

        # Build include paths list
        protoc3194_includes = [protoc3194_include, otel_proto_dir]
        protoc5260_includes = [protoc5260_include, otel_proto_dir]

        gen_python_protos(protoc3194, protoc3194_includes, proto3194_out)
        gen_python_protos(protoc5260, protoc5260_includes, proto5260_out)

        # Merge generated code from both protoc versions
        for proto_file in python_proto_files + test_proto_files:
            gencode_path = _get_python_output_path(proto_file)

            generate_final_python_gencode(
                proto3194_out / gencode_path,
                proto5260_out / gencode_path,
                gencode_path,  # Output path already includes full path from repo root
            )

    # generate java gencode using pinned protoc 3.19.4 version.
    gen_protos(
        REPO_ROOT,
        basic_proto_files,
        "java",
        protoc3194,
        protoc3194_includes,
        Path("mlflow/java/client/src/main/java"),
    )

    gen_stub_files(
        REPO_ROOT,
        python_proto_files,
        protoc5260,
        protoc5260_includes,
        REPO_ROOT,
    )


if __name__ == "__main__":
    main()

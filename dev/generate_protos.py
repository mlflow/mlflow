import platform
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Literal

SYSTEM = platform.system()
MACHINE = platform.machine()
CACHE_DIR = Path(".cache/protobuf_cache")
MLFLOW_PROTOS_DIR = Path("mlflow/protos")
TEST_PROTOS_DIR = Path("tests/protos")


def gen_protos(
    proto_dir: Path,
    proto_files: list[str],
    lang: Literal["python", "java"],
    protoc_bin: Path,
    protoc_include_path: Path,
    out_dir: Path,
) -> None:
    assert lang in ["python", "java"]
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(
        [
            protoc_bin,
            f"-I={protoc_include_path}",
            f"-I={proto_dir}",
            f"--{lang}_out={out_dir}",
            *[f"{proto_dir}/{proto_file}" for proto_file in proto_files],
        ]
    )


def apply_python_gencode_replacement(file_path: Path) -> None:
    content = file_path.read_text()

    for old, new in python_gencode_replacements:
        content = content.replace(old, new)

    file_path.write_text(content, encoding="UTF-8")


def _get_python_output_filename(proto_file_name: str) -> Path:
    file_path = Path(proto_file_name)
    return file_path.parent / (Path(file_path.name).stem + "_pb2.py")


basic_proto_files = [
    "databricks.proto",
    "service.proto",
    "model_registry.proto",
    "databricks_artifacts.proto",
    "mlflow_artifacts.proto",
    "internal.proto",
    "scalapb/scalapb.proto",
    "assessments.proto",
]
uc_proto_files = [
    "databricks_managed_catalog_messages.proto",
    "databricks_managed_catalog_service.proto",
    "databricks_uc_registry_messages.proto",
    "databricks_uc_registry_service.proto",
    "databricks_filesystem_service.proto",
    "unity_catalog_oss_messages.proto",
    "unity_catalog_oss_service.proto",
    "unity_catalog_prompt_messages.proto",
    "unity_catalog_prompt_service.proto",
]
tracing_proto_files = [
    "databricks_trace_server.proto",
]
facet_proto_files = ["facet_feature_statistics.proto"]
python_proto_files = basic_proto_files + uc_proto_files + facet_proto_files + tracing_proto_files
test_proto_files = ["test_message.proto"]


python_gencode_replacements = [
    (
        "from scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2",
        "from .scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2",
    ),
    (
        "import databricks_pb2 as databricks__pb2",
        "from . import databricks_pb2 as databricks__pb2",
    ),
    (
        "import databricks_uc_registry_messages_pb2 as databricks__uc__registry__messages__pb2",
        "from . import databricks_uc_registry_messages_pb2 as databricks_uc_registry_messages_pb2",
    ),
    (
        "import databricks_managed_catalog_messages_pb2 as databricks__managed__catalog__"
        "messages__pb2",
        "from . import databricks_managed_catalog_messages_pb2 as databricks_managed_"
        "catalog_messages_pb2",
    ),
    (
        "import unity_catalog_oss_messages_pb2 as unity__catalog__oss__messages__pb2",
        "from . import unity_catalog_oss_messages_pb2 as unity_catalog_oss_messages_pb2",
    ),
    (
        "import unity_catalog_prompt_messages_pb2 as unity__catalog__prompt__messages__pb2",
        "from . import unity_catalog_prompt_messages_pb2 as unity_catalog_prompt_messages_pb2",
    ),
    (
        "import service_pb2 as service__pb2",
        "from . import service_pb2 as service__pb2",
    ),
    (
        "import assessments_pb2 as assessments__pb2",
        "from . import assessments_pb2 as assessments__pb2",
    ),
]


def gen_python_protos(protoc_bin: Path, protoc_include_path: Path, out_dir: Path) -> None:
    gen_protos(
        MLFLOW_PROTOS_DIR,
        python_proto_files,
        "python",
        protoc_bin,
        protoc_include_path,
        out_dir,
    )

    gen_protos(
        TEST_PROTOS_DIR,
        test_proto_files,
        "python",
        protoc_bin,
        protoc_include_path,
        out_dir,
    )

    for file_name in python_proto_files:
        apply_python_gencode_replacement(out_dir / _get_python_output_filename(file_name))


def build_protoc_from_source(version: Literal["3.19.4", "26.0"]) -> tuple[Path, Path]:
    """
    Build and install protoc from source for macOS arm64 version 3.19.4 only.
    """
    assert SYSTEM == "Darwin" and MACHINE == "arm64" and version == "3.19.4", (
        "This function is intended for macOS arm64 only and version 3.19.4."
    )

    src_dir = CACHE_DIR / f"protobuf-{version}"
    build_dir = src_dir / "cmake" / "build"

    if not build_dir.exists():
        build_dir.mkdir(parents=True, exist_ok=True)

        # Download the source code
        subprocess.check_call(
            [
                "curl",
                "-L",
                f"https://github.com/protocolbuffers/protobuf/archive/refs/tags/v{version}.tar.gz",
                "-o",
                CACHE_DIR / f"protobuf-{version}.tar.gz",
            ]
        )

        # Extract the source code
        subprocess.check_call(
            [
                "tar",
                "-xzf",
                CACHE_DIR / f"protobuf-{version}.tar.gz",
                "-C",
                CACHE_DIR,
            ]
        )

        # Build protoc from source
        subprocess.check_call(["./autogen.sh"], cwd=src_dir)
        subprocess.check_call(["./configure"], cwd=src_dir)
        subprocess.check_call(["make", "-j4"], cwd=src_dir)

    protoc_bin = src_dir / "src" / "protoc"
    protoc_include_path = src_dir / "src"
    return protoc_bin, protoc_include_path


def download_file(url: str, output_path: Path) -> None:
    """
    Download a file using wget on Linux and curl on macOS.
    """
    if SYSTEM == "Darwin":
        subprocess.check_call(
            [
                "curl",
                "-L",
                url,
                "-o",
                output_path,
            ]
        )
    else:
        subprocess.check_call(
            [
                "wget",
                url,
                "-O",
                output_path,
            ]
        )


def download_and_extract_protoc(version: Literal["3.19.4", "26.0"]) -> tuple[Path, Path]:
    """
    Download and extract specific version protoc tool, return extracted protoc executable file path
    and include path.
    """
    assert SYSTEM in ["Darwin", "Linux"], "The script only supports MacOS or Linux system."
    assert MACHINE in ["x86_64", "aarch64", "arm64"], (
        "The script only supports x86_64, arm64 or aarch64 CPU."
    )

    if SYSTEM == "Darwin" and MACHINE == "arm64" and version == "3.19.4":
        return build_protoc_from_source(version)

    os_type = "osx" if SYSTEM == "Darwin" else "linux"
    cpu_type = "x86_64" if MACHINE == "x86_64" else "aarch_64"

    if SYSTEM == "Darwin" and MACHINE == "arm64":
        protoc_zip_filename = f"protoc-{version}-osx-universal_binary.zip"
    else:
        protoc_zip_filename = f"protoc-{version}-{os_type}-{cpu_type}.zip"

    downloaded_protoc_bin = CACHE_DIR / f"protoc-{version}" / "bin" / "protoc"
    downloaded_protoc_include_path = CACHE_DIR / f"protoc-{version}" / "include"

    if not (downloaded_protoc_bin.is_file() and downloaded_protoc_include_path.is_dir()):
        download_file(
            f"https://github.com/protocolbuffers/protobuf/releases/download/v{version}/{protoc_zip_filename}",
            CACHE_DIR / protoc_zip_filename,
        )
        subprocess.check_call(
            [
                "unzip",
                "-o",
                "-d",
                CACHE_DIR / f"protoc-{version}",
                CACHE_DIR / protoc_zip_filename,
            ]
        )
        (CACHE_DIR / protoc_zip_filename).unlink()
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

        gen_python_protos(protoc3194, protoc3194_include, proto3194_out)
        gen_python_protos(protoc5260, protoc5260_include, proto5260_out)

        for proto_files, protos_dir in [
            (python_proto_files, MLFLOW_PROTOS_DIR),
            (test_proto_files, TEST_PROTOS_DIR),
        ]:
            for file_name in proto_files:
                gencode_filename = _get_python_output_filename(file_name)

                generate_final_python_gencode(
                    proto3194_out / gencode_filename,
                    proto5260_out / gencode_filename,
                    Path(protos_dir, gencode_filename),
                )

    # generate java gencode using pinned protoc 3.19.4 version.
    gen_protos(
        MLFLOW_PROTOS_DIR,
        basic_proto_files,
        "java",
        protoc3194,
        protoc3194_include,
        Path("mlflow/java/client/src/main/java"),
    )

    # Graphql code generation.
    subprocess.check_call([sys.executable, "./dev/proto_to_graphql/code_generator.py"])


if __name__ == "__main__":
    main()

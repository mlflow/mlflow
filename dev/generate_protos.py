import os
import platform
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

cache_dir = ".cache/protobuf_cache"

mlflow_protos_dir = "mlflow/protos"
test_protos_dir = "tests/protos"


def gen_protos(proto_dir, proto_files, lang, protoc_bin, protoc_include_path, out_dir):
    assert lang in ["python", "java"]

    subprocess.check_call(
        [
            protoc_bin,
            f"-I={protoc_include_path}",
            f"-I={proto_dir}",
            f"--{lang}_out={out_dir}",
            *[f"{proto_dir}/{proto_file}" for proto_file in proto_files],
        ]
    )


def apply_python_gencode_replacement(file_path):
    content = file_path.read_text()

    for old, new in python_gencode_replacements:
        content = content.replace(old, new)

    file_path.write_text(content, encoding="UTF-8")


def _get_python_output_filename(proto_file_name):
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
]
uc_proto_files = [
    "databricks_managed_catalog_messages.proto",
    "databricks_managed_catalog_service.proto",
    "databricks_uc_registry_messages.proto",
    "databricks_uc_registry_service.proto",
    "databricks_filesystem_service.proto",
]
facet_proto_files = ["facet_feature_statistics.proto"]
python_proto_files = basic_proto_files + uc_proto_files + facet_proto_files
test_proto_files = ["test_message.proto"]


python_gencode_replacements = [
    (
        "from scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2",
        "from .scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2",
    ),
    ("import databricks_pb2 as databricks__pb2", "from . import databricks_pb2 as databricks__pb2"),
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
]


def gen_python_protos(protoc_bin, protoc_include_path, out_dir):
    gen_protos(
        mlflow_protos_dir, python_proto_files, "python", protoc_bin, protoc_include_path, out_dir
    )

    gen_protos(
        test_protos_dir, test_proto_files, "python", protoc_bin, protoc_include_path, out_dir
    )

    for file_name in python_proto_files:
        apply_python_gencode_replacement(out_dir / _get_python_output_filename(file_name))


def download_and_extract_protoc(version):
    """
    Download and extract specific version protoc tool, return extracted protoc executable file path
    and include path.
    """
    assert platform.system() in [
        "Darwin",
        "Linux",
    ], "The script only supports MacOS or Linux system."
    assert platform.machine() in [
        "x86_64",
        "aarch64",
    ], "The script only supports x86_64 or aarch64 CPU."

    os_type = "osx" if platform.system() == "Darwin" else "linux"
    cpu_type = "x86_64" if platform.machine() == "x86_64" else "aarch_64"

    downloaded_protoc_bin = f"{cache_dir}/protoc-{version}/bin/protoc"
    downloaded_protoc_include_path = f"{cache_dir}/protoc-{version}/include"

    protoc_zip_filename = f"protoc-{version}-{os_type}-{cpu_type}.zip"
    if not (
        os.path.isfile(downloaded_protoc_bin) and os.path.isdir(downloaded_protoc_include_path)
    ):
        subprocess.check_call(
            [
                "wget",
                f"https://github.com/protocolbuffers/protobuf/releases/download/v{version}/{protoc_zip_filename}",
                "-O",
                f"{cache_dir}/{protoc_zip_filename}",
            ]
        )
        subprocess.check_call(
            [
                "unzip",
                "-o",
                "-d",
                f"{cache_dir}/protoc-{version}",
                f"{cache_dir}/{protoc_zip_filename}",
            ]
        )
        Path(cache_dir, protoc_zip_filename).unlink()
    return downloaded_protoc_bin, downloaded_protoc_include_path


def generate_final_python_gencode(gencode3194_path, gencode5260_path, out_path):
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


def main():
    os.makedirs(cache_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_gencode_dir:
        temp_gencode_path = Path(temp_gencode_dir)
        proto3194_out = temp_gencode_path / "3.19.4"
        proto5260_out = temp_gencode_path / "5.26.0"
        proto3194_out.mkdir(exist_ok=True)
        proto5260_out.mkdir(exist_ok=True)

        protoc3194, protoc3194_include = download_and_extract_protoc("3.19.4")
        protoc5260, protoc5260_include = download_and_extract_protoc("26.0")

        gen_python_protos(protoc3194, protoc3194_include, proto3194_out)
        gen_python_protos(protoc5260, protoc5260_include, proto5260_out)

        for proto_files, protos_dir in [
            (python_proto_files, mlflow_protos_dir),
            (test_proto_files, test_protos_dir),
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
        mlflow_protos_dir,
        basic_proto_files,
        "java",
        protoc3194,
        protoc3194_include,
        "mlflow/java/client/src/main/java",
    )

    # Graphql code generation.
    subprocess.check_call([sys.executable, "./dev/proto_to_graphql/code_generator.py"])


if __name__ == "__main__":
    main()

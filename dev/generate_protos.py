from collections import namedtuple
import subprocess
import sys
import shutil
import os
import platform
from pathlib import Path


cache_dir = "build/cache"
temp_gencode_dir = "build/proto_gencode"

mlflow_protos_dir = "mlflow/protos"


def gen_protos(proto_dir, proto_files, lang, protoc_bin, protoc_include_path, out_dir):
    assert lang in ["python", "java"]

    subprocess.check_call([
        protoc_bin,
        f"-I={protoc_include_path}",
        f"-I={proto_dir}",
        f"--{lang}_out={out_dir}",
        *[f"{proto_dir}/{proto_file}" for proto_file in proto_files]
    ])


def apply_python_gencode_replacement(file_path):
    content = file_path.read_text()

    for old, new in python_gencode_replacements:
        content = content.replace(old, new)

    file_path.write_text(content, encoding='UTF-8')


def _get_python_output_filename(proto_file_name):
    return proto_file_name.split(".")[0] + "_pb2.py"


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

python_gencode_replacements = [
    ("from scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2",
     "from .scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2"),
    ("import databricks_pb2 as databricks__pb2", "from . import databricks_pb2 as databricks__pb2"),
    ("import databricks_uc_registry_messages_pb2 as databricks__uc__registry__messages__pb2",
     "from . import databricks_uc_registry_messages_pb2 as databricks_uc_registry_messages_pb2"),
    ("import databricks_managed_catalog_messages_pb2 as databricks__managed__catalog__messages__pb2",
     "from . import databricks_managed_catalog_messages_pb2 as databricks_managed_catalog_messages_pb2")
]


def gen_python_protos(protoc_bin, protoc_include_path, out_dir):
    gen_protos(
        mlflow_protos_dir,
        python_proto_files,
        "python",
        protoc_bin, protoc_include_path, out_dir
    )

    for file_name in python_proto_files:
        apply_python_gencode_replacement(
            Path(out_dir, _get_python_output_filename(file_name))
        )


def download_and_extract_protoc(version):
    """
    Download and extract specific version protoc tool, return extracted protoc executable file path
    and include path.
    """
    downloaded_protoc_bin = f"{cache_dir}/protoc-{version}/bin/protoc"
    downloaded_protoc_include_path = f"{cache_dir}/protoc-{version}/include"
    if not (os.path.isfile(downloaded_protoc_bin) and os.path.isdir(downloaded_protoc_include_path)):
        subprocess.check_call([
            "wget",
            f"https://github.com/protocolbuffers/protobuf/releases/download/v{version}/protoc-{version}-osx-x86_64.zip",
            "-O", f"{cache_dir}/protoc-{version}-osx-x86_64.zip"
        ])
        subprocess.check_call([
            "unzip", "-o", "-d", f"{cache_dir}/protoc-{version}", f"{cache_dir}/protoc-{version}-osx-x86_64.zip"
        ])
    return downloaded_protoc_bin, downloaded_protoc_include_path


def prepend_intent(content):
    lines = content.split("\n")
    lines = ["  " + line for line in lines]
    return "\n".join(lines)


def generate_final_python_gencode(gencode3194_path, gencode5260_path, out_path):
    gencode3194 = gencode3194_path.read_text()
    gencode5260 = gencode5260_path.read_text()

    merged_code = f"""
import google.protobuf
from packaging.version import Version
if Version(google.protobuf.__version__) >= Version("5.26.0"):
{prepend_intent(gencode5260)}
else:
{prepend_intent(gencode3194)}
"""
    out_path.write_text(merged_code, encoding='UTF-8')


def main():
    assert platform.system() == 'Darwin', "The script only supports MacOS system."
    os.makedirs(cache_dir, exist_ok=True)
    shutil.rmtree(temp_gencode_dir, ignore_errors=True)

    proto3194_out = f"{temp_gencode_dir}/3.19.4"
    proto5260_out = f"{temp_gencode_dir}/5.26.0"
    os.makedirs(proto3194_out, exist_ok=True)
    os.makedirs(proto5260_out, exist_ok=True)

    protoc3194, protoc3194_include = download_and_extract_protoc("3.19.4")
    protoc5260, protoc5260_include = download_and_extract_protoc("26.0")

    gen_python_protos(protoc3194, protoc3194_include, proto3194_out)
    gen_python_protos(protoc5260, protoc5260_include, proto5260_out)

    for file_name in python_proto_files:
        gencode_filename = _get_python_output_filename(file_name)

        generate_final_python_gencode(
            Path(proto3194_out, gencode_filename),
            Path(proto5260_out, gencode_filename),
            Path(mlflow_protos_dir, gencode_filename),
        )

    # generate java gencode using pinned protoc 3.19.4 version.
    gen_protos(
        mlflow_protos_dir,
        basic_proto_files,
        "java",
        protoc3194, protoc3194_include,
        "mlflow/java/client/src/main/java",
    )

    # Graphql code generation.
    subprocess.check_call([sys.executable, "./dev/proto_to_graphql/code_generator.py"])


if __name__ == "__main__":
    main()

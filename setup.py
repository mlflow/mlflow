import os
import shutil
import subprocess
from glob import glob
from typing import List, Tuple

from setuptools import Distribution, setup


def _is_go_installed() -> bool:
    try:
        subprocess.check_call(
            ["go", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False


def _get_platform() -> str:
    os = subprocess.check_output(["go", "env", "GOOS"]).strip().decode("utf-8")
    arch = subprocess.check_output(["go", "env", "GOARCH"]).strip().decode("utf-8")
    plat = f"{os}_{arch}"
    if plat == "darwin_amd64":
        return "macosx_10_12_x86_64"
    elif plat == "darwin_arm64":
        return "macosx_11_0_arm64"
    elif plat == "linux_amd64":
        return "linux_x86_64"
    elif plat == "linux_arm64":
        return "linux_aarch64"
    elif plat == "windows_amd64":
        return "win_amd64"
    elif plat == "windows_arm64":
        return "win_arm64"
    else:
        raise ValueError("not supported platform.")


def finalize_distribution_options(dist: Distribution) -> None:
    go_installed = _is_go_installed()

    dist.has_ext_modules = lambda: super(Distribution, dist).has_ext_modules() or go_installed

    # this allows us to set the tag for the wheel based on GOOS and GOARCH
    if go_installed:
        bdist_wheel_base_class = dist.get_command_class("bdist_wheel")

        class bdist_wheel_go(bdist_wheel_base_class):
            def get_tag(self) -> Tuple[str, str, str]:
                return "py3", "none", _get_platform()

        dist.cmdclass["bdist_wheel"] = bdist_wheel_go

    # this allows us to build the go binary and add the Go source files to the sdist
    build_base_class = dist.get_command_class("build")

    class build_go(build_base_class):
        def initialize_options(self) -> None:
            self.editable_mode = False
            self.build_lib = None

        def finalize_options(self) -> None:
            self.set_undefined_options("build_py", ("build_lib", "build_lib"))

        def run(self) -> None:
            if not self.editable_mode:
                shutil.rmtree(os.path.join(self.build_lib, "mlflow", "go"), ignore_errors=True)
                if go_installed:
                    subprocess.check_call(
                        [
                            "go",
                            "build",
                            "-ldflags",
                            "-w -s",
                            "-o",
                            os.path.join(self.build_lib, "mlflow", "go", "server"),
                            "./mlflow/go",
                        ]
                    )

        def get_source_files(self) -> List[str]:
            return ["go.mod", "go.sum", *glob("mlflow/go/**/*.go", recursive=True)]

    dist.cmdclass["build_go"] = build_go
    build_base_class.sub_commands.append(("build_go", None))


Distribution.finalize_options = finalize_distribution_options

setup()

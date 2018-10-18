import os
from subprocess import check_output, Popen

def get_conda_bin_path():
    conda_symlink_cmd = "which conda"
    symlink_path = check_output(conda_symlink_cmd.split(" ")).decode("utf-8").rstrip()
    conda_full_path_cmd = "realpath {symlink_path}".format(symlink_path=symlink_path)
    full_path = check_output(conda_full_path_cmd.split(" ")).decode("utf-8").rstrip()
    return os.path.dirname(full_path)

def main():
    conda_bin_path = get_conda_bin_path()
    conda_activate_path = os.path.join(conda_bin_path, "activate")
    cmd = "source {activate_path} sklearn_env && python -c \"import sys; print(sys.version_info)\"".format(
            activate_path=conda_activate_path)
    cmd = ["/bin/bash", "-c", cmd] 
    # result = Popen(cmd.split(" "))
    result = check_output(cmd)
    print(result)
    # result = Popen(cmd)
    # result.wait()

if __name__ == "__main__":
    main()


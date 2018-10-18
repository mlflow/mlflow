import os
from subprocess import check_output, Popen, PIPE

from azureml.core.model import Model

def get_conda_bin_path():
    conda_symlink_cmd = "which conda"
    symlink_path = check_output(conda_symlink_cmd.split(" ")).decode("utf-8").rstrip()
    conda_full_path_cmd = "realpath {symlink_path}".format(symlink_path=symlink_path)
    full_path = check_output(conda_full_path_cmd.split(" ")).decode("utf-8").rstrip()
    return os.path.dirname(full_path)

def init():
    # global model
    model_path = Model.get_model_path("mlflow-jyhztv47tss2y0blvpdlsa")
    # model = load_pyfunc(model_path)
    conda_bin_path = get_conda_bin_path()
    conda_activate_path = os.path.join(conda_bin_path, "activate")
    cmd = "source {activate_path} goodenv && python py27server.py {model_path}".format(
            activate_path=conda_activate_path, model_path=model_path)
    cmd = ["/bin/bash", "-c", cmd]

    with open("json_content.json", "r") as f:
        json_content = f.read()

    proc = Popen(cmd, stdin=PIPE, stdout=PIPE)
    response_text, _ = proc.communicate(json_content.encode("utf-8"))
    response_text = response_text.decode("utf-8")
    print(response_text)
    # result = Popen(cmd)
    # result.wait()


def run(s):
    input_df = pd.read_json(s, orient="records")
    return get_jsonable_obj(model.predict(input_df))


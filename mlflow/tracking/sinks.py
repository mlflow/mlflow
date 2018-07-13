import distutils.dir_util as dir_util
import os
import shutil


_TRACKING_DIR_ENV_VAR = "MLFLOW_TRACKING_DIR"


def _get_tracking_dir():
    if _TRACKING_DIR_ENV_VAR in os.environ:
        return os.environ[_TRACKING_DIR_ENV_VAR]
    else:
        return "mlruns"


class FileSink(object):
    def __init__(self, run_id, root_dir=_get_tracking_dir()):
        self.run_id = run_id
        self.run_dir = os.path.join(root_dir, run_id)
        dir_util.mkpath(self.run_dir)

    def log_param(self, key, value):
        # TODO: prevent keys from containing funky values like ".."
        fn = os.path.join(self.run_dir, "parameters", key)
        dir_util.mkpath(os.path.dirname(fn))
        with open(fn, "w") as f:
            f.write("%s\n" % value)

    def log_metric(self, key, value):
        # TODO: prevent keys from containing funky values like ".."
        fn = os.path.join(self.run_dir, "metrics", key)
        dir_util.mkpath(os.path.dirname(fn))
        with open(fn, "a") as f:
            f.write("%s\n" % value)

    def log_artifact(self, local_path, artifact_path=None):
        if artifact_path is None:
            artifact_path = os.path.basename(local_path)
        if os.path.exists(local_path):
            dst_path = os.path.join(self.run_dir, "outputs", artifact_path)
            if not os.path.exists(os.path.dirname(dst_path)):
                dir_util.mkpath(os.path.dirname(dst_path))
            shutil.copy(local_path, dst_path)

    def log_output_files(self, output_dir, path):
        if os.path.exists(output_dir):
            if path is not None:
                dst_dir = os.path.join(self.run_dir, "outputs", path)
            else:
                dst_dir = os.path.join(self.run_dir, "outputs")
            if not os.path.exists(dst_dir):
                dir_util.mkpath(dst_dir)
            dir_util.copy_tree(src=output_dir, dst=dst_dir)

    def set_status(self, status):
        fn = os.path.join(self.run_dir, "status")
        with open(fn, "w") as f:
            f.write("%s\n" % status)

    def set_source(self, source):
        fn = os.path.join(self.run_dir, "source")
        with open(fn, "w") as f:
            f.write("%s\n" % source)

    def set_git_commit(self, commit):
        fn = os.path.join(self.run_dir, "git_commit")
        with open(fn, "w") as f:
            f.write("%s\n" % commit)

    def set_start_date(self, utc_date_time):
        fn = os.path.join(self.run_dir, "start_date")
        with open(fn, "w") as f:
            f.write("%s\n" % utc_date_time.isoformat())

    def set_end_date(self, utc_date_time):
        fn = os.path.join(self.run_dir, "end_date")
        with open(fn, "w") as f:
            f.write("%s\n" % utc_date_time.isoformat())

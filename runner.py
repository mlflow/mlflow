import uvicorn
import os
import sys
from reloading_app import CONFIG_PATH_ENV_VAR


if __name__ == "__main__":
    # python runner.py config.yml
    config_path = sys.argv[1]
    os.environ[CONFIG_PATH_ENV_VAR] = "config.yml"
    uvicorn.run(
        "reloading_app:create_app",
        reload=True,
        reload_includes="config.yml",
        factory=True,
    )

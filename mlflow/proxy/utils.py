import yaml
from starlette.responses import JSONResponse

from mlflow.proxy.proxy_service_constants import _Route


def _error_response(code: int, message: str):
    return JSONResponse(status_code=code, content={"message": message})


def _load_route_config(conf_file: str):
    with open(conf_file, "r") as f:
        conf = yaml.safe_load(f)
    return [_Route.from_dict(item) for item in conf]


def _get_route_limits(config):
    return {x.route: x.limits_per_minute for x in config}


def _parse_request_path(request):
    return request.url.path[1:]


def _parse_url_path_for_base_url(url_string):
    split_url = url_string.split("/")
    return "/".join(split_url[:-1])

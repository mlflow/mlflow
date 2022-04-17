import logging
import subprocess
import typing as t

from ..types import PathLike

logger = logging.getLogger(__name__)


class CompletedProcess(subprocess.CompletedProcess):
    @property
    def success(self) -> bool:
        return self.returncode == 0


def run_cmd(args: t.Sequence[PathLike], **kwargs: t.Any) -> CompletedProcess:
    # pylint: disable-next=subprocess-run-check
    prc = subprocess.run(list(map(str, args)), text=True, **kwargs)
    return CompletedProcess(prc.args, prc.returncode, prc.stdout, prc.stderr)

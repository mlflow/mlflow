"""Custom uvicorn event-loop factory for Windows.

The in-app Assistant's CLI providers spawn their agent with
``asyncio.create_subprocess_exec``, which on Windows only works on a
``ProactorEventLoop``. uvicorn, however, runs multi-worker (the default is 4)
or ``--reload`` servers on a ``SelectorEventLoop`` — where the spawn raises
``NotImplementedError`` (see https://github.com/mlflow/mlflow/issues/24405).

uvicorn selects its loop via a ``loop_factory``, not the event-loop policy, so
a process-wide ``WindowsProactorEventLoopPolicy`` does not fix it. But uvicorn's
``--loop`` flag accepts a custom ``module:attr`` factory and returns it verbatim
(bypassing the ``use_subprocess`` gate that would otherwise force the selector
loop). Pointing ``--loop`` at this factory makes every worker serve on a
Proactor loop, so the subprocess spawn works. See ``_build_uvicorn_command``.
"""

import asyncio


def proactor_loop_factory() -> asyncio.AbstractEventLoop:
    """Return a Windows ``ProactorEventLoop`` for uvicorn to serve on.

    Only referenced on Windows (the ``--loop`` flag is only injected there), so
    the Windows-only ``asyncio.ProactorEventLoop`` is accessed inside the body
    to keep this module importable on all platforms.
    """
    return asyncio.ProactorEventLoop()

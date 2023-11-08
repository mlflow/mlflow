FROM python:3.10-slim-bullseye
ARG VERSION
# NB: psutil doesn't have binary for Arm64 Linux so we need to build it from 
# CPython, which causes multiple headache during build. As psutil is only 
# required for system metrics monitoring and gateway, hence we make is optional 
# and provide custom implementation in utils/process.py for Arm64 Linux. 
RUN if [ "$(uname -m)" != "aarch64" ]; then pip install --no-cache psutil; fi
RUN pip install --no-cache mlflow==$VERSION

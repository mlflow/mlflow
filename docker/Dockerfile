FROM python:3.10-slim-bullseye
ARG VERSION
ARG EXTRAS="[genai]"
RUN pip install --no-cache mlflow${EXTRAS}==$VERSION


FROM docker.io/centos/python-36-centos7

USER root

# When building pass `docker build --build-arg MLFLOW_VERSION=0.8.0`
ARG mlflow_version

LABEL architecture="x86_64"  \
      com.redhat.component="opendatahub-mlflow-tracking-server"  \
      maintainer="Zak Hassan <zak.hassan@redhat.com>" \
      mlflow="$mlflow_version"

RUN  pip install mlflow==$mlflow_version && pip install awscli
COPY entrypoint.sh /etc/entrypoint.sh
EXPOSE 5000
USER 185
ENTRYPOINT ["/etc/entrypoint.sh"]

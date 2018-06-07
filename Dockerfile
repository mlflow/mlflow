FROM continuumio/miniconda

RUN pip install numpy pandas flask pygal smalluuid zipstream python-dateutil gitpython scikit-learn

WORKDIR /app

ADD . /app

RUN pip install -r dev-requirements.txt
RUN pip install -r tox-requirements.txt
RUN pip install -e .

RUN apt-get install -y gnupg && curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y nodejs
RUN cd mlflow/server/js && npm run build

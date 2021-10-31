FROM python:3.6

WORKDIR /app

COPY dist ./dist
RUN pip install dist/*.whl
RUN pip install psycopg2 boto3 scikit-learn

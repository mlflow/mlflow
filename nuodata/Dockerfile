FROM python:3.10-slim

EXPOSE 5000

WORKDIR /app
COPY . .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -e .[extras]

RUN pip install psycopg2-binary httpx

CMD [ "mlflow", "server", "--app-name" ,"keycloak-auth", "--host", "0.0.0.0"]
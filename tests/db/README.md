This directory contains files to test MLflow using the following database types:

- PostgreSQL
- MySQL
- Microsoft SQL Server
- SQLite

## Prerequisites

- Docker
- Docker Compose

## Run tests

```bash
# Make sure the current working directory is the repository root.
docker-compose -f tests/db/docker-compose.yml build --build-arg DEPENDENCIES="$(python setup.py -q dependencies)" <service name>
docker-compose -f tests/db/docker-compose.yml run <service name> pytest /path/to/tests
```

For example, the following commands run `tests/db/test_db.py` using PostgreSQL as a backend store:

```bash
docker-compose -f tests/db/docker-compose.yml build --build-arg DEPENDENCIES="$(python setup.py -q dependencies)" mlflow-postgres
docker-compose -f tests/db/docker-compose.yml run mlflow-postgres pytest tests/db/test_database.py
```

Other useful commands:

```bash
# Build all services (this might take a while)
docker-compose -f tests/db/docker-compose.yml build

# View database logs
docker-compose -f tests/db/docker-compose.yml logs postgres
```

## Clean Up

```bash
docker-compose -f tests/db/docker-compose.yml down --volumes --remove-orphans
```

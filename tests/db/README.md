# Instructions

This directory contains files to test MLflow tracking operations using the following databases:

- PostgreSQL
- MySQL
- Microsoft SQL Server
- SQLite

## Prerequisites

- Docker
- Docker Compose V2

## Build Services

```bash
# Build a service
./tests/db/compose.sh build --build-arg DEPENDENCIES="$(python setup.py -q dependencies)" <service>

# Build all services
./tests/db/compose.sh build --build-arg DEPENDENCIES="$(python setup.py -q dependencies)"
```

## Run Services

```bash
# Run a service (`pytest tests/db` is executed by default)
./tests/db/compose.sh run --rm <service>

# Run all services
for service in $(./tests/db/compose.sh config --services | grep '^mlflow-')
do
  ./tests/db/compose.sh run --rm "$service"
done

# Run tests
./tests/db/compose.sh run --rm <service> pytest /path/to/directory/or/script

# Run a python script
./tests/db/compose.sh run --rm <service> python /path/to/script
```

## Clean Up Services

```bash
# Clean up containers, networks, and volumes
./tests/db/compose.sh down --volumes --remove-orphans

# Clean up containers, networks, volumes, and images
./tests/db/compose.sh down --volumes --remove-orphans --rmi all
```

## Other Useful Commands

```bash
# View database logs
./tests/db/compose.sh logs --follow <database service>
```

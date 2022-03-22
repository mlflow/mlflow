This directory contains files to test MLflow tracking operations using the following databases:

- PostgreSQL
- MySQL
- Microsoft SQL Server
- SQLite

## Prerequisites

- Docker
- Docker Compose

## Build Services

```bash
# Build a service
./tests/db/compose.sh build --build-arg DEPENDENCIES="$(python setup.py -q dependencies)" <service name>

# Build all services
./tests/db/compose.sh build --build-arg DEPENDENCIES="$(python setup.py -q dependencies)"
```

## Run Services

```bash
# Run a service with the default command (`pytest tests/db`)
./tests/db/compose.sh run --rm <service name>

# Run tests
./tests/db/compose.sh run --rm <service name> pytest /path/to/directory/or/script

# Run a python script
./tests/db/compose.sh run --rm <service name> python /path/to/script
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
./tests/db/compose.sh logs --follow <database service name>
```

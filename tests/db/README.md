This directory contains files to test MLflow tracking operations using the following databases:

- PostgreSQL
- MySQL
- Microsoft SQL Server
- SQLite

## Prerequisites

- Docker
- Docker Compose

# Build Services

```bash
# Build all services
DEPENDENCIES="$(python setup.py -q dependencies)"
docker-compose -f tests/db/docker-compose.yml build --build-arg DEPENDENCIES="$DEPENDENCIES"

# Build a specific service
docker-compose -f tests/db/docker-compose.yml build --build-arg DEPENDENCIES="$DEPENDENCIES" <service name>
```

# Run Services

```bash
# Run the default command (`pytest tests/db`)
docker-compose -f tests/db/docker-compose.yml run --rm <service name>

# Run other tests
docker-compose -f tests/db/docker-compose.yml run --rm <service name> pytest /path/to/directory/or/script

# Run a python script
docker-compose -f tests/db/docker-compose.yml run --rm <service name> python /path/to/script
```

## Clean Up Services

```bash
# Clean up containers, networks, and volumes
docker-compose -f tests/db/docker-compose.yml down --volumes --remove-orphans

# Clean up containers, networks, volumes, and images
docker-compose -f tests/db/docker-compose.yml down --volumes --remove-orphans --rmi all
```

## Other Useful Commands

```bash
# View database logs
docker-compose -f tests/db/docker-compose.yml logs --follow <database service name>
```

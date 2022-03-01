# How to run PostgreSQL benchmark

# Build wheel

```
$(git rev-parse --show-toplevel)/dev/build_wheel.sh
```

# Run benchmark

```
QUERY=$(cat benchmark.sql) NUM_RUNS=1000 docker-compose run postgres-benchmark
```

# Clean up

```
docker-compose down --volumes --remove-orphans
```

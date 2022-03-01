# How to run PostgreSQL benchmark

# Build wheel

```
$(git rev-parse --show-toplevel)/dev/build_wheel.sh
```

# Run Benchmark

```
QUERY=$(cat join_params_and_metrics.sql) NUM_RUNS=1000 docker-compose run postgres-benchmark
```

# Clean Up

```
docker-compose down --volumes --remove-orphans
```

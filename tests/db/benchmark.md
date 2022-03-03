# How to run PostgreSQL benchmark

# Build wheel

```
./build_wheel.sh
```

# Run benchmark

```
QUERY=$(cat benchmark.sql) NUM_RUNS=1000 docker-compose run postgres-benchmark
```

# Clean up

```
docker-compose down --volumes --remove-orphans
```

|     | run_id                           | experiment_id | status   | artifact_uri                                          | start_time                       | end_time                         | metrics.m_a7f3be45aab04bfdb4ff4c46b17a3994 | metrics.m_b990e42cd51b4547bc3dedab5c9b79d9 | params.p_651b3cefe83b44e99f6fed60eb3342da | params.p_fd710391d1f14ca6bdf36a35856ac558 | tags.mlflow.user | tags.mlflow.source.name | tags.mlflow.source.type |
| --: | :------------------------------- | ------------: | :------- | :---------------------------------------------------- | :------------------------------- | :------------------------------- | -----------------------------------------: | -----------------------------------------: | ----------------------------------------: | ----------------------------------------: | :--------------- | :---------------------- | :---------------------- |
|   0 | 18d992f7fe79418fb4043aff5883621b |             0 | FINISHED | ./mlruns/0/18d992f7fe79418fb4043aff5883621b/artifacts | 2022-03-02 23:54:22.574000+00:00 | 2022-03-02 23:54:22.641000+00:00 |                                          0 |                                        nan |                                         0 |                                           | root             | benchmark.py            | LOCAL                   |
|   1 | 8d2fab50203b47018fa7246d4de84294 |             0 | FINISHED | ./mlruns/0/8d2fab50203b47018fa7246d4de84294/artifacts | 2022-03-02 23:53:58.497000+00:00 | 2022-03-02 23:53:58.557000+00:00 |                                        nan |                                          0 |                                           |                                         0 | root             | benchmark.py            | LOCAL                   |

-- BENCHMARK-ONLY. Grant the bench app user server-wide rights so the benchmark can
-- CREATE/DROP its own per-run `bench_<uuid>` databases (see docker-compose.yml).
GRANT ALL PRIVILEGES ON *.* TO 'mlflow'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;

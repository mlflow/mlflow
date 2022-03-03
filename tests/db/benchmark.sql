
SELECT *
FROM runs
JOIN params on params.run_uuid = runs.run_uuid
JOIN metrics on metrics.run_uuid = params.run_uuid
WHERE runs.experiment_id = 0
-- 
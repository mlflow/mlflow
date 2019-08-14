
CREATE TABLE alembic_version (
	version_num VARCHAR(32) NOT NULL, 
	CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
)


CREATE TABLE experiments (
	experiment_id INTEGER NOT NULL, 
	name VARCHAR(256) NOT NULL, 
	artifact_location VARCHAR(256), 
	lifecycle_stage VARCHAR(32), 
	CONSTRAINT experiment_pk PRIMARY KEY (experiment_id), 
	UNIQUE (name), 
	CONSTRAINT experiments_lifecycle_stage CHECK (lifecycle_stage IN ('active', 'deleted'))
)


CREATE TABLE experiment_tags (
	key VARCHAR(250) NOT NULL, 
	value VARCHAR(5000), 
	experiment_id INTEGER NOT NULL, 
	CONSTRAINT experiment_tag_pk PRIMARY KEY (key, experiment_id), 
	FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
)


CREATE TABLE runs (
	run_uuid VARCHAR(32) NOT NULL, 
	name VARCHAR(250), 
	source_type VARCHAR(20), 
	source_name VARCHAR(500), 
	entry_point_name VARCHAR(50), 
	user_id VARCHAR(256), 
	status VARCHAR(20), 
	start_time BIGINT, 
	end_time BIGINT, 
	source_version VARCHAR(50), 
	lifecycle_stage VARCHAR(20), 
	artifact_uri VARCHAR(200), 
	experiment_id INTEGER, 
	CONSTRAINT run_pk PRIMARY KEY (run_uuid), 
	FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id), 
	CONSTRAINT source_type CHECK (source_type IN ('NOTEBOOK', 'JOB', 'LOCAL', 'UNKNOWN', 'PROJECT')), 
	CONSTRAINT status CHECK (status IN ('SCHEDULED', 'FAILED', 'FINISHED', 'RUNNING')), 
	CONSTRAINT runs_lifecycle_stage CHECK (lifecycle_stage IN ('active', 'deleted'))
)


CREATE TABLE metrics (
	key VARCHAR(250) NOT NULL, 
	value FLOAT NOT NULL, 
	timestamp BIGINT NOT NULL, 
	run_uuid VARCHAR(32) NOT NULL, 
	step BIGINT DEFAULT '0' NOT NULL, 
	is_nan BOOLEAN DEFAULT '0' NOT NULL, 
	CONSTRAINT metric_pk PRIMARY KEY (key, value, timestamp, run_uuid, step, is_nan), 
	FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
)


CREATE TABLE params (
	key VARCHAR(250) NOT NULL, 
	value VARCHAR(250) NOT NULL, 
	run_uuid VARCHAR(32) NOT NULL, 
	CONSTRAINT param_pk PRIMARY KEY (key, run_uuid), 
	FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
)


CREATE TABLE tags (
	key VARCHAR(250) NOT NULL, 
	value VARCHAR(5000), 
	run_uuid VARCHAR(32) NOT NULL, 
	CONSTRAINT tag_pk PRIMARY KEY (key, run_uuid), 
	FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
)


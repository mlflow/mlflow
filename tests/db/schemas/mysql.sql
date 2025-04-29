
CREATE TABLE alembic_version (
	version_num VARCHAR(32) NOT NULL,
	PRIMARY KEY (version_num)
)


CREATE TABLE experiments (
	experiment_id INTEGER NOT NULL,
	name VARCHAR(256) NOT NULL,
	artifact_location VARCHAR(256),
	lifecycle_stage VARCHAR(32),
	creation_time BIGINT,
	last_update_time BIGINT,
	PRIMARY KEY (experiment_id),
	CONSTRAINT experiments_lifecycle_stage CHECK ((`lifecycle_stage` in (_utf8mb4'active',_utf8mb4'deleted')))
)


CREATE TABLE input_tags (
	input_uuid VARCHAR(36) NOT NULL,
	name VARCHAR(255) NOT NULL,
	value VARCHAR(500) NOT NULL,
	PRIMARY KEY (input_uuid, name)
)


CREATE TABLE inputs (
	input_uuid VARCHAR(36) NOT NULL,
	source_type VARCHAR(36) NOT NULL,
	source_id VARCHAR(36) NOT NULL,
	destination_type VARCHAR(36) NOT NULL,
	destination_id VARCHAR(36) NOT NULL,
	step BIGINT DEFAULT '0' NOT NULL,
	PRIMARY KEY (source_type, source_id, destination_type, destination_id)
)


CREATE TABLE registered_models (
	name VARCHAR(256) NOT NULL,
	creation_time BIGINT,
	last_updated_time BIGINT,
	description VARCHAR(5000),
	PRIMARY KEY (name)
)


CREATE TABLE datasets (
	dataset_uuid VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	name VARCHAR(500) NOT NULL,
	digest VARCHAR(36) NOT NULL,
	dataset_source_type VARCHAR(36) NOT NULL,
	dataset_source TEXT NOT NULL,
	dataset_schema MEDIUMTEXT,
	dataset_profile MEDIUMTEXT,
	PRIMARY KEY (experiment_id, name, digest),
	CONSTRAINT fk_datasets_experiment_id_experiments FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE
)


CREATE TABLE experiment_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	experiment_id INTEGER NOT NULL,
	PRIMARY KEY (key, experiment_id),
	CONSTRAINT experiment_tags_ibfk_1 FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
)


CREATE TABLE logged_models (
	model_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	name VARCHAR(500) NOT NULL,
	artifact_location VARCHAR(1000) NOT NULL,
	creation_timestamp_ms BIGINT NOT NULL,
	last_updated_timestamp_ms BIGINT NOT NULL,
	status INTEGER NOT NULL,
	lifecycle_stage VARCHAR(32),
	model_type VARCHAR(500),
	source_run_id VARCHAR(32),
	status_message VARCHAR(1000),
	PRIMARY KEY (model_id),
	CONSTRAINT fk_logged_models_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id) ON DELETE CASCADE,
	CONSTRAINT logged_models_lifecycle_stage_check CHECK ((`lifecycle_stage` in (_utf8mb4'active',_utf8mb4'deleted')))
)


CREATE TABLE model_versions (
	name VARCHAR(256) NOT NULL,
	version INTEGER NOT NULL,
	creation_time BIGINT,
	last_updated_time BIGINT,
	description VARCHAR(5000),
	user_id VARCHAR(256),
	current_stage VARCHAR(20),
	source VARCHAR(500),
	run_id VARCHAR(32),
	status VARCHAR(20),
	status_message VARCHAR(500),
	run_link VARCHAR(500),
	storage_location VARCHAR(500),
	PRIMARY KEY (name, version),
	CONSTRAINT model_versions_ibfk_1 FOREIGN KEY(name) REFERENCES registered_models (name) ON UPDATE CASCADE
)


CREATE TABLE registered_model_aliases (
	alias VARCHAR(256) NOT NULL,
	version INTEGER NOT NULL,
	name VARCHAR(256) NOT NULL,
	PRIMARY KEY (name, alias),
	CONSTRAINT registered_model_alias_name_fkey FOREIGN KEY(name) REFERENCES registered_models (name) ON DELETE CASCADE ON UPDATE CASCADE
)


CREATE TABLE registered_model_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	name VARCHAR(256) NOT NULL,
	PRIMARY KEY (key, name),
	CONSTRAINT registered_model_tags_ibfk_1 FOREIGN KEY(name) REFERENCES registered_models (name) ON UPDATE CASCADE
)


CREATE TABLE runs (
	run_uuid VARCHAR(32) NOT NULL,
	name VARCHAR(250),
	source_type VARCHAR(20),
	source_name VARCHAR(500),
	entry_point_name VARCHAR(50),
	user_id VARCHAR(256),
	status VARCHAR(9),
	start_time BIGINT,
	end_time BIGINT,
	source_version VARCHAR(50),
	lifecycle_stage VARCHAR(20),
	artifact_uri VARCHAR(200),
	experiment_id INTEGER,
	deleted_time BIGINT,
	PRIMARY KEY (run_uuid),
	CONSTRAINT runs_ibfk_1 FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT runs_chk_1 CHECK ((`status` in (_utf8mb4'SCHEDULED',_utf8mb4'FAILED',_utf8mb4'FINISHED',_utf8mb4'RUNNING',_utf8mb4'KILLED'))),
	CONSTRAINT runs_lifecycle_stage CHECK ((`lifecycle_stage` in (_utf8mb4'active',_utf8mb4'deleted'))),
	CONSTRAINT source_type CHECK ((`source_type` in (_utf8mb4'NOTEBOOK',_utf8mb4'JOB',_utf8mb4'LOCAL',_utf8mb4'UNKNOWN',_utf8mb4'PROJECT')))
)


CREATE TABLE trace_info (
	request_id VARCHAR(50) NOT NULL,
	experiment_id INTEGER NOT NULL,
	timestamp_ms BIGINT NOT NULL,
	execution_time_ms BIGINT,
	status VARCHAR(50) NOT NULL,
	PRIMARY KEY (request_id),
	CONSTRAINT fk_trace_info_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
)


CREATE TABLE latest_metrics (
	key VARCHAR(250) NOT NULL,
	value DOUBLE NOT NULL,
	timestamp BIGINT,
	step BIGINT NOT NULL,
	is_nan TINYINT NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	PRIMARY KEY (key, run_uuid),
	CONSTRAINT latest_metrics_ibfk_1 FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid),
	CONSTRAINT latest_metrics_chk_1 CHECK ((`is_nan` in (0,1)))
)


CREATE TABLE logged_model_metrics (
	model_id VARCHAR(36) NOT NULL,
	metric_name VARCHAR(500) NOT NULL,
	metric_timestamp_ms BIGINT NOT NULL,
	metric_step BIGINT NOT NULL,
	metric_value DOUBLE,
	experiment_id INTEGER NOT NULL,
	run_id VARCHAR(32) NOT NULL,
	dataset_uuid VARCHAR(36),
	dataset_name VARCHAR(500),
	dataset_digest VARCHAR(36),
	PRIMARY KEY (model_id, metric_name, metric_timestamp_ms, metric_step, run_id),
	CONSTRAINT fk_logged_model_metrics_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT fk_logged_model_metrics_model_id FOREIGN KEY(model_id) REFERENCES logged_models (model_id) ON DELETE CASCADE,
	CONSTRAINT fk_logged_model_metrics_run_id FOREIGN KEY(run_id) REFERENCES runs (run_uuid) ON DELETE CASCADE
)


CREATE TABLE logged_model_params (
	model_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	param_key VARCHAR(255) NOT NULL,
	param_value TEXT NOT NULL,
	PRIMARY KEY (model_id, param_key),
	CONSTRAINT fk_logged_model_params_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT fk_logged_model_params_model_id FOREIGN KEY(model_id) REFERENCES logged_models (model_id) ON DELETE CASCADE
)


CREATE TABLE logged_model_tags (
	model_id VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	tag_key VARCHAR(255) NOT NULL,
	tag_value TEXT NOT NULL,
	PRIMARY KEY (model_id, tag_key),
	CONSTRAINT fk_logged_model_tags_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT fk_logged_model_tags_model_id FOREIGN KEY(model_id) REFERENCES logged_models (model_id) ON DELETE CASCADE
)


CREATE TABLE metrics (
	key VARCHAR(250) NOT NULL,
	value DOUBLE NOT NULL,
	timestamp BIGINT NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	step BIGINT DEFAULT '0' NOT NULL,
	is_nan TINYINT DEFAULT '0' NOT NULL,
	PRIMARY KEY (key, timestamp, step, run_uuid, value, is_nan),
	CONSTRAINT metrics_ibfk_1 FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid),
	CONSTRAINT metrics_chk_1 CHECK ((`is_nan` in (0,1))),
	CONSTRAINT metrics_chk_2 CHECK ((`is_nan` in (0,1)))
)


CREATE TABLE model_version_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	name VARCHAR(256) NOT NULL,
	version INTEGER NOT NULL,
	PRIMARY KEY (key, name, version),
	CONSTRAINT model_version_tags_ibfk_1 FOREIGN KEY(name, version) REFERENCES model_versions (name, version) ON UPDATE CASCADE
)


CREATE TABLE params (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(8000) NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	PRIMARY KEY (key, run_uuid),
	CONSTRAINT params_ibfk_1 FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
)


CREATE TABLE tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(8000),
	run_uuid VARCHAR(32) NOT NULL,
	PRIMARY KEY (key, run_uuid),
	CONSTRAINT tags_ibfk_1 FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
)


CREATE TABLE trace_request_metadata (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(8000),
	request_id VARCHAR(50) NOT NULL,
	PRIMARY KEY (key, request_id),
	CONSTRAINT fk_trace_request_metadata_request_id FOREIGN KEY(request_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
)


CREATE TABLE trace_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(8000),
	request_id VARCHAR(50) NOT NULL,
	PRIMARY KEY (key, request_id),
	CONSTRAINT fk_trace_tags_request_id FOREIGN KEY(request_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
)

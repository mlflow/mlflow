
CREATE TABLE alembic_version (
	version_num VARCHAR(32) NOT NULL,
	CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
)


CREATE TABLE experiments (
	experiment_id INTEGER DEFAULT nextval('experiments_experiment_id_seq'::regclass) NOT NULL,
	name VARCHAR(256) NOT NULL,
	artifact_location VARCHAR(256),
	lifecycle_stage VARCHAR(32),
	creation_time BIGINT,
	last_update_time BIGINT,
	CONSTRAINT experiment_pk PRIMARY KEY (experiment_id),
	CONSTRAINT experiments_name_key UNIQUE (name),
	CONSTRAINT experiments_lifecycle_stage CHECK (lifecycle_stage::text = ANY (ARRAY['active'::character varying, 'deleted'::character varying]::text[]))
)


CREATE TABLE input_tags (
	input_uuid VARCHAR(36) NOT NULL,
	name VARCHAR(255) NOT NULL,
	value VARCHAR(500) NOT NULL,
	CONSTRAINT input_tags_pk PRIMARY KEY (input_uuid, name)
)


CREATE TABLE inputs (
	input_uuid VARCHAR(36) NOT NULL,
	source_type VARCHAR(36) NOT NULL,
	source_id VARCHAR(36) NOT NULL,
	destination_type VARCHAR(36) NOT NULL,
	destination_id VARCHAR(36) NOT NULL,
	CONSTRAINT inputs_pk PRIMARY KEY (source_type, source_id, destination_type, destination_id)
)


CREATE TABLE registered_models (
	name VARCHAR(256) NOT NULL,
	creation_time BIGINT,
	last_updated_time BIGINT,
	description VARCHAR(5000),
	CONSTRAINT registered_model_pk PRIMARY KEY (name)
)


CREATE TABLE datasets (
	dataset_uuid VARCHAR(36) NOT NULL,
	experiment_id INTEGER NOT NULL,
	name VARCHAR(500) NOT NULL,
	digest VARCHAR(36) NOT NULL,
	dataset_source_type VARCHAR(36) NOT NULL,
	dataset_source TEXT NOT NULL,
	dataset_schema TEXT,
	dataset_profile TEXT,
	CONSTRAINT dataset_pk PRIMARY KEY (experiment_id, name, digest),
	CONSTRAINT datasets_experiment_id_fkey FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
)


CREATE TABLE experiment_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	experiment_id INTEGER NOT NULL,
	CONSTRAINT experiment_tag_pk PRIMARY KEY (key, experiment_id),
	CONSTRAINT experiment_tags_experiment_id_fkey FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
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
	CONSTRAINT model_version_pk PRIMARY KEY (name, version),
	CONSTRAINT model_versions_name_fkey FOREIGN KEY(name) REFERENCES registered_models (name) ON UPDATE CASCADE
)


CREATE TABLE registered_model_aliases (
	alias VARCHAR(256) NOT NULL,
	version INTEGER NOT NULL,
	name VARCHAR(256) NOT NULL,
	CONSTRAINT registered_model_alias_pk PRIMARY KEY (name, alias),
	CONSTRAINT registered_model_alias_name_fkey FOREIGN KEY(name) REFERENCES registered_models (name) ON DELETE CASCADE ON UPDATE CASCADE
)


CREATE TABLE registered_model_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	name VARCHAR(256) NOT NULL,
	CONSTRAINT registered_model_tag_pk PRIMARY KEY (key, name),
	CONSTRAINT registered_model_tags_name_fkey FOREIGN KEY(name) REFERENCES registered_models (name) ON UPDATE CASCADE
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
	CONSTRAINT run_pk PRIMARY KEY (run_uuid),
	CONSTRAINT runs_experiment_id_fkey FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id),
	CONSTRAINT runs_lifecycle_stage CHECK (lifecycle_stage::text = ANY (ARRAY['active'::character varying, 'deleted'::character varying]::text[])),
	CONSTRAINT runs_status_check CHECK (status::text = ANY (ARRAY['SCHEDULED'::character varying, 'FAILED'::character varying, 'FINISHED'::character varying, 'RUNNING'::character varying, 'KILLED'::character varying]::text[])),
	CONSTRAINT source_type CHECK (source_type::text = ANY (ARRAY['NOTEBOOK'::character varying, 'JOB'::character varying, 'LOCAL'::character varying, 'UNKNOWN'::character varying, 'PROJECT'::character varying]::text[]))
)


CREATE TABLE trace_info (
	request_id VARCHAR(50) NOT NULL,
	experiment_id INTEGER NOT NULL,
	timestamp_ms BIGINT NOT NULL,
	execution_time_ms BIGINT,
	status VARCHAR(50) NOT NULL,
	CONSTRAINT trace_info_pk PRIMARY KEY (request_id),
	CONSTRAINT fk_trace_info_experiment_id FOREIGN KEY(experiment_id) REFERENCES experiments (experiment_id)
)


CREATE TABLE latest_metrics (
	key VARCHAR(250) NOT NULL,
	value DOUBLE PRECISION NOT NULL,
	timestamp BIGINT,
	step BIGINT NOT NULL,
	is_nan BOOLEAN NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	CONSTRAINT latest_metric_pk PRIMARY KEY (key, run_uuid),
	CONSTRAINT latest_metrics_run_uuid_fkey FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
)


CREATE TABLE metrics (
	key VARCHAR(250) NOT NULL,
	value DOUBLE PRECISION NOT NULL,
	timestamp BIGINT NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	step BIGINT DEFAULT '0'::bigint NOT NULL,
	is_nan BOOLEAN DEFAULT false NOT NULL,
	CONSTRAINT metric_pk PRIMARY KEY (key, timestamp, step, run_uuid, value, is_nan),
	CONSTRAINT metrics_run_uuid_fkey FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
)


CREATE TABLE model_version_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	name VARCHAR(256) NOT NULL,
	version INTEGER NOT NULL,
	CONSTRAINT model_version_tag_pk PRIMARY KEY (key, name, version),
	CONSTRAINT model_version_tags_name_version_fkey FOREIGN KEY(name, version) REFERENCES model_versions (name, version) ON UPDATE CASCADE
)


CREATE TABLE params (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(8000) NOT NULL,
	run_uuid VARCHAR(32) NOT NULL,
	CONSTRAINT param_pk PRIMARY KEY (key, run_uuid),
	CONSTRAINT params_run_uuid_fkey FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
)


CREATE TABLE tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(5000),
	run_uuid VARCHAR(32) NOT NULL,
	CONSTRAINT tag_pk PRIMARY KEY (key, run_uuid),
	CONSTRAINT tags_run_uuid_fkey FOREIGN KEY(run_uuid) REFERENCES runs (run_uuid)
)


CREATE TABLE trace_request_metadata (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(8000),
	request_id VARCHAR(50) NOT NULL,
	CONSTRAINT trace_request_metadata_pk PRIMARY KEY (key, request_id),
	CONSTRAINT fk_trace_request_metadata_request_id FOREIGN KEY(request_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
)


CREATE TABLE trace_tags (
	key VARCHAR(250) NOT NULL,
	value VARCHAR(8000),
	request_id VARCHAR(50) NOT NULL,
	CONSTRAINT trace_tag_pk PRIMARY KEY (key, request_id),
	CONSTRAINT fk_trace_tags_request_id FOREIGN KEY(request_id) REFERENCES trace_info (request_id) ON DELETE CASCADE
)

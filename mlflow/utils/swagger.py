swagger_object_dict = {
    "definitions": {
        "Experiment": {
            "type": "object",
            "properties": {
                "experiment_id": {
                    "description": "Unique identifier for the experiment.",
                    "type": "integer",
                    "format": "int64"
                },
                "name": {
                    "description": "Human readable name that identifies this experiment.",
                    "type": "string"
                },
                "artifact_location": {
                    "description": "Location where artifacts for this experiment are stored.",
                    "type": "string"
                }
            }
        },
        "Experiment_details": {
            "type": "object",
            "properties": {
                "experiment": {
                    "description": "Experiment details.",
                    "$ref": "#/definitions/Experiment"
                },
                "runs": {
                    "description": "All (max limit to be imposed) runs associated with this experiment.",
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/RunInfo"
                    }
                }
            }
        },
        "SourceType": {
            "description": "Originating source for a run.",
            "type": "string",
            "enum": ["NOTEBOOK", "JOB", "PROJECT", "LOCAL", "UNKNOWN"]
        },
        "RunStatus": {
            "description": "Status of a run.",
            "type": "string",
            "enum": ["RUNNING", "SCHEDULED", "FINISHED", "FAILED", "KILLED"]
        },
        "RunTag": {
            "description": "Tag for a run.",
            "type": "object",
            "properties": {
                "key": {
                    "description": "The tag key.",
                    "type": "string"
                },
                "value": {
                    "description": "The tag value.",
                    "type": "string"
                }
            }
        },
        "UpdateRunQuerySchema": {
            "type": "object",
            "properties": {
                "run_uuid": {
                    "description": "Run UUID. This field is required.",
                    "required": "true",
                    "type": "string"
                },
                "status": {
                    "description": "Updated status of the run.",
                    "$ref": "#/definitions/RunStatus"
                },
                "end_time": {
                    "description": "Unix timestamp of when the run ended in milliseconds.",
                    "type": "integer",
                    "format": "int64"
                }
            }
        },
        "RunInfo": {
            "type": "object",
            "properties": {
                "run_uuid": {
                    "description": "Unique identifier for the run.",
                    "type": "string"
                },
                "experiment_id": {
                    "description": "The experiment ID.",
                    "type": "integer",
                    "format": "int64"
                },
                "name": {
                    "description": "Human readable name that identifies this run.",
                    "type": "string"
                },
                "source_type": {
                    "description": "Source type",
                    "$ref": "#/definitions/SourceType"
                },
                "source_name": {
                    "description": "Source identifier: GitHub URL, name of notebook, name of job, etc.",
                    "type": "string"
                },
                "user_id": {
                    "description": "User who initiated the run.",
                    "type": "string"
                },
                "status": {
                    "description": "Current status of the run.",
                    "$ref": "#/definitions/RunStatus"
                },
                "start_time": {
                    "description": "Unix timestamp of when the run started in milliseconds.",
                    "type": "integer",
                    "format": "int64"
                },
                "end_time": {
                    "description": "Unix timestamp of when the run ended in milliseconds.",
                    "type": "integer",
                    "format": "int64"
                },
                "source_version": {
                    "description": "Git commit of the code used for the run.",
                    "type": "string"
                },
                "entry_point_name": {
                    "description": "Name of the entry point for the run.",
                    "type": "string"
                },
                "tags": {
                    "description": "Additional metadata key-value pairs.",
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/RunTag"
                    }
                },
                "artifact_uri": {
                    "description": "URI of the directory where artifacts should be uploaded. "
                                    "This can be a local path (starting with \"/\"), or a "
                                    "distributed file system (DFS) path, like s3://bucket/directory"
                                    "or dbfs:/my/directory. If not set, the local ./mlruns "
                                    "directory is chosen.",
                    "type": "string"
                }
            }
        },
        "RunData": {
            "description": "Data logged for the run.",
            "type": "object",
            "properties": {
                "metrics": {
                    "description": "Metrics.",
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/Metric"
                    }
                },
                "params": {
                    "description": "Parameters.",
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/Param"
                    }
                }
            }
        },
        "FileInfo": {
            "description": "File info.",
            "type": "object",
            "properties": {
                "path": {
                    "description": "The relative path to the root_output_uri for the run.",
                    "type": "string"
                },
                "is_dir": {
                    "description": "Whether the file is a directory.",
                    "type": "bool"
                },
                "file_size": {
                    "description": "File size in bytes. Unset for directories.",
                    "type": "integer",
                    "format": "int64"
                }
            }
        },
        "Artifacts": {
            "description": "Artifacts structure.",
            "type": "object",
            "properties": {
                "root_uri": {
                    "description": "The root output directory for the run.",
                    "type": "string"
                },
                "files": {
                    "description": "File location and metadata for artifacts.",
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/FileInfo"
                    }
                }
            }
        },
        "Run": {
            "description": "Run with RunInfo and RunData.",
            "type": "object",
            "properties": {
                "info": {
                    "description": "Run info.",
                    "$ref": "#/definitions/RunInfo"
                },
                "data": {
                    "description": "Run data.",
                    "$ref": "#/definitions/RunData"
                }
            }
        },
        "Param": {
            "description": "Parameters associated with a run: key-value pair of strings.",
            "type": "object",
            "properties": {
                "key": {
                    "description": "Key identifying this parameter.",
                    "type": "string"
                },
                "value": {
                    "description": "Value for this parameter.",
                    "type": "string"
                }
            }
        },
        "Metric": {
            "description": "Metric associated with a run. It is represented as a key-value pair.",
            "type": "object",
            "properties": {
                "key": {
                    "description": "Key identifying this metric.",
                    "type": "string"
                },
                "value": {
                    "description": "Value associated with this metric.",
                    "type": "number",
                    "format": "float"
                },
                "timestamp": {
                    "description": "The timestamp at which this metric was recorded.",
                    "type": "integer",
                    "format": "int64"
                }
            }
        },
        "MetricDataSchema": {
            "description": "Metric with metadata for logging.",
            "type": "object",
            "properties": {
                "run_uuid": {
                    "description": "Unique ID for the run for which metric is recorded.",
                    "type": "string"
                },
                "key": {
                    "description": "Name of the metric.",
                    "type": "string"
                },
                "value": {
                    "description": "Float value for the metric being logged.",
                    "type": "number",
                    "format": "float"
                },
                "timestamp": {
                    "description": "Unix timestamp in milliseconds at the time metric was logged.",
                    "type": "integer",
                    "format": "int64"
                }
            }
        },
        "ParamDataSchema": {
            "description": "Parameter with metadata for logging.",
            "type": "object",
            "properties": {
                "run_uuid": {
                    "description": "Unique ID for the run for which parameter is recorded.",
                    "type": "string"
                },
                "key": {
                    "description": "Name of the parameter.",
                    "type": "string"
                },
                "value": {
                    "description": "String value of the parameter.",
                    "type": "string"
                }
            }
        },
        "ParamQuerySchema": {
            "description": "Parameter identifier.",
            "type": "object",
            "properties": {
                "run_uuid": {
                    "description": "Unique ID for the run for which parameter is recorded.",
                    "type": "string"
                },
                "param_name": {
                    "description": "Name of the parameter.",
                    "type": "string"
                }
            }
        },
        "MetricQuerySchema": {
            "description": "Metric identifier.",
            "type": "object",
            "properties": {
                "run_uuid": {
                    "description": "Unique ID for the run for which metric is recorded.",
                    "type": "string"
                },
                "key": {
                    "description": "Name of the metric.",
                    "type": "string"
                }
            }
        },
        "RunSearchQuerySchema": {
            "description": "Search expression.",
            "type": "object",
            "properties": {
                "experiment_ids": {
                    "description": "Identifier to get an experiment.",
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "format": "int64"
                    }
                },
                "anded_expressions": {
                    "description": "Expressions describing runs.",
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/SearchExpression"
                    }
                }
            }
        },
        "RunInfoDataSchema": {
            "type": "object",
            "properties": {
                "experiment_id": {
                    "description": "Unique identifier for the associated experiment.",
                    "type": "integer",
                    "format": "int64"
                },
                "user_id": {
                    "description": "User ID or LDAP for the user executing the run.",
                    "type": "string"
                },
                "run_name": {
                    "description": "Human readable name for a run.",
                    "type": "string"
                },
                "source_type": {
                    "description": "Originating source for this run. One of Notebook, Job, Project, Local or Unknown.",
                    "$ref": "#/definitions/SourceType"
                },
                "source_name": {
                    "description": "String descriptor for source. For example, name or description of the notebook, or job name.",
                    "type": "string"
                },
                "status": {
                    "description": "Current status of the run. One of RUNNING, SCHEDULE, FINISHED, FAILED, KILLED.",
                    "$ref": "#/definitions/RunStatus"
                },
                "start_time": {
                    "description": "Unix timestamp of when the run started in milliseconds.",
                    "type": "integer",
                    "format": "int64"
                },
                "end_time": {
                    "description": "Unix timestamp of when the run ended in milliseconds.",
                    "type": "integer",
                    "format": "int64"
                },
                "source_version": {
                    "description": "Git version of the source code used to create run.",
                    "type": "string"
                },
                "artifact_uri": {
                    "description": "URI of the directory where artifacts should be uploaded. "
                                   "This can be a local path (starting with \"/\"), or a "
                                   "distributed file system (DFS) path, like "
                                   "s3://bucket/directory or dbfs:/my/directory. If not set, "
                                   "the local ./mlruns directory will be chosen by default.",
                    "type": "string"
                },
                "entry_point_name": {
                    "description": "Name of the entry point for the run.",
                    "type": "string"
                },
                "run_tags": {
                    "description": "Additional metadata for run in key-value pairs.",
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/RunTag"
                    }
                },

            }
        },
        "ExperimentIdQuerySchema": {
            "type": "object",
            "properties": {
                "experiment_id": {
                    "description": "Identifier to get an experiment. This field is required.",
                    "type": "integer",
                    "format": "int64"
                }
            }
        },
        "RunUUIDQuerySchema": {
            "type": "object",
            "properties": {
                "run_uuid": {
                    "description": "Run UUID. This field is required.",
                    "type": "string"
                }
            }
        },
        "MetricSearchExpression": {
            "description": "Metric search expression",
            "type": "object",
            "properties": {
                "float": {
                    "description": "Float clause for comparison.",
                    "$ref": "#/definitions/FloatClause"
                },
                "key": {
                    "description": "Metric key for search",
                    "type": "string"
                }
            }
        },
        "ParameterSearchExpression": {
            "description": "Parameter search expression",
            "type": "object",
            "properties": {
                "string": {
                    "description": "String clause for comparison.",
                    "$ref": "#/definitions/StringClause"
                },
                "key": {
                    "description": "Param key for search",
                    "type": "string"
                }
            }
        },
        "StringClause": {
            "description": "String clause",
            "type": "object",
            "properties": {
                "comparator": {
                    "description": "OneOf (==, !=, ~)",
                    "type": "string"
                },
                "value": {
                    "description": "String value for comparison.",
                    "type": "string"
                }
            }
        },
        "FloatClause": {
            "description": "Float clause",
            "type": "object",
            "properties": {
                "comparator": {
                    "description": "OneOf (>, >=, ==, !=, <=, <)",
                    "type": "string"
                },
                "value": {
                    "description": "Float value for comparison.",
                    "type": "number",
                    "format": "float"
                }
            }
        }
    }
}
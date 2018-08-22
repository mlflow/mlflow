swagger_object_dict = {
    "definitions": {
        "Artifacts": {
            "description": "Artifacts structure.",
            "type": "object",
            "properties": {
                "root_uri": {
                    "description": "Root artifact directory for the run.",
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
        "Experiment": {
            "type": "object",
            "description": "Experiment",
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
        "ExperimentBody": {
            "type": "object",
            "properties": {
                "name": {
                    "description": "Experiment name. This field is required.",
                    "type": "string"
                },
                "artifact_location": {
                    "description": "Location where all artifacts for this experiment are stored."
                                   " If not provided, the remote server will select an "
                                   "appropriate default.",
                    "type": "string"
                }
            }
        },
        "ExperimentDetails": {
            "type": "object",
            "properties": {
                "experiment": {
                    "description": "Returns experiment details.",
                    "$ref": "#/definitions/Experiment"
                },
                "runs": {
                    "description": "All (max limit to be imposed) runs associated with "
                                   "this experiment.",
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/RunInfoBase"
                    }
                }
            }
        },
        "ExperimentId": {
            "type": "object",
            "properties": {
                "experiment_id": {
                    "description": "Unique identifier for created experiment.",
                    "type": "integer",
                    "format": "int64"
                }
            }
        },
        "Experiments": {
            "type": "object",
            "properties": {
                "experiments": {
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/Experiment"
                    }
                }
            }
        },
        "FileInfo": {
            "description": "Metadata of a single artifact file or directory.",
            "type": "object",
            "properties": {
                "path": {
                    "description": "Path relative to the root artifact directory run.",
                    "type": "string"
                },
                "is_dir": {
                    "description": "Whether the file is a directory.",
                    "type": "boolean"
                },
                "file_size": {
                    "description": "Size in bytes. Unset for directories.",
                    "type": "integer",
                    "format": "int64"
                }
            }
        },
        "FloatClause": {
            "description": "Float clause.",
            "type": "object",
            "properties": {
                "comparator": {
                    "description": "OneOf (\">\", \">=\", \"==\", \"!=\", \"<=\", \"<\")",
                    "type": "string"
                },
                "value": {
                    "description": "Float value for comparison.",
                    "type": "number",
                    "format": "float"
                }
            }
        },
        "Metric": {
            "description": "Metric associated with a run, represented as a key-value pair.",
            "type": "object",
            "properties": {
                "metric": {
                    "type": "object",
                    "$ref": "#/definitions/MetricBase"
                }
            }
        },
        "MetricBase": {
            "description": "Metric associated with a run, represented as a key-value pair.",
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
        "MetricBody": {
            "description": "Metric with metadata for logging.",
            "type": "object",
            "properties": {
                "run_uuid": {
                    "description": "ID of the run under which to log the metric. "
                                   "This field is required.",
                    "type": "string"
                },
                "key": {
                    "description": "Name of the metric. This field is required.",
                    "type": "string"
                },
                "value": {
                    "description": "Float value of the metric being logged. "
                                   "This field is required.",
                    "type": "number",
                    "format": "float"
                },
                "timestamp": {
                    "description": "Unix timestamp in milliseconds at the time metric was logged."
                                   "This field is required.",
                    "type": "integer",
                    "format": "int64"
                }
            }
        },
        "MetricHistory": {
            "description": "Metric historical values.",
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/MetricBase"
                    }
                }
            }
        },
        "MetricSearchExpression": {
            "description": "Metric search expression.",
            "type": "object",
            "properties": {
                "metric": {
                    "description": "Metric search expression.",
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
                }
            }
        },
        "ParamBase": {
            "description": "Param associated with a run.",
            "type": "object",
            "properties": {
                "key": {
                    "description": "Key identifying this param.",
                    "type": "string"
                },
                "value": {
                    "description": "Value associated with this param.",
                    "type": "string"
                }
            }
        },
        "ParamBody": {
            "description": "Parameter with metadata for logging.",
            "type": "object",
            "properties": {
                "run_uuid": {
                    "description": "ID of the run under which to log the param. "
                                   "This field is required.",
                    "type": "string"
                },
                "key": {
                    "description": "Name of the param. "
                                   "This field is required.",
                    "type": "string"
                },
                "value": {
                    "description": "String value of the param being logged. "
                                   "This field is required.",
                    "type": "string"
                }
            }
        },
        "Parameter": {
            "description": "Parameters associated with a run: key-value pair of strings.",
            "type": "object",
            "properties": {
                "parameter": {
                    "type": "object",
                    "$ref": "#/definitions/ParamBase"
                }
            }
        },
        "ParameterSearchExpression": {
            "description": "Parameter search expression.",
            "type": "object",
            "properties": {
                "parameter": {
                    "description": "Parameter search expression.",
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
                }
            }
        },
        "Run": {
            "description": "Metadata of the newly created run.",
            "type": "object",
            "properties": {
                "run": {
                    "type": "object",
                    "$ref": "#/definitions/RunInfo"
                }
            }
        },
        "RunData": {
            "description": "Run data (metrics, params, etc).",
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
                    "description": "Params.",
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/ParamBase"
                    }
                }
            }
        },
        "RunInfo": {
            "type": "object",
            "properties": {
                "info": {
                    "type": "object",
                    "$ref": "#/definitions/RunInfoBase"
                }
            }
        },
        "RunInfo2": {
            "type": "object",
            "properties": {
                "run_info": {
                    "type": "object",
                    "$ref": "#/definitions/RunInfoBase"
                }
            }
        },
        "RunInfoAndData": {
            "description": "Run with RunInfo and RunData.",
            "type": "object",
            "properties": {
                "run": {
                    "type": "object",
                    "$ref": "#/definitions/RunInfoAndDataBase"
                }
            }
        },
        "RunInfoAndDataBase": {
            "description": "A single run with RunInfo and RunData.",
            "type": "object",
            "properties": {
                "info": {
                    "description": "Run metadata.",
                    "$ref": "#/definitions/RunInfoBase"
                },
                "data": {
                    "description": "Run data.",
                    "$ref": "#/definitions/RunData"
                }
            }
        },
        "RunInfoBase": {
            "type": "object",
            "description": "Metadata of a single run.",
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
                    "description": "Source identifier: GitHub URL, name of notebook, "
                                   "name of job, etc.",
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
        "RunInfoBody": {
            "type": "object",
            "properties": {
                "experiment_id": {
                    "description": "ID of the associated experiment.",
                    "type": "integer",
                    "format": "int64"
                },
                "user_id": {
                    "description": "ID of the user executing the run.",
                    "type": "string"
                },
                "run_name": {
                    "description": "Human readable name for a run.",
                    "type": "string"
                },
                "source_type": {
                    "description": "Originating source for the run.",
                    "$ref": "#/definitions/SourceType"
                },
                "source_name": {
                    "description": "String descriptor for the run's source. For example, name or "
                                   "description of a notebook, or the URL or path to a project.",
                    "type": "string"
                },
                "entry_point_name": {
                    "description": "Name of the project entry point associated with the current "
                                   "run, if any.",
                    "type": "string"
                },
                "start_time": {
                    "description": "Unix timestamp of when the run started in milliseconds.",
                    "type": "integer",
                    "format": "int64"
                },
                "source_version": {
                    "description": "Git version of the source code used to create run.",
                    "type": "string"
                },
                "tags": {
                    "description": "Additional metadata for run.",
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/RunTag"
                    }
                }
            }
        },
        "RunStatus": {
            "description": "Status of a run.\n"
                           "* RUNNING - Has been initiated.\n"
                           "* SCHEDULED - Scheduled to run at a later time.\n"
                           "* FINISHED - Run has completed.\n"
                           "* FAILED - Execution failed.\n"
                           "* KILLED - Run killed by user.",
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
        "Runs": {
            "description": "Run with RunInfo and RunData.",
            "type": "object",
            "properties": {
                "runs": {
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/RunInfoAndDataBase"
                    }
                }
            }
        },
        "SourceType": {
            "description": "Description of the source that generated a run.\n"
                           "* NOTEBOOK - Within Databricks Notebook environment.\n"
                           "* JOB - Scheduled or Run Now Job.\n"
                           "* PROJECT - As a prepackaged project: either a docker image or "
                           "github source, ... etc.\n"
                           "* LOCAL - Local run: Using CLI, IDE, or local notebook.\n"
                           "* UNKNOWN - Unknown source type.",
            "type": "string",
            "enum": ["NOTEBOOK", "JOB", "PROJECT", "LOCAL", "UNKNOWN"]
        },
        "StringClause": {
            "description": "String clause",
            "type": "object",
            "properties": {
                "comparator": {
                    "description": "OneOf (\"==\", \"!=\", \"~\")",
                    "type": "string"
                },
                "value": {
                    "description": "String value for comparison.",
                    "type": "string"
                }
            }
        },
        "UpdateRunBody": {
            "type": "object",
            "properties": {
                "run_uuid": {
                    "description": "Run UUID. This field is required.",
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
        }
    }
}

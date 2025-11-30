/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

/* eslint-disable */

/**
 * DO NOT EDIT!!!
 *
 * @NOTE(dli) 12-21-2016
 *   This file is generated. For now, it is a snapshot of the proto messages as of
 *   Sep 17, 2018 6:48:22 PM. We will update the generation pipeline to actually
 *   place these generated records in the correct location shortly.
 */

import Immutable from 'immutable';
import { RecordUtils } from '../../common/sdk/RecordUtils';
import { ModelBuilder } from '../../common/sdk/ModelBuilder';

export const Metric = Immutable.Record(
  {
    // optional STRING
    key: undefined,

    // optional FLOAT
    value: undefined,

    // optional INT64
    timestamp: undefined,

    // optional INT64
    step: undefined,
  },
  'Metric',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(Metric as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_Metric = ModelBuilder.extend(Metric, {
  getKey() {
    return this.key !== undefined ? this.key : '';
  },
  getValue() {
    return this.value !== undefined ? this.value : 0.0;
  },
  getTimestamp() {
    return this.timestamp !== undefined ? this.timestamp : 0;
  },
  getStep() {
    return this.step !== undefined ? this.step : 0;
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = Metric.fromJs(pojo);
 */
(Metric as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (Metric as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_Metric(pojoWithNestedImmutables);
};

export const Param = Immutable.Record(
  {
    // optional STRING
    key: undefined,

    // optional STRING
    value: undefined,
  },
  'Param',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(Param as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_Param = ModelBuilder.extend(Param, {
  getKey() {
    return this.key !== undefined ? this.key : '';
  },
  getValue() {
    return this.value !== undefined ? this.value : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = Param.fromJs(pojo);
 */
(Param as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (Param as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_Param(pojoWithNestedImmutables);
};

const RunInfo = Immutable.Record(
  {
    // optional STRING
    run_uuid: undefined,

    // optional STRING
    experiment_id: undefined,

    // optional STRING
    user_id: undefined,

    // optional STRING
    run_name: undefined,

    // optional RunStatus
    status: undefined,

    // optional INT64
    start_time: undefined,

    // optional INT64
    end_time: undefined,

    // optional STRING
    artifact_uri: undefined,

    // optional STRING
    lifecycle_stage: undefined,
  },
  'RunInfo',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(RunInfo as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_RunInfo = ModelBuilder.extend(RunInfo, {
  getRunUuid() {
    return this.run_uuid !== undefined ? this.run_uuid : '';
  },
  getExperimentId() {
    return this.experiment_id !== undefined ? this.experiment_id : '0';
  },
  getStatus() {
    return this.status !== undefined ? this.status : 'RUNNING';
  },
  getStartTime() {
    return this.start_time !== undefined ? this.start_time : 0;
  },
  getEndTime() {
    return this.end_time !== undefined ? this.end_time : 0;
  },
  getArtifactUri() {
    return this.artifact_uri !== undefined ? this.artifact_uri : '';
  },
  getLifecycleStage() {
    return this.lifecycle_stage !== undefined ? this.lifecycle_stage : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = RunInfo.fromJs(pojo);
 */
(RunInfo as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (RunInfo as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_RunInfo(pojoWithNestedImmutables);
};

const RunData = Immutable.Record(
  {
    // repeated Metric
    metrics: Immutable.List(),

    // repeated Param
    params: Immutable.List(),

    // repeated RunTag
    tags: Immutable.List(),
  },
  'RunData',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(RunData as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    case 'metrics':
      return Immutable.List(value.map((element: any) => (Metric as any).fromJs(element)));
    case 'params':
      return Immutable.List(value.map((element: any) => (Param as any).fromJs(element)));
    case 'tags':
      return Immutable.List(value.map((element: any) => (RunTag as any).fromJs(element)));
    default:
      return Immutable.fromJS(value);
  }
};

const extended_RunData = ModelBuilder.extend(RunData, {});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = RunData.fromJs(pojo);
 */
(RunData as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (RunData as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_RunData(pojoWithNestedImmutables);
};

const Run = Immutable.Record(
  {
    // optional RunInfo
    info: undefined,

    // optional RunData
    data: undefined,
  },
  'Run',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(Run as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    case 'info':
      return (RunInfo as any).fromJs(value);
    case 'data':
      return (RunData as any).fromJs(value);
    default:
      return Immutable.fromJS(value);
  }
};

const extended_Run = ModelBuilder.extend(Run, {
  getInfo() {
    return this.info !== undefined ? this.info : (RunInfo as any).fromJs({});
  },
  getData() {
    return this.data !== undefined ? this.data : (RunData as any).fromJs({});
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = Run.fromJs(pojo);
 */
(Run as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (Run as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_Run(pojoWithNestedImmutables);
};

const Experiment = Immutable.Record(
  {
    // optional STRING
    experiment_id: undefined,

    // optional STRING
    name: undefined,

    // optional STRING
    artifact_location: undefined,

    // optional STRING
    lifecycle_stage: undefined,

    // optional INT64
    last_update_time: undefined,

    // optional INT64
    creation_time: undefined,

    // repeated ExperimentTag
    tags: Immutable.List(),
  },
  'Experiment',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(Experiment as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    case 'tags':
      return Immutable.List(value.map((element: any) => (ExperimentTag as any).fromJs(element)));
    default:
      return Immutable.fromJS(value);
  }
};

const extended_Experiment = ModelBuilder.extend(Experiment, {
  getExperimentId() {
    return this.experiment_id !== undefined ? this.experiment_id : '0';
  },
  getName() {
    return this.name !== undefined ? this.name : '';
  },
  getArtifactLocation() {
    return this.artifact_location !== undefined ? this.artifact_location : '';
  },
  getLifecycleStage() {
    return this.lifecycle_stage !== undefined ? this.lifecycle_stage : '';
  },
  getLastUpdateTime() {
    return this.last_update_time !== undefined ? this.last_update_time : 0;
  },
  getCreationTime() {
    return this.creation_time !== undefined ? this.creation_time : 0;
  },
  getTags() {
    return this.tags !== undefined ? this.tags : [];
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = Experiment.fromJs(pojo);
 */
(Experiment as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (Experiment as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_Experiment(pojoWithNestedImmutables);
};

const CreateExperiment = Immutable.Record(
  {
    // required STRING
    name: undefined,

    // optional STRING
    artifact_location: undefined,
  },
  'CreateExperiment',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(CreateExperiment as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_CreateExperiment = ModelBuilder.extend(CreateExperiment, {
  getName() {
    return this.name !== undefined ? this.name : '';
  },
  getArtifactLocation() {
    return this.artifact_location !== undefined ? this.artifact_location : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = CreateExperiment.fromJs(pojo);
 */
(CreateExperiment as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (CreateExperiment as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_CreateExperiment(pojoWithNestedImmutables);
};

const SearchExperiments = Immutable.Record(
  {
    // optional ViewType
    view_type: undefined,
  },
  'SearchExperiments',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(SearchExperiments as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_SearchExperiments = ModelBuilder.extend(SearchExperiments, {
  getViewType() {
    return this.view_type !== undefined ? this.view_type : 'ACTIVE_ONLY';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = SearchExperiments.fromJs(pojo);
 */
(SearchExperiments as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (SearchExperiments as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_SearchExperiments(pojoWithNestedImmutables);
};

const GetExperiment = Immutable.Record(
  {
    // required STRING
    experiment_id: undefined,
  },
  'GetExperiment',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(GetExperiment as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_GetExperiment = ModelBuilder.extend(GetExperiment, {
  getExperimentId() {
    return this.experiment_id !== undefined ? this.experiment_id : '0';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = GetExperiment.fromJs(pojo);
 */
(GetExperiment as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (GetExperiment as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_GetExperiment(pojoWithNestedImmutables);
};

const GetRun = Immutable.Record(
  {
    // required STRING
    run_uuid: undefined,
  },
  'GetRun',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(GetRun as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_GetRun = ModelBuilder.extend(GetRun, {
  getRunUuid() {
    return this.run_uuid !== undefined ? this.run_uuid : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = GetRun.fromJs(pojo);
 */
(GetRun as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (GetRun as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_GetRun(pojoWithNestedImmutables);
};

const MetricSearchExpression = Immutable.Record(
  {
    // optional STRING
    key: undefined,

    // optional FloatClause
    float: undefined,
  },
  'MetricSearchExpression',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(MetricSearchExpression as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    case 'float':
      return (FloatClause as any).fromJs(value);
    default:
      return Immutable.fromJS(value);
  }
};

const extended_MetricSearchExpression = ModelBuilder.extend(MetricSearchExpression, {
  getKey() {
    return this.key !== undefined ? this.key : '';
  },
  getFloat() {
    return this.float !== undefined ? this.float : (FloatClause as any).fromJs({});
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = MetricSearchExpression.fromJs(pojo);
 */
(MetricSearchExpression as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (MetricSearchExpression as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_MetricSearchExpression(pojoWithNestedImmutables);
};

const ParameterSearchExpression = Immutable.Record(
  {
    // optional STRING
    key: undefined,

    // optional StringClause
    string: undefined,
  },
  'ParameterSearchExpression',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(ParameterSearchExpression as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    case 'string':
      return (StringClause as any).fromJs(value);
    default:
      return Immutable.fromJS(value);
  }
};

const extended_ParameterSearchExpression = ModelBuilder.extend(ParameterSearchExpression, {
  getKey() {
    return this.key !== undefined ? this.key : '';
  },
  getString() {
    return this.string !== undefined ? this.string : (StringClause as any).fromJs({});
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = ParameterSearchExpression.fromJs(pojo);
 */
(ParameterSearchExpression as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (ParameterSearchExpression as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_ParameterSearchExpression(pojoWithNestedImmutables);
};

const SearchExpression = Immutable.Record(
  {
    // optional MetricSearchExpression
    metric: undefined,

    // optional ParameterSearchExpression
    parameter: undefined,
  },
  'SearchExpression',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(SearchExpression as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    case 'metric':
      return (MetricSearchExpression as any).fromJs(value);
    case 'parameter':
      return (ParameterSearchExpression as any).fromJs(value);
    default:
      return Immutable.fromJS(value);
  }
};

const extended_SearchExpression = ModelBuilder.extend(SearchExpression, {
  getMetric() {
    return this.metric !== undefined ? this.metric : (MetricSearchExpression as any).fromJs({});
  },
  getParameter() {
    return this.parameter !== undefined ? this.parameter : (ParameterSearchExpression as any).fromJs({});
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = SearchExpression.fromJs(pojo);
 */
(SearchExpression as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (SearchExpression as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_SearchExpression(pojoWithNestedImmutables);
};

const FloatClause = Immutable.Record(
  {
    // optional STRING
    comparator: undefined,

    // optional FLOAT
    value: undefined,
  },
  'FloatClause',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(FloatClause as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_FloatClause = ModelBuilder.extend(FloatClause, {
  getComparator() {
    return this.comparator !== undefined ? this.comparator : '';
  },
  getValue() {
    return this.value !== undefined ? this.value : 0.0;
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = FloatClause.fromJs(pojo);
 */
(FloatClause as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (FloatClause as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_FloatClause(pojoWithNestedImmutables);
};

const StringClause = Immutable.Record(
  {
    // optional STRING
    comparator: undefined,

    // optional STRING
    value: undefined,
  },
  'StringClause',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(StringClause as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_StringClause = ModelBuilder.extend(StringClause, {
  getComparator() {
    return this.comparator !== undefined ? this.comparator : '';
  },
  getValue() {
    return this.value !== undefined ? this.value : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = StringClause.fromJs(pojo);
 */
(StringClause as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (StringClause as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_StringClause(pojoWithNestedImmutables);
};

const SearchRuns = Immutable.Record(
  {
    // repeated STRING
    experiment_ids: Immutable.List(),

    // optional ViewType
    run_view_type: 'ACTIVE_ONLY',
  },
  'SearchRuns',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(SearchRuns as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    case 'experiment_ids':
      return Immutable.List(value);
    default:
      return Immutable.fromJS(value);
  }
};

const extended_SearchRuns = ModelBuilder.extend(SearchRuns, {
  getRunViewType() {
    return this.run_view_type !== undefined ? this.run_view_type : 'ACTIVE_ONLY';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = SearchRuns.fromJs(pojo);
 */
(SearchRuns as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (SearchRuns as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_SearchRuns(pojoWithNestedImmutables);
};

const FileInfo = Immutable.Record(
  {
    // optional STRING
    path: undefined,

    // optional BOOL
    is_dir: undefined,

    // optional INT64
    file_size: undefined,
  },
  'FileInfo',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(FileInfo as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_FileInfo = ModelBuilder.extend(FileInfo, {
  getPath() {
    return this.path !== undefined ? this.path : '';
  },
  getIsDir() {
    return this.is_dir !== undefined ? this.is_dir : false;
  },
  getFileSize() {
    return this.file_size !== undefined ? this.file_size : 0;
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = FileInfo.fromJs(pojo);
 */
(FileInfo as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (FileInfo as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_FileInfo(pojoWithNestedImmutables);
};

const ListArtifacts = Immutable.Record(
  {
    // optional STRING
    run_uuid: undefined,

    // optional STRING
    path: undefined,
  },
  'ListArtifacts',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(ListArtifacts as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_ListArtifacts = ModelBuilder.extend(ListArtifacts, {
  getRunUuid() {
    return this.run_uuid !== undefined ? this.run_uuid : '';
  },
  getPath() {
    return this.path !== undefined ? this.path : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = ListArtifacts.fromJs(pojo);
 */
(ListArtifacts as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (ListArtifacts as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_ListArtifacts(pojoWithNestedImmutables);
};

const GetArtifact = Immutable.Record(
  {
    // optional STRING
    run_uuid: undefined,

    // optional STRING
    path: undefined,
  },
  'GetArtifact',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(GetArtifact as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_GetArtifact = ModelBuilder.extend(GetArtifact, {
  getRunUuid() {
    return this.run_uuid !== undefined ? this.run_uuid : '';
  },
  getPath() {
    return this.path !== undefined ? this.path : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = GetArtifact.fromJs(pojo);
 */
(GetArtifact as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (GetArtifact as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_GetArtifact(pojoWithNestedImmutables);
};

const GetMetricHistory = Immutable.Record(
  {
    // required STRING
    run_uuid: undefined,

    // required STRING
    metric_key: undefined,
  },
  'GetMetricHistory',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(GetMetricHistory as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_GetMetricHistory = ModelBuilder.extend(GetMetricHistory, {
  getRunUuid() {
    return this.run_uuid !== undefined ? this.run_uuid : '';
  },
  getMetricKey() {
    return this.metric_key !== undefined ? this.metric_key : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = GetMetricHistory.fromJs(pojo);
 */
(GetMetricHistory as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (GetMetricHistory as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_GetMetricHistory(pojoWithNestedImmutables);
};

export const RunTag = Immutable.Record(
  {
    // optional STRING
    key: undefined,

    // optional STRING
    value: undefined,
  },
  'RunTag',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(RunTag as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_RunTag = ModelBuilder.extend(RunTag, {
  getKey() {
    return this.key !== undefined ? this.key : '';
  },
  getValue() {
    return this.value !== undefined ? this.value : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = RunTag.fromJs(pojo);
 */
(RunTag as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (RunTag as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_RunTag(pojoWithNestedImmutables);
};

export const ExperimentTag = Immutable.Record(
  {
    // optional STRING
    key: undefined,

    // optional STRING
    value: undefined,
  },
  'ExperimentTag',
);

/**
 * By default Immutable.fromJS will translate an object field in JSON into Immutable.Map.
 * This reviver allow us to keep the Immutable.Record type when serializing JSON message
 * into nested Immutable Record class.
 */
(ExperimentTag as any).fromJsReviver = function fromJsReviver(key: any, value: any) {
  switch (key) {
    default:
      return Immutable.fromJS(value);
  }
};

const extended_ExperimentTag = ModelBuilder.extend(ExperimentTag, {
  getKey() {
    return this.key !== undefined ? this.key : '';
  },
  getValue() {
    return this.value !== undefined ? this.value : '';
  },
});

/**
 * This is a customized fromJs function used to translate plain old Javascript
 * objects into this Immutable Record.  Example usage:
 *
 *   // The pojo is your javascript object
 *   const record = RunTag.fromJs(pojo);
 */
(ExperimentTag as any).fromJs = function fromJs(pojo: any) {
  const pojoWithNestedImmutables = RecordUtils.fromJs(pojo, (RunTag as any).fromJsReviver);
  // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
  return new extended_ExperimentTag(pojoWithNestedImmutables);
};

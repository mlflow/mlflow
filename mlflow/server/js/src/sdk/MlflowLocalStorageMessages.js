import Immutable from "immutable";

export const ExperimentPageState = Immutable.Record({
  // required STRING
  paramKeyFilterString: "",

  // required STRING
  metricKeyFilterString: "",

  // required STRING
  getExperimentRequestId: "",

  // required STRING
  searchRunsRequestId: "",

  // required STRING
  searchInput: "",

  // required INT
  lastExperimentId: undefined,

  // required STRING
  lifecycleFilter: "",

}, 'ExperimentPageState');


export const ExperimentViewState = Immutable.Record({
  runsHiddenByExpander: {},
  // By default all runs are expanded. In this state, runs are explicitly expanded or unexpanded.
  runsExpanded: {},
  runsSelected: {},
  paramKeyFilterInput: '',
  metricKeyFilterInput: '',
  lifecycleFilterInput: '',
  searchInput: '',
  searchErrorMessage: undefined,
  sort: {
    ascending: false,
    isMetric: false,
    isParam: false,
    key: "start_time"
  },
  showMultiColumns: true,
  showDeleteRunModal: false,
  showRestoreRunModal: false,
  // Arrays of "unbagged", or split-out metrics and parameters. We maintain these as lists to help
  // keep them ordered (i.e. splitting out a column shouldn't change the ordering of columns
  // that have already been split out)
  unbaggedMetrics: [],
  unbaggedParams: [],
}, 'ExperimentViewState');
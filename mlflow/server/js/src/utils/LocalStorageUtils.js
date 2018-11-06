import _ from 'lodash';
import {LIFECYCLE_FILTER} from "../components/ExperimentPage";
import { ExperimentViewState, ExperimentPageState} from "../sdk/MlflowLocalStorageMessages";

/**
 * Utils for working with local storage.
 */
export default class LocalStorageUtils {
  /**
   * Protocol version of MLflow's local storage. Should be incremented on any breaking change in how
   * data persisted in local storage is used, to prevent old (invalid) cached data from being loaded
   * and breaking the application.
   */
  static version = "1.0";

  /**
   * Return a LocalStorageStore corresponding to the specified component and ID, where the ID
   * can be used to disambiguate between multiple instances of cached data for the same component
   * (e.g. cached data for multiple experiments).
   */
  static getStore(componentName, id) {
    return new LocalStorageStore([componentName, id].join("-"));
  }
}

/**
 * Interface to browser local storage that allows for setting key-value pairs under the specified
 * "scope".
 */
class LocalStorageStore {
  constructor(scope) {
    this.scope = scope;
  }
  static reactComponentStateKey = "ReactComponentState";

  /**
   * Loads React component state cached in local storage into a vanilla JS object.
   */
  loadComponentState(defaultState) {
    const cachedState = JSON.parse(this.getItem(LocalStorageStore.reactComponentStateKey));
    if (cachedState) {
      return {
        ...defaultState,
        ...cachedState,
      };
    }
    return _.cloneDeep(defaultState);
  }

  /**
   * Save React component state in local storage.
   * @param stateRecord: Immutable.Record instance containing component state.
   */
  saveComponentState(stateRecord) {
    this.setItem(
      LocalStorageStore.reactComponentStateKey, JSON.stringify(stateRecord.toJSON()));
  }

  /**
   * Helper method for constructing a scoped key to use for setting/getting values in
   * local storage.
   */
  withScopePrefix(key) {
    return ["MLflowLocalStorage", LocalStorageUtils.version, this.scope, key].join("-");
  }

  /** Save the specified key-value pair in local storage. */
  setItem(key, value) {
    window.localStorage.setItem(this.withScopePrefix(key), value);
  }

  /** Fetch the value corresponding to the passed-in key from local storage. */
  getItem(key) {
    return window.localStorage.getItem(this.withScopePrefix(key));
  }
}

export class ExperimentPageState2 {
  constructor({
    paramKeyFilterString,
    metricKeyFilterString,
    getExperimentRequestId,
    searchRunsRequestId,
    searchInput,
    lastExperimentId,
    lifecycleFilter,
  }) {
    this.paramKeyFilterString = paramKeyFilterString;
    this.metricKeyFilterString = metricKeyFilterString;
    this.getExperimentRequestId = getExperimentRequestId;
    this.searchRunsRequestId = searchRunsRequestId;
    this.searchInput = searchInput;
    this.lastExperimentId = lastExperimentId;
    this.lifecycleFilter = lifecycleFilter;
  }

  toDict() {
    return {
      paramKeyFilterString: this.paramKeyFilterString,
      metricKeyFilterString: this.metricKeyFilterString,
      getExperimentRequestId: this.getExperimentRequestId,
      searchRunsRequestId: this.searchRunsRequestId,
      searchInput: this.searchInput,
      lastExperimentId: this.lastExperimentId,
      lifecycleFilter: this.lifecycleFilter,
    }
  }
}

export class ExperimentViewState2 {
  constructor({
    runsHiddenByExpander,
    // By default all runs are expanded. In this state, runs are explicitly expanded or unexpanded.
    runsExpanded,
    runsSelected,
    paramKeyFilterInput,
    metricKeyFilterInput,
    lifecycleFilterInput,
    searchInput,
    searchErrorMessage,
    sort,
    showMultiColumns,
    showDeleteRunModal,
    showRestoreRunModal,
    // Arrays of "unbagged", or split-out metrics and parameters. We maintain these as lists to help
    // keep them ordered (i.e. splitting out a column shouldn't change the ordering of columns
    // that have already been split out)
    unbaggedMetrics,
    unbaggedParams,
  }) {
    this.runsHiddenByExpander = runsHiddenByExpander;
    this.runsExpanded = runsExpanded;
    this.runsSelected = runsSelected;
    this.paramKeyFilterInput = paramKeyFilterInput;
    this.metricKeyFilterInput = metricKeyFilterInput;
    this.lifecycleFilterInput = lifecycleFilterInput;
    this.searchInput = searchInput;
    this.searchErrorMessage = searchErrorMessage;
    this.sort = sort;
    this.showMultiColumns = showMultiColumns;
    this.showDeleteRunModal = showDeleteRunModal;
    this.showRestoreRunModal = showRestoreRunModal;
    this.unbaggedMetrics = unbaggedMetrics;
    this.unbaggedParams = unbaggedParams;
  }

  toDict() {
    return {
      runsHiddenByExpander: this.runsHiddenByExpander,
      runsExpanded: this.runsExpanded,
      runsSelected: this.runsSelected,
      paramKeyFilterInput: this.paramKeyFilterInput,
      metricKeyFilterInput: this.metricKeyFilterInput,
      lifecycleFilterInput: this.lifecycleFilterInput,
      searchInput: this.searchInput,
      searchErrorMessage: this.searchErrorMessage,
      sort: this.sort,
      showMultiColumns: this.showMultiColumns,
      showDeleteRunModal: this.showDeleteRunModal,
      showRestoreRunModal: this.showRestoreRunModal,
      unbaggedMetrics: this.unbaggedMetrics,
      unbaggedParams: this.unbaggedParams,
    }
  }

}

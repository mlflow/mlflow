import _ from 'lodash';

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
  static reactComponentStateKey = "ReactComponentState";

  /**
   * Return a LocalStorageStore corresponding to the specified component and ID, where the ID
   * can be used to disambiguate between multiple instances of cached data for the same component
   * (e.g. cached data for multiple experiments).
   */
  static getStore(componentName, id) {
    return new LocalStorageStore([componentName, id].join("-"));
  }

  /** Loads React component state cached in local storage. */
  static loadComponentState(store, defaultState) {
    const cachedState = JSON.parse(store.getItem(LocalStorageUtils.reactComponentStateKey));
    if (cachedState) {
      return {
        ...defaultState,
        ...cachedState,
      };
    }
    return _.cloneDeep(defaultState);
  }

  /** Save React component state in local storage. */
  static saveComponentState(store, state) {
    store.setItem(LocalStorageUtils.reactComponentStateKey, JSON.stringify(state));
  }
}

/**
 * Interface to browser local storage that allows for setting key-value pairs under the specified
 * "scope".
 */
export class LocalStorageStore {
  constructor(scope) {
    this.scope = scope;
  }

  /**
   * Helper method for constructing a scoped key to use for setting/getting values in
   * local storage.
   */
  withScopePrefix(key) {
    return ["MLflow", LocalStorageUtils.version, this.scope, key].join("-");
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


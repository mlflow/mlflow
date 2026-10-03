/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

/**
 * Utils for working with local storage.
 */
// Keep one UI-state entry from consuming most common 5 MiB per-origin quotas.
const MAX_STORAGE_ITEM_SIZE = 1_000_000;

const isQuotaExceededError = (error: unknown) =>
  error instanceof DOMException &&
  (error.name === 'QuotaExceededError' ||
    error.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
    error.code === 22 ||
    error.code === 1014);

export const getStorageItem = (storage: Storage, key: string) => {
  const value = storage.getItem(key);
  if (value !== null && value.length > MAX_STORAGE_ITEM_SIZE) {
    storage.removeItem(key);
    return null;
  }
  return value;
};

export const setStorageItem = (storage: Storage, key: string, value: string) => {
  if (value.length > MAX_STORAGE_ITEM_SIZE) {
    return;
  }

  try {
    storage.setItem(key, value);
  } catch (error) {
    if (!isQuotaExceededError(error)) {
      throw error;
    }
  }
};

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export default class LocalStorageUtils {
  /**
   * Protocol version of MLflow's local storage. Should be incremented on any breaking change in how
   * data persisted in local storage is used, to prevent old (invalid) cached data from being loaded
   * and breaking the application.
   */
  static version = '1.1';

  /**
   * Return a LocalStorageStore corresponding to the specified component and ID, where the ID
   * can be used to disambiguate between multiple instances of cached data for the same component
   * (e.g. cached data for multiple experiments).
   */
  static getStoreForComponent(componentName: any, id: any) {
    return new LocalStorageStore([componentName, id].join('-'), 'localStorage');
  }

  static getSessionScopedStoreForComponent(componentName: any, id: any) {
    return new LocalStorageStore([componentName, id].join('-'), 'sessionStorage');
  }
}

/**
 * Interface to browser local storage that allows for setting key-value pairs under the specified
 * "scope".
 */
class LocalStorageStore {
  constructor(scope: any, type: any) {
    this.scope = scope;
    if (type === 'localStorage') {
      this.storageObj = window.localStorage;
    } else {
      this.storageObj = window.sessionStorage;
    }
  }
  static reactComponentStateKey = 'ReactComponentState';

  scope: any;
  storageObj: any;

  /**
   * Loads React component state cached in local storage into a vanilla JS object.
   */
  loadComponentState() {
    const storedVal = this.getItem(LocalStorageStore.reactComponentStateKey);
    if (storedVal) {
      return JSON.parse(storedVal);
    }
    return {};
  }

  /**
   * Save React component state in local storage.
   * @param stateRecord: Immutable.Record instance or plain object containing component state.
   */
  saveComponentState(stateRecord: any) {
    const targetValue = typeof stateRecord.toJSON === 'function' ? stateRecord.toJSON() : stateRecord;
    this.setItem(LocalStorageStore.reactComponentStateKey, JSON.stringify(targetValue));
  }

  /**
   * Helper method for constructing a scoped key to use for setting/getting values in
   * local storage.
   */
  withScopePrefix(key: any) {
    return ['MLflowLocalStorage', LocalStorageUtils.version, this.scope, key].join('-');
  }

  /** Save the specified key-value pair in local storage. */
  setItem(key: any, value: any) {
    setStorageItem(this.storageObj, this.withScopePrefix(key), value);
  }

  /** Fetch the value corresponding to the passed-in key from local storage. */
  getItem(key: any): any {
    return getStorageItem(this.storageObj, this.withScopePrefix(key));
  }
}

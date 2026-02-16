// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class LocalStorageUtils {
  static version: string;

  static getStoreForComponent(componentName: any, id: any): LocalStorageStore;
  static getSessionScopedStoreForComponent(componentName: any, id: any): LocalStorageStore;
}

export class LocalStorageStore {
  constructor(scope: string, type: string);
  static reactComponentStateKey: string;
  loadComponentState(): any;
  saveComponentState(stateRecord: any): void;
  withScopePrefix(key: any): any;
  setItem(key: any, value: any): void;
  getItem(key: any): any;
}

export default LocalStorageUtils;

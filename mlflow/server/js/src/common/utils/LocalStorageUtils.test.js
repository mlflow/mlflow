import LocalStorageUtils from './LocalStorageUtils';
import { ExperimentPagePersistedState } from '../../experiment-tracking/sdk/MlflowLocalStorageMessages';

test('Setting key-value pairs in one scope does not affect the other', () => {
  const store0 = LocalStorageUtils.getStoreForComponent('SomeTestComponent', 1);
  const store1 = LocalStorageUtils.getStoreForComponent('AnotherTestComponent', 1);
  const store2 = LocalStorageUtils.getStoreForComponent('SomeTestComponent', 2);
  const persistedState0 = new ExperimentPagePersistedState({ searchInput: 'params.ollKorrect' });
  const persistedState1 = new ExperimentPagePersistedState({ searchInput: 'metrics.ok' });
  [store1, store2].forEach((otherStore) => {
    store0.setItem('myKey', 'myCoolVal');
    otherStore.setItem('myKey', 'thisValIsBetterYo');
    expect(store0.getItem('myKey')).toEqual('myCoolVal');
    expect(otherStore.getItem('myKey')).toEqual('thisValIsBetterYo');

    store0.saveComponentState(persistedState0);
    otherStore.saveComponentState(persistedState1);
    expect(store0.loadComponentState().searchInput).toEqual('params.ollKorrect');
    expect(otherStore.loadComponentState().searchInput).toEqual('metrics.ok');
  });
});

test('Overwriting key-value pairs is possible', () => {
  const store = LocalStorageUtils.getStoreForComponent('SomeTestComponent', 1);
  store.setItem('a', 'b');
  expect(store.getItem('a')).toEqual('b');
  store.setItem('a', 'c');
  expect(store.getItem('a')).toEqual('c');
  store.saveComponentState(new ExperimentPagePersistedState({ searchInput: 'params.ollKorrect' }));
  expect(store.loadComponentState().searchInput).toEqual('params.ollKorrect');
  store.saveComponentState(new ExperimentPagePersistedState({ searchInput: 'params.okay' }));
  expect(store.loadComponentState().searchInput).toEqual('params.okay');
});

test('Session scoped storage works', () => {
  const store = LocalStorageUtils.getSessionScopedStoreForComponent('SomeTestComponent', 1);
  store.setItem('a', 'b');
  expect(store.getItem('a')).toEqual('b');
  store.setItem('a', 'c');
  expect(store.getItem('a')).toEqual('c');
  store.saveComponentState(new ExperimentPagePersistedState({ searchInput: 'params.ollKorrect' }));
  expect(store.loadComponentState().searchInput).toEqual('params.ollKorrect');
  store.saveComponentState(new ExperimentPagePersistedState({ searchInput: 'params.okay' }));
  expect(store.loadComponentState().searchInput).toEqual('params.okay');

  const store1 = LocalStorageUtils.getSessionScopedStoreForComponent('AnotherTestComponent', 1);
  const store2 = LocalStorageUtils.getSessionScopedStoreForComponent('SomeTestComponent', 2);
  const persistedState0 = new ExperimentPagePersistedState({ searchInput: 'params.ollKorrect' });
  const persistedState1 = new ExperimentPagePersistedState({ searchInput: 'metrics.ok' });
  [store1, store2].forEach((otherStore) => {
    store.setItem('myKey', 'myCoolVal');
    otherStore.setItem('myKey', 'thisValIsBetterYo');
    expect(store.getItem('myKey')).toEqual('myCoolVal');
    expect(otherStore.getItem('myKey')).toEqual('thisValIsBetterYo');

    store.saveComponentState(persistedState0);
    otherStore.saveComponentState(persistedState1);
    expect(store.loadComponentState().searchInput).toEqual('params.ollKorrect');
    expect(otherStore.loadComponentState().searchInput).toEqual('metrics.ok');
  });
});

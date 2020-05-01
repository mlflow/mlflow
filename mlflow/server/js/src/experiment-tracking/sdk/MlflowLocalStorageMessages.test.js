import { ExperimentPagePersistedState } from './MlflowLocalStorageMessages';

test('Local storage messages ignore unknown fields', () => {
  const persistedState = ExperimentPagePersistedState({ heyYallImAnUnknownField: 'value' });
  expect(persistedState.paramKeyFilterString).toEqual('');
  expect(persistedState.metricKeyFilterString).toEqual('');
  expect(persistedState.searchInput).toEqual('');
});

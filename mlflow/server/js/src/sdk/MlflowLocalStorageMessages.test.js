import React from 'react';
import { ExperimentPagePersistedState, ExperimentViewPersistedState } from '../sdk/MlflowLocalStorageMessages';

test('Local storage messages ignore unknown fields', () => {
  const persistedState = ExperimentPagePersistedState({heyYallImAnUnknownField: "value"});
  expect(persistedState.paramKeyFilterString).toEqual("");
  expect(persistedState.metricKeyFilterString).toEqual("");
  expect(persistedState.searchInput).toEqual("");
});

test('Local storage messages set default values for unspecified fields', () => {
  const persistedState = ExperimentViewPersistedState({});
  expect(persistedState.sort === {
    ascending: false,
    isMetric: false,
    isParam: false,
    key: "start_time",
  });
});

import React, { useState } from 'react';
import { IntlProvider } from 'react-intl';
import { Provider } from 'react-redux';
import { StaticRouter } from 'react-router-dom';
import { applyMiddleware, compose, createStore } from 'redux';
import promiseMiddleware from 'redux-promise-middleware';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { ExperimentViewRunsControlsFilters } from './ExperimentViewRunsControlsFilters';
import { experimentRunsSelector } from '../../utils/experimentRuns.selector';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { UpdateExperimentSearchFacetsFn } from '../../../../types';
import { GetExperimentRunsContextProvider } from '../../contexts/GetExperimentRunsContext';

const MOCK_EXPERIMENT = EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789'];

const MOCK_RUNS_DATA = experimentRunsSelector(EXPERIMENT_RUNS_MOCK_STORE, {
  experiments: [MOCK_EXPERIMENT],
});

export default {
  title: 'ExperimentView/ExperimentViewRunsControlsFilters',
  component: ExperimentViewRunsControlsFilters,
  argTypes: {},
};

export const Default = () => {
  const [searchFacetsState, setSearchFacetsState] = useState<SearchExperimentRunsFacetsState>(
    new SearchExperimentRunsFacetsState(),
  );
  const [messages, setMessages] = useState<string[]>([]);
  const updateSearchFacets = ((updatedFacetsState: Partial<SearchExperimentRunsFacetsState>) => {
    setSearchFacetsState((s) => ({ ...s, ...updatedFacetsState }));
    setMessages((currentMessages) => [
      `updateSearchFacets() called while updating state ${JSON.stringify(updatedFacetsState)}`,
      ...currentMessages,
    ]);
  }) as UpdateExperimentSearchFacetsFn;

  return (
    <Provider
      store={createStore(
        // Identity reducer
        (s) => s as any,
        EXPERIMENT_RUNS_MOCK_STORE,
        compose(applyMiddleware(promiseMiddleware())),
      )}
    >
      <IntlProvider locale='en'>
        <StaticRouter location='/'>
          <GetExperimentRunsContextProvider actions={{} as any}>
            <div
              css={{
                marginBottom: 20,
                paddingBottom: 10,
                borderBottom: '1px solid #ccc',
              }}
            >
              <h2>Component:</h2>
            </div>
            <ExperimentViewRunsControlsFilters
              runsData={MOCK_RUNS_DATA}
              searchFacetsState={searchFacetsState}
              updateSearchFacets={updateSearchFacets}
              viewState={{} as any}
              updateViewState={() => {}}
            />
            <div
              css={{
                marginTop: 20,
                paddingTop: 10,
                borderTop: '1px solid #ccc',
              }}
            >
              <h2>Debug info:</h2>
              <h3>Current search-sort-filter state:</h3>
              <div css={{ fontFamily: 'monospace', marginBottom: 10 }}>
                {JSON.stringify(searchFacetsState)}
              </div>
              <h3>Log:</h3>
              {messages.map((m, i) => (
                <div key={i} css={{ fontFamily: 'monospace' }}>
                  - {m}
                </div>
              ))}
            </div>
          </GetExperimentRunsContextProvider>
        </StaticRouter>
      </IntlProvider>
    </Provider>
  );
};

import React, { useState } from 'react';
import { IntlProvider } from 'react-intl';
import { Provider } from 'react-redux';
import { StaticRouter } from 'react-router-dom';
import { applyMiddleware, compose, createStore } from 'redux';
import promiseMiddleware from 'redux-promise-middleware';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { ExperimentViewRunsControlsActions } from './ExperimentViewRunsControlsActions';
import { experimentRunsSelector } from '../../utils/experimentRuns.selector';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { UpdateExperimentSearchFacetsFn } from '../../../../types';
import { GetExperimentRunsContextProvider } from '../../contexts/GetExperimentRunsContext';

const MOCK_EXPERIMENT = EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789'];
const DEFAULT_VIEW_STATE = new SearchExperimentRunsViewState();

const MOCK_RUNS_DATA = experimentRunsSelector(EXPERIMENT_RUNS_MOCK_STORE, {
  experiments: [MOCK_EXPERIMENT],
});

export default {
  title: 'ExperimentView/ExperimentViewRunsControlsActions',
  component: ExperimentViewRunsControlsActions,
  argTypes: {},
};

const createComponentWrapper = () => () => {
  const [searchFacetsState, setSearchFacetsState] = useState<SearchExperimentRunsFacetsState>(
    new SearchExperimentRunsFacetsState(),
  );
  const [messages, setMessages] = useState<string[]>([]);
  const updateSearchFacets = ((
    updatedFacetsState: Partial<SearchExperimentRunsFacetsState>,
    refresh?: boolean,
  ) => {
    setSearchFacetsState((s) => ({ ...s, ...updatedFacetsState }));
    setMessages((currentMessages) => [
      refresh
        ? 'updateSearchFacets() called requesting refresh data'
        : `updateSearchFacets() called while updating state ${JSON.stringify(updatedFacetsState)}`,
      ...currentMessages,
    ]);
  }) as UpdateExperimentSearchFacetsFn;

  return (
    <Provider
      store={createStore(
        (s) => s as any,
        EXPERIMENT_RUNS_MOCK_STORE,
        compose(applyMiddleware(promiseMiddleware())),
      )}
    >
      <StaticRouter location='/'>
        <GetExperimentRunsContextProvider actions={{} as any}>
          <IntlProvider locale='en'>
            <div
              css={{
                marginBottom: 20,
                paddingBottom: 10,
                borderBottom: '1px solid #ccc',
              }}
            >
              <h2>Component:</h2>
            </div>
            <ExperimentViewRunsControlsActions
              runsData={MOCK_RUNS_DATA}
              searchFacetsState={searchFacetsState}
              viewState={DEFAULT_VIEW_STATE}
              updateSearchFacets={updateSearchFacets}
            />
            <div
              css={{
                marginTop: 20,
                paddingTop: 10,
                borderTop: '1px solid #ccc',
              }}
            >
              <h2>Debug info:</h2>
              <h3>Current search-facets state:</h3>
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
          </IntlProvider>
        </GetExperimentRunsContextProvider>
      </StaticRouter>
    </Provider>
  );
};

export const Default = createComponentWrapper();

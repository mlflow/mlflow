import React, { useState } from 'react';
import { IntlProvider } from 'react-intl';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../../../../common/utils/RoutingUtils';
import { applyMiddleware, compose, createStore } from 'redux';
import promiseMiddleware from 'redux-promise-middleware';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { ExperimentViewRunsControlsActions } from './ExperimentViewRunsControlsActions';
import { experimentRunsSelector } from '../../utils/experimentRuns.selector';
import { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import { useRunSortOptions } from '../../hooks/useRunSortOptions';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';

const MOCK_EXPERIMENT = EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789'];

const MOCK_RUNS_DATA = experimentRunsSelector(EXPERIMENT_RUNS_MOCK_STORE, {
  experiments: [MOCK_EXPERIMENT],
});

export default {
  title: 'ExperimentView/ExperimentViewRunsControlsActions',
  component: ExperimentViewRunsControlsActions,
  argTypes: {},
};

const createComponentWrapper = (viewState: ExperimentPageViewState) => () => {
  const [searchFacetsState] = useState(() => createExperimentPageSearchFacetsState());

  const sortOptions = useRunSortOptions(['metric1'], ['param1']);

  return (
    <Provider
      store={createStore((s) => s as any, EXPERIMENT_RUNS_MOCK_STORE, compose(applyMiddleware(promiseMiddleware())))}
    >
      <MemoryRouter>
        <IntlProvider locale="en">
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
            viewState={viewState}
            refreshRuns={() => {}}
          />
        </IntlProvider>
      </MemoryRouter>
    </Provider>
  );
};

export const Default = createComponentWrapper(new ExperimentPageViewState());
export const WithOneRunSelected = createComponentWrapper(
  Object.assign(new ExperimentPageViewState(), {
    runsSelected: {
      experiment123456789_run1: true,
    },
  }),
);

export const WithTwoRunsSelected = createComponentWrapper(
  Object.assign(new ExperimentPageViewState(), {
    runsSelected: {
      experiment123456789_run1: true,
      experiment123456789_run2: true,
    },
  }),
);

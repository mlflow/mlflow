import { useState } from 'react';
import type { ReactWrapper } from 'enzyme';
import { IntlProvider } from 'react-intl';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../../../../common/utils/RoutingUtils';
import { applyMiddleware, compose, createStore } from 'redux';
import promiseMiddleware from 'redux-promise-middleware';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
// import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import { experimentRunsSelector } from '../../utils/experimentRuns.selector';
import type { ExperimentViewRunsControlsActionsProps } from './ExperimentViewRunsControlsActions';
import { ExperimentViewRunsControlsActions } from './ExperimentViewRunsControlsActions';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';

const MOCK_EXPERIMENT = EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789'];

const MOCK_RUNS_DATA = experimentRunsSelector(EXPERIMENT_RUNS_MOCK_STORE, {
  experiments: [MOCK_EXPERIMENT],
});

const DEFAULT_VIEW_STATE = new ExperimentPageViewState();

const doMock = (additionalProps: Partial<ExperimentViewRunsControlsActionsProps> = {}) => {
  const mockUpdateSearchFacets = jest.fn();
  let currentState: any;

  const getCurrentState = () => currentState;

  const Component = () => {
    const [searchFacetsState, setSearchFacetsState] = useState<ExperimentPageSearchFacetsState>(
      createExperimentPageSearchFacetsState(),
    );

    currentState = searchFacetsState;

    const props: ExperimentViewRunsControlsActionsProps = {
      runsData: MOCK_RUNS_DATA,
      searchFacetsState,
      viewState: DEFAULT_VIEW_STATE,
      refreshRuns: jest.fn(),
      ...additionalProps,
    };
    return (
      <Provider
        store={createStore((s) => s as any, EXPERIMENT_RUNS_MOCK_STORE, compose(applyMiddleware(promiseMiddleware())))}
      >
        <MemoryRouter>
          <IntlProvider locale="en">
            <ExperimentViewRunsControlsActions {...props} />
          </IntlProvider>
        </MemoryRouter>
      </Provider>
    );
  };
  return {
    wrapper: mountWithIntl(<Component />),
    mockUpdateSearchFacets,
    getCurrentState,
  };
};

// @ts-expect-error TS(2709): Cannot use namespace 'ReactWrapper' as a type.
const getActionButtons = (wrapper: ReactWrapper) => {
  const deleteButton = wrapper.find("button[data-testid='runs-delete-button']");
  const compareButton = wrapper.find("button[data-testid='runs-compare-button']");
  const renameButton = wrapper.find("button[data-testid='run-rename-button']");
  return { deleteButton, compareButton, renameButton };
};

describe('ExperimentViewRunsControlsFilters', () => {
  test('should render with given search facets model properly', () => {
    const { wrapper } = doMock();
    expect(wrapper).toBeTruthy();
  });

  test('should enable delete buttons when there is single row selected', () => {
    const { wrapper } = doMock({
      viewState: {
        runsSelected: { '123': true },
        hiddenChildRunsSelected: {},
        columnSelectorVisible: false,
        previewPaneVisible: false,
        artifactViewState: {},
      },
    });

    const { deleteButton, compareButton, renameButton } = getActionButtons(wrapper);

    // All buttons should be visible
    expect(deleteButton.length).toBe(1);
    expect(compareButton.length).toBe(1);
    expect(renameButton.length).toBe(1);

    // Only compare button should be disabled
    expect(deleteButton.getDOMNode().getAttribute('disabled')).toBeNull();
    expect(compareButton.getDOMNode().getAttribute('disabled')).not.toBeNull();
    expect(renameButton.getDOMNode().getAttribute('disabled')).toBeNull();
  });

  test('should enable delete and compare buttons when there are multiple rows selected', () => {
    const { wrapper } = doMock({
      viewState: {
        runsSelected: { '123': true, '321': true },
        hiddenChildRunsSelected: {},
        columnSelectorVisible: false,
        previewPaneVisible: false,
        artifactViewState: {},
      },
    });

    const { deleteButton, compareButton, renameButton } = getActionButtons(wrapper);

    // All buttons should be visible
    expect(deleteButton.length).toBe(1);
    expect(compareButton.length).toBe(1);
    expect(renameButton.length).toBe(1);

    // Only rename button should be disabled
    expect(deleteButton.getDOMNode().getAttribute('disabled')).toBeNull();
    expect(compareButton.getDOMNode().getAttribute('disabled')).toBeNull();
    expect(renameButton.getDOMNode().getAttribute('disabled')).not.toBeNull();
  });

  test('should enable rename button when necessary', () => {
    const { wrapper } = doMock({
      viewState: {
        runsSelected: { '123': true },
        hiddenChildRunsSelected: {},
        columnSelectorVisible: false,
        previewPaneVisible: false,
        artifactViewState: {},
      },
    });

    const deleteButton = wrapper.find("button[data-testid='run-rename-button']");
    expect(deleteButton.getDOMNode().getAttribute('disabled')).toBeNull();
  });

  test('should disable rename button when necessary', () => {
    const { wrapper } = doMock({
      viewState: {
        runsSelected: { '123': true, '321': true },
        hiddenChildRunsSelected: {},
        columnSelectorVisible: false,
        previewPaneVisible: false,
        artifactViewState: {},
      },
    });

    const deleteButton = wrapper.find("button[data-testid='run-rename-button']");
    expect(deleteButton.getDOMNode().getAttribute('disabled')).not.toBeNull();
  });
});

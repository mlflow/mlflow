import { Provider } from 'react-redux';
import type { EvaluationDataReduxState } from '../../reducers/EvaluationDataReducer';
import { EvaluationArtifactCompareView } from './EvaluationArtifactCompareView';
import configureStore from 'redux-mock-store';
import type { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { ExperimentPageViewState } from '../experiment-page/models/ExperimentPageViewState';
import { renderWithIntl, act, within, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { getEvaluationTableArtifact } from '../../actions';
import { MLFLOW_LOGGED_ARTIFACTS_TAG, MLFLOW_RUN_SOURCE_TYPE_TAG, MLflowRunSourceType } from '../../constants';
import type { EvaluationArtifactCompareTableProps } from './components/EvaluationArtifactCompareTable';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import userEventGlobal, { PointerEventsCheckLevel } from '@testing-library/user-event';
import { useState } from 'react';

// Disable pointer events check for DialogCombobox which masks the elements we want to click
const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

jest.mock('../../actions', () => ({
  getEvaluationTableArtifact: jest
    .fn()
    .mockReturnValue({ type: 'GETEVALUATIONTABLEARTIFACT', payload: Promise.resolve() }),
}));

jest.mock('./components/EvaluationArtifactCompareTable', () => ({
  EvaluationArtifactCompareTable: ({
    visibleRuns,
    resultList,
    groupByColumns,
    onCellClick,
  }: EvaluationArtifactCompareTableProps) => (
    <div data-testid="mock-compare-table">
      {/* Render a super simple but functional variant of results table */}
      {resultList.map((result) => (
        <div key={result.key}>
          {groupByColumns.map((groupByCol) => (
            <div key={`groupby-${groupByCol}-${result.key}`} data-testid="group-by-cell">
              {result.groupByCellValues[groupByCol]}
            </div>
          ))}
          {visibleRuns.map(({ runUuid }) => (
            <button
              key={`result-${runUuid}-${result.key}`}
              data-testid={`result-${runUuid}-${result.key}`}
              onClick={() => onCellClick?.(result.cellValues[runUuid].toString(), runUuid)}
            >
              {`row ${result.key}, run ${runUuid}, result ${result.cellValues[runUuid] || '(empty)'}`}
            </button>
          ))}
        </div>
      ))}
    </div>
  ),
}));

const SAMPLE_COMPARED_RUNS: RunRowType[] = [
  {
    runUuid: 'run_a',
    tags: {
      [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
        key: MLFLOW_LOGGED_ARTIFACTS_TAG,
        value: '[{"path":"/table.json","type":"table"}]',
      },
    },
  },
  {
    runUuid: 'run_b',
    tags: {
      [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
        key: MLFLOW_LOGGED_ARTIFACTS_TAG,
        value: '[{"path":"/table.json","type":"table"}]',
      },
    },
  },
  {
    runUuid: 'run_c',
    tags: {
      [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
        key: MLFLOW_LOGGED_ARTIFACTS_TAG,
        value: '[{"path":"/table_c.json","type":"table"}]',
      },
    },
  },
] as any;

const SAMPLE_STATE = {
  evaluationArtifactsBeingUploaded: {},
  evaluationArtifactsByRunUuid: {
    run_a: {
      '/table.json': {
        columns: ['col_group', 'col_output'],
        path: '/table.json',
        entries: [
          { col_group: 'question_1', col_output: 'answer_1_run_a' },
          { col_group: 'question_2', col_output: 'answer_2_run_a' },
        ],
      },
    },
    run_b: {
      '/table.json': {
        columns: ['col_group', 'col_output'],
        path: '/table.json',
        entries: [
          { col_group: 'question_1', col_output: 'answer_1_run_b' },
          { col_group: 'question_2', col_output: 'answer_2_run_b' },
        ],
      },
    },
    run_c: {
      '/table_c.json': {
        columns: ['col_group', 'col_output'],
        path: '/table_c.json',
        entries: [
          { col_group: 'question_1', col_output: 'answer_1_run_c' },
          { col_group: 'question_2', col_output: 'answer_2_run_c' },
        ],
      },
    },
  },

  evaluationDraftInputValues: [],
  evaluationPendingDataByRunUuid: {},
  evaluationPendingDataLoadingByRunUuid: {},
  evaluationArtifactsErrorByRunUuid: {},
  evaluationArtifactsLoadingByRunUuid: {},
};

const IMAGE_JSON = {
  type: 'image',
  filepath: 'fakePathUncompressed',
  compressed_filepath: 'fakePath',
};

const SAMPLE_STATE_WITH_IMAGES = {
  evaluationArtifactsBeingUploaded: {},
  evaluationArtifactsByRunUuid: {
    run_a: {
      '/table.json': {
        columns: ['col_group', 'col_group_2', 'col_output'],
        path: '/table.json',
        entries: [
          { col_group: 'question_1', col_group_2: 'question_1', col_output: IMAGE_JSON },
          { col_group: 'question_2', col_group_2: 'question_2', col_output: IMAGE_JSON },
        ],
      },
    },
    run_b: {
      '/table.json': {
        columns: ['col_group', 'col_output'],
        path: '/table.json',
        entries: [
          { col_group: 'question_1', col_group_2: 'question_1', col_output: IMAGE_JSON },
          { col_group: 'question_2', col_group_2: 'question_2', col_output: IMAGE_JSON },
        ],
      },
    },
  },

  evaluationDraftInputValues: [],
  evaluationPendingDataByRunUuid: {},
  evaluationPendingDataLoadingByRunUuid: {},
  evaluationArtifactsErrorByRunUuid: {},
  evaluationArtifactsLoadingByRunUuid: {},
};

describe('EvaluationArtifactCompareView', () => {
  const mountTestComponent = ({
    comparedRuns = SAMPLE_COMPARED_RUNS,
    mockState = SAMPLE_STATE,
    viewState = new ExperimentPageViewState(),
  }: {
    viewState?: ExperimentPageViewState;
    mockState?: EvaluationDataReduxState;
    comparedRuns?: RunRowType[];
  } = {}) => {
    const mockStore = configureStore([thunk, promiseMiddleware()])({
      evaluationData: mockState,
      modelGateway: { modelGatewayRoutesLoading: {} },
    });
    const updateSearchFacetsMock = jest.fn();
    const updateViewStateMock = jest.fn();
    let setVisibleRunsFn: React.Dispatch<React.SetStateAction<RunRowType[]>>;
    const TestComponent = () => {
      const [visibleRuns, setVisibleRuns] = useState(comparedRuns);
      setVisibleRunsFn = setVisibleRuns;
      return (
        <Provider store={mockStore}>
          <EvaluationArtifactCompareView
            comparedRuns={visibleRuns}
            updateViewState={updateViewStateMock}
            viewState={viewState}
            onDatasetSelected={() => {}}
          />
        </Provider>
      );
    };
    const renderResult = renderWithIntl(<TestComponent />);
    return {
      updateSearchFacetsMock,
      updateViewStateMock,
      renderResult,
      setVisibleRuns: (runs: RunRowType[]) => setVisibleRunsFn(runs),
    };
  };

  test('checks if the initial tables are properly fetched', async () => {
    mountTestComponent();

    expect(getEvaluationTableArtifact).toHaveBeenCalledWith('run_a', '/table.json', false);
    expect(getEvaluationTableArtifact).toHaveBeenCalledWith('run_b', '/table.json', false);
  });
  test('checks if the newly selected table is being fetched', async () => {
    const { renderResult } = mountTestComponent();

    await userEvent.click(renderResult.getByTestId('dropdown-tables'));

    await userEvent.click(within(screen.getByRole('listbox')).getByLabelText('/table_c.json'));

    expect(getEvaluationTableArtifact).toHaveBeenCalledWith('run_c', '/table_c.json', false);
  });

  test('checks if the fetch artifact is properly called for differing tables', async () => {
    const { renderResult } = mountTestComponent({
      comparedRuns: [
        {
          runUuid: 'run_a',
          tags: {
            [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
              key: MLFLOW_LOGGED_ARTIFACTS_TAG,
              value: '[{"path":"/table_a.json","type":"table"}]',
            },
          },
        },
        {
          runUuid: 'run_b',
          tags: {
            [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
              key: MLFLOW_LOGGED_ARTIFACTS_TAG,
              value: '[{"path":"/table_b.json","type":"table"}]',
            },
          },
        },
      ] as any,
    });

    await userEvent.click(renderResult.getByTestId('dropdown-tables'));
    await userEvent.click(within(screen.getByRole('listbox')).getByLabelText('/table_a.json'));

    expect(getEvaluationTableArtifact).toHaveBeenCalledWith('run_a', '/table_a.json', false);
    expect(getEvaluationTableArtifact).not.toHaveBeenCalledWith('run_a', '/table_b.json', false);
    expect(getEvaluationTableArtifact).not.toHaveBeenCalledWith('run_b', '/table_a.json', false);
    expect(getEvaluationTableArtifact).not.toHaveBeenCalledWith('run_b', '/table_b.json', false);
  });

  test('checks if the table component receives proper result set based on the store data and selected table', async () => {
    mountTestComponent();

    // Check if the table "group by" column cells were properly populated
    expect(screen.getByText('question_1', { selector: '[data-testid="group-by-cell"]' })).toBeInTheDocument();
    expect(screen.getByText('question_2', { selector: '[data-testid="group-by-cell"]' })).toBeInTheDocument();

    // Check if the table output cells were properly populated
    [
      'row question_1, run run_a, result answer_1_run_a',
      'row question_1, run run_b, result answer_1_run_b',
      'row question_1, run run_c, result (empty)',
      'row question_2, run run_a, result answer_2_run_a',
      'row question_2, run run_b, result answer_2_run_b',
      'row question_2, run run_c, result (empty)',
    ].forEach((cellText) => {
      expect(screen.getByText(cellText, { selector: 'button' })).toBeInTheDocument();
    });

    expect.assertions(8);
  });

  test('checks if the preview sidebar renders proper details', async () => {
    const { renderResult } = mountTestComponent({
      viewState: Object.assign(new ExperimentPageViewState(), { previewPaneVisible: true }),
    });

    const previewSidebar = renderResult.getByTestId('preview-sidebar-content');
    expect(previewSidebar).toHaveTextContent(/Select a cell to display preview/);

    const run_a_question_1 = renderResult.getByTestId('result-run_a-question_1');
    const run_b_question_2 = renderResult.getByTestId('result-run_b-question_2');

    await userEvent.click(run_a_question_1);
    expect(previewSidebar).toHaveTextContent(/answer_1_run_a/);

    await userEvent.click(run_b_question_2);
    expect(previewSidebar).toHaveTextContent(/answer_2_run_b/);
  });

  test('checks if the component initializes with proper "group by" and "output" columns when evaluating prompt engineering artifacts', async () => {
    mountTestComponent({
      mockState: {
        ...SAMPLE_STATE,
        evaluationArtifactsByRunUuid: {
          run_a: {
            '/table_a.json': {
              columns: ['input_a', 'input_b', 'output'],
              path: '/table_a.json',
              entries: [],
            },
          },
          run_b: {
            '/table_b.json': {
              columns: ['input_b', 'output'],
              path: '/table_b.json',
              entries: [],
            },
          },
        },
      },
      comparedRuns: [
        {
          runUuid: 'run_a',
          params: [{ key: 'prompt_template', value: 'prompt template with {{input_a}} and {{input_b}}' }],
          tags: {
            [MLFLOW_RUN_SOURCE_TYPE_TAG]: {
              key: MLFLOW_RUN_SOURCE_TYPE_TAG,
              value: MLflowRunSourceType.PROMPT_ENGINEERING,
            },
            [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
              key: MLFLOW_LOGGED_ARTIFACTS_TAG,
              value: '[{"path":"/table_a.json","type":"table"}]',
            },
          },
        },
        {
          runUuid: 'run_b',
          params: [{ key: 'prompt_template', value: 'prompt template with {{input_c}}' }],
          tags: {
            [MLFLOW_RUN_SOURCE_TYPE_TAG]: {
              key: MLFLOW_RUN_SOURCE_TYPE_TAG,
              value: MLflowRunSourceType.PROMPT_ENGINEERING,
            },
            [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
              key: MLFLOW_LOGGED_ARTIFACTS_TAG,
              value: '[{"path":"/table_b.json","type":"table"}]',
            },
          },
          hidden: true,
        },
      ] as any,
    });

    await userEvent.click(screen.getByLabelText('Select "group by" columns'));

    expect(within(screen.getByRole('listbox')).getByLabelText('input_a')).toBeChecked();
    expect(within(screen.getByRole('listbox')).getByLabelText('input_b')).toBeChecked();

    expect(
      screen.getByRole('combobox', {
        name: 'Dialog Combobox, selected option: output',
      }),
    ).toBeInTheDocument();
  });

  test('checks if component behaves correctly if user deselects all "group by" columns', async () => {
    mountTestComponent({
      mockState: {
        ...SAMPLE_STATE,
        evaluationArtifactsByRunUuid: {
          run_a: {
            '/table_a.json': {
              columns: ['input_a', 'input_b', 'output'],
              path: '/table_a.json',
              entries: [],
            },
          },
          run_b: {
            '/table_b.json': {
              columns: ['input_b', 'output'],
              path: '/table_b.json',
              entries: [],
            },
          },
        },
      },
      comparedRuns: [
        {
          runUuid: 'run_a',
          params: [{ key: 'prompt_template', value: 'prompt template with {{input_a}} and {{input_b}}' }],
          tags: {
            [MLFLOW_RUN_SOURCE_TYPE_TAG]: {
              key: MLFLOW_RUN_SOURCE_TYPE_TAG,
              value: MLflowRunSourceType.PROMPT_ENGINEERING,
            },
            [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
              key: MLFLOW_LOGGED_ARTIFACTS_TAG,
              value: '[{"path":"/table_a.json","type":"table"}]',
            },
          },
        },
        {
          runUuid: 'run_b',
          params: [{ key: 'prompt_template', value: 'prompt template with {{input_c}}' }],
          tags: {
            [MLFLOW_RUN_SOURCE_TYPE_TAG]: {
              key: MLFLOW_RUN_SOURCE_TYPE_TAG,
              value: MLflowRunSourceType.PROMPT_ENGINEERING,
            },
            [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
              key: MLFLOW_LOGGED_ARTIFACTS_TAG,
              value: '[{"path":"/table_b.json","type":"table"}]',
            },
          },
          hidden: true,
        },
      ] as any,
    });

    await userEvent.click(screen.getByLabelText('Select "group by" columns'));

    // Expect two "group by" columns to be initially selected
    expect(within(screen.getByRole('listbox')).getByLabelText('input_a')).toBeChecked();
    expect(within(screen.getByRole('listbox')).getByLabelText('input_b')).toBeChecked();

    expect(screen.queryByText('No group by columns selected')).not.toBeInTheDocument();

    // Deselect both columns
    await userEvent.click(within(screen.getByRole('listbox')).getByLabelText('input_a'));
    await userEvent.click(within(screen.getByRole('listbox')).getByLabelText('input_b'));

    // Expect proper message to appear
    expect(screen.getByText('No group by columns selected')).toBeInTheDocument();
  });

  test('checks if the component automatically re-selects "group by" columns when changing visible prompt engineering runs', async () => {
    const comparedRuns = [
      {
        runUuid: 'run_a',
        params: [{ key: 'prompt_template', value: 'prompt template with {{input_a}} and {{input_b}}' }],
        tags: {
          [MLFLOW_RUN_SOURCE_TYPE_TAG]: {
            key: MLFLOW_RUN_SOURCE_TYPE_TAG,
            value: MLflowRunSourceType.PROMPT_ENGINEERING,
          },
          [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
            key: MLFLOW_LOGGED_ARTIFACTS_TAG,
            value: '[{"path":"/eval_results_table.json","type":"table"}]',
          },
        },
        hidden: true,
      },
      {
        runUuid: 'run_b',
        params: [{ key: 'prompt_template', value: 'prompt template with {{input_b}}' }],
        tags: {
          [MLFLOW_RUN_SOURCE_TYPE_TAG]: {
            key: MLFLOW_RUN_SOURCE_TYPE_TAG,
            value: MLflowRunSourceType.PROMPT_ENGINEERING,
          },
          [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
            key: MLFLOW_LOGGED_ARTIFACTS_TAG,
            value: '[{"path":"/eval_results_table.json","type":"table"}]',
          },
        },
      },
    ];
    const { setVisibleRuns } = mountTestComponent({
      mockState: {
        ...SAMPLE_STATE,
        evaluationArtifactsByRunUuid: {
          run_a: {
            '/eval_results_table.json': {
              columns: ['input_a', 'input_b', 'output'],
              path: '/eval_results_table.json',
              entries: [],
            },
          },
          run_b: {
            '/eval_results_table.json': {
              columns: ['input_b', 'output'],
              path: '/eval_results_table.json',
              entries: [],
            },
          },
        },
      },
      comparedRuns: comparedRuns as any,
    });

    await userEvent.click(screen.getByLabelText('Select "group by" columns'));

    expect(within(screen.getByRole('listbox')).queryByLabelText('input_a')).not.toBeInTheDocument();
    expect(within(screen.getByRole('listbox')).getByLabelText('input_b')).toBeChecked();

    expect(
      screen.getByRole('combobox', {
        name: 'Dialog Combobox, selected option: output',
      }),
    ).toBeInTheDocument();

    // Unhide all runs
    await act(async () => {
      setVisibleRuns(comparedRuns.map((run) => ({ ...run, hidden: false })) as any);
    });

    expect(within(screen.getByRole('listbox')).getByLabelText('input_a')).toBeChecked();
    expect(within(screen.getByRole('listbox')).getByLabelText('input_b')).toBeChecked();
  });

  test('checks if relevant empty message is displayed when there are no logged evaluation tables', async () => {
    const { renderResult } = mountTestComponent({
      mockState: {
        ...SAMPLE_STATE,
        evaluationArtifactsByRunUuid: {
          run_a: {},
          run_b: {},
        },
      },
      comparedRuns: [
        {
          runUuid: 'run_a',
          params: [],
          tags: {},
        },
        {
          runUuid: 'run_b',
          params: [],
          tags: {},
        },
      ] as any,
    });

    expect(renderResult.getByTestId('dropdown-tables')).toBeDisabled();
    expect(renderResult.getByText(/No evaluation tables logged/)).toBeInTheDocument();
  });

  test('checks that image columns are correctly assigned to only be output columns', async () => {
    mountTestComponent({ mockState: SAMPLE_STATE_WITH_IMAGES });

    expect(getEvaluationTableArtifact).toHaveBeenCalledWith('run_a', '/table.json', false);
    expect(getEvaluationTableArtifact).toHaveBeenCalledWith('run_b', '/table.json', false);

    await userEvent.click(screen.getByLabelText('Select "group by" columns'));

    // Check that the group by columns are properly populated and recognizes image columns as non-groupable
    expect(within(screen.getByRole('listbox')).queryByLabelText('col_group')).toBeChecked();
    expect(within(screen.getByRole('listbox')).queryByLabelText('col_group')).toBeInTheDocument();
    expect(within(screen.getByRole('listbox')).queryByLabelText('col_group_2')).toBeInTheDocument();
    expect(within(screen.getByRole('listbox')).queryByLabelText('col_group_2')).not.toBeChecked();
    expect(within(screen.getByRole('listbox')).queryByLabelText('col_output')).not.toBeInTheDocument();
  });
});

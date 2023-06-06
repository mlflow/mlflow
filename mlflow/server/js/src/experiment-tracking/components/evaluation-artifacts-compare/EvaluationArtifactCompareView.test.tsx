import { Provider } from 'react-redux';
import type { EvaluationDataReduxState } from '../../reducers/EvaluationDataReducer';
import { EvaluationArtifactCompareView } from './EvaluationArtifactCompareView';
import configureStore from 'redux-mock-store';
import { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { SearchExperimentRunsViewState } from '../experiment-page/models/SearchExperimentRunsViewState';
import { mountWithIntl } from '../../../common/utils/TestUtils';
import { getEvaluationTableArtifact } from '../../actions';
import { MLFLOW_LOGGED_ARTIFACTS_TAG } from '../../constants';
import { act } from 'react-dom/test-utils';
import {
  EvaluationArtifactCompareTable,
  EvaluationArtifactCompareTableProps,
} from './components/EvaluationArtifactCompareTable';

jest.mock('../../actions', () => ({
  getEvaluationTableArtifact: jest.fn().mockReturnValue({ type: 'GETEVALUATIONTABLEARTIFACT' }),
}));

jest.mock('./components/EvaluationArtifactCompareTable', () => ({
  EvaluationArtifactCompareTable: () => <div />,
}));

describe('EvaluationArtifactCompareView', () => {
  const mountTestComponent = ({
    comparedRuns = [
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
    ] as any,
    mockState = {
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
      },
      evaluationArtifactsErrorByRunUuid: {},
      evaluationArtifactsLoadingByRunUuid: {},
    },
    viewState = new SearchExperimentRunsViewState(),
  }: {
    viewState?: SearchExperimentRunsViewState;
    mockState?: EvaluationDataReduxState;
    comparedRuns?: RunRowType[];
  } = {}) => {
    const mockStore = configureStore()({ evaluationData: mockState });
    const updateSearchFacetsMock = jest.fn();
    const updateViewStateMock = jest.fn();
    const wrapper = mountWithIntl(
      <Provider store={mockStore}>
        <EvaluationArtifactCompareView
          visibleRuns={comparedRuns}
          updateSearchFacets={updateSearchFacetsMock}
          updateViewState={updateViewStateMock}
          viewState={viewState}
          onDatasetSelected={() => {}}
        />
      </Provider>,
    );
    return { updateSearchFacetsMock, updateViewStateMock, wrapper };
  };

  const openCombobox = (element: any) =>
    act(async () => {
      element.simulate('click', { button: 0, ctrlKey: false });
    });

  beforeAll(() => {
    // Polyfill missing objects, should be fixed globally later
    global.DOMRect = {
      fromRect: () => ({
        top: 0,
        left: 0,
        bottom: 0,
        right: 0,
        width: 0,
        height: 0,
        x: 0,
        y: 0,
        toJSON: () => {},
      }),
    } as any;
    global.ResizeObserver = class ResizeObserver {
      constructor(cb: any) {
        (this as any).cb = cb;
      }
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  });

  test('checks if the fetch artifact is properly called for common tables', async () => {
    const { wrapper } = mountTestComponent();

    await openCombobox(wrapper.find("button[data-testid='dropdown-tables']"));
    wrapper.update();

    const tableOption = wrapper
      .find("div[data-testid='dropdown-tables-option']")
      .filterWhere((node: any) => node.text().includes('/table.json'));

    tableOption.simulate('click');

    expect(getEvaluationTableArtifact).toBeCalledWith('run_a', '/table.json', false);
    expect(getEvaluationTableArtifact).toBeCalledWith('run_b', '/table.json', false);
  });

  test('checks if the fetch artifact is properly called for differing tables', async () => {
    const { wrapper } = mountTestComponent({
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

    await openCombobox(wrapper.find("button[data-testid='dropdown-tables']"));
    wrapper.update();

    const tableOption = wrapper
      .find("div[data-testid='dropdown-tables-option']")
      .filterWhere((node: any) => node.text().includes('/table_a.json'));

    tableOption.simulate('click');

    expect(getEvaluationTableArtifact).toBeCalledWith('run_a', '/table_a.json', false);
    expect(getEvaluationTableArtifact).not.toBeCalledWith('run_a', '/table_b.json', false);
    expect(getEvaluationTableArtifact).not.toBeCalledWith('run_b', '/table_a.json', false);
    expect(getEvaluationTableArtifact).not.toBeCalledWith('run_b', '/table_b.json', false);
  });

  test('checks if the table component receives proper result set based on the store data and selected table', async () => {
    const { wrapper } = mountTestComponent();

    await openCombobox(wrapper.find("button[data-testid='dropdown-tables']"));
    wrapper.update();

    const tableOption = wrapper
      .find("div[data-testid='dropdown-tables-option']")
      .filterWhere((node: any) => node.text().includes('/table.json'));

    tableOption.simulate('click');

    const tableProps: EvaluationArtifactCompareTableProps = wrapper
      .find(EvaluationArtifactCompareTable)
      .props();
    expect(tableProps.comparedRuns).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ runUuid: 'run_a' }),
        expect.objectContaining({ runUuid: 'run_b' }),
      ]),
    );

    expect(tableProps.groupByColumns).toEqual(['col_group']);

    expect(tableProps.resultList).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: 'question_1',
          groupByCellValues: { col_group: 'question_1' },
          cellValues: { run_a: 'answer_1_run_a', run_b: 'answer_1_run_b' },
        }),
        expect.objectContaining({
          key: 'question_2',
          groupByCellValues: { col_group: 'question_2' },
          cellValues: { run_a: 'answer_2_run_a', run_b: 'answer_2_run_b' },
        }),
      ]),
    );
  });
});

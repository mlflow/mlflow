import { Provider } from 'react-redux';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { EvaluationArtifactCompareTable } from './EvaluationArtifactCompareTable';
import { screen, waitFor, renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import type { UseEvaluationArtifactTableDataResult } from '../hooks/useEvaluationArtifactTableData';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { BrowserRouter } from '../../../../common/utils/RoutingUtils';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(90000);

jest.mock('../../experiment-page/hooks/useExperimentRunColor', () => ({
  useGetExperimentRunColor: jest.fn().mockReturnValue((_: string) => '#000000'),
}));

describe('EvaluationArtifactCompareTable', () => {
  let originalImageSrc: any;

  beforeAll(() => {
    // Mock <img> src setter to trigger load callback
    originalImageSrc = Object.getOwnPropertyDescriptor(window.Image.prototype, 'src');
    Object.defineProperty(window.Image.prototype, 'src', {
      set() {
        setTimeout(() => this.onload?.());
      },
      get() {},
    });
  });

  afterAll(() => {
    Object.defineProperty(window.Image.prototype, 'src', originalImageSrc);
  });

  const renderComponent = (
    resultList: UseEvaluationArtifactTableDataResult,
    groupByColumns: string[],
    outputColumnName: string,
    isImageColumn: boolean,
  ) => {
    const visibleRuns: RunRowType[] = [
      {
        runUuid: 'run_a',
        runName: 'able-panda-761',
        rowUuid: '9b1b553bb1ca4e948c248c5ca426ae52',
        runInfo: {
          runUuid: 'run_a',
          status: 'FINISHED',
          artifactUri: 'dbfs:/databricks/mlflow-tracking/676587362364997/9b1b553bb1ca4e948c248c5ca426ae52/artifacts',
          endTime: 1717111275344,
          experimentId: '676587362364997',
          lifecycleStage: 'active',
          runName: 'able-panda-761',
          startTime: 1717111273526,
        },
        duration: '1.8s',
        tags: {
          'mlflow.loggedArtifacts': {
            key: 'mlflow.loggedArtifacts',
            value: '[{"path": "table.json", "type": "table"}]',
          },
        },
        models: {
          registeredModels: [],
          loggedModels: [],
          experimentId: '676587362364997',
          runUuid: 'run_a',
        },
        pinnable: true,
        hidden: false,
        pinned: false,
        datasets: [],
      },
      {
        runUuid: 'run_b',
        runName: 'able-panda-762',
        rowUuid: '9b1b553bb1ca4e948c248c5ca426ae52',
        runInfo: {
          runUuid: 'run_b',
          status: 'FINISHED',
          artifactUri: 'dbfs:/databricks/mlflow-tracking/676587362364997/9b1b553bb1ca4e948c248c5ca426ae52/artifacts',
          endTime: 1717111275344,
          experimentId: '676587362364997',
          lifecycleStage: 'active',
          runName: 'able-panda-762',
          startTime: 1717111273526,
        },
        duration: '1.8s',
        tags: {
          'mlflow.loggedArtifacts': {
            key: 'mlflow.loggedArtifacts',
            value: '[{"path": "table.json", "type": "table"}]',
          },
        },
        models: {
          registeredModels: [],
          loggedModels: [],
          experimentId: '676587362364997',
          runUuid: 'run_b',
        },
        pinnable: true,
        hidden: false,
        pinned: false,
        datasets: [],
      },
      {
        runUuid: 'run_c',
        runName: 'able-panda-763',
        rowUuid: '9b1b553bb1ca4e948c248c5ca426ae52',
        runInfo: {
          runUuid: 'run_c',
          status: 'FINISHED',
          artifactUri: 'dbfs:/databricks/mlflow-tracking/676587362364997/9b1b553bb1ca4e948c248c5ca426ae52/artifacts',
          endTime: 1717111275344,
          experimentId: '676587362364997',
          lifecycleStage: 'active',
          runName: 'able-panda-763',
          startTime: 1717111273526,
        },
        duration: '1.8s',
        tags: {
          'mlflow.loggedArtifacts': {
            key: 'mlflow.loggedArtifacts',
            value: '[{"path": "table_c.json", "type": "table"}]',
          },
        },
        models: {
          registeredModels: [],
          loggedModels: [],
          experimentId: '676587362364997',
          runUuid: 'run_c',
        },
        pinnable: true,
        hidden: false,
        pinned: false,
        datasets: [],
      },
    ];
    const onHideRun = jest.fn();
    const onDatasetSelected = jest.fn();
    const highlightedText = '';

    const SAMPLE_STATE = {
      evaluationArtifactsBeingUploaded: {},
      evaluationArtifactsByRunUuid: {
        run_a: {
          '/table.json': {
            columns: ['data', 'output'],
            path: '/table.json',
            entries: [
              { data: 'question_1', output: 'answer_1_run_a' },
              { data: 'question_2', output: 'answer_2_run_a' },
            ],
          },
        },
        run_b: {
          '/table.json': {
            columns: ['data', 'output'],
            path: '/table.json',
            entries: [
              { data: 'question_1', output: 'answer_1_run_b' },
              { data: 'question_2', output: 'answer_2_run_b' },
            ],
          },
        },
        run_c: {
          '/table_c.json': {
            columns: ['data', 'output'],
            path: '/table_c.json',
            entries: [
              { data: 'question_1', output: 'answer_1_run_c' },
              { data: 'question_2', output: 'answer_2_run_c' },
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

    const mockStore = configureStore([thunk, promiseMiddleware()])({
      evaluationData: SAMPLE_STATE,
      modelGateway: { modelGatewayRoutesLoading: {} },
    });

    renderWithIntl(
      <BrowserRouter>
        <Provider store={mockStore}>
          <EvaluationArtifactCompareTable
            resultList={resultList}
            visibleRuns={visibleRuns}
            groupByColumns={groupByColumns}
            onHideRun={onHideRun}
            onDatasetSelected={onDatasetSelected}
            highlightedText={highlightedText}
            outputColumnName={outputColumnName}
            isImageColumn={isImageColumn}
          />
        </Provider>
      </BrowserRouter>,
    );
  };

  it('should render the component', async () => {
    const resultList: UseEvaluationArtifactTableDataResult = [
      {
        key: 'question_1',
        groupByCellValues: {
          data: 'question_1',
        },
        cellValues: {
          run_c: 'answer_1_run_c',
          run_b: 'answer_1_run_b',
          run_a: 'answer_1_run_a',
        },
        isPendingInputRow: false,
      },
      {
        key: 'question_2',
        groupByCellValues: {
          data: 'question_2',
        },
        cellValues: {
          run_c: 'answer_2_run_c',
          run_b: 'answer_2_run_b',
          run_a: 'answer_2_run_a',
        },
        isPendingInputRow: false,
      },
    ];

    renderComponent(resultList, ['data'], 'output', false);

    await waitFor(
      () => {
        expect(screen.getByRole('columnheader', { name: 'data' })).toBeInTheDocument();
        expect(screen.getByRole('columnheader', { name: new RegExp('output', 'i') })).toBeInTheDocument();
        ['able-panda-761', 'able-panda-762', 'able-panda-763'].forEach((value) => {
          expect(screen.getByRole('columnheader', { name: new RegExp(value, 'i') })).toBeInTheDocument();
        });
        resultList.forEach((result) => {
          Object.values(result.groupByCellValues).forEach((value) => {
            expect(screen.getByRole('gridcell', { name: value })).toBeInTheDocument();
          });
          Object.values(result.cellValues).forEach((value) => {
            if (typeof value === 'string') {
              expect(screen.getByRole('gridcell', { name: value })).toBeInTheDocument();
            }
          });
        });
      },
      { timeout: 90000 },
    );
  });

  it('should render the component with multiple groups', async () => {
    const resultList: UseEvaluationArtifactTableDataResult = [
      {
        key: 'question_1.answer_1_run_c',
        groupByCellValues: {
          data: 'question_1',
          output: 'answer_1_run_c',
        },
        cellValues: {},
        isPendingInputRow: false,
      },
      {
        key: 'question_2.answer_2_run_c',
        groupByCellValues: {
          data: 'question_2',
          output: 'answer_2_run_c',
        },
        cellValues: {},
        isPendingInputRow: false,
      },
      {
        key: 'question_1.answer_1_run_b',
        groupByCellValues: {
          data: 'question_1',
          output: 'answer_1_run_b',
        },
        cellValues: {},
        isPendingInputRow: false,
      },
      {
        key: 'question_2.answer_2_run_b',
        groupByCellValues: {
          data: 'question_2',
          output: 'answer_2_run_b',
        },
        cellValues: {},
        isPendingInputRow: false,
      },
      {
        key: 'question_1.answer_1_run_a',
        groupByCellValues: {
          data: 'question_1',
          output: 'answer_1_run_a',
        },
        cellValues: {},
        isPendingInputRow: false,
      },
      {
        key: 'question_2.answer_2_run_a',
        groupByCellValues: {
          data: 'question_2',
          output: 'answer_2_run_a',
        },
        cellValues: {},
        isPendingInputRow: false,
      },
    ];

    renderComponent(resultList, ['data', 'output'], '', false);

    await waitFor(() => {
      expect(screen.getByRole('columnheader', { name: 'data' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'output' })).toBeInTheDocument();
      ['able-panda-761', 'able-panda-762', 'able-panda-763'].forEach((value) => {
        expect(screen.getByRole('columnheader', { name: new RegExp(value, 'i') })).toBeInTheDocument();
      });
      resultList.forEach((result) => {
        Object.values(result.groupByCellValues).forEach((value) => {
          const cells = screen.getAllByRole('gridcell', { name: value });
          expect(cells.length).toBeGreaterThan(0);
        });
      });
    });
  });

  const renderMultiTypeComponent = (
    resultList: UseEvaluationArtifactTableDataResult,
    groupByColumns: string[],
    outputColumnName: string,
    isImageColumn: boolean,
  ) => {
    const visibleRuns: RunRowType[] = [
      {
        runUuid: 'run_a',
        runName: 'able-panda-761',
        rowUuid: '9b1b553bb1ca4e948c248c5ca426ae52',
        runInfo: {
          runUuid: 'run_a',
          status: 'FINISHED',
          artifactUri: 'dbfs:/databricks/mlflow-tracking/676587362364997/9b1b553bb1ca4e948c248c5ca426ae52/artifacts',
          endTime: 1717111275344,
          experimentId: '676587362364997',
          lifecycleStage: 'active',
          runName: 'able-panda-761',
          startTime: 1717111273526,
        },
        duration: '1.8s',
        tags: {
          'mlflow.loggedArtifacts': {
            key: 'mlflow.loggedArtifacts',
            value: '[{"path": "table.json", "type": "table"}]',
          },
        },
        models: {
          registeredModels: [],
          loggedModels: [],
          experimentId: '676587362364997',
          runUuid: 'run_a',
        },
        pinnable: true,
        hidden: false,
        pinned: false,
        datasets: [],
      },
      {
        runUuid: 'run_b',
        runName: 'able-panda-762',
        rowUuid: '9b1b553bb1ca4e948c248c5ca426ae52',
        runInfo: {
          runUuid: 'run_b',
          status: 'FINISHED',
          artifactUri: 'dbfs:/databricks/mlflow-tracking/676587362364997/9b1b553bb1ca4e948c248c5ca426ae52/artifacts',
          endTime: 1717111275344,
          experimentId: '676587362364997',
          lifecycleStage: 'active',
          runName: 'able-panda-762',
          startTime: 1717111273526,
        },
        duration: '1.8s',
        tags: {
          'mlflow.loggedArtifacts': {
            key: 'mlflow.loggedArtifacts',
            value: '[{"path": "table.json", "type": "table"}]',
          },
        },
        models: {
          registeredModels: [],
          loggedModels: [],
          experimentId: '676587362364997',
          runUuid: 'run_b',
        },
        pinnable: true,
        hidden: false,
        pinned: false,
        datasets: [],
      },
      {
        runUuid: 'run_c',
        runName: 'able-panda-763',
        rowUuid: '9b1b553bb1ca4e948c248c5ca426ae52',
        runInfo: {
          runUuid: 'run_c',
          status: 'FINISHED',
          artifactUri: 'dbfs:/databricks/mlflow-tracking/676587362364997/9b1b553bb1ca4e948c248c5ca426ae52/artifacts',
          endTime: 1717111275344,
          experimentId: '676587362364997',
          lifecycleStage: 'active',
          runName: 'able-panda-763',
          startTime: 1717111273526,
        },
        duration: '1.8s',
        tags: {
          'mlflow.loggedArtifacts': {
            key: 'mlflow.loggedArtifacts',
            value: '[{"path": "table_c.json", "type": "table"}]',
          },
        },
        models: {
          registeredModels: [],
          loggedModels: [],
          experimentId: '676587362364997',
          runUuid: 'run_c',
        },
        pinnable: true,
        hidden: false,
        pinned: false,
        datasets: [],
      },
    ];
    const onHideRun = jest.fn();
    const onDatasetSelected = jest.fn();
    const highlightedText = '';

    const arrayJson = JSON.stringify(['arr1', 'arr2']);

    const sampleState = {
      evaluationArtifactsBeingUploaded: {},
      evaluationArtifactsByRunUuid: {
        run_a: {
          '/table.json': {
            columns: [1234, 4321],
            path: '/table.json',
            entries: [
              { 1234: 1, 4321: arrayJson },
              { 1234: null, 4321: 'text' },
              { 1234: 'text', 4321: null },
              { 1234: arrayJson, 4321: 1 },
            ],
          },
        },
        run_b: {
          '/table.json': {
            columns: [1234, 'output'],
            path: '/table.json',
            entries: [
              { 1234: 1, output: 1 },
              { 1234: null, output: null },
              { 1234: 'text', output: 'text' },
              { 1234: arrayJson, output: arrayJson },
            ],
          },
        },
        run_c: {
          '/table_c.json': {
            columns: ['data', null],
            path: '/table_c.json',
            entries: [
              { data: 1, null: 'text' },
              { data: null, null: 'text2' },
              { data: 'text', null: 'text3' },
              { data: arrayJson, null: 4 },
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

    const mockStore = configureStore([thunk, promiseMiddleware()])({
      evaluationData: sampleState,
      modelGateway: { modelGatewayRoutesLoading: {} },
    });

    renderWithIntl(
      <BrowserRouter>
        <Provider store={mockStore}>
          <EvaluationArtifactCompareTable
            resultList={resultList}
            visibleRuns={visibleRuns}
            groupByColumns={groupByColumns}
            onHideRun={onHideRun}
            onDatasetSelected={onDatasetSelected}
            highlightedText={highlightedText}
            outputColumnName={outputColumnName}
            isImageColumn={isImageColumn}
          />
        </Provider>
      </BrowserRouter>,
    );
  };

  it('should render the component with different types', async () => {
    const resultList: UseEvaluationArtifactTableDataResult = [
      {
        key: '1',
        groupByCellValues: {
          '1234': '1',
        },
        cellValues: {
          run_a: 1 as any,
        },
        isPendingInputRow: false,
      },
      {
        key: 'null',
        groupByCellValues: {
          '1234': 'null',
        },
        cellValues: {
          run_a: null as any,
        },
        isPendingInputRow: false,
      },
      {
        key: 'text',
        groupByCellValues: {
          '1234': 'text',
        },
        cellValues: {
          run_a: 'text',
        },
        isPendingInputRow: false,
      },
      {
        key: '["arr1","arr2"]',
        groupByCellValues: {
          '1234': '["arr1","arr2"]',
        },
        cellValues: {
          run_a: '["arr1","arr2"]',
        },
        isPendingInputRow: false,
      },
    ];

    renderMultiTypeComponent(resultList, ['data'], 'null', false);

    await waitFor(() => {
      expect(screen.getByRole('columnheader', { name: 'data' })).toBeInTheDocument();
      ['able-panda-761', 'able-panda-762', 'able-panda-763'].forEach((value) => {
        expect(screen.getByRole('columnheader', { name: new RegExp(value, 'i') })).toBeInTheDocument();
      });
      resultList.forEach((result) => {
        Object.values(result.groupByCellValues).forEach((value) => {
          let cells = [];
          if (value === 'null') {
            cells = screen.getAllByRole('gridcell', { name: '(empty)' });
            expect(cells.length).toBeGreaterThan(0);
          } else if (value === '["arr1","arr2"]') {
            // Becomes a code block
            const complexDiv = screen.getByRole('gridcell', {
              name: /arr1/i,
            });
            expect(complexDiv).toBeInTheDocument();
          } else {
            cells = screen.getAllByRole('gridcell', { name: value });
            expect(cells.length).toBeGreaterThan(0);
          }
        });
      });
    });
  });

  const renderImageComponent = (
    resultList: UseEvaluationArtifactTableDataResult,
    groupByColumns: string[],
    outputColumnName: string,
    isImageColumn: boolean,
  ) => {
    const visibleRuns: RunRowType[] = [
      {
        runUuid: 'run_a',
        runName: 'able-panda-761',
        rowUuid: '9b1b553bb1ca4e948c248c5ca426ae52',
        runInfo: {
          runUuid: 'run_a',
          status: 'FINISHED',
          artifactUri: 'dbfs:/databricks/mlflow-tracking/676587362364997/9b1b553bb1ca4e948c248c5ca426ae52/artifacts',
          endTime: 1717111275344,
          experimentId: '676587362364997',
          lifecycleStage: 'active',
          runName: 'able-panda-761',
          startTime: 1717111273526,
        },
        duration: '1.8s',
        tags: {
          'mlflow.loggedArtifacts': {
            key: 'mlflow.loggedArtifacts',
            value: '[{"path": "table.json", "type": "table"}]',
          },
        },
        models: {
          registeredModels: [],
          loggedModels: [],
          experimentId: '676587362364997',
          runUuid: 'run_a',
        },
        pinnable: true,
        hidden: false,
        pinned: false,
        datasets: [],
      },
      {
        runUuid: 'run_b',
        runName: 'able-panda-762',
        rowUuid: '9b1b553bb1ca4e948c248c5ca426ae52',
        runInfo: {
          runUuid: 'run_b',
          status: 'FINISHED',
          artifactUri: 'dbfs:/databricks/mlflow-tracking/676587362364997/9b1b553bb1ca4e948c248c5ca426ae52/artifacts',
          endTime: 1717111275344,
          experimentId: '676587362364997',
          lifecycleStage: 'active',
          runName: 'able-panda-762',
          startTime: 1717111273526,
        },
        duration: '1.8s',
        tags: {
          'mlflow.loggedArtifacts': {
            key: 'mlflow.loggedArtifacts',
            value: '[{"path": "table.json", "type": "table"}]',
          },
        },
        models: {
          registeredModels: [],
          loggedModels: [],
          experimentId: '676587362364997',
          runUuid: 'run_b',
        },
        pinnable: true,
        hidden: false,
        pinned: false,
        datasets: [],
      },
      {
        runUuid: 'run_c',
        runName: 'able-panda-763',
        rowUuid: '9b1b553bb1ca4e948c248c5ca426ae52',
        runInfo: {
          runUuid: 'run_c',
          status: 'FINISHED',
          artifactUri: 'dbfs:/databricks/mlflow-tracking/676587362364997/9b1b553bb1ca4e948c248c5ca426ae52/artifacts',
          endTime: 1717111275344,
          experimentId: '676587362364997',
          lifecycleStage: 'active',
          runName: 'able-panda-763',
          startTime: 1717111273526,
        },
        duration: '1.8s',
        tags: {
          'mlflow.loggedArtifacts': {
            key: 'mlflow.loggedArtifacts',
            value: '[{"path": "table_c.json", "type": "table"}]',
          },
        },
        models: {
          registeredModels: [],
          loggedModels: [],
          experimentId: '676587362364997',
          runUuid: 'run_c',
        },
        pinnable: true,
        hidden: false,
        pinned: false,
        datasets: [],
      },
    ];
    const onHideRun = jest.fn();
    const onDatasetSelected = jest.fn();
    const highlightedText = '';

    const imageJson = {
      type: 'image',
      filepath: 'fakePathUncompressed',
      compressed_filepath: 'fakePath',
    };
    const sampleState = {
      evaluationArtifactsBeingUploaded: {},
      evaluationArtifactsByRunUuid: {
        run_a: {
          '/table.json': {
            columns: [1234, 4321],
            path: '/table.json',
            entries: [
              { image: imageJson, text: 1 },
              { image: imageJson, text: 2 },
              { image: imageJson, text: 3 },
            ],
          },
        },
        run_b: {
          '/table.json': {
            columns: [1234, 'output'],
            path: '/table.json',
            entries: [
              { image: imageJson, text: 1 },
              { image: imageJson, text: 2 },
              { image: imageJson, text: 3 },
            ],
          },
        },
        run_c: {
          '/table_c.json': {
            columns: ['data', null],
            path: '/table_c.json',
            entries: [
              { image: imageJson, text: 1 },
              { image: imageJson, text: 2 },
              { image: imageJson, text: 3 },
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

    const mockStore = configureStore([thunk, promiseMiddleware()])({
      evaluationData: sampleState,
      modelGateway: { modelGatewayRoutesLoading: {} },
    });

    renderWithIntl(
      <BrowserRouter>
        <Provider store={mockStore}>
          <EvaluationArtifactCompareTable
            resultList={resultList}
            visibleRuns={visibleRuns}
            groupByColumns={groupByColumns}
            onHideRun={onHideRun}
            onDatasetSelected={onDatasetSelected}
            highlightedText={highlightedText}
            outputColumnName={outputColumnName}
            isImageColumn={isImageColumn}
          />
        </Provider>
      </BrowserRouter>,
    );
  };

  it('should render the component with images', async () => {
    const imageJson = {
      url: 'get-artifact?path=fakePathUncompressed&run_uuid=test-run-uuid',
      compressed_url: 'get-artifact?path=fakePath&run_uuid=test-run-uuid',
    };
    const resultList: UseEvaluationArtifactTableDataResult = [
      {
        key: '1',
        groupByCellValues: {
          text: '1',
        },
        cellValues: {
          run_a: imageJson,
          run_b: imageJson,
          run_c: imageJson,
        },
        isPendingInputRow: false,
      },
      {
        key: '2',
        groupByCellValues: {
          text: '2',
        },
        cellValues: {
          run_a: imageJson,
          run_b: imageJson,
          run_c: imageJson,
        },
        isPendingInputRow: false,
      },
      {
        key: '3',
        groupByCellValues: {
          text: '3',
        },
        cellValues: {
          run_a: imageJson,
          run_b: imageJson,
          run_c: imageJson,
        },
        isPendingInputRow: false,
      },
    ];

    renderImageComponent(resultList, ['text'], 'image', true);

    await waitFor(() => {
      expect(screen.getByRole('columnheader', { name: 'text' })).toBeInTheDocument();
      ['able-panda-761', 'able-panda-762', 'able-panda-763'].forEach((value) => {
        expect(screen.getByRole('columnheader', { name: new RegExp(value, 'i') })).toBeInTheDocument();
      });
    });

    await waitFor(() => {
      const image = screen.getAllByRole('img');
      expect(image.length).toBeGreaterThan(0);
      expect(image[0]).toBeInTheDocument();
      expect(image[0]).toHaveAttribute(
        'src',
        expect.stringContaining('get-artifact?path=fakePath&run_uuid=test-run-uuid'),
      );
    });
  });
});

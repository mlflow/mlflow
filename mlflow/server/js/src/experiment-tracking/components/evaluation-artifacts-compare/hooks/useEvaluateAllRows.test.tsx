import { act, renderHook } from '@testing-library/react';
import { useEvaluateAllRows } from './useEvaluateAllRows';
import { Provider } from 'react-redux';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import configureStore from 'redux-mock-store';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { evaluatePromptTableValue } from '../../../actions/PromptEngineeringActions';
import { cloneDeep } from 'lodash';
import type { UseEvaluationArtifactTableDataResult } from './useEvaluationArtifactTableData';
import { IntlProvider } from 'react-intl';

// Mock the evaluation action, it simulates taking 1000 ms to evaluate a single value
jest.mock('../../../actions/PromptEngineeringActions', () => ({
  evaluatePromptTableValue: jest.fn(() => ({
    type: 'evaluatePromptTableValue',
    payload: new Promise((resolve) => setTimeout(resolve, 1000)),
    meta: {},
  })),
}));

jest.useFakeTimers();

// Sample table with evaluation data. There are two input sets.
// - run_1 has recorded outputs for both inputs
// - run_2 has recorded output for one input
// - run_3 has no recorded outputs
const mockEvaluationTable: UseEvaluationArtifactTableDataResult = [
  {
    cellValues: { run_1: 'answer_alpha_1', run_2: '', run_3: '' },
    groupByCellValues: {
      col_question: 'question_alpha',
    },
    key: 'question',
  },
  {
    cellValues: { run_1: 'answer_beta_1', run_2: 'answer_beta_2', run_3: '' },
    groupByCellValues: {
      col_question: 'question_beta',
    },
    key: 'question_beta',
  },
];

// Utility function: creates a mocked run row (column in the evaluation table)
const createMockRun = (name: string): RunRowType =>
  ({
    runUuid: name,
    runName: name,
    params: [
      { key: 'model_route', value: 'model-route' },
      { key: 'prompt_template', value: 'this is a prompt template with {{ col_question }}' },
      { key: 'max_tokens', value: '100' },
      { key: 'temperature', value: '0.5' },
    ],
  } as any);

// Create three mocked runs
const mockRun1 = createMockRun('run_1');
const mockRun2 = createMockRun('run_2');
const mockRun3 = createMockRun('run_3');

// Utility function: creates a new result evaluation table with updated row
const updateResultTable = (
  sourceTable = mockEvaluationTable,
  questionValue: string,
  runUuid: string,
  newValue: string,
) => {
  const updatedTable = cloneDeep(sourceTable);
  const row = updatedTable.find(({ groupByCellValues }) => groupByCellValues['col_question'] === questionValue);

  if (row) {
    row.cellValues[runUuid] = newValue;
  }
  return updatedTable;
};

describe('useEvaluateAllRows', () => {
  const render = () => {
    const mockStore = configureStore([thunk, promiseMiddleware()])({});
    const { result, rerender } = renderHook((props) => useEvaluateAllRows(props.evaluationTable, 'col_output'), {
      initialProps: {
        evaluationTable: mockEvaluationTable,
      },
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <Provider store={mockStore}>{children}</Provider>
        </IntlProvider>
      ),
    });
    return { getResult: () => result.current, rerender };
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('it should ignore evaluation of the already fully evaluated run', async () => {
    const { getResult } = render();
    expect(getResult().runColumnsBeingEvaluated).toEqual([]);
    await act(async () => {
      getResult().startEvaluatingRunColumn(mockRun1);
    });
    expect(getResult().runColumnsBeingEvaluated).toEqual([]);
  });

  test('it should properly kickstart evaluation of partially evaluated run', async () => {
    const { getResult, rerender } = render();
    expect(getResult().runColumnsBeingEvaluated).toEqual([]);
    await act(async () => {
      getResult().startEvaluatingRunColumn(mockRun2);
    });
    expect(getResult().runColumnsBeingEvaluated).toEqual(['run_2']);
    expect(evaluatePromptTableValue).toHaveBeenCalledWith(
      expect.objectContaining({
        compiledPrompt: 'this is a prompt template with question_alpha',
        inputValues: {
          col_question: 'question_alpha',
        },
        outputColumn: 'col_output',
        routeName: 'model-route',
        rowKey: 'question',
        run: mockRun2,
      }),
    );

    await act(async () => {
      rerender({
        evaluationTable: updateResultTable(mockEvaluationTable, 'question_alpha', 'run_2', 'newly_evaluated_data'),
      });
    });

    await act(async () => {
      jest.advanceTimersByTime(2000);
    });

    expect(getResult().runColumnsBeingEvaluated).toEqual([]);
  });

  test('it should properly process a run column with multiple values to be evaluated', async () => {
    const { getResult, rerender } = render();
    expect(getResult().runColumnsBeingEvaluated).toEqual([]);
    await act(async () => {
      getResult().startEvaluatingRunColumn(mockRun3);
    });
    expect(getResult().runColumnsBeingEvaluated).toEqual(['run_3']);
    expect(evaluatePromptTableValue).toHaveBeenCalledTimes(1);

    expect(evaluatePromptTableValue).toHaveBeenCalledWith(
      expect.objectContaining({
        compiledPrompt: 'this is a prompt template with question_alpha',
        inputValues: {
          col_question: 'question_alpha',
        },
        run: mockRun3,
      }),
    );

    const updatedTable = updateResultTable(mockEvaluationTable, 'question_alpha', 'run_3', 'newly_evaluated_data');

    rerender({
      evaluationTable: updatedTable,
    });

    await act(async () => {
      jest.advanceTimersByTime(2000);
    });

    expect(evaluatePromptTableValue).toHaveBeenCalledTimes(2);

    expect(evaluatePromptTableValue).toHaveBeenCalledWith(
      expect.objectContaining({
        compiledPrompt: 'this is a prompt template with question_beta',
        inputValues: {
          col_question: 'question_beta',
        },
        run: mockRun3,
      }),
    );

    rerender({
      evaluationTable: updateResultTable(updatedTable, 'question_beta', 'run_3', 'newly_evaluated_data'),
    });

    await act(async () => {
      jest.advanceTimersByTime(2000);
    });

    expect(getResult().runColumnsBeingEvaluated).toEqual([]);
  });

  test('it should properly stop evaluating', async () => {
    const { getResult, rerender } = render();
    expect(getResult().runColumnsBeingEvaluated).toEqual([]);
    await act(async () => {
      getResult().startEvaluatingRunColumn(mockRun3);
    });
    expect(getResult().runColumnsBeingEvaluated).toEqual(['run_3']);
    expect(evaluatePromptTableValue).toHaveBeenCalledTimes(1);

    expect(evaluatePromptTableValue).toHaveBeenCalledWith(
      expect.objectContaining({
        compiledPrompt: 'this is a prompt template with question_alpha',
        inputValues: {
          col_question: 'question_alpha',
        },
        run: mockRun3,
      }),
    );

    const updatedTable = updateResultTable(mockEvaluationTable, 'question_alpha', 'run_3', 'newly_evaluated_data');

    rerender({
      evaluationTable: updatedTable,
    });

    await act(async () => {
      getResult().stopEvaluatingRunColumn(mockRun3);
    });
    await act(async () => {
      jest.advanceTimersByTime(2000);
    });

    // Contrary to previous test, we don't get additional action call
    expect(evaluatePromptTableValue).toHaveBeenCalledTimes(1);
  });
});

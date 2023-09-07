import { rejected } from '../../common/utils/ActionUtils';
import { fulfilled, pending } from '../../common/utils/ActionUtils';
import { GET_EVALUATION_TABLE_ARTIFACT, GetEvaluationTableArtifactAction } from '../actions';
import { EvaluationArtifactTable, EvaluationArtifactTableEntry } from '../types';
import { evaluationDataReducer } from './EvaluationDataReducer';

describe('evaluationDataReducer', () => {
  const emptyState: ReturnType<typeof evaluationDataReducer> = {
    evaluationArtifactsByRunUuid: {},
    evaluationArtifactsErrorByRunUuid: {},
    evaluationArtifactsLoadingByRunUuid: {},
  };

  const MOCK_ENTRY_A: EvaluationArtifactTableEntry = {
    input: 'input_a',
    output: 'output_a',
    target: 'target_a',
  };

  const MOCK_ENTRY_B: EvaluationArtifactTableEntry = {
    input: 'input_b',
    output: 'output_b',
    target: 'target_b',
  };

  const MOCK_ENTRY_C: EvaluationArtifactTableEntry = {
    input: 'input_c',
    output: 'output_c',
    target: 'target_c',
  };

  const MOCK_ENTRIES = [MOCK_ENTRY_A, MOCK_ENTRY_B, MOCK_ENTRY_C];
  const MOCK_COLUMNS = ['input', 'output', 'target'];

  const mockFulfilledAction = (
    runUuid: string,
    artifactPath: string,
    payload: Omit<EvaluationArtifactTable, 'path'> | Error = {
      entries: MOCK_ENTRIES,
      columns: MOCK_COLUMNS,
    },
  ): GetEvaluationTableArtifactAction => ({
    type: fulfilled(GET_EVALUATION_TABLE_ARTIFACT),
    meta: { runUuid, artifactPath },
    payload: { ...payload, path: artifactPath },
  });

  const mockPendingAction = (runUuid: string, artifactPath: string) => ({
    type: pending(GET_EVALUATION_TABLE_ARTIFACT),
    meta: { runUuid, artifactPath },
    payload: {} as any,
  });

  const mockRejectedAction = (runUuid: string, artifactPath: string) => ({
    type: rejected(GET_EVALUATION_TABLE_ARTIFACT),
    meta: { runUuid, artifactPath },
    payload: new Error('Mock error'),
  });

  it('artifact entries are correctly populated for multiple runs', () => {
    let state = emptyState;
    state = evaluationDataReducer(
      state,
      mockFulfilledAction('run_1', '/some/artifact', {
        entries: [MOCK_ENTRY_A, MOCK_ENTRY_B],
        columns: MOCK_COLUMNS,
      }),
    );
    state = evaluationDataReducer(
      state,
      mockFulfilledAction('run_2', '/some/other/artifact', {
        entries: [MOCK_ENTRY_C],
        columns: MOCK_COLUMNS,
      }),
    );
    const { evaluationArtifactsByRunUuid } = state;
    expect(evaluationArtifactsByRunUuid['run_1']).toEqual({
      '/some/artifact': {
        columns: ['input', 'output', 'target'],
        entries: [
          { input: 'input_a', output: 'output_a', target: 'target_a' },
          { input: 'input_b', output: 'output_b', target: 'target_b' },
        ],
        path: '/some/artifact',
      },
    });
    expect(evaluationArtifactsByRunUuid['run_2']).toEqual({
      '/some/other/artifact': {
        columns: ['input', 'output', 'target'],
        entries: [{ input: 'input_c', output: 'output_c', target: 'target_c' }],
        path: '/some/other/artifact',
      },
    });
  });

  it('correctly sets loading state', () => {
    let state = emptyState;
    state = evaluationDataReducer(state, mockPendingAction('run_1', '/some/artifact'));
    state = evaluationDataReducer(state, mockPendingAction('run_2', '/some/artifact'));
    state = evaluationDataReducer(state, mockFulfilledAction('run_1', '/some/artifact'));
    const { evaluationArtifactsLoadingByRunUuid } = state;

    expect(evaluationArtifactsLoadingByRunUuid['run_1']['/some/artifact']).toEqual(false);
    expect(evaluationArtifactsLoadingByRunUuid['run_2']['/some/artifact']).toEqual(true);
  });

  it('correctly marks failed attempts to fetch artifacts', () => {
    let state = emptyState;
    state = evaluationDataReducer(state, mockRejectedAction('run_1', '/some/artifact'));
    const { evaluationArtifactsErrorByRunUuid } = state;

    expect(evaluationArtifactsErrorByRunUuid['run_1']['/some/artifact']).toMatch(/Mock error/);
  });
});

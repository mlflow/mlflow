import { rejected } from '../../common/utils/ActionUtils';
import { fulfilled, pending } from '../../common/utils/ActionUtils';
import { AsyncRejectedAction } from '../../redux-types';
import { GET_EVALUATION_TABLE_ARTIFACT, GetEvaluationTableArtifactAction, UPLOAD_ARTIFACT_API } from '../actions';
import { WRITE_BACK_EVALUATION_ARTIFACTS } from '../actions/PromptEngineeringActions';
import type { EvaluationArtifactTable, EvaluationArtifactTableEntry } from '../types';
import { evaluationDataReducer } from './EvaluationDataReducer';

describe('evaluationDataReducer', () => {
  const emptyState: ReturnType<typeof evaluationDataReducer> = {
    evaluationArtifactsByRunUuid: {},
    evaluationArtifactsErrorByRunUuid: {},
    evaluationArtifactsLoadingByRunUuid: {},
    evaluationDraftInputValues: [],
    evaluationPendingDataByRunUuid: {},
    evaluationPendingDataLoadingByRunUuid: {},
    evaluationArtifactsBeingUploaded: {},
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
  ) => ({
    type: fulfilled(GET_EVALUATION_TABLE_ARTIFACT),
    meta: { runUuid, artifactPath },
    payload: { ...payload, path: artifactPath } as any,
  });

  const mockPendingAction = (runUuid: string, artifactPath: string) => ({
    type: pending(GET_EVALUATION_TABLE_ARTIFACT),
    meta: { runUuid, artifactPath },
    payload: {} as any,
  });

  const mockRejectedAction = (runUuid: string, artifactPath: string) => ({
    type: rejected(GET_EVALUATION_TABLE_ARTIFACT),
    meta: { runUuid, artifactPath },
    payload: new Error('Mock error') as any,
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

  it('correctly indicates artifacts being currently uploaded', () => {
    let state = emptyState;
    state = evaluationDataReducer(state, {
      type: pending(WRITE_BACK_EVALUATION_ARTIFACTS),
      meta: { runUuidsToUpdate: ['run_1', 'run_2'], artifactPath: '/some/artifact' },
    });

    expect(state.evaluationArtifactsBeingUploaded).toEqual({
      run_1: { '/some/artifact': true },
      run_2: { '/some/artifact': true },
    });

    state = evaluationDataReducer(state, {
      type: fulfilled(UPLOAD_ARTIFACT_API),
      meta: { id: '1', runUuid: 'run_1', filePath: '/some/artifact' },
      payload: {},
    });

    expect(state.evaluationArtifactsBeingUploaded).toEqual({
      run_1: { '/some/artifact': false },
      run_2: { '/some/artifact': true },
    });

    state = evaluationDataReducer(state, {
      type: rejected(UPLOAD_ARTIFACT_API),
      meta: { id: '1', runUuid: 'run_2', filePath: '/some/artifact' },
      payload: new Error() as any,
    });

    expect(state.evaluationArtifactsBeingUploaded).toEqual({
      run_1: { '/some/artifact': false },
      run_2: { '/some/artifact': false },
    });
  });

  it('correctly saves newly written data', () => {
    let state = emptyState;
    const newEvaluationTable = {
      columns: MOCK_COLUMNS,
      entries: MOCK_ENTRIES,
      path: '/some/artifact',
    };

    state = evaluationDataReducer(state, {
      type: fulfilled(WRITE_BACK_EVALUATION_ARTIFACTS),
      payload: [
        {
          runUuid: 'run_1',
          newEvaluationTable,
        },
      ],
      meta: { runUuidsToUpdate: ['run_1'], artifactPath: '/some/artifact' },
    });

    expect(state.evaluationArtifactsByRunUuid).toEqual({
      run_1: { '/some/artifact': newEvaluationTable },
    });
  });
});

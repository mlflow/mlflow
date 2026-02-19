import { combineReducers } from 'redux';
import { fulfilled, pending, rejected } from '../../common/utils/ActionUtils';
import type { EvaluationArtifactTable, PendingEvaluationArtifactTableEntry } from '../types';

import type { AsyncAction, AsyncFulfilledAction, AsyncPendingAction, AsyncRejectedAction } from '../../redux-types';
import type {
  DiscardPendingEvaluationDataAction,
  EvaluateAddInputValues,
  EvaluatePromptTableValueAction,
  WriteBackEvaluationArtifactsAction,
} from '../actions/PromptEngineeringActions';
import type { GetEvaluationTableArtifactAction, UploadArtifactApiAction } from '../actions';
import {
  DEFAULT_PROMPTLAB_OUTPUT_COLUMN,
  DEFAULT_PROMPTLAB_PROMPT_COLUMN,
} from '../components/prompt-engineering/PromptEngineering.utils';

export interface EvaluationDataReduxState {
  /**
   * Stores artifact data indexed by run UUID and by the artifact path
   */
  evaluationArtifactsByRunUuid: {
    [runUuid: string]: { [artifactPath: string]: EvaluationArtifactTable };
  };
  /**
   * Determines if particular artifact is being loaded, indexed by run UUID and by the artifact path
   */
  evaluationArtifactsLoadingByRunUuid: {
    [runUuid: string]: { [artifactPath: string]: boolean };
  };
  /**
   * Stores errors of fetching artifacts, indexed by run UUID and by the artifact path
   */
  evaluationArtifactsErrorByRunUuid: {
    [runUuid: string]: { [artifactPath: string]: string };
  };

  evaluationPendingDataByRunUuid: {
    [runUuid: string]: PendingEvaluationArtifactTableEntry[];
  };

  evaluationPendingDataLoadingByRunUuid: {
    [runUuid: string]: { [inputHash: string]: boolean };
  };

  evaluationDraftInputValues: Record<string, string>[];

  evaluationArtifactsBeingUploaded: {
    [runUuid: string]: { [artifactPath: string]: boolean };
  };
}

const evaluationArtifactsBeingUploaded = (
  state: {
    [runUuid: string]: { [artifactPath: string]: boolean };
  } = {},
  action:
    | AsyncPendingAction<WriteBackEvaluationArtifactsAction>
    | AsyncFulfilledAction<UploadArtifactApiAction>
    | AsyncRejectedAction<UploadArtifactApiAction>,
) => {
  if (action.type === pending('WRITE_BACK_EVALUATION_ARTIFACTS') && action.meta) {
    const { runUuidsToUpdate, artifactPath } = action.meta;
    return runUuidsToUpdate.reduce<{
      [runUuid: string]: { [artifactPath: string]: boolean };
    }>(
      (aggregate, runUuid) => ({
        ...aggregate,
        [runUuid]: { [artifactPath]: true },
      }),
      {},
    );
  }
  if (
    (action.type === fulfilled('UPLOAD_ARTIFACT_API') || action.type === rejected('UPLOAD_ARTIFACT_API')) &&
    action.meta
  ) {
    const { filePath, runUuid } = action.meta;
    return { ...state, [runUuid]: { ...state[runUuid], [filePath]: false } };
  }
  return state;
};

const evaluationDraftInputValues = (
  state: Record<string, string>[] = [],
  action:
    | EvaluateAddInputValues
    | AsyncFulfilledAction<WriteBackEvaluationArtifactsAction>
    | DiscardPendingEvaluationDataAction,
) => {
  if (action.type === 'DISCARD_PENDING_EVALUATION_DATA') {
    return [];
  }
  if (action.type === 'EVALUATE_ADD_INPUT_VALUES') {
    return [...state, action.payload];
  }
  if (action.type === fulfilled('WRITE_BACK_EVALUATION_ARTIFACTS')) {
    return [];
  }
  return state;
};

const evaluationArtifactsByRunUuid = (
  state: { [runUuid: string]: EvaluationArtifactTable[] } = {},
  action:
    | AsyncFulfilledAction<GetEvaluationTableArtifactAction>
    | AsyncFulfilledAction<WriteBackEvaluationArtifactsAction>,
) => {
  if (action.type === fulfilled('WRITE_BACK_EVALUATION_ARTIFACTS') && action.meta) {
    const { artifactPath } = action.meta;
    const updatedRunTables = action.payload;

    const newState = { ...state };

    for (const { runUuid, newEvaluationTable } of updatedRunTables) {
      newState[runUuid] = { ...newState[runUuid], [artifactPath]: newEvaluationTable };
    }

    return newState;
  }
  if (action.type === fulfilled('GET_EVALUATION_TABLE_ARTIFACT') && action.meta) {
    const { runUuid, artifactPath } = action.meta;
    return { ...state, [runUuid]: { ...state[runUuid], [artifactPath]: action.payload } };
  }
  return state;
};

const evaluationPendingDataLoadingByRunUuid = (
  state: {
    [runUuid: string]: { [inputHash: string]: boolean };
  } = {},
  action: AsyncAction,
) => {
  if (action.meta && action.type === pending('EVALUATE_PROMPT_TABLE_VALUE')) {
    const { rowKey, run } = action.meta;
    const runEntries = state[run.runUuid] || {};
    runEntries[rowKey] = true;
    return { ...state, [run.runUuid]: runEntries };
  }
  if (
    action.meta &&
    (action.type === fulfilled('EVALUATE_PROMPT_TABLE_VALUE') ||
      action.type === rejected('EVALUATE_PROMPT_TABLE_VALUE'))
  ) {
    const { rowKey, run } = action.meta;
    const runEntries = state[run.runUuid] || {};
    runEntries[rowKey] = false;
    return { ...state, [run.runUuid]: runEntries };
  }
  return state;
};
const evaluationPendingDataByRunUuid = (
  state: {
    [runUuid: string]: PendingEvaluationArtifactTableEntry[];
  } = {},
  action:
    | AsyncFulfilledAction<EvaluatePromptTableValueAction>
    | AsyncFulfilledAction<WriteBackEvaluationArtifactsAction>
    | DiscardPendingEvaluationDataAction,
) => {
  if (action.type === 'DISCARD_PENDING_EVALUATION_DATA') {
    return {};
  }
  if (action.type === fulfilled('WRITE_BACK_EVALUATION_ARTIFACTS')) {
    const newState = { ...state };
    for (const runUuid of action.meta?.runUuidsToUpdate || []) {
      delete newState[runUuid];
    }
    return newState;
  }
  if (action.type === fulfilled('EVALUATE_PROMPT_TABLE_VALUE') && action.meta) {
    const { run, inputValues, startTime, compiledPrompt, gatewayRoute } = action.meta;

    const evaluationTime = !startTime ? 0 : performance.now() - startTime;

    const { metadata, text } = action.payload;

    const newEntry: PendingEvaluationArtifactTableEntry = {
      entryData: {
        ...inputValues,
        [DEFAULT_PROMPTLAB_OUTPUT_COLUMN]: text,
        [DEFAULT_PROMPTLAB_PROMPT_COLUMN]: compiledPrompt,
      },
      evaluationTime,
      totalTokens: metadata.total_tokens,
      isPending: true,
    };

    const runEntries = state[run.runUuid] || [];
    const existingEntry = runEntries.find(({ entryData: entry }) =>
      Object.entries(inputValues).every(([key, value]) => entry[key] === value),
    );
    const runEntriesWithoutDuplicate = existingEntry ? runEntries.filter((e) => e !== existingEntry) : runEntries;

    runEntriesWithoutDuplicate.push(newEntry);
    return { ...state, [run.runUuid]: runEntriesWithoutDuplicate };
  }
  return state;
};

const evaluationArtifactsLoadingByRunUuid = (
  state: {
    [runUuid: string]: { [artifactPath: string]: boolean };
  } = {},
  action:
    | AsyncPendingAction<GetEvaluationTableArtifactAction>
    | AsyncRejectedAction<GetEvaluationTableArtifactAction>
    | AsyncFulfilledAction<GetEvaluationTableArtifactAction>,
) => {
  if (action.type === pending('GET_EVALUATION_TABLE_ARTIFACT') && action.meta) {
    const { runUuid, artifactPath } = action.meta;
    return { ...state, [runUuid]: { ...state[runUuid], [artifactPath]: true } };
  }
  if (
    action.type === rejected('GET_EVALUATION_TABLE_ARTIFACT') ||
    action.type === fulfilled('GET_EVALUATION_TABLE_ARTIFACT')
  ) {
    if (!action.meta) {
      return state;
    }
    const { runUuid, artifactPath } = action.meta;
    return { ...state, [runUuid]: { ...state[runUuid], [artifactPath]: false } };
  }
  return state;
};

const evaluationArtifactsErrorByRunUuid = (
  state: {
    [runUuid: string]: { [artifactPath: string]: string };
  } = {},
  action: AsyncRejectedAction<GetEvaluationTableArtifactAction>,
) => {
  if (action.type === rejected('GET_EVALUATION_TABLE_ARTIFACT') && action.meta) {
    const { runUuid, artifactPath } = action.meta;
    const error = action.payload;
    return { ...state, [runUuid]: { ...state[runUuid], [artifactPath]: error?.toString() } };
  }
  return state;
};

export const evaluationDataReducer = combineReducers({
  evaluationDraftInputValues,
  evaluationArtifactsByRunUuid,
  evaluationArtifactsLoadingByRunUuid,
  evaluationArtifactsErrorByRunUuid,
  evaluationPendingDataByRunUuid,
  evaluationPendingDataLoadingByRunUuid,
  evaluationArtifactsBeingUploaded,
});

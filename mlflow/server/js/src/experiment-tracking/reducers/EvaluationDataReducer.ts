import { Action, combineReducers } from 'redux';
import { fulfilled, pending, rejected } from '../../common/utils/ActionUtils';
import { EvaluationArtifactTable } from '../types';
import { GET_EVALUATION_TABLE_ARTIFACT, GetEvaluationTableArtifactAction } from '../actions';

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
}

const evaluationArtifactsByRunUuid = (
  state: { [runUuid: string]: EvaluationArtifactTable[] } = {},
  action: GetEvaluationTableArtifactAction,
) => {
  if (action.type === fulfilled(GET_EVALUATION_TABLE_ARTIFACT)) {
    const { runUuid, artifactPath } = action.meta;
    return { ...state, [runUuid]: { ...state[runUuid], [artifactPath]: action.payload } };
  }
  return state;
};

const evaluationArtifactsLoadingByRunUuid = (
  state: {
    [runUuid: string]: { [artifactPath: string]: boolean };
  } = {},
  action: GetEvaluationTableArtifactAction,
) => {
  if (action.type === pending(GET_EVALUATION_TABLE_ARTIFACT)) {
    const { runUuid, artifactPath } = action.meta;
    return { ...state, [runUuid]: { ...state[runUuid], [artifactPath]: true } };
  }
  if (
    action.type === rejected(GET_EVALUATION_TABLE_ARTIFACT) ||
    action.type === fulfilled(GET_EVALUATION_TABLE_ARTIFACT)
  ) {
    const { runUuid, artifactPath } = action.meta;
    return { ...state, [runUuid]: { ...state[runUuid], [artifactPath]: false } };
  }
  return state;
};

const evaluationArtifactsErrorByRunUuid = (
  state: {
    [runUuid: string]: { [artifactPath: string]: string };
  } = {},
  action: GetEvaluationTableArtifactAction,
) => {
  if (action.type === rejected(GET_EVALUATION_TABLE_ARTIFACT)) {
    const { runUuid, artifactPath } = action.meta;
    const error = action.payload;
    return { ...state, [runUuid]: { ...state[runUuid], [artifactPath]: error?.toString() } };
  }
  return state;
};

export const evaluationDataReducer = combineReducers({
  evaluationArtifactsByRunUuid,
  evaluationArtifactsLoadingByRunUuid,
  evaluationArtifactsErrorByRunUuid,
});

import { cloneDeep, fromPairs } from 'lodash';
import type { Action } from 'redux';
import Utils from '../../common/utils/Utils';
import type { AsyncAction, ReduxState, ThunkDispatch } from '../../redux-types';
import { uploadArtifactApi } from '../actions';
import type { RunRowType } from '../components/experiment-page/utils/experimentPage.row-types';
import { MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME } from '../constants';
import type { RawEvaluationArtifact } from '../sdk/EvaluationArtifactService';
import { parseEvaluationTableArtifact } from '../sdk/EvaluationArtifactService';
import type { ModelGatewayQueryPayload, ModelGatewayRouteType, ModelGatewayRoute } from '../sdk/ModelGatewayService';
import { ModelGatewayService } from '../sdk/ModelGatewayService';
import type { EvaluationArtifactTable } from '../types';
import { searchMlflowDeploymentsRoutesApi } from './ModelGatewayActions';
import {
  PROMPTLAB_METADATA_COLUMN_LATENCY,
  PROMPTLAB_METADATA_COLUMN_TOTAL_TOKENS,
} from '../components/prompt-engineering/PromptEngineering.utils';

const EVALUATE_PROMPT_TABLE_VALUE = 'EVALUATE_PROMPT_TABLE_VALUE';
export interface EvaluatePromptTableValueAction
  extends AsyncAction<
    { metadata: any; text: string },
    {
      inputValues: Record<string, string>;
      run: RunRowType;
      compiledPrompt: string;
      rowKey: string;
      startTime: number;
      gatewayRoute: ModelGatewayRoute;
    }
  > {
  type: 'EVALUATE_PROMPT_TABLE_VALUE';
}
const evaluatePromptTableValueUnified =
  ({
    routeName,
    routeType,
    compiledPrompt,
    inputValues,
    parameters,
    outputColumn,
    rowKey,
    run,
  }: {
    routeName: string;
    routeType: ModelGatewayRouteType;
    compiledPrompt: string;
    inputValues: Record<string, string>;
    parameters: ModelGatewayQueryPayload['parameters'];
    outputColumn: string;
    rowKey: string;
    run: RunRowType;
  }) =>
  async (dispatch: ThunkDispatch, getState: () => ReduxState) => {
    // Check if model gateway routes have been fetched. If not, fetch them first.
    const { modelGateway } = getState();
    if (!modelGateway.modelGatewayRoutesLoading.loading && Object.keys(modelGateway.modelGatewayRoutes).length === 0) {
      await dispatch(searchAllPromptLabAvailableEndpoints());
    }
    // If the gateway is not present in the store, it means that it was deleted
    // recently. Display relevant error in this scenario.
    const gatewayRoute = getState().modelGateway.modelGatewayRoutes[`${routeType}:${routeName}`];
    if (!gatewayRoute) {
      const errorMessage = `MLflow deployment endpoint ${routeName} does not exist anymore!`;
      Utils.logErrorAndNotifyUser(errorMessage);
      throw new Error(errorMessage);
    }
    const modelGatewayRequestPayload: ModelGatewayQueryPayload = {
      inputText: compiledPrompt,
      parameters,
    };

    const action = {
      type: EVALUATE_PROMPT_TABLE_VALUE,
      payload: ModelGatewayService.queryModelGatewayRoute(gatewayRoute, modelGatewayRequestPayload),
      meta: {
        inputValues,
        run,
        compiledPrompt,
        rowKey,
        startTime: performance.now(),
      },
    };
    return dispatch(action);
  };

const DISCARD_PENDING_EVALUATION_DATA = 'DISCARD_PENDING_EVALUATION_DATA';
export type DiscardPendingEvaluationDataAction = Action<'DISCARD_PENDING_EVALUATION_DATA'>;
export const discardPendingEvaluationData = () => ({
  type: DISCARD_PENDING_EVALUATION_DATA,
});

export const WRITE_BACK_EVALUATION_ARTIFACTS = 'WRITE_BACK_EVALUATION_ARTIFACTS';

export interface WriteBackEvaluationArtifactsAction
  extends AsyncAction<
    { runUuid: string; newEvaluationTable: EvaluationArtifactTable }[],
    { runUuidsToUpdate: string[]; artifactPath: string }
  > {
  type: 'WRITE_BACK_EVALUATION_ARTIFACTS';
}

export const writeBackEvaluationArtifactsAction = () => async (dispatch: ThunkDispatch, getState: () => ReduxState) => {
  const { evaluationPendingDataByRunUuid, evaluationArtifactsByRunUuid } = getState().evaluationData;
  const runUuidsToUpdate = Object.keys(evaluationPendingDataByRunUuid);
  const originalRunArtifacts = fromPairs(
    Object.entries(evaluationArtifactsByRunUuid)
      .filter(
        ([runUuid, artifactTableRecords]) =>
          runUuidsToUpdate.includes(runUuid) && artifactTableRecords[MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME],
      )
      .map(([runUuid, artifactTableRecords]) => [
        runUuid,
        artifactTableRecords[MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME],
      ]),
  );

  const updatedArtifactFiles = runUuidsToUpdate.map((runUuid) => {
    const originalTableRecord = originalRunArtifacts[runUuid];

    if (!originalTableRecord) {
      throw new Error(`Cannot find existing prompt engineering artifact for run ${runUuid}`);
    }

    const transformedEntries = evaluationPendingDataByRunUuid[runUuid].map(
      ({ entryData, evaluationTime, totalTokens }) => {
        return originalTableRecord.columns.map((columnName) => {
          if (columnName === PROMPTLAB_METADATA_COLUMN_LATENCY) {
            return evaluationTime.toString();
          } else if (columnName === PROMPTLAB_METADATA_COLUMN_TOTAL_TOKENS && totalTokens) {
            return totalTokens.toString();
          } else {
            return entryData[columnName] || '';
          }
        });
      },
    );

    const updatedArtifactFile = cloneDeep(originalRunArtifacts[runUuid].rawArtifactFile) as RawEvaluationArtifact;
    updatedArtifactFile?.data.unshift(...transformedEntries);

    return { runUuid, updatedArtifactFile };
  });

  const promises = updatedArtifactFiles.map(({ runUuid, updatedArtifactFile }) =>
    dispatch(uploadArtifactApi(runUuid, MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME, updatedArtifactFile)).then(() => {
      const newEvaluationTable = parseEvaluationTableArtifact(
        MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME,
        updatedArtifactFile,
      );
      return { runUuid, newEvaluationTable };
    }),
  );

  return dispatch({
    type: 'WRITE_BACK_EVALUATION_ARTIFACTS',
    payload: Promise.all(promises),
    meta: { runUuidsToUpdate, artifactPath: MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME },
  });
};
const EVALUATE_ADD_INPUT_VALUES = 'EVALUATE_ADD_INPUT_VALUES';
export interface EvaluateAddInputValues extends Action<'EVALUATE_ADD_INPUT_VALUES'> {
  payload: Record<string, string>;
}
export const evaluateAddInputValues = (inputValues: Record<string, string>) => ({
  type: EVALUATE_ADD_INPUT_VALUES,
  payload: inputValues,
  meta: {},
});

export const evaluatePromptTableValue = ({
  routeName,
  routeType,
  compiledPrompt,
  inputValues,
  parameters,
  outputColumn,
  rowKey,
  run,
}: {
  routeName: string;
  routeType: ModelGatewayRouteType;
  compiledPrompt: string;
  inputValues: Record<string, string>;
  parameters: ModelGatewayQueryPayload['parameters'];
  outputColumn: string;
  rowKey: string;
  run: RunRowType;
}) => {
  const evaluateParams = {
    routeName,
    compiledPrompt,
    inputValues,
    parameters,
    outputColumn,
    rowKey,
    run,
  };

  return evaluatePromptTableValueUnified({
    ...evaluateParams,
    routeType,
  });
};

export const searchAllPromptLabAvailableEndpoints = () => async (dispatch: ThunkDispatch) => {
  return dispatch(searchMlflowDeploymentsRoutesApi());
};

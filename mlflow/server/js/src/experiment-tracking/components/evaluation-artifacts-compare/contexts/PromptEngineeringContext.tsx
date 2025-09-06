import React, { useCallback, useContext, useMemo, useState } from 'react';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import type { UseEvaluationArtifactTableDataResult } from '../hooks/useEvaluationArtifactTableData';
import { useDispatch, useSelector } from 'react-redux';
import type { ReduxState, ThunkDispatch } from '../../../../redux-types';
import type { EvaluationDataReduxState } from '../../../reducers/EvaluationDataReducer';
import { evaluatePromptTableValue } from '../../../actions/PromptEngineeringActions';
import {
  DEFAULT_PROMPTLAB_OUTPUT_COLUMN,
  canEvaluateOnRun,
  compilePromptInputText,
  extractEvaluationPrerequisitesForRun,
  extractPromptInputVariables,
} from '../../prompt-engineering/PromptEngineering.utils';
import Utils from '../../../../common/utils/Utils';
import { useEvaluateAllRows } from '../hooks/useEvaluateAllRows';
import { useIntl } from 'react-intl';
import type { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import { getPromptEngineeringErrorMessage } from '../utils/PromptEngineeringErrorUtils';
import type { GatewayErrorWrapper } from '../../../utils/LLMGatewayUtils';

export interface PromptEngineeringContextType {
  getMissingParams: (run: RunRowType, rowKey: string) => string[] | null;
  getEvaluableRowCount: (run: RunRowType) => number;
  pendingDataLoading: EvaluationDataReduxState['evaluationPendingDataLoadingByRunUuid'];
  evaluateCell: (run: RunRowType, rowKey: string) => void;
  evaluateAllClick: (run: RunRowType) => void;
  runColumnsBeingEvaluated: string[];
  canEvaluateInRunColumn: (run: RunRowType) => boolean;
  toggleExpandedHeader: () => void;
  isHeaderExpanded: boolean;
}

const PromptEngineeringContext = React.createContext<PromptEngineeringContextType>({
  getMissingParams: () => [],
  pendingDataLoading: {},
  getEvaluableRowCount: () => 0,
  evaluateCell: () => {},
  evaluateAllClick: () => {},
  runColumnsBeingEvaluated: [],
  canEvaluateInRunColumn: () => false,
  toggleExpandedHeader: () => {},
  isHeaderExpanded: false,
});

export const PromptEngineeringContextProvider = ({
  tableData,
  outputColumn,
  children,
}: React.PropsWithChildren<{
  tableData: UseEvaluationArtifactTableDataResult;
  outputColumn: string;
}>) => {
  const intl = useIntl();

  const [isHeaderExpanded, setIsHeaderExpanded] = useState(false);
  const toggleExpandedHeader = useCallback(() => setIsHeaderExpanded((expanded) => !expanded), []);

  const getMissingParams = useCallback(
    (run: RunRowType, rowKey: string) => {
      if (!canEvaluateOnRun(run)) {
        return null;
      }
      const row = tableData.find((x) => x.key === rowKey);
      if (!row) {
        return null;
      }

      const { promptTemplate } = extractEvaluationPrerequisitesForRun(run);

      if (!promptTemplate) {
        return null;
      }

      const requiredInputs = extractPromptInputVariables(promptTemplate);

      const missingInputParams = requiredInputs.filter((requiredInput) => !row.groupByCellValues[requiredInput]);

      return missingInputParams;
    },
    [tableData],
  );

  const dispatch = useDispatch<ThunkDispatch>();
  const { startEvaluatingRunColumn, stopEvaluatingRunColumn, runColumnsBeingEvaluated } = useEvaluateAllRows(
    tableData,
    outputColumn,
  );

  const pendingDataLoading = useSelector(
    ({ evaluationData }: ReduxState) => evaluationData.evaluationPendingDataLoadingByRunUuid,
  );

  const canEvaluateInRunColumn = useCallback(
    (run?: RunRowType) => outputColumn === DEFAULT_PROMPTLAB_OUTPUT_COLUMN && canEvaluateOnRun(run),
    [outputColumn],
  );

  const getEvaluableRowCount = useCallback(
    (run: RunRowType) => {
      const evaluatableRows = tableData.filter((tableRow) => {
        if (tableRow.cellValues[run.runUuid]) {
          return false;
        }
        const missingParams = getMissingParams(run, tableRow.key);
        return missingParams?.length === 0;
      });

      return evaluatableRows.length;
    },
    [tableData, getMissingParams],
  );

  const evaluateAllClick = useCallback(
    (run: RunRowType) => {
      if (runColumnsBeingEvaluated.includes(run.runUuid)) {
        stopEvaluatingRunColumn(run);
      } else {
        startEvaluatingRunColumn(run);
      }
    },
    [runColumnsBeingEvaluated, startEvaluatingRunColumn, stopEvaluatingRunColumn],
  );

  const evaluateCell = useCallback(
    (run: RunRowType, rowKey: string) => {
      const row = tableData.find(({ key }) => key === rowKey);
      if (!row) {
        return;
      }
      const inputValues = row.groupByCellValues;

      const { parameters, promptTemplate, routeName, routeType } = extractEvaluationPrerequisitesForRun(run);

      if (!promptTemplate) {
        return;
      }

      const compiledPrompt = compilePromptInputText(promptTemplate, inputValues);

      if (routeName) {
        const getAction = () => {
          return evaluatePromptTableValue({
            routeName,
            routeType,
            compiledPrompt,
            inputValues,
            outputColumn,
            rowKey,
            parameters,
            run,
          });
        };

        dispatch(getAction()).catch((e: Error | ErrorWrapper | GatewayErrorWrapper) => {
          const errorMessage = getPromptEngineeringErrorMessage(e);

          const wrappedMessage = intl.formatMessage(
            {
              defaultMessage: 'MLflow deployment returned the following error: "{errorMessage}"',
              description: 'Experiment page > MLflow deployment error message',
            },
            {
              errorMessage,
            },
          );
          Utils.logErrorAndNotifyUser(wrappedMessage);
        });
      }
    },
    [tableData, dispatch, outputColumn, intl],
  );
  const contextValue = useMemo(
    () => ({
      getMissingParams,
      getEvaluableRowCount,
      evaluateCell,
      evaluateAllClick,
      pendingDataLoading,
      canEvaluateInRunColumn,
      runColumnsBeingEvaluated,
      isHeaderExpanded,
      toggleExpandedHeader,
    }),
    [
      getMissingParams,
      getEvaluableRowCount,
      evaluateAllClick,
      evaluateCell,
      pendingDataLoading,
      canEvaluateInRunColumn,
      runColumnsBeingEvaluated,
      isHeaderExpanded,
      toggleExpandedHeader,
    ],
  );
  return <PromptEngineeringContext.Provider value={contextValue}>{children}</PromptEngineeringContext.Provider>;
};

export const usePromptEngineeringContext = () => useContext(PromptEngineeringContext);

import { useCallback, useEffect, useRef, useState } from 'react';
import { useIntl } from 'react-intl';
import type { UseEvaluationArtifactTableDataResult } from './useEvaluationArtifactTableData';
import { useDispatch } from 'react-redux';
import type { ThunkDispatch } from '../../../../redux-types';
import {
  compilePromptInputText,
  extractEvaluationPrerequisitesForRun,
  extractPromptInputVariables,
} from '../../prompt-engineering/PromptEngineering.utils';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { evaluatePromptTableValue } from '../../../actions/PromptEngineeringActions';
import Utils from '../../../../common/utils/Utils';
import { getPromptEngineeringErrorMessage } from '../utils/PromptEngineeringErrorUtils';

/**
 * Local utilility function, confirms if all param values
 * are provided for a particular evaluation table data row.
 */
const containsAllParamValuesForRow = (row: UseEvaluationArtifactTableDataResult[0], requiredInputs: string[]) => {
  const missingInputParams = requiredInputs.filter((requiredInput) => !row.groupByCellValues[requiredInput]);

  return missingInputParams.length === 0;
};

/**
 * A hook containing complete toolset supporting "Evaluate all" button
 */
export const useEvaluateAllRows = (evaluationTableData: UseEvaluationArtifactTableDataResult, outputColumn: string) => {
  const currentTableData = useRef<UseEvaluationArtifactTableDataResult>(evaluationTableData);
  const currentRunsBeingEvaluated = useRef<string[]>([]);
  const intl = useIntl();

  useEffect(() => {
    currentTableData.current = evaluationTableData;
  }, [evaluationTableData]);

  const [runColumnsBeingEvaluated, setEvaluatedRuns] = useState<string[]>([]);

  useEffect(() => {
    currentRunsBeingEvaluated.current = runColumnsBeingEvaluated;
  }, [runColumnsBeingEvaluated]);

  const dispatch = useDispatch<ThunkDispatch>();

  // Processes single run's evaluation queue.
  const processQueueForRun = useCallback(
    (run: RunRowType) => {
      const tableData = currentTableData.current;
      const { parameters, promptTemplate, routeName, routeType } = extractEvaluationPrerequisitesForRun(run);

      if (!promptTemplate) {
        return;
      }

      const requiredInputs = extractPromptInputVariables(promptTemplate);

      // Try to find the next row in the table that can be evaluated for a particular table
      const nextEvaluableRow = tableData.find(
        (tableRow) => !tableRow.cellValues[run.runUuid] && containsAllParamValuesForRow(tableRow, requiredInputs),
      );

      // If there's no row, close the queue and return
      if (!nextEvaluableRow) {
        setEvaluatedRuns((runs) => runs.filter((existingRunUuid) => existingRunUuid !== run.runUuid));
        return;
      }
      const rowKey = nextEvaluableRow.key;
      const inputValues = nextEvaluableRow.groupByCellValues;

      if (!promptTemplate) {
        return;
      }

      const compiledPrompt = compilePromptInputText(promptTemplate, inputValues);

      if (routeName) {
        dispatch(
          evaluatePromptTableValue({
            routeName,
            routeType,
            compiledPrompt,
            inputValues,
            outputColumn,
            rowKey,
            parameters,
            run,
          }),
        )
          .then(() => {
            // If the current queue for the run is still active, continue with processing
            if (currentRunsBeingEvaluated.current.includes(run.runUuid)) {
              processQueueForRun(run);
            }
          })
          .catch((e) => {
            const errorMessage = getPromptEngineeringErrorMessage(e);

            // In case of error, notify the user and close the queue
            const wrappedMessage = intl.formatMessage(
              {
                defaultMessage: 'Gateway returned the following error: "{errorMessage}"',
                description: 'Experiment page > gateway error message',
              },
              {
                errorMessage,
              },
            );
            Utils.logErrorAndNotifyUser(wrappedMessage);
            setEvaluatedRuns((runs) => runs.filter((existingRunUuid) => existingRunUuid !== run.runUuid));
          });
      }
    },
    [dispatch, outputColumn, intl],
  );

  // Enables run's evaluation queue and starts its processing
  const startEvaluatingRunColumn = useCallback(
    (run: RunRowType) => {
      setEvaluatedRuns((runs) => [...runs, run.runUuid]);
      processQueueForRun(run);
    },
    [processQueueForRun],
  );

  // Removes the run from evaluation queue so it will gracefully stop after currently pending evaluation
  const stopEvaluatingRunColumn = useCallback((run: RunRowType) => {
    setEvaluatedRuns((runs) => runs.filter((existingRunUuid) => existingRunUuid !== run.runUuid));
  }, []);

  return { runColumnsBeingEvaluated, startEvaluatingRunColumn, stopEvaluatingRunColumn };
};

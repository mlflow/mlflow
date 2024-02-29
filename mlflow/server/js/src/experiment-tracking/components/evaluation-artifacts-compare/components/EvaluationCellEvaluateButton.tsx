import { Button, InfoIcon, PlayIcon, RefreshIcon, Tooltip } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { usePromptEngineeringContext } from '../contexts/PromptEngineeringContext';
import { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { canEvaluateOnRun } from '../../prompt-engineering/PromptEngineering.utils';

/**
 * Displays multiple variants of "(re)evaluate" button within the artifact comparison table
 */
export const EvaluationCellEvaluateButton = ({
  disabled,
  isLoading,
  run,
  rowKey,
}: {
  disabled?: boolean;
  isLoading: boolean;
  rowKey: string;
  run: RunRowType;
}) => {
  const isRunEvaluable = canEvaluateOnRun(run);
  const { evaluateCell, getMissingParams } = usePromptEngineeringContext();

  const missingParamsToEvaluate = (run && getMissingParams(run, rowKey)) || null;

  if (missingParamsToEvaluate && missingParamsToEvaluate.length > 0) {
    return (
      <Tooltip
        title={
          <FormattedMessage
            description="Experiment page > artifact compare view > text cell > missing evaluation parameter values tooltip"
            defaultMessage='Evaluation is not possible because values for the following inputs cannot be determined: {missingParamList}. Add input columns to the "group by" settings or use "Add row" button to define new parameter set.'
            values={{
              missingParamList: <code>{missingParamsToEvaluate.join(', ')}</code>,
            }}
          />
        }
      >
        <InfoIcon />
      </Tooltip>
    );
  }

  if (!isRunEvaluable) {
    return (
      <Tooltip
        title={
          <FormattedMessage
            description="Experiment page > artifact compare view > text cell > run not evaluable tooltip"
            defaultMessage="You cannot evaluate this cell, this run was not created using served LLM model route"
          />
        }
      >
        <InfoIcon />
      </Tooltip>
    );
  }
  return (
    <Button
      componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_components_evaluationcellevaluatebutton.tsx_59"
      loading={isLoading}
      disabled={disabled}
      size="small"
      onMouseDownCapture={(e) => e.stopPropagation()}
      onClickCapture={(e) => {
        e.stopPropagation();
        evaluateCell(run, rowKey);
      }}
      icon={<PlayIcon />}
    >
      <>Evaluate</>
    </Button>
  );
};

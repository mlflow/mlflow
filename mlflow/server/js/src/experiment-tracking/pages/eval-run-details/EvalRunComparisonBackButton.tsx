import { Button } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useEvalRunsCompareMode } from '../../components/evaluations/hooks/useEvalRunsCompareMode';

export const EvalRunComparisonBackButton = () => {
  const { setCompareMode } = useEvalRunsCompareMode();

  return (
    <Button
      componentId="mlflow.eval_run_details.exit_compare"
      type="tertiary"
      size="small"
      onClick={() => setCompareMode(false)}
    >
      <FormattedMessage
        defaultMessage="Back to Reference Run"
        description="Exit comparison and show the reference evaluation run"
      />
    </Button>
  );
};

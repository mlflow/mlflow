import { CheckboxIcon, ToggleButton } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export interface EvalRunComparisonReviewButtonProps {
  onReview: () => void;
}

/** Starts pairwise review of the compared runs' traces (also R on the compare page). */
export const EvalRunComparisonReviewButton = ({ onReview }: EvalRunComparisonReviewButtonProps) => (
  <ToggleButton
    componentId="mlflow.eval_run_details.compare.review"
    size="small"
    pressed={false}
    onPressedChange={(pressed) => pressed && onReview()}
    icon={<CheckboxIcon css={{ svg: { width: 15, height: 15 } }} />}
    aria-keyshortcuts="R"
    css={{
      '&&, &&:hover, &&:active': {
        borderColor: 'transparent !important',
      },
    }}
  >
    <FormattedMessage
      defaultMessage="Review"
      description="Action that starts pairwise review for evaluation run comparison traces"
    />
  </ToggleButton>
);

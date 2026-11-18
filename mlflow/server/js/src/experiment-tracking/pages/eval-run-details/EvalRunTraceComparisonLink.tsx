import { ArrowRightIcon, Button, Tooltip } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

/** Link shown in each comparison overview table's title, opening the row-level Details tab. */
export const EvalRunTraceComparisonLink = ({ onNavigateToTraceView }: { onNavigateToTraceView: () => void }) => (
  <Tooltip
    componentId="mlflow.eval_run_details.compare.view_trace_comparison.tooltip"
    content={
      <FormattedMessage
        defaultMessage="Find which inputs the selected runs answered differently."
        description="Tooltip explaining what the trace-level comparison link shows"
      />
    }
  >
    <Button
      componentId="mlflow.eval_run_details.compare.view_trace_comparison"
      type="link"
      size="small"
      endIcon={<ArrowRightIcon />}
      onClick={() => onNavigateToTraceView()}
    >
      <FormattedMessage
        defaultMessage="Compare traces row by row"
        description="Link on an eval comparison table that opens the row-level Details tab"
      />
    </Button>
  </Tooltip>
);

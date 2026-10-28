import { Alert, Button, Empty, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

const centeredEmptyStateStyles = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  height: '100%',
  minHeight: 360,
  width: '100%',
  '& > div': {
    height: '100%',
    display: 'flex',
    flexDirection: 'column' as const,
    justifyContent: 'center',
    alignItems: 'center',
  },
};

export const EvalRunTracesEmptyState = () => (
  <div css={centeredEmptyStateStyles}>
    <Empty
      title={
        <FormattedMessage
          defaultMessage="No traces in this evaluation run"
          description="Title for an evaluation run whose trace results are empty"
        />
      }
      description={
        <FormattedMessage
          defaultMessage="This evaluation run did not produce any trace results."
          description="Description for an evaluation run whose trace results are empty"
        />
      }
    />
  </div>
);

export interface EvalRunSelectedTraceErrorAlertProps {
  onRetry: () => void;
}

export const EvalRunSelectedTraceErrorAlert = ({ onRetry }: EvalRunSelectedTraceErrorAlertProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Alert
      componentId="mlflow.eval-run.traces.selected-trace-error"
      type="error"
      closable={false}
      message={
        <FormattedMessage
          defaultMessage="Couldn't open the selected trace"
          description="Title of the error shown when an evaluation run cannot validate a selected trace"
        />
      }
      description={
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="The trace could not be verified as part of this evaluation run."
            description="Description of the error shown when an evaluation run cannot validate a selected trace"
          />
          <Button
            componentId="mlflow.eval-run.traces.selected-trace-error.retry"
            size="small"
            onClick={onRetry}
            css={{ alignSelf: 'flex-start' }}
          >
            <FormattedMessage
              defaultMessage="Retry"
              description="Button to retry validating a selected trace in an evaluation run"
            />
          </Button>
        </div>
      }
    />
  );
};

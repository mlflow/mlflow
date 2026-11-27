import { Button, Empty, SearchIcon, Spinner } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { TracesViewTableNoTracesQuickstart } from '@mlflow/mlflow/src/experiment-tracking/components/traces/quickstart/TracesViewTableNoTracesQuickstart';
import type { TracesV4EmptyStateKind } from '../hooks/useTracesV4Controller';
import { getNamedDateFilters } from '../utils/dateUtils';
import type { TracesV4TimeLabel } from '../utils/timeRange';

interface TracesV4EmptyStateProps {
  kind: TracesV4EmptyStateKind;
  timeLabel: TracesV4TimeLabel;
  onViewAll: () => void;
}

export const TracesV4EmptyState = ({ kind, timeLabel, onViewAll }: TracesV4EmptyStateProps) => {
  const intl = useIntl();

  if (kind === 'no-traces') {
    return <TracesViewTableNoTracesQuickstart baseComponentId="mlflow.traces" />;
  }

  const centeredStateStyles = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
    minHeight: 360,
    width: '100%',
    '& > div': {
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
    },
  } as const;

  if (kind === 'checking') {
    return (
      <div
        css={centeredStateStyles}
        role="status"
        aria-label={intl.formatMessage({
          defaultMessage: 'Checking for traces outside this time range',
          description: 'Status while checking whether traces exist outside the selected time range',
        })}
      >
        <Spinner size="small" />
      </div>
    );
  }

  const filterLabel = getNamedDateFilters(intl).find((filter) => filter.key === timeLabel)?.label ?? timeLabel;

  return (
    <div css={centeredStateStyles}>
      <Empty
        image={<SearchIcon />}
        title={<FormattedMessage defaultMessage="No traces found" description="No traces found message" />}
        description={
          kind === 'time-filtered' ? (
            <FormattedMessage
              defaultMessage='Some traces are hidden by your time range filter: "{filterLabel}"'
              description="Message shown when traces are hidden by time filter"
              values={{ filterLabel: <strong>{filterLabel}</strong> }}
            />
          ) : (
            <FormattedMessage
              defaultMessage="We couldn't check for traces outside this time range. Try viewing all traces."
              description="Message shown when checking for traces outside the time range fails"
            />
          )
        }
        button={
          <Button componentId="mlflow.traces-v4.empty.view-all" onClick={onViewAll}>
            <FormattedMessage defaultMessage="View All" description="View all traces button" />
          </Button>
        }
      />
    </div>
  );
};

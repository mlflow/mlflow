import { SimpleSelect, SimpleSelectOption, Spinner, Tooltip, Typography } from '@databricks/design-system';
import type { CursorPaginationProps } from '@databricks/design-system';
import type { Theme } from '@emotion/react';
import { FormattedMessage, useIntl } from 'react-intl';

interface ExperimentViewRunsTableStatusBarProps {
  isLoading: boolean;
  allRunsCount: number;
  pageSizeSelect: NonNullable<CursorPaginationProps['pageSizeSelect']>;
}

// Strongifies the i18n string, used in <FormattedMessage> below
const strong = (text: string) => <strong>{text}</strong>;

export const ExperimentViewRunsTableStatusBar = ({
  isLoading,
  allRunsCount,
  pageSizeSelect,
}: ExperimentViewRunsTableStatusBarProps) => {
  const intl = useIntl();
  const pageSizeLabel = intl.formatMessage({
    defaultMessage: 'Runs per request (including children)',
    description: 'Label for the experiment Runs page size selector',
  });
  return (
    <div css={styles.statusBar}>
      <Typography.Text size="sm" color={isLoading ? 'secondary' : undefined}>
        <FormattedMessage
          defaultMessage="<strong>{length}</strong> matching {length, plural, =0 {runs} =1 {run} other {runs}}"
          description="Message for displaying how many runs match search criteria on experiment page"
          values={{
            strong,
            length: allRunsCount,
          }}
        />
      </Typography.Text>
      {isLoading && <Spinner size="small" />}
      <div css={(theme) => ({ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: theme.spacing.sm })}>
        <Tooltip
          componentId="mlflow.experiment_runs.page_size_help"
          content={
            <FormattedMessage
              defaultMessage="This limit includes child runs even when collapsed, so parent rows may be fewer."
              description="Explanation that the Runs page size limits all matching runs rather than parent rows"
            />
          }
        >
          <Typography.Text size="sm">{pageSizeLabel}</Typography.Text>
        </Tooltip>
        <SimpleSelect
          componentId="mlflow.experiment_runs.page_size"
          label={pageSizeLabel}
          value={String(pageSizeSelect.default)}
          onChange={(event) => pageSizeSelect.onChange(Number(event.target.value))}
          disabled={isLoading}
          triggerSize="small"
        >
          {pageSizeSelect.options.map((size) => (
            <SimpleSelectOption key={size} value={String(size)}>
              {size}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </div>
    </div>
  );
};

const styles = {
  statusBar: (theme: Theme) => ({
    minHeight: theme.spacing.xl + theme.spacing.xs,
    display: 'flex',
    gap: theme.spacing.sm,
    marginTop: -1,
    position: 'relative' as const,
    alignItems: 'center',
    borderTop: `1px solid ${theme.colors.border}`,
  }),
};

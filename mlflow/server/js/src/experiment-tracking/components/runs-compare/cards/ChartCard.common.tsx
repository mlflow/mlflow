import { Button, DropdownMenu, OverflowIcon, Typography } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { PropsWithChildren } from 'react';
import { InfoTooltip } from '@databricks/design-system';

export interface ChartCardWrapperProps {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  onEdit: () => void;
  onDelete: () => void;
  fullWidth?: boolean;
  tooltip?: string;
}

/**
 * Wrapper components for all chart cards. Provides styles and adds
 * a dropdown menu with actions for configure and delete.
 */
export const RunsCompareChartCardWrapper = ({
  title,
  subtitle,
  onDelete,
  onEdit,
  children,
  fullWidth = false,
  tooltip = '',
}: PropsWithChildren<ChartCardWrapperProps>) => (
  <div css={styles.chartEntry(fullWidth)} data-testid='experiment-view-compare-runs-card'>
    <div css={styles.chartEntryTitle}>
      <div css={{ overflow: 'hidden' }}>
        <Typography.Title
          title={String(title)}
          level={4}
          css={{
            marginBottom: 0,
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            textOverflow: 'ellipsis',
          }}
        >
          {title}
        </Typography.Title>
        {subtitle && <span css={styles.subtitle}>{subtitle}</span>}
        {tooltip && <InfoTooltip title={tooltip} />}
      </div>
      <DropdownMenu.Root modal={false}>
        <DropdownMenu.Trigger asChild>
          <Button
            type='tertiary'
            icon={<OverflowIcon />}
            data-testid='experiment-view-compare-runs-card-menu'
          />
        </DropdownMenu.Trigger>
        <DropdownMenu.Content align='end' minWidth={100}>
          <DropdownMenu.Item onClick={onEdit} data-testid='experiment-view-compare-runs-card-edit'>
            Configure
          </DropdownMenu.Item>
          <DropdownMenu.Item
            onClick={onDelete}
            data-testid='experiment-view-compare-runs-card-delete'
          >
            Delete
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    </div>
    {children}
  </div>
);

const styles = {
  chartEntry:
    (fullWidth = false) =>
    (theme: Theme) => ({
      height: 360,
      maxWidth: fullWidth ? 'unset' : 800,
      overflow: 'hidden',
      display: 'grid',
      gridTemplateRows: 'auto 1fr',
      backgroundColor: theme.colors.backgroundPrimary,
      padding: theme.spacing.md,
      border: `1px solid ${theme.colors.border}`,
      borderRadius: theme.general.borderRadiusBase,
    }),
  chartComponentWrapper: () => ({
    overflow: 'hidden',
  }),
  chartEntryTitle: () => ({
    display: 'grid' as const,
    gridTemplateColumns: '1fr auto',
    alignItems: 'flex-start',
  }),
  subtitle: (theme: Theme) => ({
    color: theme.colors.textSecondary,
    fontSize: 11,
    marginRight: 4,
  }),
};

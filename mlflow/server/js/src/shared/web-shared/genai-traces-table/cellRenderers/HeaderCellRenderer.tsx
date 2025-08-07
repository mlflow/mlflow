import type { HeaderContext } from '@tanstack/react-table';

import { HoverCard, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { TracesTableColumnGroup, TracesTableColumnGroupToLabelMap, type EvalTraceComparisonEntry } from '../types';

type HeaderCellRendererMeta = {
  groupId: TracesTableColumnGroup;
  visibleCount: number;
  totalCount: number;
  enableGrouping?: boolean;
};

export const HeaderCellRenderer = (props: HeaderContext<EvalTraceComparisonEntry, unknown>) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { groupId, visibleCount, totalCount, enableGrouping } = props.column.columnDef.meta as HeaderCellRendererMeta;

  if (!enableGrouping) {
    return TracesTableColumnGroupToLabelMap[groupId as TracesTableColumnGroup];
  }

  const groupName = TracesTableColumnGroupToLabelMap[groupId as TracesTableColumnGroup];
  return (
    <div
      css={{
        height: '100%',
        width: '100%',
        display: 'flex',
        overflow: 'hidden',
        gap: theme.spacing.sm,
      }}
    >
      <div>{groupName}</div>
      {groupId === TracesTableColumnGroup.INFO ? null : visibleCount === totalCount ? (
        <div
          css={{
            color: theme.colors.textSecondary,
            fontWeight: 'normal',
          }}
        >
          ({visibleCount}/{totalCount})
        </div>
      ) : (
        <HoverCard
          trigger={
            <div
              css={{
                color: theme.colors.textSecondary,
                ':hover': {
                  textDecoration: 'underline',
                },
                fontWeight: 'normal',
              }}
            >
              ({visibleCount}/{totalCount})
            </div>
          }
          content={intl.formatMessage(
            {
              defaultMessage: 'Showing {visibleCount} out of {totalCount} {groupName}. Select columns to view more.',
              description: 'Tooltip for the group column header',
            },
            {
              visibleCount,
              totalCount,
              groupName,
            },
          )}
          align="start"
        />
      )}
    </div>
  );
};

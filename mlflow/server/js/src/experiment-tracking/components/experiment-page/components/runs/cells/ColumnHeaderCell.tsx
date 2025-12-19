import { SortAscendingIcon, SortDescendingIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useUpdateExperimentPageSearchFacets } from '../../../hooks/useExperimentPageSearchFacets';

export interface ColumnHeaderCellProps {
  enableSorting: boolean;
  displayName: string;
  canonicalSortKey: string;
  context: {
    orderByKey: string;
    orderByAsc: boolean;
  };
}

export const ColumnHeaderCell = ({
  enableSorting,
  canonicalSortKey,
  displayName,
  context: tableContext,
}: ColumnHeaderCellProps) => {
  const { orderByKey, orderByAsc } = tableContext || {};
  const updateSearchFacets = useUpdateExperimentPageSearchFacets();
  const selectedCanonicalSortKey = canonicalSortKey;

  const handleSortBy = () => {
    let newOrderByAsc = !orderByAsc;

    // If the new sortKey is not equal to the previous sortKey, reset the orderByAsc
    if (selectedCanonicalSortKey !== orderByKey) {
      newOrderByAsc = false;
    }
    updateSearchFacets({ orderByKey: selectedCanonicalSortKey, orderByAsc: newOrderByAsc });
  };

  const { theme } = useDesignSystemTheme();
  const isOrderedByClassName = 'is-ordered-by';

  return (
    <div
      role="columnheader"
      css={{
        height: '100%',
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}
    >
      <div
        css={{
          height: '100%',
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          overflow: 'hidden',
          paddingLeft: theme.spacing.xs + theme.spacing.sm,
          paddingRight: theme.spacing.xs + theme.spacing.sm,
          gap: theme.spacing.sm,
          svg: {
            color: theme.colors.textSecondary,
          },
          '&:hover': {
            color: enableSorting ? theme.colors.actionTertiaryTextHover : 'unset',
            svg: {
              color: theme.colors.actionTertiaryTextHover,
            },
          },
        }}
        className={selectedCanonicalSortKey === orderByKey ? isOrderedByClassName : ''}
        onClick={enableSorting ? handleSortBy : undefined}
      >
        <span data-testid={`sort-header-${displayName}`}>{displayName}</span>
        {enableSorting && selectedCanonicalSortKey === orderByKey ? (
          orderByAsc ? (
            <SortAscendingIcon />
          ) : (
            <SortDescendingIcon />
          )
        ) : null}
      </div>
    </div>
  );
};

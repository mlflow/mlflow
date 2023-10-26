import {
  SortAscendingIcon,
  SortDescendingIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useFetchExperimentRuns } from '../../../hooks/useFetchExperimentRuns';

export interface ColumnHeaderCellProps {
  enableSorting: boolean;
  displayName: string;
  canonicalSortKey: string;
}

export const ColumnHeaderCell = (props: ColumnHeaderCellProps) => {
  const { updateSearchFacets, searchFacetsState } = useFetchExperimentRuns();
  const { orderByAsc, orderByKey } = searchFacetsState;

  const { enableSorting, canonicalSortKey, displayName } = props;

  const handleSortBy = () => {
    let newOrderByAsc = !orderByAsc;

    // If the new sortKey is not equal to the previous sortKey, reset the orderByAsc
    if (canonicalSortKey !== orderByKey) {
      newOrderByAsc = false;
    }
    updateSearchFacets({ orderByKey: canonicalSortKey, orderByAsc: newOrderByAsc });
  };

  const { theme } = useDesignSystemTheme();

  return (
    <div
      role='columnheader'
      css={{
        height: '100%',
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 12px',
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
      className={canonicalSortKey === orderByKey ? 'is-ordered-by' : ''}
      onClick={enableSorting ? () => handleSortBy() : undefined}
    >
      <span data-test-id={`sort-header-${displayName}`}>{displayName}</span>
      {enableSorting && canonicalSortKey === orderByKey ? (
        orderByAsc ? (
          <SortAscendingIcon />
        ) : (
          <SortDescendingIcon />
        )
      ) : null}
    </div>
  );
};

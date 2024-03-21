import { SortAscendingIcon, SortDescendingIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useFetchExperimentRuns } from '../../../hooks/useFetchExperimentRuns';
import { shouldEnableShareExperimentViewByTags } from '../../../../../../common/utils/FeatureUtils';
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

/**
 * A local hook that selects the correct updateSearchFacets function based on the feature flag.
 *
 * TODO: Remove this once we migrate to the new view state model
 */
const useUpdateOrderByValues = () => {
  // We can disable this eslint rule because condition uses a stable feature flag evaluation
  /* eslint-disable react-hooks/rules-of-hooks */
  if (shouldEnableShareExperimentViewByTags()) {
    return useUpdateExperimentPageSearchFacets();
  }
  const { updateSearchFacets } = useFetchExperimentRuns();
  return updateSearchFacets;
};

/**
 * A local hook that selects the correct order by/order direction values based on the feature flag.
 *
 * TODO: Remove this once we migrate to the new view state model
 */
const useOrderByValues = (tableContext: ColumnHeaderCellProps['context']) => {
  // We can disable this eslint rule because condition uses a stable feature flag evaluation
  /* eslint-disable react-hooks/rules-of-hooks */
  if (shouldEnableShareExperimentViewByTags()) {
    const { orderByKey, orderByAsc } = tableContext || {};
    return { orderByAsc, orderByKey };
  }
  const { searchFacetsState } = useFetchExperimentRuns();
  const { orderByAsc, orderByKey } = searchFacetsState;
  return { orderByAsc, orderByKey };
};

export const ColumnHeaderCell = ({
  enableSorting,
  canonicalSortKey,
  displayName,
  context: tableContext,
}: ColumnHeaderCellProps) => {
  const { orderByKey, orderByAsc } = useOrderByValues(tableContext);
  const updateSearchFacets = useUpdateOrderByValues();

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
      role="columnheader"
      css={{
        height: '100%',
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 12px',
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

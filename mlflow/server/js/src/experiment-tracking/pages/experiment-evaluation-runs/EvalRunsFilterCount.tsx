import { Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

/**
 * The run count, right-aligned at the end of the toolbar row / footer.
 *
 * Both counts are of the runs fetched so far, not the experiment totals:
 * search_runs pages at 50 with infinite scroll and returns no total count.
 * Claiming an exact total would be wrong on any experiment past one page, so the
 * copy is hedged while more pages remain.
 */
export const EvalRunsFilterCount = ({
  isAnyFilterActive,
  visibleCount,
  totalCount,
  hasMoreRuns,
}: {
  isAnyFilterActive: boolean;
  visibleCount: number;
  totalCount: number;
  hasMoreRuns: boolean;
}) => {
  return (
    <Typography.Hint css={{ marginLeft: 'auto', whiteSpace: 'nowrap', alignSelf: 'center' }}>
      {isAnyFilterActive ? (
        hasMoreRuns ? (
          <FormattedMessage
            defaultMessage="Showing {visibleCount} of {totalCount}+ runs loaded"
            description="Count of matching evaluation runs when more runs are still unloaded"
            values={{ visibleCount, totalCount }}
          />
        ) : (
          <FormattedMessage
            defaultMessage="Showing {visibleCount} of {totalCount} runs"
            description="Count of evaluation runs currently shown versus the unfiltered total"
            values={{ visibleCount, totalCount }}
          />
        )
      ) : hasMoreRuns ? (
        <FormattedMessage
          defaultMessage="{totalCount}+ runs loaded"
          description="Count of loaded evaluation runs when more pages remain"
          values={{ totalCount }}
        />
      ) : (
        <FormattedMessage
          defaultMessage="{totalCount, plural, one {# run} other {# runs}}"
          description="Total count of evaluation runs when no filter is applied"
          values={{ totalCount }}
        />
      )}
    </Typography.Hint>
  );
};

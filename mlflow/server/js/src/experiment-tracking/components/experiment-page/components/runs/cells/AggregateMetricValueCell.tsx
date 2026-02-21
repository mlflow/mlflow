import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { RunRowType } from '../../../utils/experimentPage.row-types';
import { EXPERIMENT_FIELD_PREFIX_METRIC } from '../../../utils/experimentPage.common-utils';

export const AggregateMetricValueCell = ({
  value,
  data,
  valueFormatted,
  colDef,
}: {
  value: string;
  valueFormatted: null | string;
  data: RunRowType;
  colDef?: { field?: string };
}) => {
  const { theme } = useDesignSystemTheme();
  if (data.groupParentInfo?.aggregateFunction) {
    const metricKey = colDef?.field?.replace(`${EXPERIMENT_FIELD_PREFIX_METRIC}-`, '');
    const stats = metricKey ? data.groupParentInfo.aggregatedMetricStatistics?.[metricKey] : undefined;
    const showStddev = stats && stats.stddev > 0;

    return (
      <Typography.Text>
        {valueFormatted ?? value}
        {showStddev && (
          <span css={{ color: theme.colors.textSecondary }}>{` \u00B1${stats.stddev.toFixed(2)}`}</span>
        )}{' '}
        <span css={{ color: theme.colors.textSecondary }}>({data.groupParentInfo.aggregateFunction})</span>
      </Typography.Text>
    );
  }
  return value;
};

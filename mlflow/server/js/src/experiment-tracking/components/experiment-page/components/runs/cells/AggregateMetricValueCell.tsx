import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { RunRowType } from '../../../utils/experimentPage.row-types';

export const AggregateMetricValueCell = ({
  value,
  data,
  valueFormatted,
}: {
  value: string;
  valueFormatted: null | string;
  data: RunRowType;
}) => {
  const { theme } = useDesignSystemTheme();
  if (data.groupParentInfo?.aggregateFunction) {
    return (
      <Typography.Text>
        {valueFormatted ?? value}{' '}
        <span css={{ color: theme.colors.textSecondary }}>({data.groupParentInfo.aggregateFunction})</span>
      </Typography.Text>
    );
  }
  return value;
};

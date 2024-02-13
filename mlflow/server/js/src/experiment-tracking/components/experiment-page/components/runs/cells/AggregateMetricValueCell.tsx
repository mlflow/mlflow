import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { RunRowType } from '../../../utils/experimentPage.row-types';

export const AggregateMetricValueCell = ({ value, data }: { value: string; data: RunRowType }) => {
  const { theme } = useDesignSystemTheme();
  if (data.groupParentInfo?.aggregateFunction) {
    return (
      <Typography.Text>
        {value} <span css={{ color: theme.colors.textSecondary }}>({data.groupParentInfo.aggregateFunction})</span>
      </Typography.Text>
    );
  }
  return value;
};

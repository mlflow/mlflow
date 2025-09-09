import type { TypographyColor } from '@databricks/design-system';
import { ArrowDownIcon, ArrowUpIcon, DashIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { DifferenceChartCellDirection } from '../../utils/differenceView';

export const CellDifference = ({ label, direction }: { label: string; direction: DifferenceChartCellDirection }) => {
  const { theme } = useDesignSystemTheme();
  let paragraphColor: TypographyColor | undefined = undefined;
  let icon = null;
  switch (direction) {
    case DifferenceChartCellDirection.NEGATIVE:
      paragraphColor = 'error';
      icon = <ArrowDownIcon color="danger" data-testid="negative-cell-direction" />;
      break;
    case DifferenceChartCellDirection.POSITIVE:
      paragraphColor = 'success';
      icon = <ArrowUpIcon color="success" data-testid="positive-cell-direction" />;
      break;
    case DifferenceChartCellDirection.SAME:
      paragraphColor = 'info';
      icon = <DashIcon css={{ color: theme.colors.textSecondary }} data-testid="same-cell-direction" />;
      break;
    default:
      break;
  }

  return (
    <div css={{ display: 'inline-flex', gap: theme.spacing.xs, alignItems: 'center' }}>
      <Typography.Paragraph color={paragraphColor} css={{ margin: 0 }}>
        {label}
      </Typography.Paragraph>
      {icon}
    </div>
  );
};

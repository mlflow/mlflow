import type { TypographyColor } from '@databricks/design-system';
import { ArrowDownIcon, ArrowUpIcon, DashIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import {
  DifferenceChartCellDirection,
  differenceView,
  getDifferenceChartDisplayedValue,
} from '../../../utils/differenceView';
import { type CellContext } from '@tanstack/react-table';
import type { DifferencePlotDataColumnDef } from '../DifferenceViewPlot';
import { type DifferencePlotDataRow } from '../DifferenceViewPlot';

const CellDifference = ({ label, direction }: { label: string; direction: DifferenceChartCellDirection }) => {
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
    <div
      css={{
        display: 'inline-flex',
        backgroundColor: theme.colors.actionDisabledBackground,
        padding: `${theme.spacing.xs / 2}px ${theme.spacing.xs}px`,
        fontSize: theme.typography.fontSizeSm,
        borderRadius: theme.borders.borderRadiusSm,
        userSelect: 'none',
        gap: theme.spacing.xs,
        alignItems: 'center',
        svg: {
          width: theme.typography.fontSizeSm,
          height: theme.typography.fontSizeSm,
        },
        overflow: 'hidden',
      }}
    >
      <Typography.Text size="sm" color={paragraphColor} css={{ margin: 0 }} ellipsis>
        {label}
      </Typography.Text>
      {icon}
    </div>
  );
};

export const DifferencePlotDataCell = ({
  getValue,
  row: { original },
  column: { columnDef },
}: CellContext<DifferencePlotDataRow, DifferencePlotDataRow> & {
  column: { columnDef: DifferencePlotDataColumnDef };
  row: { original: Record<string, any> };
}) => {
  const { theme } = useDesignSystemTheme();

  const { isBaseline, baselineColumnUuid, showChangeFromBaseline } = columnDef.meta ?? {};

  const value = getValue();

  if (isBaseline) {
    return getDifferenceChartDisplayedValue(getValue());
  }
  if (value === undefined) {
    return null;
  }
  const rowDifference = baselineColumnUuid ? differenceView(value, original[baselineColumnUuid]) : null;
  return (
    <span css={{ display: 'inline-flex', overflow: 'hidden', gap: theme.spacing.sm, alignItems: 'center' }}>
      <Typography.Text ellipsis>{getDifferenceChartDisplayedValue(value)}</Typography.Text>
      {rowDifference && showChangeFromBaseline && (
        <CellDifference label={rowDifference.label} direction={rowDifference.direction} />
      )}
    </span>
  );
};

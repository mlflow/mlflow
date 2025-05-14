import { HeaderContext } from '@tanstack/react-table';
import { RunColorPill } from '../../../../experiment-page/components/RunColorPill';
import { Button, DropdownMenu, OverflowIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { type DifferencePlotDataColumnDef } from '../DifferenceViewPlot';

export const DifferencePlotRunHeaderCell = ({
  column: { columnDef },
}: {
  column: {
    columnDef: DifferencePlotDataColumnDef;
  };
}) => {
  const { theme } = useDesignSystemTheme();
  const traceData = columnDef.meta?.traceData;
  const updateBaselineColumnUuid = columnDef.meta?.updateBaselineColumnUuid;
  if (!traceData) {
    return null;
  }
  return (
    <div
      css={{
        flex: 1,
        display: 'inline-flex',
        overflow: 'hidden',
        alignItems: 'center',
        gap: theme.spacing.xs,
        fontWeight: 'normal',
      }}
    >
      <RunColorPill color={traceData.color} /> <Typography.Text ellipsis>{traceData?.displayName}</Typography.Text>
      <div css={{ flex: 1 }} />
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button
            type="link"
            size="small"
            componentId="mlflow.charts.difference_plot.overflow_menu.trigger"
            icon={<OverflowIcon />}
          />
        </DropdownMenu.Trigger>
        <DropdownMenu.Content>
          <DropdownMenu.Item
            componentId="mlflow.charts.difference_plot.overflow_menu.set_as_baseline"
            onClick={() => updateBaselineColumnUuid?.(traceData.uuid)}
          >
            <FormattedMessage
              defaultMessage="Set as baseline"
              description="In the run data difference comparison table, the label for an option to set particular experiment run as a baseline one - meaning other runs will be compared to it."
            />
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    </div>
  );
};

import { Button, DropdownMenu, PlusIcon } from '@databricks/design-system';
import { Theme } from '@emotion/react';

import { ReactComponent as ChartBarIcon } from '../../../../common/static/chart-bar.svg';
import { ReactComponent as ChartContourIcon } from '../../../../common/static/chart-contour.svg';
import { ReactComponent as ChartLineIcon } from '../../../../common/static/chart-line.svg';
import { ReactComponent as ChartParallelIcon } from '../../../../common/static/chart-parallel.svg';
import { ReactComponent as ChartScatterIcon } from '../../../../common/static/chart-scatter.svg';
import { RunsChartType } from '../runs-charts.types';

export interface RunsChartsAddChartMenuProps {
  onAddChart: (type: RunsChartType) => void;
  supportedChartTypes?: RunsChartType[];
}

export const RunsChartsAddChartMenu = ({ onAddChart, supportedChartTypes }: RunsChartsAddChartMenuProps) => {
  const isChartTypeSupported = (type: RunsChartType) => !supportedChartTypes || supportedChartTypes.includes(type);
  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-compare_runscompareaddchartmenu.tsx_19"
          css={styles.addChartButton}
          icon={<PlusIcon />}
          data-testid="experiment-view-compare-runs-add-chart"
        >
          Add chart
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        {isChartTypeSupported(RunsChartType.BAR) && (
          <DropdownMenu.Item
            onClick={() => onAddChart(RunsChartType.BAR)}
            data-testid="experiment-view-compare-runs-chart-type-bar"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartBarIcon />
            </DropdownMenu.IconWrapper>
            Bar chart
          </DropdownMenu.Item>
        )}
        {isChartTypeSupported(RunsChartType.LINE) && (
          <DropdownMenu.Item
            onClick={() => onAddChart(RunsChartType.LINE)}
            data-testid="experiment-view-compare-runs-chart-type-line"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartLineIcon />
            </DropdownMenu.IconWrapper>
            Line chart
          </DropdownMenu.Item>
        )}
        {isChartTypeSupported(RunsChartType.PARALLEL) && (
          <DropdownMenu.Item
            onClick={() => onAddChart(RunsChartType.PARALLEL)}
            data-testid="experiment-view-compare-runs-chart-type-parallel"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartParallelIcon />
            </DropdownMenu.IconWrapper>
            Parallel coordinates
          </DropdownMenu.Item>
        )}
        {isChartTypeSupported(RunsChartType.SCATTER) && (
          <DropdownMenu.Item
            onClick={() => onAddChart(RunsChartType.SCATTER)}
            data-testid="experiment-view-compare-runs-chart-type-scatter"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartScatterIcon />
            </DropdownMenu.IconWrapper>
            Scatter chart
          </DropdownMenu.Item>
        )}
        {isChartTypeSupported(RunsChartType.CONTOUR) && (
          <DropdownMenu.Item
            onClick={() => onAddChart(RunsChartType.CONTOUR)}
            data-testid="experiment-view-compare-runs-chart-type-contour"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartContourIcon />
            </DropdownMenu.IconWrapper>
            Contour chart
          </DropdownMenu.Item>
        )}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

const styles = {
  addChartButton: (theme: Theme) => ({
    // Overriden while waiting for design decision in DuBois (FEINF-1711)
    backgroundColor: `${theme.colors.backgroundPrimary} !important`,
  }),
  iconWrapper: (theme: Theme) => ({
    width: theme.general.iconSize + theme.spacing.xs,
  }),
};

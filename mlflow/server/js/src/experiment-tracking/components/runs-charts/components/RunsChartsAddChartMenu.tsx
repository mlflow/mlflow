import { Button, DropdownMenu, PlusIcon } from '@databricks/design-system';
import type { Theme } from '@emotion/react';

import { ReactComponent as ChartBarIcon } from '../../../../common/static/chart-bar.svg';
import { ReactComponent as ChartContourIcon } from '../../../../common/static/chart-contour.svg';
import { ReactComponent as ChartLineIcon } from '../../../../common/static/chart-line.svg';
import { ReactComponent as ChartParallelIcon } from '../../../../common/static/chart-parallel.svg';
import { ReactComponent as ChartScatterIcon } from '../../../../common/static/chart-scatter.svg';
import { ReactComponent as ChartDifferenceIcon } from '../../../../common/static/chart-difference.svg';
import { ReactComponent as ChartImageIcon } from '../../../../common/static/chart-image.svg';
import { RunsChartType } from '../runs-charts.types';
import { FormattedMessage } from 'react-intl';

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
          <FormattedMessage
            defaultMessage="Add chart"
            description="Experiment tracking > runs charts > add chart menu"
          />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        {isChartTypeSupported(RunsChartType.BAR) && (
          <DropdownMenu.Item
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_42"
            onClick={() => onAddChart(RunsChartType.BAR)}
            data-testid="experiment-view-compare-runs-chart-type-bar"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartBarIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Bar chart"
              description="Experiment tracking > runs charts > add chart menu > bar chart"
            />
          </DropdownMenu.Item>
        )}
        {isChartTypeSupported(RunsChartType.LINE) && (
          <DropdownMenu.Item
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_56"
            onClick={() => onAddChart(RunsChartType.LINE)}
            data-testid="experiment-view-compare-runs-chart-type-line"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartLineIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Line chart"
              description="Experiment tracking > runs charts > add chart menu > line chart"
            />
          </DropdownMenu.Item>
        )}
        {isChartTypeSupported(RunsChartType.PARALLEL) && (
          <DropdownMenu.Item
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_70"
            onClick={() => onAddChart(RunsChartType.PARALLEL)}
            data-testid="experiment-view-compare-runs-chart-type-parallel"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartParallelIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Parallel coordinates"
              description="Experiment tracking > runs charts > add chart menu > parallel coordinates"
            />
          </DropdownMenu.Item>
        )}
        {isChartTypeSupported(RunsChartType.SCATTER) && (
          <DropdownMenu.Item
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_84"
            onClick={() => onAddChart(RunsChartType.SCATTER)}
            data-testid="experiment-view-compare-runs-chart-type-scatter"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartScatterIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Scatter chart"
              description="Experiment tracking > runs charts > add chart menu > scatter plot"
            />
          </DropdownMenu.Item>
        )}
        {isChartTypeSupported(RunsChartType.CONTOUR) && (
          <DropdownMenu.Item
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_98"
            onClick={() => onAddChart(RunsChartType.CONTOUR)}
            data-testid="experiment-view-compare-runs-chart-type-contour"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartContourIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Contour chart"
              description="Experiment tracking > runs charts > add chart menu > contour chart"
            />
          </DropdownMenu.Item>
        )}
        {isChartTypeSupported(RunsChartType.DIFFERENCE) && (
          <DropdownMenu.Item
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_112"
            onClick={() => onAddChart(RunsChartType.DIFFERENCE)}
            data-testid="experiment-view-compare-runs-chart-type-difference"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartDifferenceIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Difference view"
              description="Experiment tracking > runs charts > add chart menu > difference view"
            />
          </DropdownMenu.Item>
        )}
        {isChartTypeSupported(RunsChartType.IMAGE) && (
          <DropdownMenu.Item
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsaddchartmenu.tsx_126"
            onClick={() => onAddChart(RunsChartType.IMAGE)}
            data-testid="experiment-view-compare-runs-chart-type-image"
          >
            <DropdownMenu.IconWrapper css={styles.iconWrapper}>
              <ChartImageIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Image grid"
              description="Experiment tracking > runs charts > add chart menu > image grid"
            />
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

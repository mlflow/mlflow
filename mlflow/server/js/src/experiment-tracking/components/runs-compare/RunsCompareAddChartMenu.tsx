import { Button, DropdownMenu, PlusIcon } from '@databricks/design-system';
import { Theme } from '@emotion/react';

import { ReactComponent as ChartBarIcon } from '../../../common/static/chart-bar.svg';
import { ReactComponent as ChartContourIcon } from '../../../common/static/chart-contour.svg';
import { ReactComponent as ChartLineIcon } from '../../../common/static/chart-line.svg';
import { ReactComponent as ChartParallelIcon } from '../../../common/static/chart-parallel.svg';
import { ReactComponent as ChartScatterIcon } from '../../../common/static/chart-scatter.svg';
import { RunsCompareChartType } from './runs-compare.types';

export interface RunsCompareAddChartMenuProps {
  onAddChart: (type: RunsCompareChartType) => void;
}

export const RunsCompareAddChartMenu = ({ onAddChart }: RunsCompareAddChartMenuProps) => {
  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <Button
          css={styles.addChartButton}
          icon={<PlusIcon />}
          data-testid='experiment-view-compare-runs-add-chart'
        >
          Add chart
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align='end'>
        <DropdownMenu.Item
          onClick={() => onAddChart(RunsCompareChartType.BAR)}
          data-testid='experiment-view-compare-runs-chart-type-bar'
        >
          <DropdownMenu.IconWrapper css={styles.iconWrapper}>
            <ChartBarIcon />
          </DropdownMenu.IconWrapper>
          Bar chart
        </DropdownMenu.Item>
        <DropdownMenu.Item
          onClick={() => onAddChart(RunsCompareChartType.LINE)}
          data-testid='experiment-view-compare-runs-chart-type-line'
        >
          <DropdownMenu.IconWrapper css={styles.iconWrapper}>
            <ChartLineIcon />
          </DropdownMenu.IconWrapper>
          Line chart
        </DropdownMenu.Item>
        <DropdownMenu.Item
          onClick={() => onAddChart(RunsCompareChartType.PARALLEL)}
          data-testid='experiment-view-compare-runs-chart-type-parallel'
        >
          <DropdownMenu.IconWrapper css={styles.iconWrapper}>
            <ChartParallelIcon />
          </DropdownMenu.IconWrapper>
          Parallel coordinates
        </DropdownMenu.Item>
        <DropdownMenu.Item
          onClick={() => onAddChart(RunsCompareChartType.SCATTER)}
          data-testid='experiment-view-compare-runs-chart-type-scatter'
        >
          <DropdownMenu.IconWrapper css={styles.iconWrapper}>
            <ChartScatterIcon />
          </DropdownMenu.IconWrapper>
          Scatter chart
        </DropdownMenu.Item>
        <DropdownMenu.Item
          onClick={() => onAddChart(RunsCompareChartType.CONTOUR)}
          data-testid='experiment-view-compare-runs-chart-type-contour'
        >
          <DropdownMenu.IconWrapper css={styles.iconWrapper}>
            <ChartContourIcon />
          </DropdownMenu.IconWrapper>
          Contour chart
        </DropdownMenu.Item>
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

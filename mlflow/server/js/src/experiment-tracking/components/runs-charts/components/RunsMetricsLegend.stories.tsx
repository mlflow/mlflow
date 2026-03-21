import { lineDashStyles } from './RunsCharts.common';
import { chartColors, getRandomRunName } from './RunsCharts.stories-common';
import type { LegendLabelData } from './RunsMetricsLegend';
import RunsMetricsLegend from './RunsMetricsLegend';

const createData = (withDashStyle: boolean): LegendLabelData[] => {
  const data = [];

  for (let i = 0; i < 10; i++) {
    data.push({
      label: getRandomRunName(),
      color: chartColors[i],
      dashStyle: withDashStyle ? lineDashStyles[i] : undefined,
    });
  }

  return data;
};

export const WithDashStyles = () => <RunsMetricsLegend labelData={createData(true)} height={500} />;
export const NoDashStyles = () => <RunsMetricsLegend labelData={createData(false)} height={500} />;

WithDashStyles.storyName = 'With dash styles';
NoDashStyles.storyName = 'No dash styles';

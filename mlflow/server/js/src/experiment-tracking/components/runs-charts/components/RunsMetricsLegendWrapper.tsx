import React from 'react';

import type { LegendLabelData } from './RunsMetricsLegend';
import RunsMetricsLegend from './RunsMetricsLegend';
import { useDesignSystemTheme } from '@databricks/design-system';

const RunsMetricsLegendWrapper = ({
  labelData,
  fullScreen,
  children,
}: React.PropsWithChildren<{
  labelData: LegendLabelData[];
  fullScreen?: boolean;
}>) => {
  const { theme } = useDesignSystemTheme();

  const FULL_SCREEN_LEGEND_HEIGHT = 100;
  const LEGEND_HEIGHT = 32;

  const height = fullScreen ? FULL_SCREEN_LEGEND_HEIGHT : LEGEND_HEIGHT;
  const heightBuffer = fullScreen ? theme.spacing.lg : theme.spacing.md;

  return (
    <>
      <div css={{ height: `calc(100% - ${height + heightBuffer}px)` }}>{children}</div>
      <RunsMetricsLegend labelData={labelData} height={height} fullScreen={fullScreen} />
    </>
  );
};

export default RunsMetricsLegendWrapper;

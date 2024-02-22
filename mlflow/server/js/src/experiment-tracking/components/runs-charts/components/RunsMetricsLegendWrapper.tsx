import React from 'react';

import RunsMetricsLegend, { LegendLabelData } from './RunsMetricsLegend';
import { useDesignSystemTheme } from '@databricks/design-system';

const RunsMetricsLegendWrapper = ({
  labelData,
  children,
}: React.PropsWithChildren<{
  labelData: LegendLabelData[];
}>) => {
  const { theme } = useDesignSystemTheme();
  return (
    <>
      <div css={{ height: `calc(100% - 40px)` }}>{children}</div>
      <RunsMetricsLegend labelData={labelData} />
    </>
  );
};

export default RunsMetricsLegendWrapper;

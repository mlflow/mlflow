import { FormattedMessage } from 'react-intl';
import {
  DifferenceCardConfigCompareGroup,
  RunsChartsCardConfig,
  RunsChartsDifferenceCardConfig,
} from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import { DifferenceViewPlot } from '../charts/DifferenceViewPlot';
import { DifferenceViewPlotV2 } from '../charts/DifferenceViewPlotV2';

import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import { shouldEnableNewDifferenceViewCharts } from '@mlflow/mlflow/src/common/utils/FeatureUtils';

export const RunsChartsConfigureDifferenceChartPreview = ({
  previewData,
  groupBy,
  cardConfig,
  setCardConfig,
}: {
  previewData: RunsChartsRunData[];
  groupBy: RunsGroupByConfig | null;
  cardConfig: RunsChartsDifferenceCardConfig;
  setCardConfig: (setter: (current: RunsChartsCardConfig) => RunsChartsDifferenceCardConfig) => void;
}) => {
  if (shouldEnableNewDifferenceViewCharts()) {
    return (
      <DifferenceViewPlotV2
        previewData={previewData}
        groupBy={groupBy}
        cardConfig={cardConfig}
        setCardConfig={setCardConfig}
      />
    );
  }
  return (
    <DifferenceViewPlot
      previewData={previewData}
      groupBy={groupBy}
      cardConfig={cardConfig}
      setCardConfig={setCardConfig}
    />
  );
};

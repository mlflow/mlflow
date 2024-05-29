import { useCallback, useMemo } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import {
  DISABLED_GROUP_WHEN_GROUPBY,
  DifferenceCardConfigCompareGroup,
  type RunsChartsCardConfig,
  type RunsChartsDifferenceCardConfig,
} from '../../runs-charts.types';
import {
  type RunsChartCardReorderProps,
  RunsChartCardWrapper,
  RunsChartsChartsDragGroup,
  RunsChartCardFullScreenProps,
  ChartRunsCountIndicator,
} from './ChartCard.common';
import { shouldUseNewRunRowsVisibilityModel } from '../../../../../common/utils/FeatureUtils';
import { DifferenceViewPlot } from '../charts/DifferenceViewPlot';
import { useConfirmChartCardConfigurationFn } from '../../hooks/useRunsChartsUIConfiguration';
import { useIntl, FormattedMessage } from 'react-intl';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';

export interface RunsChartsDifferenceChartCardProps extends RunsChartCardReorderProps, RunsChartCardFullScreenProps {
  config: RunsChartsDifferenceCardConfig;
  chartRunData: RunsChartsRunData[];

  onDelete: () => void;
  onEdit: () => void;
  groupBy: RunsGroupByConfig | null;
}

export const RunsChartsDifferenceChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  onReorderWith,
  canMoveDown,
  canMoveUp,
  onMoveDown,
  onMoveUp,
  groupBy,
  fullScreen,
  setFullScreenChart,
}: RunsChartsDifferenceChartCardProps) => {
  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title: config.chartName,
      subtitle: <ChartRunsCountIndicator runsOrGroups={chartRunData} />,
    });
  };

  const slicedRuns = useMemo(() => {
    if (shouldUseNewRunRowsVisibilityModel()) {
      return chartRunData.filter(({ hidden }) => !hidden).reverse();
    }
    return chartRunData.slice(0, config.runsCountToCompare || 10).reverse();
  }, [chartRunData, config]);

  const confirmChartCardConfiguration = useConfirmChartCardConfigurationFn();

  const setCardConfig = (setter: (current: RunsChartsCardConfig) => RunsChartsDifferenceCardConfig) => {
    confirmChartCardConfiguration(setter(config));
  };

  const showChangeFromBaselineToggle = useCallback(() => {
    confirmChartCardConfiguration({
      ...config,
      showChangeFromBaseline: !config.showChangeFromBaseline,
    } as RunsChartsCardConfig);
  }, [config, confirmChartCardConfiguration]);

  const showDifferencesOnlyToggle = useCallback(() => {
    confirmChartCardConfiguration({
      ...config,
      showDifferencesOnly: !config.showDifferencesOnly,
    } as RunsChartsCardConfig);
  }, [config, confirmChartCardConfiguration]);

  const { formatMessage } = useIntl();

  const chartBody = (
    <div
      css={{
        display: 'flex',
        overflow: 'auto hidden',
        cursor: 'pointer',
        height: fullScreen ? '100%' : undefined,
        width: '100%',
      }}
    >
      <div css={{ width: '100%' }}>
        <DifferenceViewPlot
          previewData={slicedRuns}
          groupBy={groupBy}
          cardConfig={config}
          setCardConfig={setCardConfig}
        />
      </div>
    </div>
  );

  let showTooltip = undefined;
  if (groupBy && DISABLED_GROUP_WHEN_GROUPBY.some((group) => config.compareGroups.includes(group))) {
    showTooltip = formatMessage({
      defaultMessage: 'Disable grouped runs to compare parameters, tag, or attributes',
      description:
        'Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > disable group runs tooltip message',
    });
  }
  if (fullScreen) {
    return chartBody;
  }

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={config.chartName}
      subtitle={<ChartRunsCountIndicator runsOrGroups={slicedRuns} />}
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
      onReorderWith={onReorderWith}
      canMoveDown={canMoveDown}
      canMoveUp={canMoveUp}
      onMoveDown={onMoveDown}
      onMoveUp={onMoveUp}
      toggleFullScreenChart={toggleFullScreenChart}
      toggles={[
        {
          toggleLabel: formatMessage({
            defaultMessage: 'Show change from baseline',
            description:
              'Runs charts > components > cards > RunsChartsDifferenceChartCard > Show change from baseline toggle label',
          }),
          currentToggle: config.showChangeFromBaseline,
          setToggle: showChangeFromBaselineToggle,
        },
        {
          toggleLabel: formatMessage({
            defaultMessage: 'Show differences only',
            description:
              'Runs charts > components > cards > RunsChartsDifferenceChartCard > Show differences only toggle label',
          }),
          currentToggle: config.showDifferencesOnly,
          setToggle: showDifferencesOnlyToggle,
        },
      ]}
      tooltip={showTooltip}
    >
      {chartBody}
    </RunsChartCardWrapper>
  );
};

import { Button, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useCallback, useMemo } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import {
  DISABLED_GROUP_WHEN_GROUPBY,
  DifferenceCardConfigCompareGroup,
  type RunsChartsCardConfig,
  type RunsChartsDifferenceCardConfig,
} from '../../runs-charts.types';
import type { RunsChartCardFullScreenProps } from './ChartCard.common';
import { type RunsChartCardReorderProps, RunsChartCardWrapper, RunsChartsChartsDragGroup } from './ChartCard.common';
import { useConfirmChartCardConfigurationFn } from '../../hooks/useRunsChartsUIConfiguration';
import { useIntl, FormattedMessage } from 'react-intl';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import { DifferenceViewPlot } from '../charts/DifferenceViewPlot';

export interface RunsChartsDifferenceChartCardProps extends RunsChartCardReorderProps, RunsChartCardFullScreenProps {
  config: RunsChartsDifferenceCardConfig;
  chartRunData: RunsChartsRunData[];

  hideEmptyCharts?: boolean;

  onDelete: () => void;
  onEdit: () => void;
  groupBy: RunsGroupByConfig | null;
}

/**
 * A placeholder component displayed before runs difference chart is being configured by user
 */
const NotConfiguredDifferenceChartPlaceholder = ({ onEdit }: { onEdit: () => void }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', maxWidth: 360 }}>
        <Typography.Title css={{ marginTop: theme.spacing.md }} color="secondary" level={3}>
          <FormattedMessage
            defaultMessage="Compare runs"
            description="Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > chart not configured warning > title"
          />
        </Typography.Title>
        <Typography.Text css={{ marginBottom: theme.spacing.md }} color="secondary">
          <FormattedMessage
            defaultMessage="Use the runs difference view to compare model and system metrics, parameters, attributes,
            and tags across runs."
            description="Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > chart not configured warning > description"
          />
        </Typography.Text>
        <Button componentId="mlflow.charts.difference_chart_configure_button" type="primary" onClick={onEdit}>
          <FormattedMessage
            defaultMessage="Configure chart"
            description="Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > configure chart button"
          />
        </Button>
      </div>
    </div>
  );
};

export const RunsChartsDifferenceChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  groupBy,
  fullScreen,
  setFullScreenChart,
  hideEmptyCharts,
  ...reorderProps
}: RunsChartsDifferenceChartCardProps) => {
  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title: config.chartName,
      subtitle: null,
    });
  };

  const [isConfigured, slicedRuns] = useMemo(() => {
    const configured = Boolean(config.compareGroups?.length);
    return [configured, chartRunData.filter(({ hidden }) => !hidden).reverse()];
  }, [chartRunData, config]);

  const isEmptyDataset = useMemo(() => {
    return !isConfigured;
  }, [isConfigured]);

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
    <>
      {!isConfigured ? (
        <NotConfiguredDifferenceChartPlaceholder onEdit={onEdit} />
      ) : (
        <DifferenceViewPlot
          previewData={slicedRuns}
          groupBy={groupBy}
          cardConfig={config}
          setCardConfig={setCardConfig}
        />
      )}
    </>
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

  // Do not render the card if the chart is empty and the user has enabled hiding empty charts
  if (hideEmptyCharts && isEmptyDataset) {
    return null;
  }

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={config.chartName}
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
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
      {...reorderProps}
    >
      {chartBody}
    </RunsChartCardWrapper>
  );
};

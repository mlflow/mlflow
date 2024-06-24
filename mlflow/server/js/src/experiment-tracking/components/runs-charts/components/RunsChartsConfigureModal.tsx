/**
 * TODO: implement actual UI for this modal, it's a crude placeholder with minimal logic for now
 */
import { Modal, LegacySelect, useDesignSystemTheme } from '@databricks/design-system';
import { Interpolation, Theme } from '@emotion/react';
import React, { useCallback, useMemo, useState } from 'react';
import { useIntl, FormattedMessage } from 'react-intl';
import {
  RunsChartsBarCardConfig,
  RunsChartsCardConfig,
  RunsChartType,
  RunsChartsLineCardConfig,
  RunsChartsContourCardConfig,
  RunsChartsScatterCardConfig,
  RunsChartsParallelCardConfig,
  RunsChartsDifferenceCardConfig,
  RunsChartsImageCardConfig,
} from '../runs-charts.types';

import { ReactComponent as ChartBarIcon } from '../../../../common/static/chart-bar.svg';
import { ReactComponent as ChartContourIcon } from '../../../../common/static/chart-contour.svg';
import { ReactComponent as ChartLineIcon } from '../../../../common/static/chart-line.svg';
import { ReactComponent as ChartParallelIcon } from '../../../../common/static/chart-parallel.svg';
import { ReactComponent as ChartScatterIcon } from '../../../../common/static/chart-scatter.svg';
import { ReactComponent as ChartDifferenceIcon } from '../../../../common/static/chart-difference.svg';
import { ReactComponent as ChartImageIcon } from '../../../../common/static/chart-image.svg';
import { RunsChartsConfigureBarChart } from './config/RunsChartsConfigureBarChart';
import { RunsChartsConfigureParallelChart } from './config/RunsChartsConfigureParallelChart';
import type { RunsChartsRunData } from './RunsCharts.common';
import { RunsChartsConfigureField } from './config/RunsChartsConfigure.common';
import { RunsChartsConfigureLineChart } from './config/RunsChartsConfigureLineChart';
import { RunsChartsConfigureLineChartPreview } from './config/RunsChartsConfigureLineChart.preview';
import { RunsChartsConfigureBarChartPreview } from './config/RunsChartsConfigureBarChart.preview';
import { RunsChartsConfigureContourChartPreview } from './config/RunsChartsConfigureContourChart.preview';
import { RunsChartsConfigureScatterChartPreview } from './config/RunsChartsConfigureScatterChart.preview';
import { RunsChartsConfigureParallelChartPreview } from './config/RunsChartsConfigureParallelChart.preview';
import { RunsChartsConfigureContourChart } from './config/RunsChartsConfigureContourChart';
import { RunsChartsConfigureScatterChart } from './config/RunsChartsConfigureScatterChart';
import { RunsChartsTooltipBody } from './RunsChartsTooltipBody';
import { RunsChartsTooltipWrapper } from '../hooks/useRunsChartsTooltip';
import {
  shouldEnableDifferenceViewCharts,
  shouldEnableImageGridCharts,
  shouldUseNewRunRowsVisibilityModel,
} from 'common/utils/FeatureUtils';
import { RunsChartsConfigureDifferenceChartPreview } from './config/RunsChartsConfigureDifferenceChart.preview';
import { RunsChartsConfigureDifferenceChart } from './config/RunsChartsConfigureDifferenceChart';
import type { RunsGroupByConfig } from '../../experiment-page/utils/experimentPage.group-row-utils';
import { RunsChartsConfigureImageChart } from './config/RunsChartsConfigureImageChart';
import { RunsChartsConfigureImageChartPreview } from './config/RunsChartsConfigureImageChart.preview';

const previewComponentsMap: Record<
  RunsChartType,
  React.FC<{
    previewData: RunsChartsRunData[];
    cardConfig: any;
    groupBy: RunsGroupByConfig | null;
    setCardConfig: (
      setter: (current: RunsChartsCardConfig) => RunsChartsDifferenceCardConfig | RunsChartsImageCardConfig,
    ) => void;
  }>
> = {
  [RunsChartType.BAR]: RunsChartsConfigureBarChartPreview,
  [RunsChartType.CONTOUR]: RunsChartsConfigureContourChartPreview,
  [RunsChartType.LINE]: RunsChartsConfigureLineChartPreview,
  [RunsChartType.PARALLEL]: RunsChartsConfigureParallelChartPreview,
  [RunsChartType.SCATTER]: RunsChartsConfigureScatterChartPreview,
  [RunsChartType.DIFFERENCE]: RunsChartsConfigureDifferenceChartPreview,
  [RunsChartType.IMAGE]: RunsChartsConfigureImageChartPreview,
};

export const RunsChartsConfigureModal = ({
  onCancel,
  onSubmit,
  config,
  chartRunData,
  metricKeyList,
  paramKeyList,
  groupBy,
  supportedChartTypes,
}: {
  metricKeyList: string[];
  paramKeyList: string[];
  config: RunsChartsCardConfig;
  chartRunData: RunsChartsRunData[];
  onCancel: () => void;
  groupBy: RunsGroupByConfig | null;
  onSubmit: (formData: Partial<RunsChartsCardConfig>) => void;
  supportedChartTypes?: RunsChartType[] | undefined;
}) => {
  const isChartTypeSupported = (type: RunsChartType) => !supportedChartTypes || supportedChartTypes.includes(type);
  const { theme } = useDesignSystemTheme();

  const [currentFormState, setCurrentFormState] = useState<RunsChartsCardConfig>(config);

  const isEditing = Boolean(currentFormState.uuid);

  const updateChartType = useCallback((type?: RunsChartType) => {
    if (!type) {
      return;
    }
    const emptyChartCard = RunsChartsCardConfig.getEmptyChartCardByType(type, true);
    if (emptyChartCard) {
      setCurrentFormState(emptyChartCard);
    }
  }, []);

  const previewData = useMemo(() => {
    if (shouldUseNewRunRowsVisibilityModel()) {
      return chartRunData.filter(({ hidden }) => !hidden).reverse();
    }
    return chartRunData.slice(0, currentFormState.runsCountToCompare).reverse();
  }, [chartRunData, currentFormState.runsCountToCompare]);

  const imageKeyList = useMemo(() => {
    const imageKeys = new Set<string>();
    previewData.forEach((run) => {
      Object.keys(run.images).forEach((imageKey) => {
        imageKeys.add(imageKey);
      });
    });
    return Array.from(imageKeys).sort();
  }, [previewData]);

  const renderConfigOptionsforChartType = (type?: RunsChartType) => {
    if (type === RunsChartType.BAR) {
      return (
        <RunsChartsConfigureBarChart
          metricKeyList={metricKeyList}
          state={currentFormState as RunsChartsBarCardConfig}
          onStateChange={setCurrentFormState}
        />
      );
    }
    if (type === RunsChartType.CONTOUR) {
      return (
        <RunsChartsConfigureContourChart
          metricKeyList={metricKeyList}
          paramKeyList={paramKeyList}
          state={currentFormState as RunsChartsContourCardConfig}
          onStateChange={setCurrentFormState}
        />
      );
    }
    if (type === RunsChartType.LINE) {
      return (
        <RunsChartsConfigureLineChart
          metricKeyList={metricKeyList}
          state={currentFormState as RunsChartsLineCardConfig}
          onStateChange={setCurrentFormState}
        />
      );
    }
    if (type === RunsChartType.PARALLEL) {
      return (
        <RunsChartsConfigureParallelChart
          metricKeyList={metricKeyList}
          paramKeyList={paramKeyList}
          state={currentFormState as RunsChartsParallelCardConfig}
          onStateChange={setCurrentFormState}
        />
      );
    }
    if (type === RunsChartType.SCATTER) {
      return (
        <RunsChartsConfigureScatterChart
          metricKeyList={metricKeyList}
          paramKeyList={paramKeyList}
          state={currentFormState as RunsChartsScatterCardConfig}
          onStateChange={setCurrentFormState}
        />
      );
    }
    if (shouldEnableDifferenceViewCharts() && type === RunsChartType.DIFFERENCE) {
      return (
        <RunsChartsConfigureDifferenceChart
          metricKeyList={metricKeyList}
          paramKeyList={paramKeyList}
          state={currentFormState as RunsChartsDifferenceCardConfig}
          onStateChange={setCurrentFormState}
          groupBy={groupBy}
        />
      );
    }
    if (shouldEnableImageGridCharts() && type === RunsChartType.IMAGE) {
      return (
        <RunsChartsConfigureImageChart
          previewData={previewData}
          imageKeyList={imageKeyList}
          state={currentFormState as RunsChartsImageCardConfig}
          onStateChange={setCurrentFormState}
        />
      );
    }
    return null;
  };

  const renderPreviewChartType = (type?: RunsChartType) => {
    if (!type) {
      return null;
    }
    const PreviewComponent = previewComponentsMap[type];
    if (!PreviewComponent) {
      return null;
    }
    return (
      <PreviewComponent
        previewData={previewData}
        cardConfig={currentFormState}
        groupBy={groupBy}
        setCardConfig={setCurrentFormState}
      />
    );
  };

  const { formatMessage } = useIntl();

  let disableSaveButton = false;
  if (currentFormState.type === RunsChartType.LINE) {
    const lineCardConfig = currentFormState as RunsChartsLineCardConfig;
    disableSaveButton = (lineCardConfig.selectedMetricKeys ?? []).length === 0;
  }

  return (
    <Modal
      visible
      onCancel={onCancel}
      onOk={() => onSubmit(currentFormState)}
      title={
        isEditing
          ? formatMessage({
              defaultMessage: 'Edit chart',
              description: 'Title of the modal when editing a runs comparison chart',
            })
          : formatMessage({
              defaultMessage: 'Add new chart',
              description: 'Title of the modal when adding a new runs comparison chart',
            })
      }
      okButtonProps={{
        'data-testid': 'experiment-view-compare-runs-chart-modal-confirm',
        disabled: disableSaveButton,
      }}
      cancelText={formatMessage({
        defaultMessage: 'Cancel',
        description: 'Cancel button label within a modal for adding/editing a new runs comparison chart',
      })}
      okText={
        isEditing
          ? formatMessage({
              defaultMessage: 'Save changes',
              description: 'Confirm button label within a modal when editing a runs comparison chart',
            })
          : formatMessage({
              defaultMessage: 'Add chart',
              description: 'Confirm button label within a modal when adding a new runs comparison chart',
            })
      }
      size="wide"
      css={{ width: 1280 }}
    >
      <div css={styles.wrapper}>
        <div>
          {!isEditing && (
            <RunsChartsConfigureField title="Type">
              <LegacySelect<RunsChartType>
                css={{ width: '100%' }}
                value={currentFormState.type}
                onChange={updateChartType}
              >
                {isChartTypeSupported(RunsChartType.BAR) && (
                  <LegacySelect.Option value={RunsChartType.BAR}>
                    <div css={styles.chartTypeOption(theme)}>
                      <ChartBarIcon />
                      <FormattedMessage
                        defaultMessage="Bar chart"
                        description="Experiment tracking > runs charts > add chart menu > bar chart"
                      />
                    </div>
                  </LegacySelect.Option>
                )}
                {isChartTypeSupported(RunsChartType.SCATTER) && (
                  <LegacySelect.Option value={RunsChartType.SCATTER}>
                    <div css={styles.chartTypeOption(theme)}>
                      <ChartScatterIcon />
                      <FormattedMessage
                        defaultMessage="Scatter chart"
                        description="Experiment tracking > runs charts > add chart menu > scatter plot"
                      />
                    </div>
                  </LegacySelect.Option>
                )}
                {isChartTypeSupported(RunsChartType.LINE) && (
                  <LegacySelect.Option value={RunsChartType.LINE}>
                    <div css={styles.chartTypeOption(theme)}>
                      <ChartLineIcon />
                      <FormattedMessage
                        defaultMessage="Line chart"
                        description="Experiment tracking > runs charts > add chart menu > line chart"
                      />
                    </div>
                  </LegacySelect.Option>
                )}
                {isChartTypeSupported(RunsChartType.PARALLEL) && (
                  <LegacySelect.Option value={RunsChartType.PARALLEL}>
                    <div css={styles.chartTypeOption(theme)}>
                      <ChartParallelIcon />
                      <FormattedMessage
                        defaultMessage="Parallel coordinates"
                        description="Experiment tracking > runs charts > add chart menu > parallel coordinates"
                      />
                    </div>
                  </LegacySelect.Option>
                )}
                {isChartTypeSupported(RunsChartType.CONTOUR) && (
                  <LegacySelect.Option value={RunsChartType.CONTOUR}>
                    <div css={styles.chartTypeOption(theme)}>
                      <ChartContourIcon />
                      <FormattedMessage
                        defaultMessage="Contour chart"
                        description="Experiment tracking > runs charts > add chart menu > contour chart"
                      />
                    </div>
                  </LegacySelect.Option>
                )}
                {shouldEnableDifferenceViewCharts() && isChartTypeSupported(RunsChartType.DIFFERENCE) && (
                  <LegacySelect.Option value={RunsChartType.DIFFERENCE}>
                    <div css={styles.chartTypeOption(theme)}>
                      <ChartDifferenceIcon />
                      <FormattedMessage
                        defaultMessage="Difference view"
                        description="Experiment tracking > runs charts > add chart menu > difference view"
                      />
                    </div>
                  </LegacySelect.Option>
                )}
                {shouldEnableImageGridCharts() && isChartTypeSupported(RunsChartType.IMAGE) && (
                  <LegacySelect.Option value={RunsChartType.IMAGE}>
                    <div css={styles.chartTypeOption(theme)}>
                      <ChartImageIcon />
                      <FormattedMessage
                        defaultMessage="Image grid"
                        description="Experiment tracking > runs charts > add chart menu > image grid"
                      />
                    </div>
                  </LegacySelect.Option>
                )}
              </LegacySelect>
            </RunsChartsConfigureField>
          )}
          {renderConfigOptionsforChartType(currentFormState.type)}
        </div>
        <RunsChartsTooltipWrapper contextData={{ runs: chartRunData }} component={RunsChartsTooltipBody} hoverOnly>
          <div css={styles.chartWrapper}>{renderPreviewChartType(currentFormState.type)}</div>
        </RunsChartsTooltipWrapper>
      </div>
    </Modal>
  );
};

const styles = {
  chartTypeOption: (theme: Theme) =>
    ({
      display: 'grid',
      gridTemplateColumns: `${theme.general.iconSize + theme.spacing.xs}px 1fr`,
      gap: theme.spacing.xs,
      alignItems: 'center',
    } as Interpolation<Theme>),
  wrapper: {
    // TODO: wait for modal dimensions decision
    display: 'grid',
    gridTemplateColumns: '300px 1fr',
    gap: 32,
  } as Interpolation<Theme>,
  field: {
    // TODO: wait for modal dimensions decision
    display: 'grid',
    gridTemplateColumns: '80px 1fr',
    marginBottom: 16,
  } as Interpolation<Theme>,
  chartWrapper: {
    height: 400,
    width: 500,
  },
};

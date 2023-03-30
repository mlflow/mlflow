/**
 * TODO: implement actual UI for this modal, it's a crude placeholder with minimal logic for now
 */
import { Modal, Select } from '@databricks/design-system';
import { Interpolation, Theme } from '@emotion/react';
import React, { useCallback, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import {
  RunsCompareBarCardConfig,
  RunsCompareCardConfig,
  RunsCompareChartType,
  RunsCompareLineCardConfig,
  RunsCompareContourCardConfig,
  RunsCompareScatterCardConfig,
  RunsCompareParallelCardConfig,
} from './runs-compare.types';

import { ReactComponent as ChartBarIcon } from '../../../common/static/chart-bar.svg';
import { ReactComponent as ChartContourIcon } from '../../../common/static/chart-contour.svg';
import { ReactComponent as ChartLineIcon } from '../../../common/static/chart-line.svg';
import { ReactComponent as ChartParallelIcon } from '../../../common/static/chart-parallel.svg';
import { ReactComponent as ChartScatterIcon } from '../../../common/static/chart-scatter.svg';
import { RunsCompareConfigureBarChart } from './config/RunsCompareConfigureBarChart';
import { RunsCompareConfigureParallelChart } from './config/RunsCompareConfigureParallelChart';
import type { CompareChartRunData } from './charts/CompareRunsCharts.common';
import { RunsCompareConfigureField } from './config/RunsCompareConfigure.common';
import { RunsCompareConfigureLineChart } from './config/RunsCompareConfigureLineChart';
import { RunsCompareConfigureLineChartPreview } from './config/RunsCompareConfigureLineChart.preview';
import { RunsCompareConfigureBarChartPreview } from './config/RunsCompareConfigureBarChart.preview';
import { RunsCompareConfigureContourChartPreview } from './config/RunsCompareConfigureContourChart.preview';
import { RunsCompareConfigureScatterChartPreview } from './config/RunsCompareConfigureScatterChart.preview';
import { RunsCompareConfigureParallelChartPreview } from './config/RunsCompareConfigureParallelChart.preview';
import { RunsCompareConfigureContourChart } from './config/RunsCompareConfigureContourChart';
import { RunsCompareConfigureScatterChart } from './config/RunsCompareConfigureScatterChart';
import { RunsCompareTooltipBody } from './RunsCompareTooltipBody';
import { CompareRunsTooltipWrapper } from './hooks/useCompareRunsTooltip';

const previewComponentsMap: Record<
  RunsCompareChartType,
  React.FC<{
    previewData: CompareChartRunData[];
    cardConfig: any;
  }>
> = {
  [RunsCompareChartType.BAR]: RunsCompareConfigureBarChartPreview,
  [RunsCompareChartType.CONTOUR]: RunsCompareConfigureContourChartPreview,
  [RunsCompareChartType.LINE]: RunsCompareConfigureLineChartPreview,
  [RunsCompareChartType.PARALLEL]: RunsCompareConfigureParallelChartPreview,
  [RunsCompareChartType.SCATTER]: RunsCompareConfigureScatterChartPreview,
};

export const RunsCompareConfigureModal = ({
  onCancel,
  onSubmit,
  config,
  chartRunData,
  metricKeyList,
  paramKeyList,
}: {
  metricKeyList: string[];
  paramKeyList: string[];
  config: RunsCompareCardConfig;
  chartRunData: CompareChartRunData[];
  onCancel: () => void;
  onSubmit: (formData: Partial<RunsCompareCardConfig>) => void;
}) => {
  const [currentFormState, setCurrentFormState] = useState<RunsCompareCardConfig>(config);

  const isEditing = Boolean(currentFormState.uuid);

  const updateChartType = useCallback((type?: RunsCompareChartType) => {
    if (!type) {
      return;
    }
    const emptyChartCard = RunsCompareCardConfig.getEmptyChartCardByType(type);
    if (emptyChartCard) {
      setCurrentFormState(emptyChartCard);
    }
  }, []);

  const renderConfigOptionsforChartType = (type?: RunsCompareChartType) => {
    if (type === RunsCompareChartType.BAR) {
      return (
        <RunsCompareConfigureBarChart
          metricKeyList={metricKeyList}
          state={currentFormState as RunsCompareBarCardConfig}
          onStateChange={setCurrentFormState}
        />
      );
    }
    if (type === RunsCompareChartType.CONTOUR) {
      return (
        <RunsCompareConfigureContourChart
          metricKeyList={metricKeyList}
          paramKeyList={paramKeyList}
          state={currentFormState as RunsCompareContourCardConfig}
          onStateChange={setCurrentFormState}
        />
      );
    }
    if (type === RunsCompareChartType.LINE) {
      return (
        <RunsCompareConfigureLineChart
          metricKeyList={metricKeyList}
          state={currentFormState as RunsCompareLineCardConfig}
          onStateChange={setCurrentFormState}
        />
      );
    }
    if (type === RunsCompareChartType.PARALLEL) {
      return (
        <RunsCompareConfigureParallelChart
          metricKeyList={metricKeyList}
          paramKeyList={paramKeyList}
          state={currentFormState as RunsCompareParallelCardConfig}
          onStateChange={setCurrentFormState}
        />
      );
    }
    if (type === RunsCompareChartType.SCATTER) {
      return (
        <RunsCompareConfigureScatterChart
          metricKeyList={metricKeyList}
          paramKeyList={paramKeyList}
          state={currentFormState as RunsCompareScatterCardConfig}
          onStateChange={setCurrentFormState}
        />
      );
    }
    return null;
  };

  const previewData = useMemo(
    () => chartRunData.slice(0, currentFormState.runsCountToCompare).reverse(),
    [chartRunData, currentFormState.runsCountToCompare],
  );

  const renderPreviewChartType = (type?: RunsCompareChartType) => {
    if (!type) {
      return null;
    }
    const PreviewComponent = previewComponentsMap[type];
    if (!PreviewComponent) {
      return null;
    }
    return <PreviewComponent previewData={previewData} cardConfig={currentFormState} />;
  };

  const { formatMessage } = useIntl();

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
      okButtonProps={{ 'data-testid': 'experiment-view-compare-runs-chart-modal-confirm' }}
      cancelText={formatMessage({
        defaultMessage: 'Cancel',
        description:
          'Cancel button label within a modal for adding/editing a new runs comparison chart',
      })}
      okText={
        isEditing
          ? formatMessage({
              defaultMessage: 'Save changes',
              description:
                'Confirm button label within a modal when editing a runs comparison chart',
            })
          : formatMessage({
              defaultMessage: 'Add chart',
              description:
                'Confirm button label within a modal when adding a new runs comparison chart',
            })
      }
      size='wide'
      css={{ width: 1280 }}
    >
      <div css={styles.wrapper}>
        <div>
          {!isEditing && (
            <RunsCompareConfigureField title='Type'>
              <Select<RunsCompareChartType>
                css={{ width: '100%' }}
                value={currentFormState.type}
                onChange={updateChartType}
              >
                <Select.Option value={RunsCompareChartType.BAR}>
                  <div css={styles.chartTypeOption}>
                    <ChartBarIcon />
                    Bar chart
                  </div>
                </Select.Option>
                <Select.Option value={RunsCompareChartType.SCATTER}>
                  <div css={styles.chartTypeOption}>
                    <ChartScatterIcon />
                    Scatter chart
                  </div>
                </Select.Option>
                <Select.Option value={RunsCompareChartType.LINE}>
                  <div css={styles.chartTypeOption}>
                    <ChartLineIcon />
                    Line chart
                  </div>
                </Select.Option>
                <Select.Option value={RunsCompareChartType.PARALLEL}>
                  <div css={styles.chartTypeOption}>
                    <ChartParallelIcon />
                    Parallel coordinates
                  </div>
                </Select.Option>
                <Select.Option value={RunsCompareChartType.CONTOUR}>
                  <div css={styles.chartTypeOption}>
                    <ChartContourIcon />
                    Contour chart
                  </div>
                </Select.Option>
              </Select>
            </RunsCompareConfigureField>
          )}
          {renderConfigOptionsforChartType(currentFormState.type)}
        </div>
        <CompareRunsTooltipWrapper
          contextData={{ runs: chartRunData }}
          component={RunsCompareTooltipBody}
          hoverOnly
        >
          <div css={styles.chartWrapper}>{renderPreviewChartType(currentFormState.type)}</div>
        </CompareRunsTooltipWrapper>
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
  },
};

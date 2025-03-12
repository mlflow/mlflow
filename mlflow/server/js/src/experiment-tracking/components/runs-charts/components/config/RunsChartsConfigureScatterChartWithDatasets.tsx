import { useCallback, useEffect, useMemo } from 'react';
import { useIntl } from 'react-intl';
import type {
  RunsChartsCardConfig,
  RunsChartsMetricByDatasetEntry,
  RunsChartsScatterCardConfig,
} from '../../runs-charts.types';
import { RunsChartsConfigureField, RunsChartsMetricParamSelectV2 } from './RunsChartsConfigure.common';

type ValidAxis = keyof Pick<RunsChartsScatterCardConfig, 'xaxis' | 'yaxis'>;

/**
 * Form containing configuration controls for scatter runs compare chart.
 */
export const RunsChartsConfigureScatterChartWithDatasets = ({
  state,
  onStateChange,
  paramKeyList,
  metricKeysByDataset,
}: {
  paramKeyList: string[];
  metricKeysByDataset: RunsChartsMetricByDatasetEntry[] | undefined;
  state: RunsChartsScatterCardConfig;
  onStateChange: (setter: (current: RunsChartsCardConfig) => RunsChartsScatterCardConfig) => void;
}) => {
  const { formatMessage } = useIntl();

  const metricOptions = useMemo(
    () =>
      metricKeysByDataset?.map(({ dataAccessKey, metricKey, datasetName }) => ({
        key: JSON.stringify(['METRIC', dataAccessKey]),
        dataAccessKey,
        datasetName,
        metricKey,
      })) ?? [],
    [metricKeysByDataset],
  );

  const paramOptions = useMemo(
    () =>
      paramKeyList?.map((paramKey) => ({
        key: JSON.stringify(['PARAM', paramKey]),
        paramKey,
      })) ?? [],
    [paramKeyList],
  );

  /**
   * Callback for updating X or Y axis
   */
  const handleChange = useCallback(
    (axis: ValidAxis) => (value: string) => {
      const foundMetric = metricOptions.find(({ key }) => key === value);
      if (foundMetric) {
        const { dataAccessKey, datasetName, metricKey } = foundMetric;
        onStateChange((current) => ({
          ...(current as RunsChartsScatterCardConfig),
          [axis]: { key: metricKey, type: 'METRIC', datasetName, dataAccessKey },
        }));
      }
      const foundParam = paramOptions.find(({ key }) => key === value);
      if (foundParam) {
        onStateChange((current) => ({
          ...(current as RunsChartsScatterCardConfig),
          [axis]: { key: foundParam.paramKey, type: 'PARAM' },
        }));
      }
    },
    [onStateChange, metricOptions, paramOptions],
  );

  useEffect(() => {
    // For each axis: if there is no selected value, select the first available option
    for (const axis of ['xaxis', 'yaxis'] as const) {
      if (!state[axis]?.key) {
        if (metricOptions?.[0]) {
          handleChange(axis)(metricOptions[0].key);
        } else if (paramOptions?.[0]) {
          handleChange(axis)(paramOptions[0].key);
        }
      }
    }
  }, [state, metricOptions, paramOptions, handleChange]);

  const getSelectedValue = useCallback(
    (axis: ValidAxis) => {
      if (state[axis].type === 'METRIC') {
        const foundMetricOption = metricOptions.find(
          ({ dataAccessKey }) => dataAccessKey === state[axis].dataAccessKey,
        );
        if (foundMetricOption) {
          return foundMetricOption.key;
        }
      }
      if (state[axis].type === 'PARAM') {
        const foundParamOption = paramOptions.find(({ paramKey }) => paramKey === state[axis].key);
        if (foundParamOption) {
          return foundParamOption.key;
        }
      }
      return '';
    },
    [state, metricOptions, paramOptions],
  );

  const selectedXValue = useMemo(() => getSelectedValue('xaxis'), [getSelectedValue]);
  const selectedYValue = useMemo(() => getSelectedValue('yaxis'), [getSelectedValue]);

  return (
    <>
      <RunsChartsConfigureField
        title={formatMessage({
          defaultMessage: 'X axis',
          description: 'Label for X axis in scatter chart configurator in compare runs chart config modal',
        })}
      >
        <RunsChartsMetricParamSelectV2
          value={selectedXValue}
          onChange={handleChange('xaxis')}
          metricOptions={metricOptions}
          paramOptions={paramOptions}
          id="mlflow.charts.chart_configure.scatter.x_axis"
        />
      </RunsChartsConfigureField>
      <RunsChartsConfigureField
        title={formatMessage({
          defaultMessage: 'Y axis',
          description: 'Label for Y axis in scatter chart configurator in compare runs chart config modal',
        })}
      >
        <RunsChartsMetricParamSelectV2
          value={selectedYValue}
          onChange={handleChange('yaxis')}
          metricOptions={metricOptions}
          paramOptions={paramOptions}
          id="mlflow.charts.chart_configure.scatter.y_axis"
        />
      </RunsChartsConfigureField>
    </>
  );
};

import { SimpleSelect, SimpleSelectOption, Tag } from '@databricks/design-system';
import type { RunsChartsMetricByDatasetEntry } from '../../runs-charts.types';

export const RunsChartsConfigureMetricWithDatasetSelect = ({
  metricKeysByDataset,
  value,
  onChange,
}: {
  metricKeysByDataset?: RunsChartsMetricByDatasetEntry[];
  value?: string;
  onChange: (metricByDatasetEntry: RunsChartsMetricByDatasetEntry) => void;
}) => {
  return (
    <SimpleSelect
      css={{ width: '100%' }}
      componentId="mlflow.charts.chart_configure.metric_with_dataset_select"
      id="mlflow.charts.chart_configure.metric_with_dataset_select"
      value={value}
      onChange={({ target }) => {
        const entry = metricKeysByDataset?.find(({ dataAccessKey }) => dataAccessKey === target.value);
        if (entry) {
          onChange(entry);
        }
      }}
      contentProps={{
        matchTriggerWidth: true,
        maxHeight: 400,
      }}
    >
      {metricKeysByDataset?.map(({ datasetName, metricKey, dataAccessKey }) => (
        <SimpleSelectOption key={dataAccessKey} value={dataAccessKey}>
          {datasetName && (
            <Tag componentId="mlflow.charts.chart_configure.metric_with_dataset_select.tag">{datasetName}</Tag>
          )}{' '}
          {metricKey}
        </SimpleSelectOption>
      ))}
    </SimpleSelect>
  );
};

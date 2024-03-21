/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Button, LegacySelect } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

type Props = {
  paramKeys: string[];
  metricKeys: string[];
  selectedParamKeys: string[];
  selectedMetricKeys: string[];
  handleParamsSelectChange: (...args: any[]) => any;
  handleMetricsSelectChange: (...args: any[]) => any;
  onClearAllSelect: (...args: any[]) => any;
};

export class ParallelCoordinatesPlotControls extends React.Component<Props> {
  render() {
    const {
      paramKeys,
      metricKeys,
      selectedParamKeys,
      selectedMetricKeys,
      handleParamsSelectChange,
      handleMetricsSelectChange,
      onClearAllSelect,
    } = this.props;
    return (
      <div css={styles.wrapper}>
        <div>
          <FormattedMessage
            defaultMessage="Parameters:"
            description="Label text for parameters in parallel coordinates plot in MLflow"
          />
        </div>
        <LegacySelect
          mode="multiple"
          css={styles.select}
          placeholder={
            <FormattedMessage
              defaultMessage="Please select parameters"
              description="Placeholder text for parameters in parallel coordinates plot in MLflow"
            />
          }
          value={selectedParamKeys}
          onChange={handleParamsSelectChange}
        >
          {paramKeys.map((key) => (
            <LegacySelect.Option value={key} key={key}>
              {key}
            </LegacySelect.Option>
          ))}
        </LegacySelect>
        <div style={{ marginTop: 20 }}>
          <FormattedMessage
            defaultMessage="Metrics:"
            description="Label text for metrics in parallel coordinates plot in MLflow"
          />
        </div>
        <LegacySelect
          mode="multiple"
          css={styles.select}
          placeholder={
            <FormattedMessage
              defaultMessage="Please select metrics"
              description="Placeholder text for metrics in parallel coordinates plot in MLflow"
            />
          }
          value={selectedMetricKeys}
          onChange={handleMetricsSelectChange}
        >
          {metricKeys.map((key) => (
            <LegacySelect.Option value={key} key={key}>
              {key}
            </LegacySelect.Option>
          ))}
        </LegacySelect>
        <div style={{ marginTop: 20 }}>
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_parallelcoordinatesplotcontrols.tsx_84"
            data-test-id="clear-button"
            onClick={onClearAllSelect}
          >
            <FormattedMessage
              defaultMessage="Clear All"
              description="String for the clear button to clear any selected parameters and metrics"
            />
          </Button>
        </div>
      </div>
    );
  }
}

const styles = {
  wrapper: (theme: any) => ({
    padding: `0 ${theme.spacing.xs}px`,
  }),
  select: { width: '100%' },
};

import React, { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';

import {
  Typography,
  Row,
  Col,
  SimpleSelect,
  SimpleSelectOptionGroup,
  SimpleSelectOption,
  FormUI,
} from '@databricks/design-system';
import { LazyPlot } from './LazyPlot';
import type { RunInfoEntity } from '../types';

type Props = {
  runUuids: string[];
  runInfos: RunInfoEntity[];
  metricLists: any[][];
  paramLists: any[][];
};

type Axis = {
  key?: string;
  isParam?: boolean;
};

const paramOptionPrefix = 'param-';
const metricOptionPrefix = 'metric-';

// Note: This component does not pass the value of the parent component to the child component.
// Doing so will cause weird rendering issues with the label and updating of the value.
const Selector = ({
  id,
  onChange,
  paramKeys,
  metricKeys,
}: {
  id: string;
  onChange: (axis: Axis) => void;
  paramKeys: string[];
  metricKeys: string[];
}) => {
  const intl = useIntl();
  return (
    <SimpleSelect
      componentId="codegen_mlflow_app_src_experiment-tracking_components_comparerunbox.tsx_46"
      id={id}
      css={{ width: '100%', marginBottom: '16px' }}
      placeholder={intl.formatMessage({
        defaultMessage: 'Select parameter or metric',
        description: 'Placeholder text for parameter/metric selector in box plot comparison in MLflow',
      })}
      onChange={({ target }) => {
        const { value } = target;
        const [_prefix, key] = value.split('-');
        const isParam = value.startsWith(paramOptionPrefix);
        onChange({ key, isParam });
      }}
    >
      <SimpleSelectOptionGroup label="Parameters">
        {paramKeys.map((key) => (
          <SimpleSelectOption key={key} value={paramOptionPrefix + key}>
            {key}
          </SimpleSelectOption>
        ))}
      </SimpleSelectOptionGroup>
      <SimpleSelectOptionGroup label="Metrics">
        {metricKeys.map((key) => (
          <SimpleSelectOption key={key} value={metricOptionPrefix + key}>
            {key}
          </SimpleSelectOption>
        ))}
      </SimpleSelectOptionGroup>
    </SimpleSelect>
  );
};

export const CompareRunBox = ({ runInfos, metricLists, paramLists }: Props) => {
  const [xAxis, setXAxis] = useState<Axis>({ key: undefined, isParam: undefined });
  const [yAxis, setYAxis] = useState<Axis>({ key: undefined, isParam: undefined });

  const paramKeys = Array.from(new Set(paramLists.flat().map(({ key }) => key))).sort();
  const metricKeys = Array.from(new Set(metricLists.flat().map(({ key }) => key))).sort();

  const getBoxPlotData = () => {
    const data = {};
    runInfos.forEach((_, index) => {
      const params = paramLists[index];
      const metrics = metricLists[index];
      const x = (xAxis.isParam ? params : metrics).find(({ key }) => key === xAxis.key);
      const y = (yAxis.isParam ? params : metrics).find(({ key }) => key === yAxis.key);
      if (x === undefined || y === undefined) {
        return;
      }

      if (x.value in data) {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        data[x.value].push(y.value);
      } else {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        data[x.value] = [y.value];
      }
    });

    return Object.entries(data).map(([key, values]) => ({
      y: values,
      type: 'box',
      name: key,
      jitter: 0.3,
      pointpos: -1.5,
      boxpoints: 'all',
    }));
  };

  const renderPlot = () => {
    if (!(xAxis.key && yAxis.key)) {
      return (
        <div
          css={{
            display: 'flex',
            width: '100%',
            height: '100%',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <Typography.Text size="xl">
            <FormattedMessage
              defaultMessage="Select parameters/metrics to plot."
              description="Text to show when x or y axis is not selected on box plot"
            />
          </Typography.Text>
        </div>
      );
    }

    return (
      <LazyPlot
        css={{
          width: '100%',
          height: '100%',
          minHeight: '35vw',
        }}
        data={getBoxPlotData()}
        layout={{
          margin: {
            t: 30,
          },
          hovermode: 'closest',
          xaxis: {
            title: xAxis.key,
          },
          yaxis: {
            title: yAxis.key,
          },
        }}
        config={{
          responsive: true,
          displaylogo: false,
          scrollZoom: true,
          modeBarButtonsToRemove: [
            'sendDataToCloud',
            'select2d',
            'lasso2d',
            'resetScale2d',
            'hoverClosestCartesian',
            'hoverCompareCartesian',
          ],
        }}
        useResizeHandler
      />
    );
  };

  return (
    <Row>
      <Col span={6}>
        <div css={styles.borderSpacer}>
          <div>
            <FormUI.Label htmlFor="x-axis-selector">
              <FormattedMessage
                defaultMessage="X-axis:"
                description="Label text for X-axis in box plot comparison in MLflow"
              />
            </FormUI.Label>
          </div>
          <Selector id="x-axis-selector" onChange={setXAxis} paramKeys={paramKeys} metricKeys={metricKeys} />

          <div>
            <FormUI.Label htmlFor="y-axis-selector">
              <FormattedMessage
                defaultMessage="Y-axis:"
                description="Label text for Y-axis in box plot comparison in MLflow"
              />
            </FormUI.Label>
          </div>
          <Selector id="y-axis-selector" onChange={setYAxis} paramKeys={paramKeys} metricKeys={metricKeys} />
        </div>
      </Col>
      <Col span={18}>{renderPlot()}</Col>
    </Row>
  );
};

const styles = {
  borderSpacer: (theme: any) => ({
    paddingLeft: theme.spacing.xs,
  }),
};

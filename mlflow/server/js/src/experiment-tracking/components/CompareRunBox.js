import React, { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import PropTypes from 'prop-types';
import { Row, Col, Select } from 'antd';
import { Typography } from '@databricks/design-system';
import { RunInfo } from '../sdk/MlflowMessages';
import { LazyPlot } from './LazyPlot';

const { Option, OptGroup } = Select;

export const CompareRunBox = ({ runUuids, runInfos, metricLists, paramLists }) => {
  const [xAxis, setXAxis] = useState({ key: undefined, isParam: undefined });
  const [yAxis, setYAxis] = useState({ key: undefined, isParam: undefined });

  const paramKeys = Array.from(new Set(paramLists.flat().map(({ key }) => key))).sort();
  const metricKeys = Array.from(new Set(metricLists.flat().map(({ key }) => key))).sort();

  const paramOptionPrefix = 'param-';
  const metricOptionPrefix = 'metric-';

  const handleXAxisChange = (_, { value, key }) => {
    const isParam = value.startsWith(paramOptionPrefix);
    setXAxis({ key, isParam });
  };

  const handleYAxisChange = (_, { value, key }) => {
    const isParam = value.startsWith(paramOptionPrefix);
    setYAxis({ key, isParam });
  };

  const renderSelector = (onChange, selectedValue) => (
    <Select
      css={{ width: '100%', marginBottom: '16px' }}
      placeholder='Select'
      onChange={onChange}
      value={selectedValue}
    >
      <OptGroup label='Parameters' key='parameters'>
        {paramKeys.map((key) => (
          <Option key={key} value={paramOptionPrefix + key}>
            <div data-test-id='axis-option'>{key}</div>
          </Option>
        ))}
      </OptGroup>
      <OptGroup label='Metrics'>
        {metricKeys.map((key) => (
          <Option key={key} value={metricOptionPrefix + key}>
            <div data-test-id='axis-option'>{key}</div>
          </Option>
        ))}
      </OptGroup>
    </Select>
  );

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
        data[x.value].push(y.value);
      } else {
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
          <Typography.Text size='xl'>
            <FormattedMessage
              defaultMessage='Select parameters/metrics to plot.'
              description='Text to show when x or y axis is not selected on box plot'
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
        <div>
          <label htmlFor='x-axis-selector'>
            <FormattedMessage
              defaultMessage='X-axis:'
              description='Label text for X-axis in box plot comparison in MLflow'
            />
          </label>
        </div>
        {renderSelector(handleXAxisChange, xAxis.value)}

        <div>
          <label htmlFor='y-axis-selector'>
            <FormattedMessage
              defaultMessage='Y-axis:'
              description='Label text for Y-axis in box plot comparison in MLflow'
            />
          </label>
        </div>
        {renderSelector(handleYAxisChange, yAxis.value)}
      </Col>
      <Col span={18}>{renderPlot()}</Col>
    </Row>
  );
};

CompareRunBox.propTypes = {
  runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
  runInfos: PropTypes.arrayOf(PropTypes.instanceOf(RunInfo)).isRequired,
  metricLists: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
  paramLists: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
};

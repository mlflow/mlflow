import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from 'recharts';
import Utils from '../utils/Utils';
import _ from 'lodash';
import PropTypes from 'prop-types';
import { X_AXIS_STEP, X_AXIS_RELATIVE, X_AXIS_WALL } from './MetricsPlotControls';
import { CHART_TYPE_BAR } from './MetricsPlotPanel';

const COLORS = [
  '#A3C3D9',
  '#82ca9d',
  '#8884d8',
  '#FF82A9',
  '#FFC0BE',
  '#AE76A6',
  '#993955',
  '#364958',
];

export class MetricsPlotView extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(String).isRequired,
    metrics: PropTypes.arrayOf(Object).isRequired,
    xAxis: PropTypes.string.isRequired,
    metricKeys: PropTypes.arrayOf(String).isRequired,
    showDot: PropTypes.bool.isRequired,
    chartType: PropTypes.string.isRequired,
  };

  static MAX_RUN_NAME_DISPLAY_LENGTH = 36;

  getIdForRunMetricData = (runUuid, metricKey) => `${runUuid}_${metricKey}`;

  parseTimestamp = (timestampStr, baseTimestamp, xAxis) => {
    const timestamp = Number.parseFloat(timestampStr);
    return xAxis === X_AXIS_RELATIVE ? (timestamp - baseTimestamp) / 1000 : timestamp;
  };

  getDataForLineChart = () =>
    _.flatMap(this.props.metrics, (metric) => {
      const { metricKey, runUuid, runName, history } = metric;
      const baseTimestamp = Number.parseFloat(history[0] && history[0].timestamp);
      return history.map((entry) => ({
        [this.getIdForRunMetricData(runUuid, metricKey)]: entry.value,
        runUuid,
        runName,
        metricKey,
        value: entry.value,
        step: Number.parseInt(entry.step, 10),
        timestamp: this.parseTimestamp(entry.timestamp, baseTimestamp, this.props.xAxis),
      }));
    });

  getDataForBarChart = () => {
    const map = this.props.metrics.reduce((map, metric) => {
      const { runUuid, metricKey, history } = metric;
      const value = history[0] && history[0].value;
      if (!map[metricKey]) {
        map[metricKey] = {
          metricKey,
          [runUuid]: value,
        };
      } else {
        map[metricKey][runUuid] = value;
      }
      return map;
    }, {});
    return _.sortBy(Object.values(map), 'metricKey');
  };

  // Returns payload to use in recharts Legend component
  // Legend type must be one of the values in
  // https://github.com/recharts/recharts/blob/1b523c1/src/util/ReactUtils.js#L139
  getLegendPayload = (legendType) => {
    const { metrics } = this.props;
    metrics.map(({ runName, metricKey }, index) => {
      const truncatedRunName = Utils.truncateString(
        runName,
        MetricsPlotView.MAX_RUN_NAME_DISPLAY_LENGTH,
      );
      return {
        value: `${truncatedRunName} - ${metricKey}`,
        id: index,
        type: legendType,
        // Must specify legend item color, see https://github.com/recharts/recharts/issues/818
        color: COLORS[index % COLORS.length],
      };
    });
  };

  getTickFormatter = () => {
    let tickFormatter;
    const { xAxis } = this.props;
    if (xAxis === X_AXIS_WALL) {
      tickFormatter = (timestamp) => Utils.formatTimestamp(timestamp, 'HH:MM:ss');
    }
    return tickFormatter;
  };

  render() {
    const { metrics, showDot, runUuids, chartType } = this.props;
    const showBarChart = chartType === CHART_TYPE_BAR;
    const data = showBarChart ? this.getDataForBarChart() : this.getDataForLineChart();
    const isStep = this.props.xAxis === X_AXIS_STEP;
    return (
      <ResponsiveContainer width='100%' aspect={1.55}>
        {showBarChart ? (
          <BarChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
            <Tooltip isAnimationActive={false} labelStyle={{ display: 'none' }} />
            <XAxis dataKey='metricKey' />
            <CartesianGrid strokeDasharray='3 3' />
            <Legend verticalAlign='bottom' payload={this.getLegendPayload('rect')} />
            <YAxis />
            {runUuids.map((runUuid, idx) => (
              <Bar
                dataKey={runUuid}
                key={runUuid}
                name={runUuid}
                isAnimationActive={false}
                fill={COLORS[idx % COLORS.length]}
              />
            ))}
          </BarChart>
        ) : (
          <LineChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
            <XAxis
              dataKey={isStep ? 'step' : 'timestamp'}
              type='number'
              name='Time'
              domain={['auto', 'auto']}
              tickFormatter={this.getTickFormatter()}
            />
            <YAxis />
            <Tooltip isAnimationActive={false} labelStyle={{ display: 'none' }} />
            <CartesianGrid strokeDasharray='3 3' />
            <Legend verticalAlign='bottom' payload={this.getLegendPayload('line')} />
            {metrics.map((metric, index) => {
              const { runUuid, metricKey } = metric;
              const id = this.getIdForRunMetricData(runUuid, metricKey);
              return (
                <Line
                  dataKey={id}
                  type='linear'
                  key={id}
                  name={id}
                  isAnimationActive={false}
                  connectNulls
                  stroke={COLORS[index % COLORS.length]}
                  strokeWidth={2}
                  dot={showDot}
                />
              );
            })}
          </LineChart>
        )}
      </ResponsiveContainer>
    );
  }
}

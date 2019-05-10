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
import Plot from 'react-plotly.js';

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
    isComparing: PropTypes.bool.isRequired,
    yAxisLogScale: PropTypes.bool.isRequired,
    lineSmoothness: PropTypes.number,
  };

  static MAX_RUN_NAME_DISPLAY_LENGTH = 36;

  getLegend = (metricKey, runName) =>
    `${metricKey}, ` + (this.props.isComparing ? runName.slice(0, 8) : '');

  parseTimestamp = (timestampStr, baseTimestamp, xAxis) => {
    const timestamp = Number.parseFloat(timestampStr);
    if (xAxis === X_AXIS_RELATIVE) {
      return (timestamp - baseTimestamp) / 1000;
    }
    return Utils.formatTimestamp(timestamp, 'HH:MM:ss');
  };

  getPlotPropsForLineChart = () => {
    const { metrics, xAxis, showDot, yAxisLogScale, lineSmoothness } = this.props;
    const data = metrics.map((metric) => {
      const { metricKey, runName, history } = metric;
      const baseTimestamp = Number.parseFloat(history[0] && history[0].timestamp);
      const isSingleHistory = history.length === 0;
      return {
        name: this.getLegend(metricKey, runName),
        x: history.map((entry) =>
          xAxis === X_AXIS_STEP
            ? Number.parseInt(entry.step, 10)
            : this.parseTimestamp(entry.timestamp, baseTimestamp, xAxis)
        ),
        y: history.map((entry) => entry.value),
        type: 'scatter',
        mode: isSingleHistory ? 'markers' : (showDot ? 'lines+markers' : 'lines'),

        line: { shape: 'spline', smoothing: lineSmoothness },
      };
    });
    const props = { data };
    if (yAxisLogScale) {
      props.layout = {
        yaxis: { type: 'log', autorange: true },
      };
    }
    return props;
  }

  getDataForBarChart = () => {
    /* eslint-disable no-param-reassign */
    const { runUuids, yAxisLogScale } = this.props;
    const runMap = {};
    const metricsMap = this.props.metrics.reduce((map, metric) => {
      const { runUuid, runName, metricKey, history } = metric;
      runMap[runUuid] = runName;
      const value = history[0] && history[0].value;
      if (!map[metricKey]) {
        map[metricKey] = {
          metricKey,
          [runUuid]: { runName, value },
        };
      } else {
        map[metricKey][runUuid] = { runName, value };
      }
      return map;
    }, {});
    // TODO(Zangr) ^^ remove this map
    const sortedMetrics = _.sortBy(Object.values(metricsMap), 'metricKey');
    const sortedMetricKeys = sortedMetrics.map((m) => m.metricKey);
    console.log(sortedMetrics);
    const data =  runUuids.map((runUuid) => ({
      name: runMap[runUuid].slice(0, 8),
      x: sortedMetricKeys,
      y: sortedMetrics.map((m) => m[runUuid] && m[runUuid].value),
      type: 'bar',
    }));
    const layout = { barmode: 'group' };
    const props = { data, layout };
    if (yAxisLogScale) {
      props.layout.yaxis = { type: 'log', autorange: true };
    }
    return props;
  };

  render() {
    const plotProps = this.props.chartType === CHART_TYPE_BAR
      ? this.getDataForBarChart()
      : this.getPlotPropsForLineChart();
    return (
      <div className='metrics-plot-view-container'>
        <Plot
          layout={{ autosize: true }}
          useResizeHandler
          style={{width: '100%', height: '100%'}}
          {...plotProps}
        />
      </div>
    );
  }
}

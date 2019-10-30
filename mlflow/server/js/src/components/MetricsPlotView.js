import React from 'react';
import Utils from '../utils/Utils';
import _ from 'lodash';
import PropTypes from 'prop-types';
import { X_AXIS_STEP, X_AXIS_RELATIVE } from './MetricsPlotControls';
import { CHART_TYPE_BAR } from './MetricsPlotPanel';
import Plot from 'react-plotly.js';

const MAX_RUN_NAME_DISPLAY_LENGTH = 36;

export class MetricsPlotView extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(String).isRequired,
    runDisplayNames: PropTypes.arrayOf(String).isRequired,
    metrics: PropTypes.arrayOf(Object).isRequired,
    xAxis: PropTypes.string.isRequired,
    metricKeys: PropTypes.arrayOf(String).isRequired,
    // Whether or not to show point markers on the line chart
    showPoint: PropTypes.bool.isRequired,
    chartType: PropTypes.string.isRequired,
    isComparing: PropTypes.bool.isRequired,
    yAxisLogScale: PropTypes.bool.isRequired,
    lineSmoothness: PropTypes.number,
  };

  static getLineLegend = (metricKey, runDisplayName, isComparing) => {
    let legend = metricKey;
    if (isComparing) {
      legend += `, ${Utils.truncateString(runDisplayName, MAX_RUN_NAME_DISPLAY_LENGTH)}`;
    }
    return legend;
  };

  static parseTimestamp = (timestamp, history, xAxis) => {
    if (xAxis === X_AXIS_RELATIVE) {
      const minTimestamp = _.minBy(history, 'timestamp').timestamp;
      return (timestamp - minTimestamp) / 1000;
    }
    return Utils.formatTimestamp(timestamp);
  };

  getPlotPropsForLineChart = () => {
    const { metrics, xAxis, showPoint, yAxisLogScale, lineSmoothness, isComparing } = this.props;
    const data = metrics.map((metric) => {
      const { metricKey, runDisplayName, history } = metric;
      const isSingleHistory = history.length === 0;
      return {
        name: MetricsPlotView.getLineLegend(metricKey, runDisplayName, isComparing),
        x: history.map((entry) => {
          if (xAxis === X_AXIS_STEP) {
            return entry.step;
          }
          return MetricsPlotView.parseTimestamp(entry.timestamp, history, xAxis);
        }),
        y: history.map((entry) => entry.value),
        type: 'scatter',
        mode: isSingleHistory ? 'markers' : 'lines+markers',
        line: { shape: 'spline', smoothing: lineSmoothness },
        marker: {opacity: isSingleHistory || showPoint ? 1 : 0 }
      };
    });
    const props = { data };
    if (yAxisLogScale) {
      props.layout = {
        yaxis: { type: 'log', autorange: true },
      };
    }
    return props;
  };

  getPlotPropsForBarChart = () => {
    /* eslint-disable no-param-reassign */
    const { runUuids, runDisplayNames, yAxisLogScale } = this.props;

    // A reverse lookup of `metricKey: { runUuid: value, metricKey }`
    const historyByMetricKey = this.props.metrics.reduce((map, metric) => {
      const { runUuid, metricKey, history } = metric;
      const value = history[0] && history[0].value;
      if (!map[metricKey]) {
        map[metricKey] = { metricKey, [runUuid]: value };
      } else {
        map[metricKey][runUuid] = value;
      }
      return map;
    }, {});

    const arrayOfHistorySortedByMetricKey = _.sortBy(
      Object.values(historyByMetricKey),
      'metricKey',
    );

    const sortedMetricKeys = arrayOfHistorySortedByMetricKey.map((history) => history.metricKey);

    const data = runUuids.map((runUuid, i) => ({
      name: Utils.truncateString(runDisplayNames[i], MAX_RUN_NAME_DISPLAY_LENGTH),
      x: sortedMetricKeys,
      y: arrayOfHistorySortedByMetricKey.map((history) => history[runUuid]),
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
    const plotProps =
      this.props.chartType === CHART_TYPE_BAR
        ? this.getPlotPropsForBarChart()
        : this.getPlotPropsForLineChart();
    return (
      <div className='metrics-plot-view-container'>
        <Plot
          {...plotProps}
          useResizeHandler
          style={{ width: '100%', height: '100%' }}
          layout={{ ...plotProps.layout, ...{ autosize: true } }}
          config={{
            displaylogo: false,
            scrollZoom: true,
            modeBarButtonsToRemove: ['sendDataToCloud'],
          }}
        />
      </div>
    );
  }
}

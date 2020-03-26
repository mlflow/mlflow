import React from 'react';
import Utils from '../../common/utils/Utils';
import _ from 'lodash';
import PropTypes from 'prop-types';
import { X_AXIS_STEP, X_AXIS_RELATIVE } from './MetricsPlotControls';
import { CHART_TYPE_BAR } from './MetricsPlotPanel';
import Plot from '../../../node_modules/react-plotly.js/react-plotly';

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
    lineSmoothness: PropTypes.number,
    extraLayout: PropTypes.object,
    onLayoutChange: PropTypes.func.isRequired,
    onClick: PropTypes.func.isRequired,
    onLegendClick: PropTypes.func.isRequired,
    onLegendDoubleClick: PropTypes.func.isRequired,
    deselectedCurves: PropTypes.arrayOf(String).isRequired,
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
    const { metrics, xAxis, showPoint, lineSmoothness, isComparing,
      deselectedCurves } = this.props;
    const deselectedCurvesSet = new Set(deselectedCurves);
    const data = metrics.map((metric) => {
      const { metricKey, runDisplayName, history, runUuid } = metric;
      const isSingleHistory = history.length === 0;
      const visible = !deselectedCurvesSet.has(Utils.getCurveKey(runUuid, metricKey)) ?
          true : "legendonly";
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
        marker: {opacity: isSingleHistory || showPoint ? 1 : 0 },
        visible: visible,
        runId: runUuid,
        metricName: metricKey,
      };
    });
    const props = { data };
    props.layout = {
      ...props.layout,
      ...this.props.extraLayout,
    };
    return props;
  };

  getPlotPropsForBarChart = () => {
    /* eslint-disable no-param-reassign */
    const { runUuids, runDisplayNames, deselectedCurves } = this.props;

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
    const deselectedCurvesSet = new Set(deselectedCurves);
    const data = runUuids.map((runUuid, i) => {
      const visibility = deselectedCurvesSet.has(runUuid) ?
        { visible: 'legendonly' } : {};
      return {
        name: Utils.truncateString(runDisplayNames[i], MAX_RUN_NAME_DISPLAY_LENGTH),
        x: sortedMetricKeys,
        y: arrayOfHistorySortedByMetricKey.map((history) => history[runUuid]),
        type: 'bar',
        runId: runUuid,
        ...visibility,
      };
    });

    const layout = { barmode: 'group' };
    const props = { data, layout };
    props.layout = {
      ...props.layout,
      ...this.props.extraLayout,
    };
    return props;
  };

  render() {
    const { onLayoutChange, onClick, onLegendClick, onLegendDoubleClick } = this.props;
    const plotProps =
      this.props.chartType === CHART_TYPE_BAR
        ? this.getPlotPropsForBarChart()
        : this.getPlotPropsForLineChart();
    return (
      <div className='metrics-plot-view-container'>
        <Plot
          {...plotProps}
          useResizeHandler
          onRelayout={onLayoutChange}
          onClick={onClick}
          onLegendClick={onLegendClick}
          onLegendDoubleClick={onLegendDoubleClick}
          style={{ width: '100%', height: '100%' }}
          layout={_.cloneDeep(plotProps.layout)}
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

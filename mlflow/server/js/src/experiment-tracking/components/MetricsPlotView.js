import React from 'react';
import Utils from '../../common/utils/Utils';
import _ from 'lodash';
import PropTypes from 'prop-types';
import { X_AXIS_STEP, X_AXIS_RELATIVE, MAX_LINE_SMOOTHNESS } from './MetricsPlotControls';
import { CHART_TYPE_BAR } from './MetricsPlotPanel';
import Plot from 'react-plotly.js';

const MAX_RUN_NAME_DISPLAY_LENGTH = 24;

const EMA = (mArray, smoothingWeight) => {
  // If all elements in the set of metric values are constant, or if
  // the degree of smoothing is set to the minimum value, return the
  // original set of metric values
  if (smoothingWeight <= 1 || mArray.every((v) => v === mArray[0])) {
    return mArray;
  }

  const smoothness = smoothingWeight / (MAX_LINE_SMOOTHNESS + 1);
  const smoothedArray = [];
  let biasedElement = 0;
  for (let i = 0; i < mArray.length; i++) {
    if (!isNaN(mArray[i])) {
      biasedElement = biasedElement * smoothness + (1 - smoothness) * mArray[i];
      // To avoid biasing earlier elements toward smaller-than-accurate values, we divide
      // all elements by a `debiasedWeight` that asymptotically increases and approaches
      // 1 as the element index increases
      const debiasWeight = 1.0 - Math.pow(smoothness, i + 1);
      const debiasedElement = biasedElement / debiasWeight;
      smoothedArray.push(debiasedElement);
    } else {
      smoothedArray.push(mArray[i]);
    }
  }
  return smoothedArray;
};

export class MetricsPlotView extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    runDisplayNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    metrics: PropTypes.arrayOf(PropTypes.object).isRequired,
    xAxis: PropTypes.string.isRequired,
    metricKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
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
    deselectedCurves: PropTypes.arrayOf(PropTypes.string).isRequired,
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
    const { metrics, xAxis, showPoint, lineSmoothness, isComparing, deselectedCurves } = this.props;
    const deselectedCurvesSet = new Set(deselectedCurves);
    const data = metrics.map((metric) => {
      const { metricKey, runDisplayName, history, runUuid } = metric;
      const historyValues = history.map((entry) => entry.value);
      // For metrics with exactly one non-NaN item, we set `isSingleHistory` to `true` in order
      // to display the item as a point. For metrics with zero non-NaN items (i.e., empty metrics),
      // we also set `isSingleHistory` to `true` in order to populate the plot legend with a
      // point-style entry for each empty metric, although no data will be plotted for empty
      // metrics
      const isSingleHistory = historyValues.filter((value) => !isNaN(value)).length <= 1;
      const visible = !deselectedCurvesSet.has(Utils.getCurveKey(runUuid, metricKey))
        ? true
        : 'legendonly';
      return {
        name: MetricsPlotView.getLineLegend(metricKey, runDisplayName, isComparing),
        x: history.map((entry) => {
          if (xAxis === X_AXIS_STEP) {
            return entry.step;
          }
          return MetricsPlotView.parseTimestamp(entry.timestamp, history, xAxis);
        }),
        y: isSingleHistory ? historyValues : EMA(historyValues, lineSmoothness),
        text: historyValues.map((value) => (isNaN(value) ? value : value.toFixed(5))),
        type: 'scattergl',
        mode: isSingleHistory ? 'markers' : 'lines+markers',
        marker: { opacity: isSingleHistory || showPoint ? 1 : 0 },
        hovertemplate:
          isSingleHistory || lineSmoothness === 1 ? '%{y}' : 'Value: %{text}<br>Smoothed: %{y}',
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
      const visibility = deselectedCurvesSet.has(runUuid) ? { visible: 'legendonly' } : {};
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

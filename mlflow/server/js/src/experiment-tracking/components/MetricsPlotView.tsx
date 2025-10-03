/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import Utils from '../../common/utils/Utils';
import { cloneDeep, minBy, sortBy } from 'lodash';
import { saveAs } from 'file-saver';
import { X_AXIS_STEP, X_AXIS_RELATIVE, MAX_LINE_SMOOTHNESS } from './MetricsPlotControls';
import { CHART_TYPE_BAR, convertMetricsToCsv } from './MetricsPlotPanel';
import { LazyPlot } from './LazyPlot';
import { generateInfinityAnnotations } from '../utils/MetricsUtils';
import type { IntlShape } from 'react-intl';
import { injectIntl } from 'react-intl';

const MAX_RUN_NAME_DISPLAY_LENGTH = 24;
const EMA_THRESHOLD = 1;

export const EMA = (mArray: any, smoothingWeight: any) => {
  // If all elements in the set of metric values are constant, or if
  // the degree of smoothing is set to the minimum value, return the
  // original set of metric values
  if (smoothingWeight <= 1 || !mArray || mArray.length <= EMA_THRESHOLD || mArray.every((v: any) => v === mArray[0])) {
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

// To avoid pulling in plotly.js (unlazily) and / or using a separate package, just duplicating here
// Copied from https://github.com/plotly/plotly.js/blob/v2.5.1/src/fonts/ploticon.js#L100
const DISK_ICON = {
  width: 857.1,
  height: 1000,
  // eslint-disable-next-line max-len
  path: 'm214-7h429v214h-429v-214z m500 0h72v500q0 8-6 21t-11 20l-157 156q-5 6-19 12t-22 5v-232q0-22-15-38t-38-16h-322q-22 0-37 16t-16 38v232h-72v-714h72v232q0 22 16 38t37 16h465q22 0 38-16t15-38v-232z m-214 518v178q0 8-5 13t-13 5h-107q-7 0-13-5t-5-13v-178q0-8 5-13t13-5h107q7 0 13 5t5 13z m357-18v-518q0-22-15-38t-38-16h-750q-23 0-38 16t-16 38v750q0 22 16 38t38 16h517q23 0 50-12t42-26l156-157q16-15 27-42t11-49z',
  transform: 'matrix(1 0 0 -1 0 850)',
};

type MetricsPlotViewImplProps = {
  runUuids: string[];
  runDisplayNames: string[];
  metrics: any[];
  xAxis: string;
  metricKeys: string[];
  showPoint: boolean;
  chartType: string;
  isComparing: boolean;
  lineSmoothness?: number;
  extraLayout?: any;
  onLayoutChange: (...args: any[]) => any;
  onClick: (...args: any[]) => any;
  onLegendClick: (...args: any[]) => any;
  onLegendDoubleClick: (...args: any[]) => any;
  deselectedCurves: string[];
  intl?: any;
};

export class MetricsPlotViewImpl extends React.Component<MetricsPlotViewImplProps> {
  static getLineLegend = (metricKey: any, runDisplayName: any, isComparing: any) => {
    let legend = metricKey;
    if (isComparing) {
      legend += `, ${Utils.truncateString(runDisplayName, MAX_RUN_NAME_DISPLAY_LENGTH)}`;
    }
    return legend;
  };

  static getXValuesForLineChart(history: any, xAxisType: any, intl?: IntlShape) {
    if (history.length === 0) {
      return [];
    }
    switch (xAxisType) {
      case X_AXIS_STEP:
        return history.map(({ step }: any) => step);
      case X_AXIS_RELATIVE: {
        // @ts-expect-error TS(2339): Property 'timestamp' does not exist on type '{ toS... Remove this comment to see the full error message
        const { timestamp: minTimestamp } = minBy(history, 'timestamp');
        return history.map(({ timestamp }: any) => (timestamp - minTimestamp) / 1000);
      }
      default: // X_AXIS_WALL
        return history.map(({ timestamp }: any) => timestamp);
    }
  }

  /**
   * Regenerates annotations and shapes for infinity and NaN values.
   * Best called infrequently. Ideally should be called only when data input changes.
   */
  regenerateInfinityAnnotations = () => {
    const { metrics, xAxis, extraLayout } = this.props;
    const isYAxisLog = extraLayout?.yaxis?.type === 'log';
    const annotationData = {};

    metrics.forEach((metric) => {
      const { metricKey, history } = metric;
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      annotationData[metricKey] = generateInfinityAnnotations({
        xValues: (MetricsPlotView as any).getXValuesForLineChart(history, xAxis, this.props.intl),
        yValues: history.map((entry: any) => (typeof entry.value === 'number' ? entry.value : Number(entry.value))),
        isLogScale: isYAxisLog,
        stringFormatter: (value) => this.props.intl.formatMessage(value, { metricKey }),
      });
    });

    this.#annotationData = annotationData;
  };

  #annotationData = {};

  getPlotPropsForLineChart = () => {
    const { metrics, xAxis, showPoint, lineSmoothness, isComparing, deselectedCurves } = this.props;

    const deselectedCurvesSet = new Set(deselectedCurves);
    const shapes: any = [];
    const annotations: any = [];

    const data = metrics.map((metric) => {
      const { metricKey, runDisplayName, history, runUuid } = metric;
      const historyValues = history.map((entry: any) =>
        typeof entry.value === 'number' ? entry.value : Number(entry.value),
      );
      // For metrics with exactly one non-NaN item, we set `isSingleHistory` to `true` in order
      // to display the item as a point. For metrics with zero non-NaN items (i.e., empty metrics),
      // we also set `isSingleHistory` to `true` in order to populate the plot legend with a
      // point-style entry for each empty metric, although no data will be plotted for empty
      // metrics
      const isSingleHistory = historyValues.filter((value: any) => !isNaN(value)).length <= 1;

      const visible = !deselectedCurvesSet.has(Utils.getCurveKey(runUuid, metricKey)) ? true : 'legendonly';

      if (this.#annotationData && metricKey in this.#annotationData && visible === true) {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        shapes.push(...this.#annotationData[metricKey].shapes);
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        annotations.push(...this.#annotationData[metricKey].annotations);
      }

      return {
        name: (MetricsPlotView as any).getLineLegend(metricKey, runDisplayName, isComparing),
        x: (MetricsPlotView as any).getXValuesForLineChart(history, xAxis),
        y: (isSingleHistory ? historyValues : EMA(historyValues, lineSmoothness)).map((entry: any) =>
          !isFinite(entry) ? NaN : entry,
        ),
        text: historyValues.map((value: any) => (isNaN(value) ? value : value.toFixed(5))),
        type: 'scattergl',
        mode: isSingleHistory ? 'markers' : 'lines+markers',
        marker: { opacity: isSingleHistory || showPoint ? 1 : 0 },
        hovertemplate: isSingleHistory || lineSmoothness === 1 ? '%{y}' : 'Value: %{text}<br>Smoothed: %{y}',
        visible: visible,
        runId: runUuid,
        metricName: metricKey,
      };
    });
    const props = { data };

    (props as any).layout = {
      ...(props as any).layout,
      ...this.props.extraLayout,
      shapes,
      annotations,
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

    const arrayOfHistorySortedByMetricKey = sortBy(Object.values(historyByMetricKey), 'metricKey');

    const sortedMetricKeys = arrayOfHistorySortedByMetricKey.map((history) => (history as any).metricKey);
    const deselectedCurvesSet = new Set(deselectedCurves);
    const data = runUuids.map((runUuid, i) => {
      const visibility = deselectedCurvesSet.has(runUuid) ? { visible: 'legendonly' } : {};
      return {
        name: Utils.truncateString(runDisplayNames[i], MAX_RUN_NAME_DISPLAY_LENGTH),
        x: sortedMetricKeys,
        y: arrayOfHistorySortedByMetricKey.map((history) => (history as any)[runUuid]),
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

  componentDidMount() {
    this.regenerateInfinityAnnotations();
  }

  componentDidUpdate() {
    /**
     * TODO: make sure that annotations are regenereated only when data changes.
     * In fact, all internal recalculations should be done only then.
     */
    this.regenerateInfinityAnnotations();
  }

  render() {
    const { onLayoutChange, onClick, onLegendClick, onLegendDoubleClick } = this.props;
    const plotProps =
      this.props.chartType === CHART_TYPE_BAR ? this.getPlotPropsForBarChart() : this.getPlotPropsForLineChart();

    return (
      <div className="metrics-plot-view-container">
        <LazyPlot
          {...plotProps}
          useResizeHandler
          onRelayout={onLayoutChange}
          onClick={onClick}
          onLegendClick={onLegendClick}
          onLegendDoubleClick={onLegendDoubleClick}
          style={{ width: '100%', height: '100%' }}
          layout={cloneDeep((plotProps as any).layout)}
          config={{
            displaylogo: false,
            scrollZoom: true,
            modeBarButtonsToRemove: ['sendDataToCloud'],
            modeBarButtonsToAdd: [
              {
                name: 'Download plot data as CSV',
                icon: DISK_ICON,
                click: () => {
                  const csv = convertMetricsToCsv(this.props.metrics);
                  const blob = new Blob([csv], { type: 'application/csv;charset=utf-8' });
                  saveAs(blob, 'metrics.csv');
                },
              },
            ],
          }}
        />
      </div>
    );
  }
}

// @ts-expect-error TS(2769): No overload matches this call.
export const MetricsPlotView = injectIntl(MetricsPlotViewImpl);

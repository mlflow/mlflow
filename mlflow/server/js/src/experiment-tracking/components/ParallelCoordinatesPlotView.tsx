/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { connect } from 'react-redux';
import { findLast, invert, isEqual, max, min, range, sortBy, uniq } from 'lodash';
import { LazyPlot } from './LazyPlot';

const AXIS_LABEL_CLS = '.pcp-plot .parcoords .y-axis .axis-heading .axis-title';
export const UNKNOWN_TERM = 'unknown';

type ParallelCoordinatesPlotViewProps = {
  runUuids: string[];
  paramKeys: string[];
  metricKeys: string[];
  paramDimensions: any[];
  metricDimensions: any[];
};

type ParallelCoordinatesPlotViewState = any;

export class ParallelCoordinatesPlotView extends React.Component<
  ParallelCoordinatesPlotViewProps,
  ParallelCoordinatesPlotViewState
> {
  state = {
    // Current sequence of all axes, both parameters and metrics.
    sequence: [...this.props.paramKeys, ...this.props.metricKeys],
  };

  static getDerivedStateFromProps(props: any, state: any) {
    const keysFromProps = [...props.paramKeys, ...props.metricKeys];
    const keysFromState = state.sequence;
    if (!isEqual(sortBy(keysFromProps), sortBy(keysFromState))) {
      return { sequence: keysFromProps };
    }
    return null;
  }

  getData() {
    const { sequence } = this.state;
    const { paramDimensions, metricDimensions, metricKeys } = this.props;
    const lastMetricKey = this.findLastKeyFromState(metricKeys);
    const lastMetricDimension = this.props.metricDimensions.find((d) => d.label === lastMetricKey);
    const colorScaleConfigs = ParallelCoordinatesPlotView.getColorScaleConfigsForDimension(lastMetricDimension);
    // This make sure axis order consistency across renders.
    const orderedDimensions = ParallelCoordinatesPlotView.getDimensionsOrderedBySequence(
      [...paramDimensions, ...metricDimensions],
      sequence,
    );
    return [
      {
        type: 'parcoords',
        line: { ...colorScaleConfigs },
        dimensions: orderedDimensions,
      },
    ];
  }

  static getDimensionsOrderedBySequence(dimensions: any, sequence: any) {
    return sortBy(dimensions, [(dimension) => sequence.indexOf(dimension.label)]);
  }

  static getLabelElementsFromDom = () => Array.from(document.querySelectorAll(AXIS_LABEL_CLS));

  findLastKeyFromState(keys: any) {
    const { sequence } = this.state;
    const keySet = new Set(keys);
    return findLast(sequence, (key) => keySet.has(key));
  }

  static getColorScaleConfigsForDimension(dimension: any) {
    if (!dimension) return null;
    const cmin = min(dimension.values);
    const cmax = max(dimension.values);
    return {
      showscale: true,
      colorscale: 'Jet',
      cmin,
      cmax,
      color: dimension.values,
    };
  }

  // Update styles(green & bold) for metric axes.
  // Note(Zangr) 2019-6-25 this is needed because there is no per axis label setting available. This
  // needs to be called every time chart updates. More information about currently available label
  // setting here: https://plot.ly/javascript/reference/#parcoords-labelfont
  updateMetricAxisLabelStyle = () => {
    /* eslint-disable no-param-reassign */
    const metricsKeySet = new Set(this.props.metricKeys);
    // TODO(Zangr) 2019-06-20 This assumes name uniqueness across params & metrics. Find a way to
    // make it more deterministic. Ex. Add add different data attributes to indicate axis kind.
    ParallelCoordinatesPlotView.getLabelElementsFromDom()
      .filter((el) => metricsKeySet.has(el.innerHTML))
      .forEach((el) => {
        (el as any).style.fill = 'green';
        (el as any).style.fontWeight = 'bold';
      });
  };

  maybeUpdateStateForColorScale = (currentSequenceFromPlotly: any) => {
    const rightmostMetricKeyFromState = this.findLastKeyFromState(this.props.metricKeys);
    const metricsKeySet = new Set(this.props.metricKeys);
    const rightmostMetricKeyFromPlotly = findLast(currentSequenceFromPlotly, (key) => metricsKeySet.has(key));
    // Currently we always render color scale based on the rightmost metric axis, so if that changes
    // we need to setState with the new axes sequence to trigger a rerender.
    if (rightmostMetricKeyFromState !== rightmostMetricKeyFromPlotly) {
      this.setState({ sequence: currentSequenceFromPlotly });
    }
  };

  handlePlotUpdate = ({ data: [{ dimensions }] }: any) => {
    this.updateMetricAxisLabelStyle();
    this.maybeUpdateStateForColorScale(dimensions.map((d: any) => d.label));
  };

  render() {
    return (
      <LazyPlot
        layout={{ autosize: true, margin: { t: 50 } }}
        useResizeHandler
        css={styles.plot}
        data={this.getData()}
        onUpdate={this.handlePlotUpdate}
        className="pcp-plot"
        config={{ displayModeBar: false }}
      />
    );
  }
}

export const generateAttributesForCategoricalDimension = (labels: any) => {
  // Create a lookup from label to its own alphabetical sorted order.
  // Ex. ['A', 'B', 'C'] => { 'A': '0', 'B': '1', 'C': '2' }
  const sortedUniqLabels = uniq(labels).sort();

  // We always want the UNKNOWN_TERM to be at the top
  // of the chart which is end of the sorted label array
  // Ex. ['A', 'UNKNOWN_TERM', 'B'] => { 'A': '0', 'B': '1', 'UNKNOWN_TERM': '2' }
  let addUnknownTerm = false;
  const filteredSortedUniqLabels = sortedUniqLabels.filter((label) => {
    if (label === UNKNOWN_TERM) addUnknownTerm = true;
    return label !== UNKNOWN_TERM;
  });
  if (addUnknownTerm) {
    filteredSortedUniqLabels.push(UNKNOWN_TERM);
  }
  const labelToIndexStr = invert(filteredSortedUniqLabels);
  const attributes = {};

  // Values are assigned to their alphabetical sorted index number
  (attributes as any).values = labels.map((label: any) => Number(labelToIndexStr[label]));

  // Default to alphabetical order for categorical axis here. Ex. [0, 1, 2, 3 ...]
  (attributes as any).tickvals = range(filteredSortedUniqLabels.length);

  // Default to alphabetical order for categorical axis here. Ex. ['A', 'B', 'C', 'D' ...]
  (attributes as any).ticktext = filteredSortedUniqLabels.map((sortedUniqLabel) =>
    (sortedUniqLabel as any).substring(0, 10),
  );

  return attributes;
};

/**
 * Infer the type of data in a run. If all the values are numbers or castable to numbers, then
 * treat it as a number column.
 */
export const inferType = (key: any, runUuids: any, entryByRunUuid: any) => {
  for (let i = 0; i < runUuids.length; i++) {
    if (entryByRunUuid[runUuids[i]][key]) {
      const { value } = entryByRunUuid[runUuids[i]][key];
      if (typeof value === 'string' && isNaN(Number(value)) && value !== 'NaN') {
        return 'string';
      }
    }
  }
  return 'number';
};

export const createDimension = (key: any, runUuids: any, entryByRunUuid: any) => {
  let attributes = {};
  const dataType = inferType(key, runUuids, entryByRunUuid);
  if (dataType === 'string') {
    attributes = generateAttributesForCategoricalDimension(
      runUuids.map((runUuid: any) =>
        entryByRunUuid[runUuid][key] ? entryByRunUuid[runUuid][key].value : UNKNOWN_TERM,
      ),
    );
  } else {
    let maxValue = Number.MIN_SAFE_INTEGER;
    const values = runUuids.map((runUuid: any) => {
      if (entryByRunUuid[runUuid][key]) {
        const { value } = entryByRunUuid[runUuid][key];
        const numericValue = Number(value);
        if (maxValue < numericValue) maxValue = numericValue;
        return numericValue;
      }
      return UNKNOWN_TERM;
    });

    // For Numerical values, we take the max value of all the attribute
    // values and 0.01 to it so it is always at top of the graph.
    (attributes as any).values = values.map((value: any) => {
      if (value === UNKNOWN_TERM) return maxValue + 0.01;
      return value;
    });

    // For some reason, Plotly tries to plot these values with SI prefixes by default
    // Explicitly set to 5 fixed digits float here
    (attributes as any).tickformat = '.5f';
  }
  return {
    label: key,
    ...attributes,
  };
};

const styles = {
  plot: {
    width: '100%',
  },
};

const mapStateToProps = (state: any, ownProps: any) => {
  const { runUuids, paramKeys, metricKeys } = ownProps;
  const { latestMetricsByRunUuid, paramsByRunUuid } = state.entities;
  const paramDimensions = paramKeys.map((paramKey: any) => createDimension(paramKey, runUuids, paramsByRunUuid));
  const metricDimensions = metricKeys.map((metricKey: any) =>
    createDimension(metricKey, runUuids, latestMetricsByRunUuid),
  );
  return { paramDimensions, metricDimensions };
};

export default connect(mapStateToProps)(ParallelCoordinatesPlotView);

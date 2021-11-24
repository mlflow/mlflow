import React from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import _ from 'lodash';
import { LazyPlot } from './LazyPlot';

const AXIS_LABEL_CLS = '.pcp-plot .parcoords .y-axis .axis-heading .axis-title';
export const UNKNOWN_TERM = 'unknown';

export class ParallelCoordinatesPlotView extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    paramKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    paramDimensions: PropTypes.arrayOf(PropTypes.object).isRequired,
    metricDimensions: PropTypes.arrayOf(PropTypes.object).isRequired,
  };

  state = {
    // Current sequence of all axes, both parameters and metrics.
    sequence: [...this.props.paramKeys, ...this.props.metricKeys],
  };

  static getDerivedStateFromProps(props, state) {
    const keysFromProps = [...props.paramKeys, ...props.metricKeys];
    const keysFromState = state.sequence;
    if (!_.isEqual(_.sortBy(keysFromProps), _.sortBy(keysFromState))) {
      return { sequence: keysFromProps };
    }
    return null;
  }

  getData() {
    const { sequence } = this.state;
    const { paramDimensions, metricDimensions, metricKeys } = this.props;
    const lastMetricKey = this.findLastKeyFromState(metricKeys);
    const lastMetricDimension = this.props.metricDimensions.find((d) => d.label === lastMetricKey);
    const colorScaleConfigs = ParallelCoordinatesPlotView.getColorScaleConfigsForDimension(
      lastMetricDimension,
    );
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

  static getDimensionsOrderedBySequence(dimensions, sequence) {
    return _.sortBy(dimensions, [(dimension) => sequence.indexOf(dimension.label)]);
  }

  static getLabelElementsFromDom = () => Array.from(document.querySelectorAll(AXIS_LABEL_CLS));

  findLastKeyFromState(keys) {
    const { sequence } = this.state;
    const keySet = new Set(keys);
    return _.findLast(sequence, (key) => keySet.has(key));
  }

  static getColorScaleConfigsForDimension(dimension) {
    if (!dimension) return null;
    const cmin = _.min(dimension.values);
    const cmax = _.max(dimension.values);
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
        el.style.fill = 'green';
        el.style.fontWeight = 'bold';
      });
  };

  maybeUpdateStateForColorScale = (currentSequenceFromPlotly) => {
    const rightmostMetricKeyFromState = this.findLastKeyFromState(this.props.metricKeys);
    const metricsKeySet = new Set(this.props.metricKeys);
    const rightmostMetricKeyFromPlotly = _.findLast(currentSequenceFromPlotly, (key) =>
      metricsKeySet.has(key),
    );
    // Currently we always render color scale based on the rightmost metric axis, so if that changes
    // we need to setState with the new axes sequence to trigger a rerender.
    if (rightmostMetricKeyFromState !== rightmostMetricKeyFromPlotly) {
      this.setState({ sequence: currentSequenceFromPlotly });
    }
  };

  handlePlotUpdate = ({ data: [{ dimensions }] }) => {
    this.updateMetricAxisLabelStyle();
    this.maybeUpdateStateForColorScale(dimensions.map((d) => d.label));
  };

  render() {
    return (
      <LazyPlot
        layout={{ autosize: true, margin: { t: 50 } }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
        data={this.getData()}
        onUpdate={this.handlePlotUpdate}
        className='pcp-plot'
        config={{ displayModeBar: false }}
      />
    );
  }
}

export const generateAttributesForCategoricalDimension = (labels) => {
  // Create a lookup from label to its own alphabetical sorted order.
  // Ex. ['A', 'B', 'C'] => { 'A': '0', 'B': '1', 'C': '2' }
  const sortedUniqLabels = _.uniq(labels).sort();

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
  const labelToIndexStr = _.invert(filteredSortedUniqLabels);
  const attributes = {};

  // Values are assigned to their alphabetical sorted index number
  attributes.values = labels.map((label) => Number(labelToIndexStr[label]));

  // Default to alphabetical order for categorical axis here. Ex. [0, 1, 2, 3 ...]
  attributes.tickvals = _.range(filteredSortedUniqLabels.length);

  // Default to alphabetical order for categorical axis here. Ex. ['A', 'B', 'C', 'D' ...]
  attributes.ticktext = filteredSortedUniqLabels.map((sortedUniqLabel) =>
    sortedUniqLabel.substring(0, 10),
  );

  return attributes;
};

/**
 * Infer the type of data in a run. If all the values are numbers or castable to numbers, then
 * treat it as a number column.
 */
export const inferType = (key, runUuids, entryByRunUuid) => {
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

export const createDimension = (key, runUuids, entryByRunUuid) => {
  let attributes = {};
  const dataType = inferType(key, runUuids, entryByRunUuid);
  if (dataType === 'string') {
    attributes = generateAttributesForCategoricalDimension(
      runUuids.map((runUuid) =>
        entryByRunUuid[runUuid][key] ? entryByRunUuid[runUuid][key].value : UNKNOWN_TERM,
      ),
    );
  } else {
    let maxValue = Number.MIN_SAFE_INTEGER;
    const values = runUuids.map((runUuid) => {
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
    attributes.values = values.map((value) => {
      if (value === UNKNOWN_TERM) return maxValue + 0.01;
      return value;
    });

    // For some reason, Plotly tries to plot these values with SI prefixes by default
    // Explicitly set to 5 fixed digits float here
    attributes.tickformat = '.5f';
  }
  return {
    label: key,
    ...attributes,
  };
};

const mapStateToProps = (state, ownProps) => {
  const { runUuids, paramKeys, metricKeys } = ownProps;
  const { latestMetricsByRunUuid, paramsByRunUuid } = state.entities;
  const paramDimensions = paramKeys.map((paramKey) =>
    createDimension(paramKey, runUuids, paramsByRunUuid),
  );
  const metricDimensions = metricKeys.map((metricKey) =>
    createDimension(metricKey, runUuids, latestMetricsByRunUuid),
  );
  return { paramDimensions, metricDimensions };
};

export default connect(mapStateToProps)(ParallelCoordinatesPlotView);

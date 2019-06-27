import React from 'react';
import { connect } from 'react-redux';
import Plot from 'react-plotly.js';
import PropTypes from 'prop-types';
import Utils from '../utils/Utils';
import _ from 'lodash';

const AXIS_LABEL_CLS = '.pcp-plot .parcoords .y-axis .axis-heading .axis-title';

export class ParallelCoordinatesPlotView extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(String).isRequired,
    paramKeys: PropTypes.arrayOf(String).isRequired,
    metricKeys: PropTypes.arrayOf(String).isRequired,
    paramDimensions: PropTypes.arrayOf(Object).isRequired,
    metricDimensions: PropTypes.arrayOf(Object).isRequired,
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
    const rightmostMetricKeyFromPlotly = _.findLast(
      currentSequenceFromPlotly,
      (key) => metricsKeySet.has(key),
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
      <Plot
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
  const labelToIndexStr = _.invert(sortedUniqLabels);
  const attributes = {};

  // Values are assigned to their alphabetical sorted index number
  attributes.values = labels.map((label) => Number(labelToIndexStr[label]));

  // Default to alphabetical order for categorical axis here. Ex. [0, 1, 2, 3 ...]
  attributes.tickvals = _.range(sortedUniqLabels.length);

  // Default to alphabetical order for categorical axis here. Ex. ['A', 'B', 'C', 'D' ...]
  attributes.ticktext = sortedUniqLabels;

  return attributes;
};

// Infer type with the first run's value
export const inferType = (key, runUuids, entryByRunUuid) => {
  return isNaN(entryByRunUuid[runUuids[0]][key].value) ? 'string' : 'number';
};

export const createDimension = (key, runUuids, entryByRunUuid) => {
  let attributes = {};
  const dataType = inferType(key, runUuids, entryByRunUuid);
  if (dataType === 'string') {
    attributes = generateAttributesForCategoricalDimension(
      runUuids.map((runUuid) => entryByRunUuid[runUuid][key].value),
    );
  } else {
    attributes.values = runUuids.map((runUuid) => {
      const { value } = entryByRunUuid[runUuid][key];
      return isNaN(value) ? 0 : Number(Utils.formatMetric(value)); // Default NaN to zero here
    });
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

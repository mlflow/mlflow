import React from 'react';
import { connect } from 'react-redux';
import Plot from 'react-plotly.js';
import PropTypes from 'prop-types';
import _ from 'lodash';

const AXIS_LABEL_CLS = '.pcp-plot .parcoords .y-axis .axis-heading .axis-title';
const DIM_TYPE_PARAM = 'param';
const DIM_TYPE_METRIC = 'metric';

export class ParallelCoordinatesPlotView extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(String).isRequired,
    paramKeys: PropTypes.arrayOf(String).isRequired,
    metricKeys: PropTypes.arrayOf(String).isRequired,
    paramDimensions: PropTypes.arrayOf(Object).isRequired,
    metricDimensions: PropTypes.arrayOf(Object).isRequired,
  };

  state = {
    dimensions: [...this.props.paramDimensions, ...this.props.metricDimensions],
  };

  static getDerivedStateFromProps(props, state) {
    const dimensionsFromState = state.dimensions.map((dimension) => dimension.label);
    const dimensionsFromProps = [...props.paramKeys, ...props.metricKeys];
    if (dimensionsFromState.sort().join() !== dimensionsFromProps.sort().join()) {
      console.log('set derived state = ', [...props.paramKeys, ...props.metricKeys].join(','));
      return { dimensions: [...props.paramDimensions, ...props.metricDimensions] };
    }
    return null;
  }

  getData() {
    const { dimensions } = this.state;
    const lastMetricDimension = this.findLastMetricDimensionFromState();
    const colorScaleConfigs =
      ParallelCoordinatesPlotView.getColorScaleConfigsForDimension(lastMetricDimension);
    return [
      {
        type: 'parcoords',
        line: {
          ...colorScaleConfigs,
        },
        dimensions: _.clone(dimensions), // dimensions will be mutated by plotly
      },
    ];
  };

  static getAllAxisLabelElementsFromDom = () =>
    Array.from(document.querySelectorAll(AXIS_LABEL_CLS));

  static getSequencedAxisLabelsFromDom = () =>
    ParallelCoordinatesPlotView.getAllAxisLabelElementsFromDom().map((el) => el.innerHTML);

  findLastMetricDimensionFromState() {
    const { dimensions } = this.state;
    const metricsKeySet = new Set(this.props.metricKeys);
    return _.findLast(dimensions, (dimension) => metricsKeySet.has(dimension.label));
  }

  findLastMetricFromDom() {
    // The source of truth of current label element is in the dom
    const dimensionLabels = ParallelCoordinatesPlotView.getSequencedAxisLabelsFromDom();
    const metricsKeySet = new Set(this.props.metricKeys);
    return _.findLast(dimensionLabels, (label) => metricsKeySet.has(label));
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
    }
  }

  updateMetricAxisLabels = () => {
    const { metricDimensions } = this.props;
    const metricsLabelSet = new Set(metricDimensions.map((dimension) => dimension.label));
    const axisLabelElements = document.querySelectorAll(AXIS_LABEL_CLS);
    // Note(Zangr) 2019-06-20 This assumes name uniqueness across params & metrics. Find a way to
    // make it more deterministic. Ex. Add add different data attributes to indicate axis kind.
    Array.from(axisLabelElements)
      .filter((el) => metricsLabelSet.has(el.innerHTML))
      .forEach((el) => {
        el.style.fill = 'green';
        el.style.fontWeight = 'bold';
      });
  };

  maybeUpdateStateForColorScale = () => {
    const lastMetricDimensionFromState = this.findLastMetricDimensionFromState();
    const lastMetricFromDom = this.findLastMetricFromDom();
    // If we found diff on the last(right most) metric dimension, set state with dimension sorted
    // based on current axis label order from DOM
    if (lastMetricDimensionFromState && lastMetricDimensionFromState.label !== lastMetricFromDom) {
      const labelSequence = ParallelCoordinatesPlotView.getSequencedAxisLabelsFromDom();
      const sortedDimensions = [...this.state.dimensions.sort((d1, d2) => {
        return labelSequence.indexOf(d1.label) - labelSequence.indexOf(d2.label);
      })];
      console.log('colorUpdate state = ', sortedDimensions.map(d => d.label).join());
      this.setState({ dimensions: sortedDimensions });
    }
  };

  handlePlotUpdate = () => {
    this.updateMetricAxisLabels();
    this.maybeUpdateStateForColorScale();
  };

  render() {
    console.log('render state = ', this.state.dimensions.map(d => d.label).join());
    return (
      <Plot
        layout={{ autosize: true }}
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

const mapStateToProps = (state, ownProps) => {
  const { runUuids, paramKeys, metricKeys } = ownProps;
  const { latestMetricsByRunUuid, paramsByRunUuid } = state.entities;
  const paramDimensions = paramKeys.map((paramKey) => ({
    label: paramKey,
    values: runUuids.map((runUuid) => {
      const { value } = paramsByRunUuid[runUuid][paramKey];
      return isNaN(value) ? value : Number(value);
    }),
    dimType: DIM_TYPE_PARAM,
  }));
  const metricDimensions = metricKeys.map((metricKey) => ({
    label: metricKey,
    values: runUuids.map((runUuid) => {
      const { value } = latestMetricsByRunUuid[runUuid][metricKey];
      return isNaN(value) ? value : Number(value);
    }),
    dimType: DIM_TYPE_METRIC,
  }));
  return { paramDimensions, metricDimensions };
};

export default connect(mapStateToProps)(ParallelCoordinatesPlotView);

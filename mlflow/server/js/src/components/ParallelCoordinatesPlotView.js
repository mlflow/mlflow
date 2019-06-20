import React from 'react';
import { connect } from 'react-redux';
import Plot from 'react-plotly.js';
import PropTypes from 'prop-types';
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

  getData() {
    const { paramDimensions, metricDimensions } = this.props;
    const dimensionToScale = metricDimensions[metricDimensions.length - 1];
    const colorScaleConfigs = ParallelCoordinatesPlotView.getColorScaleConfigs(dimensionToScale);
    return [
      {
        type: 'parcoords',
        line: {
          ...colorScaleConfigs,
        },
        dimensions: [...paramDimensions, ...metricDimensions],
      },
    ];
  };

  findDimensionToColorScale(metricDimensions) {
    const axisLabelElements = document.querySelectorAll(AXIS_LABEL_CLS);
    const { length } = axisLabelElements;
    const rightMostMetricLabel = length > 0 ? axisLabelElements[length - 1].innerHTML : undefined;
    if (rightMostMetricLabel) {
      return metricDimensions.find((d) => d.label === rightMostMetricLabel);
    }
    return null;
  }

  static getColorScaleConfigs(dimension) {
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
    console.log('updateAxisLabels');
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

  render() {
    return (
      <Plot
        layout={{ autosize: true }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
        data={this.getData()}
        onUpdate={this.updateMetricAxisLabels}
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
  }));
  const metricDimensions = metricKeys.map((metricKey) => ({
    label: metricKey,
    values: runUuids.map((runUuid) => {
      const { value } = latestMetricsByRunUuid[runUuid][metricKey];
      return isNaN(value) ? value : Number(value);
    }),
  }));
  return { paramDimensions, metricDimensions };
};

export default connect(mapStateToProps)(ParallelCoordinatesPlotView);

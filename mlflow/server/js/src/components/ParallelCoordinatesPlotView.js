import React from 'react';
import Plot from 'react-plotly.js';
import PropTypes from 'prop-types';
import rows from '../pcp.json';

export class ParallelCoordinatesPlotView extends React.Component {
  static propTypes = {
    metrics: PropTypes.arrayOf(Object).isRequired,
    mockMetrics: PropTypes.arrayOf(Object).isRequired,
  };

  getData() {
    return [
      {
        type: 'parcoords',
        line: {
          color: 'blue',
        },
        dimensions: this.props.metrics,
      },
    ];
  }

  // TODO(Zangr) remove mock after testing
  getMockData() {
    return [
      {
        type: 'parcoords',
        line: {
          showscale: true,
          reversescale: true,
          colorscale: 'Jet',
          cmin: -4000,
          cmax: -100,
          color: rows.map((row) => row.colorVal),
        },
        dimensions: this.props.mockMetrics,
      },
    ];
  }

  render() {
    return (
      <Plot
        layout={{ autosize: true }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
        data={this.getMockData()}
      />
    );
  }
}

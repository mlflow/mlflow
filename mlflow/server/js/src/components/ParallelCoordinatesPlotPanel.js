import React from 'react';
import { connect } from 'react-redux';
import _ from 'lodash';
import PropTypes from 'prop-types';
import { ParallelCoordinatesPlotView } from './ParallelCoordinatesPlotView';
import { ParallelCoordinatesPlotControl } from './ParallelCoordinatesPlotControl';
import rows from '../pcp.json';

import './ParallelCoordinatesPlotPanel.css';

class ParallelCoordinatesPlotPanel extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(String).isRequired,
    sharedMetrics: PropTypes.arrayOf(String).isRequired,
    metrics: PropTypes.arrayOf(Object).isRequired,
  };

  state = {
    selectedMetricKeys: this.props.sharedMetrics,
  };

  handleMetricsSelectChange = (metricValues) => {
    this.setState({ selectedMetricKeys: metricValues });
  };

  render() {
    // TODO(rzang) remove mock after testing
    const { metrics } = this.props;
    const { selectedMetricKeys } = this.state;
    const selectedMetricKeysSet = new Set(selectedMetricKeys);
    const allMetricKeys = metrics.map((m) => ({
      title: m.label,
      value: m.label,
      key: m.label,
    }));
    return (
      <div className='parallel-coorinates-plot-panel'>
        <ParallelCoordinatesPlotControl
          allMetricKeys={allMetricKeys}
          selectedMetricKeys={selectedMetricKeys}
          handleMetricsSelectChange={this.handleMetricsSelectChange}
        />
        <ParallelCoordinatesPlotView
          metrics={metrics.filter((m) => selectedMetricKeysSet.has(m.label))}
        />
      </div>
    );
  }
}

// TODO(Zangr) remove mock after testing
const mockMetricRanges = {
  blockHeight: [32000, 227900],
  blockWidth: [0, 700000],
  cycMaterial: undefined,
  blockMaterial: [-1, 4],
  totalWeight: [134, 3154],
  assemblyPW: [9, 19984],
  HstW: [49000, 568000],
  minHW: [-28000, 196430],
  minWD: [98453, 501789],
  rfBlock: [1417, 107154],
};
// TODO(Zangr) remove mock after testing
const getMockMetrics = () =>
  Object.keys(mockMetricRanges).map((metricKey, index) => {
    const values = rows.map((row) => row[metricKey]);
    return {
      label: `metric_${index}`,
      values,
      range: mockMetricRanges[metricKey],
    };
  });

const mapStateToProps = (state, ownProps) => {
  const { runUuids } = ownProps;
  const { latestMetricsByRunUuid } = state.entities;
  const sharedMetrics = _.intersection(
    ...runUuids.map((runUuid) => Object.keys(latestMetricsByRunUuid[runUuid])),
  ).sort();
  // const metrics = sharedMetrics.map((metricKey) => {
  //   const values = runUuids.map((runUuid) => latestMetricsByRunUuid[runUuid][metricKey].value);
  //   return {
  //     label: metricKey,
  //     values,
  //     range: [_.min(values), _.max(values)],
  //   };
  // });

  // TODO(Zangr) remove mock after testing
  const metrics = getMockMetrics();
  return { metrics, sharedMetrics };
};

export default connect(mapStateToProps)(ParallelCoordinatesPlotPanel);

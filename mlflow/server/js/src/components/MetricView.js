import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import Utils from '../utils/Utils';
import './MetricView.css';
import { Experiment } from '../sdk/MlflowMessages';
import { getExperiment, getRunTags } from '../reducers/Reducers';
import BreadcrumbTitle from './BreadcrumbTitle';
import MetricsPlotPanel from './MetricsPlotPanel';

class MetricView extends Component {
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    runUuids: PropTypes.arrayOf(String).isRequired,
    runNames: PropTypes.arrayOf(String).isRequired,
    metricKey: PropTypes.string.isRequired,
  };

  render() {
    const { experiment, runUuids, runNames, metricKey } = this.props;
    return (
      <div className='MetricView'>
        <div className='header-container'>
          <BreadcrumbTitle
            experiment={experiment}
            runNames={runNames}
            runUuids={runUuids}
            title={<span>{metricKey}</span>}
          />
        </div>
        <MetricsPlotPanel {...{ runUuids, metricKey }}/>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { experimentId, runUuids } = ownProps;
  const experiment = experimentId !== null ? getExperiment(experimentId, state) : null;
  const runNames = runUuids.map((runUuid) => {
    const tags = getRunTags(runUuid, state);
    return Utils.getRunDisplayName(tags, runUuid);
  });

  return { experiment, runNames };
};

export default connect(mapStateToProps)(MetricView);

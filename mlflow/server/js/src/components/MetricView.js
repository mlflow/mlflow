import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import Utils from '../utils/Utils';
import './MetricView.css';
import { Experiment } from '../sdk/MlflowMessages';
import { getExperiment, getRunTags } from '../reducers/Reducers';
import BreadcrumbTitle from './BreadcrumbTitle';
import MetricsPlotPanel from './MetricsPlotPanel';
import { withRouter } from 'react-router-dom';

class MetricView extends Component {
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    runUuids: PropTypes.arrayOf(String).isRequired,
    runNames: PropTypes.arrayOf(String).isRequired,
    metricKey: PropTypes.string.isRequired,
    location: PropTypes.object.isRequired,
  };

  render() {
    const { experiment, runUuids, runNames, metricKey, location } = this.props;
    const plotMetricKeys = Utils.getPlotMetricKeysFromUrl(location.search);
    return (
      <div className='MetricView'>
        <div className='header-container'>
          <BreadcrumbTitle
            experiment={experiment}
            runNames={runNames}
            runUuids={runUuids}
            title={<span>{plotMetricKeys.length > 1 ? 'Metrics' : plotMetricKeys[0]}</span>}
          />
        </div>
        <MetricsPlotPanel {...{ runUuids, metricKey }} />
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

export default withRouter(connect(mapStateToProps)(MetricView));

import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import Utils from '../../common/utils/Utils';
import './MetricView.css';
import { Experiment } from '../sdk/MlflowMessages';
import { getExperiment, getRunTags } from '../reducers/Reducers';
import { BreadcrumbTitle } from './BreadcrumbTitle';
import MetricsPlotPanel from './MetricsPlotPanel';
import { withRouter } from 'react-router-dom';

export class MetricViewImpl extends Component {
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    runNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricKey: PropTypes.string.isRequired,
    location: PropTypes.object.isRequired,
  };

  render() {
    const { experiment, runUuids, runNames, metricKey, location } = this.props;
    const experimentId = experiment.experiment_id;
    const { selectedMetricKeys } = Utils.getMetricPlotStateFromUrl(location.search);
    return (
      <div className='MetricView'>
        <div className='header-container'>
          <BreadcrumbTitle
            experiment={experiment}
            runNames={runNames}
            runUuids={runUuids}
            title={<span>{selectedMetricKeys.length > 1 ? 'Metrics' : selectedMetricKeys[0]}</span>}
          />
        </div>
        <MetricsPlotPanel {...{ experimentId, runUuids, metricKey }} />
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

export const MetricView = withRouter(connect(mapStateToProps)(MetricViewImpl));

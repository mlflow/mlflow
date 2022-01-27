import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { FormattedMessage } from 'react-intl';
import Utils from '../../common/utils/Utils';
import './MetricView.css';
import { Experiment } from '../sdk/MlflowMessages';
import { getExperiment, getRunTags } from '../reducers/Reducers';
import MetricsPlotPanel from './MetricsPlotPanel';
import { withRouter, Link } from 'react-router-dom';
import { PageHeader } from '../../shared/building_blocks/PageHeader';
import Routes from '../routes';

export class MetricViewImpl extends Component {
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    runNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricKey: PropTypes.string.isRequired,
    location: PropTypes.object.isRequired,
  };

  getRunsLink() {
    const { experiment, runUuids, runNames } = this.props;
    const experimentId = experiment.getExperimentId();

    if (!runUuids || runUuids.length === 0) {
      return null;
    }

    return runUuids.length === 1 ? (
      <Link to={Routes.getRunPageRoute(experimentId, runUuids[0])}>{runNames[0]}</Link>
    ) : (
      <Link to={Routes.getCompareRunPageRoute(runUuids, experimentId)}>
        <FormattedMessage
          defaultMessage='Comparing {length} Runs'
          description='Breadcrumb title for metrics page when comparing multiple runs'
          values={{
            length: runUuids.length,
          }}
        />
      </Link>
    );
  }

  render() {
    const { experiment, runUuids, metricKey, location } = this.props;
    const experimentId = experiment.experiment_id;
    const { selectedMetricKeys } = Utils.getMetricPlotStateFromUrl(location.search);
    const title =
      selectedMetricKeys.length > 1 ? (
        <FormattedMessage defaultMessage='Metrics' description='Title for metrics page' />
      ) : (
        selectedMetricKeys[0]
      );
    const breadcrumbs = [
      <Link to={Routes.getExperimentPageRoute(experimentId)}>{experiment.getName()}</Link>,
      this.getRunsLink(),
      title,
    ];
    return (
      <div>
        <PageHeader title={title} breadcrumbs={breadcrumbs} />
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

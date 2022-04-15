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
    experiments: PropTypes.arrayOf(PropTypes.instanceOf(Experiment)).isRequired,
    experimentIds: PropTypes.arrayOf(PropTypes.string).isRequired,
    comparedExperimentIds: PropTypes.arrayOf(PropTypes.string),
    hasComparedExperimentsBefore: PropTypes.bool,
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    runNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricKey: PropTypes.string.isRequired,
    location: PropTypes.object.isRequired,
  };

  getCompareRunsPageText(numRuns, numExperiments) {
    return numExperiments > 1 ? (
      <FormattedMessage
        defaultMessage='Comparing {numRuns} Runs from {numExperiments} Experiments'
        // eslint-disable-next-line max-len
        description='Breadcrumb title for compare runs page with multiple experiments'
        values={{ numRuns, numExperiments }}
      />
    ) : (
      <FormattedMessage
        defaultMessage='Comparing {numRuns} Runs from 1 Experiment'
        description='Breadcrumb title for compare runs page with single experiment'
        values={{ numRuns }}
      />
    );
  }

  hasMultipleExperiments() {
    return this.props.experimentIds.length > 1;
  }

  getRunPageLink() {
    const { experimentIds, runUuids, runNames } = this.props;

    if (!runUuids || runUuids.length === 0) {
      return null;
    }

    if (runUuids.length === 1) {
      return <Link to={Routes.getRunPageRoute(experimentIds[0], runUuids[0])}>{runNames[0]}</Link>;
    }

    const text = this.getCompareRunsPageText(runUuids.length, experimentIds.length);
    return <Link to={Routes.getCompareRunPageRoute(runUuids, experimentIds)}>{text}</Link>;
  }

  getCompareExperimentsPageLinkText(numExperiments) {
    return (
      <FormattedMessage
        defaultMessage='Displaying Runs from {numExperiments} Experiments'
        // eslint-disable-next-line max-len
        description='Breadcrumb nav item to link to the compare-experiments page on compare runs page'
        values={{ numExperiments }}
      />
    );
  }

  getExperimentPageLink() {
    const {
      comparedExperimentIds,
      hasComparedExperimentsBefore,
      experimentIds,
      experiments,
    } = this.props;

    if (hasComparedExperimentsBefore) {
      const text = this.getCompareExperimentsPageLinkText(comparedExperimentIds.length);
      return <Link to={Routes.getCompareExperimentsPageRoute(comparedExperimentIds)}>{text}</Link>;
    }

    if (this.hasMultipleExperiments()) {
      const text = this.getCompareExperimentsPageLinkText(experimentIds.length);
      return <Link to={Routes.getCompareExperimentsPageRoute(experimentIds)}>{text}</Link>;
    }

    return (
      <Link to={Routes.getExperimentPageRoute(experimentIds[0])}>{experiments[0].getName()}</Link>
    );
  }

  render() {
    const { experimentIds, runUuids, metricKey, location } = this.props;
    const { selectedMetricKeys } = Utils.getMetricPlotStateFromUrl(location.search);
    const title =
      selectedMetricKeys.length > 1 ? (
        <FormattedMessage defaultMessage='Metrics' description='Title for metrics page' />
      ) : (
        selectedMetricKeys[0]
      );
    const breadcrumbs = [this.getExperimentPageLink(), this.getRunPageLink(), title];
    return (
      <div>
        <PageHeader title={title} breadcrumbs={breadcrumbs} />
        <MetricsPlotPanel {...{ experimentIds, runUuids, metricKey }} />
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { comparedExperimentIds, hasComparedExperimentsBefore } = state.compareExperiments;
  const { experimentIds, runUuids } = ownProps;
  const experiments =
    experimentIds !== null
      ? experimentIds.map((experimentId) => getExperiment(experimentId, state))
      : null;
  const runNames = runUuids.map((runUuid) => {
    const tags = getRunTags(runUuid, state);
    return Utils.getRunDisplayName(tags, runUuid);
  });
  return { experiments, runNames, comparedExperimentIds, hasComparedExperimentsBefore };
};

export const MetricView = withRouter(connect(mapStateToProps)(MetricViewImpl));

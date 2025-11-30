/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { Component } from 'react';
import { connect } from 'react-redux';
import { FormattedMessage } from 'react-intl';
import Utils from '../../common/utils/Utils';
import './MetricView.css';
import { getExperiment, getRunInfo } from '../reducers/Reducers';
import MetricsPlotPanel from './MetricsPlotPanel';
import { Link } from '../../common/utils/RoutingUtils';
import type { Location } from '../../common/utils/RoutingUtils';
import { PageHeader } from '../../shared/building_blocks/PageHeader';
import Routes from '../routes';
import { withRouterNext } from '../../common/utils/withRouterNext';

type MetricViewImplProps = {
  experiments: any[]; // TODO: PropTypes.instanceOf(Experiment)
  experimentIds: string[];
  comparedExperimentIds?: string[];
  hasComparedExperimentsBefore?: boolean;
  runUuids: string[];
  runNames: string[];
  metricKey: string;
  location: Location;
  navigate: (path: string, options?: { replace?: boolean }) => void;
};

export class MetricViewImpl extends Component<MetricViewImplProps> {
  getCompareRunsPageText(numRuns: any, numExperiments: any) {
    return numExperiments > 1 ? (
      <FormattedMessage
        defaultMessage="Comparing {numRuns} Runs from {numExperiments} Experiments"
        // eslint-disable-next-line max-len
        description="Breadcrumb title for compare runs page with multiple experiments"
        values={{ numRuns, numExperiments }}
      />
    ) : (
      <FormattedMessage
        defaultMessage="Comparing {numRuns} Runs from 1 Experiment"
        description="Breadcrumb title for compare runs page with single experiment"
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

  getCompareExperimentsPageLinkText(numExperiments: any) {
    return (
      <FormattedMessage
        defaultMessage="Displaying Runs from {numExperiments} Experiments"
        // eslint-disable-next-line max-len
        description="Breadcrumb nav item to link to the compare-experiments page on compare runs page"
        values={{ numExperiments }}
      />
    );
  }

  getExperimentPageLink() {
    const { comparedExperimentIds, hasComparedExperimentsBefore, experimentIds, experiments } = this.props;

    if (hasComparedExperimentsBefore && comparedExperimentIds) {
      const text = this.getCompareExperimentsPageLinkText(comparedExperimentIds.length);
      return <Link to={Routes.getCompareExperimentsPageRoute(comparedExperimentIds)}>{text}</Link>;
    }

    if (this.hasMultipleExperiments()) {
      const text = this.getCompareExperimentsPageLinkText(experimentIds.length);
      return <Link to={Routes.getCompareExperimentsPageRoute(experimentIds)}>{text}</Link>;
    }

    return <Link to={Routes.getExperimentPageRoute(experimentIds[0])}>{experiments[0].name}</Link>;
  }

  render() {
    const { experimentIds, runUuids, metricKey, location } = this.props;
    const { selectedMetricKeys } = Utils.getMetricPlotStateFromUrl(location.search);
    const title =
      selectedMetricKeys.length > 1 ? (
        <FormattedMessage defaultMessage="Metrics" description="Title for metrics page" />
      ) : (
        selectedMetricKeys[0]
      );
    const breadcrumbs = [this.getExperimentPageLink(), this.getRunPageLink()];
    return (
      <div>
        <PageHeader title={title} breadcrumbs={breadcrumbs} hideSpacer />
        <MetricsPlotPanel
          experimentIds={experimentIds}
          runUuids={runUuids}
          metricKey={metricKey}
          location={location}
          navigate={this.props.navigate}
        />
      </div>
    );
  }
}

const mapStateToProps = (state: any, ownProps: any) => {
  const { comparedExperimentIds, hasComparedExperimentsBefore } = state.compareExperiments;
  const { experimentIds, runUuids } = ownProps;
  const experiments =
    experimentIds !== null ? experimentIds.map((experimentId: any) => getExperiment(experimentId, state)) : null;
  const runNames = runUuids.map((runUuid: any) => {
    const runInfo = getRunInfo(runUuid, state);
    return Utils.getRunDisplayName(runInfo, runUuid);
  });
  return { experiments, runNames, comparedExperimentIds, hasComparedExperimentsBefore };
};

export const MetricView = withRouterNext(connect(mapStateToProps)(MetricViewImpl));

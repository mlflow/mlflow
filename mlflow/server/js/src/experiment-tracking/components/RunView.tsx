/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { IntlShape, injectIntl } from 'react-intl';
import {
  getExperiment,
  getParams,
  getRunInfo,
  getRunTags,
  getRunDatasets,
} from '../reducers/Reducers';
import { connect } from 'react-redux';
import './RunView.css';
import { getLatestMetrics } from '../reducers/MetricReducer';
import Utils from '../../common/utils/Utils';
import { RenameRunModal } from './modals/RenameRunModal';
import { NotificationInstance, withNotifications } from '@databricks/design-system';
import { setTagApi, deleteTagApi } from '../actions';
import { RunViewMetricCharts } from './run-page/RunViewMetricCharts';
import { RunViewHeader } from './run-page/RunViewHeader';
import { RunViewOverview } from './run-page/RunViewOverview';
import { RunViewArtifactTab } from './run-page/RunViewArtifactTab';
import { RunPageTabName } from '../constants';
import { shouldEnableDeepLearningUI } from '../../common/utils/FeatureUtils';
import { useRunViewActiveTab } from './run-page/useRunViewActiveTab';

type RunViewImplProps = {
  activeTab: RunPageTabName;
  runUuid: string;
  run: any;
  experiment: any; // TODO: PropTypes.instanceOf(Experiment)
  experimentId: string;
  comparedExperimentIds?: string[];
  hasComparedExperimentsBefore?: boolean;
  params: any;
  tags: any;
  latestMetrics: any;
  datasets: any;
  getMetricPagePath: (...args: any[]) => any;
  runDisplayName: string;
  runName: string;
  handleSetRunTag: (...args: any[]) => any;
  setTagApi: (...args: any[]) => any;
  deleteTagApi: (...args: any[]) => any;
  intl: IntlShape;
  notificationContextHolder: React.ReactElement;
  notificationAPI: NotificationInstance;
};

type RunViewImplState = {
  showRunRenameModal: boolean;
  mode: string;
};

export class RunViewImpl extends Component<RunViewImplProps, RunViewImplState> {
  state = {
    showRunRenameModal: false,
    mode: 'OVERVIEW',
  };

  componentDidMount() {
    const pageTitle = `${this.props.runDisplayName} - MLflow Run`;
    Utils.updatePageTitle(pageTitle);
  }

  handleRenameRunClick = () => {
    this.setState({ showRunRenameModal: true });
  };

  hideRenameRunModal = () => {
    this.setState({ showRunRenameModal: false });
  };

  renderActiveTab = () => {
    const { tags, activeTab, run, runUuid, latestMetrics } = this.props;

    if (!shouldEnableDeepLearningUI()) {
      return <RunViewOverview {...this.props} />;
    }

    if (activeTab === RunPageTabName.CHARTS) {
      return (
        <RunViewMetricCharts metricKeys={Object.keys(this.props.latestMetrics)} runInfo={run} />
      );
    }

    if (activeTab === RunPageTabName.ARTIFACTS) {
      return <RunViewArtifactTab runUuid={runUuid} runTags={tags} />;
    }

    return <RunViewOverview {...this.props} />;
  };

  render() {
    const { runUuid, activeTab } = this.props;

    // Certain tabs (e.g. artifacts) have dynamic height so we need to
    // apply special CSS treatment for them
    const usingFullHeightTab = [RunPageTabName.ARTIFACTS].includes(activeTab);

    return (
      <div
        className='RunView'
        css={{
          flex: '1',
          display: 'flex',
          flexDirection: 'column',
          overflow: usingFullHeightTab ? 'hidden' : 'unset',
        }}
      >
        <RenameRunModal
          // @ts-expect-error TS(2322): Type '{ runUuid: string; onClose: () => void; runN... Remove this comment to see the full error message
          runUuid={runUuid}
          onClose={this.hideRenameRunModal}
          runName={this.props.runName}
          isOpen={this.state.showRunRenameModal}
        />
        <RunViewHeader
          comparedExperimentIds={this.props.comparedExperimentIds}
          experiment={this.props.experiment}
          handleRenameRunClick={this.handleRenameRunClick}
          hasComparedExperimentsBefore={this.props.hasComparedExperimentsBefore}
          runUuid={this.props.runUuid}
          runDisplayName={this.props.runDisplayName}
        />
        {this.renderActiveTab()}
      </div>
    );
  }
}

const mapStateToProps = (state: any, ownProps: any) => {
  const { comparedExperimentIds, hasComparedExperimentsBefore } = state.compareExperiments;
  const { runUuid, experimentId } = ownProps;
  const run = getRunInfo(runUuid, state);
  const experiment = getExperiment(experimentId, state);
  const params = getParams(runUuid, state);
  const tags = getRunTags(runUuid, state);
  const latestMetrics = getLatestMetrics(runUuid, state);
  const datasets = getRunDatasets(runUuid, state);
  const runDisplayName = Utils.getRunDisplayName(run, runUuid);
  // @ts-expect-error TS(2554): Expected 1 arguments, but got 2.
  const runName = Utils.getRunName(run, runUuid);
  return {
    run,
    experiment,
    params,
    tags,
    latestMetrics,
    datasets,
    runDisplayName,
    runName,
    comparedExperimentIds,
    hasComparedExperimentsBefore,
  };
};
const mapDispatchToProps = { setTagApi, deleteTagApi };

const RunViewImplWithIntl = withNotifications(injectIntl(RunViewImpl));
/**
 * Class -> function retrofit component to enable hooks
 */
const RunViewWithActiveTab = (props: RunViewImplProps) => {
  const activeTab = useRunViewActiveTab();
  return <RunViewImplWithIntl {...props} activeTab={activeTab} />;
};
export const RunView = connect(mapStateToProps, mapDispatchToProps)(RunViewWithActiveTab);

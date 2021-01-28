import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { ArtifactView } from './ArtifactView';
import { Spinner } from '../../common/components/Spinner';
import { listArtifactsApi } from '../actions';
import { searchModelVersionsApi } from '../../model-registry/actions';
import { connect } from 'react-redux';
import { getArtifactRootUri } from '../reducers/Reducers';
import { MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL } from '../../model-registry/constants';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import Utils from '../../common/utils/Utils';
import { getUUID } from '../../common/utils/ActionUtils';
import './ArtifactPage.css';

export class ArtifactPageImpl extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    // An initially-selected artifact path to display in the artifact viewer, if any.
    // If no path is specified, defaults to selecting & displaying the contents of the
    // run's root artifact directory.
    initialSelectedArtifactPath: PropTypes.string,
    artifactRootUri: PropTypes.string.isRequired,
    apis: PropTypes.object.isRequired,
    listArtifactsApi: PropTypes.func.isRequired,
    searchModelVersionsApi: PropTypes.func.isRequired,
    runTags: PropTypes.object,
    modelVersions: PropTypes.arrayOf(PropTypes.object),
  };

  getFailedtoListArtifactsMsg = () => {
    return (
      <span>
        Unable to list artifacts stored under
        <code>{this.props.artifactRootUri}</code> for the current run. Please contact your tracking
        server administrator to notify them of this error, which can happen when the tracking server
        lacks permission to list artifacts under the current run's root artifact directory.
      </span>
    );
  };

  state = { activeNodeIsDirectory: false };

  searchRequestId = getUUID();

  listArtifactRequestIds = [getUUID()].concat(
    this.props.initialSelectedArtifactPath
      ? this.props.initialSelectedArtifactPath.split('/').map((s) => getUUID())
      : [],
  );

  pollModelVersionsForCurrentRun = () => {
    const { apis, runUuid } = this.props;
    const { activeNodeIsDirectory } = this.state;
    const searchRequest = apis[this.searchRequestId];
    if (activeNodeIsDirectory && !(searchRequest && searchRequest.active)) {
      this.props
        .searchModelVersionsApi({ run_id: runUuid }, this.searchRequestId)
        .catch(console.error);
    }
  };

  handleActiveNodeChange = (activeNodeIsDirectory) => {
    this.setState({ activeNodeIsDirectory });
  };

  pollArtifactsForCurrentRun = async () => {
    const { runUuid } = this.props;
    await this.props.listArtifactsApi(runUuid, undefined, this.listArtifactRequestIds[0]);
    if (this.props.initialSelectedArtifactPath) {
      const parts = this.props.initialSelectedArtifactPath.split('/');
      let pathSoFar = '';
      for (let i = 0; i < parts.length; i++) {
        pathSoFar += parts[i];
        // ML-12477: ListArtifacts API requests need to be sent and fulfilled for parent
        // directories before nested child directories, as our Reducers assume that parent
        // directories are listed before their children to construct the correct artifact tree.
        // Index i + 1 because listArtifactRequestIds[0] would have been used up by
        // root-level artifact API call above.
        // eslint-disable-next-line no-await-in-loop
        await this.props.listArtifactsApi(runUuid, pathSoFar, this.listArtifactRequestIds[i + 1]);
        pathSoFar += '/';
      }
    }
  };

  componentDidMount() {
    if (Utils.isModelRegistryEnabled()) {
      this.pollModelVersionsForCurrentRun();
      this.pollIntervalId = setInterval(this.pollModelVersionsForCurrentRun, POLL_INTERVAL);
    }
    this.pollArtifactsForCurrentRun();
  }

  componentWillUnmount() {
    if (Utils.isModelRegistryEnabled()) {
      clearInterval(this.pollIntervalId);
    }
  }

  renderArtifactView = (isLoading, shouldRenderError, requests) => {
    if (isLoading) {
      return <Spinner />;
    }
    if (shouldRenderError) {
      const failedReq = requests[0];
      if (failedReq && failedReq.error) {
        console.error(failedReq.error);
      }
      return (
        <div className='mlflow-artifact-error'>
          <div className='artifact-load-error-outer-container'>
            <div className='artifact-load-error-container'>
              <div>
                <div className='artifact-load-error-header'>Loading Artifacts Failed</div>
                <div className='artifact-load-error-info'>
                  <i className='far fa-times-circle artifact-load-error-icon' aria-hidden='true' />
                  {this.getFailedtoListArtifactsMsg()}
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }
    return <ArtifactView {...this.props} handleActiveNodeChange={this.handleActiveNodeChange} />;
  };

  render() {
    return (
      <RequestStateWrapper requestIds={this.listArtifactRequestIds}>
        {this.renderArtifactView}
      </RequestStateWrapper>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { runUuid } = ownProps;
  const { apis } = state;
  const artifactRootUri = getArtifactRootUri(runUuid, state);
  return { artifactRootUri, apis };
};

const mapDispatchToProps = {
  listArtifactsApi,
  searchModelVersionsApi,
};

export default connect(mapStateToProps, mapDispatchToProps)(ArtifactPageImpl);

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
    artifactRootUri: PropTypes.string.isRequired,
    apis: PropTypes.object.isRequired,
    listArtifactsApi: PropTypes.func.isRequired,
    searchModelVersionsApi: PropTypes.func.isRequired,
    artifactsAreLoading: PropTypes.bool.isRequired,
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

  listArtifactRequestId = getUUID();

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

  componentWillMount() {
    const { runUuid } = this.props;
    this.props.listArtifactsApi(runUuid, undefined, this.listArtifactRequestId);
  }

  componentDidMount() {
    if (Utils.isModelRegistryEnabled()) {
      this.pollModelVersionsForCurrentRun();
      this.pollIntervalId = setInterval(this.pollModelVersionsForCurrentRun, POLL_INTERVAL);
    }
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
      <RequestStateWrapper requestIds={[this.listArtifactRequestId]}>
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

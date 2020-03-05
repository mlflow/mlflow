import React, { Component } from 'react';
import PropTypes from 'prop-types';
import RequestStateWrapper from './RequestStateWrapper';
import { getExperimentApi, getRunApi, getUUID, listArtifactsApi, setTagApi } from '../Actions';
import { searchModelVersionsApi } from '../model-registry/actions';
import { connect } from 'react-redux';
import RunView from './RunView';
import Routes from '../Routes';
import Utils from '../utils/Utils';
import ErrorCodes from '../sdk/ErrorCodes';
import RunNotFoundView from './RunNotFoundView';

class RunPage extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    experimentId: PropTypes.number.isRequired,
    modelVersions: PropTypes.arrayOf(Object),
    getRunApi: PropTypes.func.isRequired,
    listArtifactsApi: PropTypes.func.isRequired,
    getExperimentApi: PropTypes.func.isRequired,
    searchModelVersionsApi: PropTypes.func.isRequired,
    setTagApi: PropTypes.func.isRequired,
  };

  getRunRequestId = getUUID();

  listArtifactRequestId = getUUID();

  getExperimentRequestId = getUUID();

  searchModelVersionsRequestId = getUUID();

  setTagRequestId = getUUID();

  componentWillMount() {
    const { experimentId, runUuid } = this.props;
    this.props.getRunApi(runUuid, this.getRunRequestId);
    this.props.listArtifactsApi(runUuid, undefined, this.listArtifactRequestId);
    this.props.getExperimentApi(experimentId, this.getExperimentRequestId);
    this.props.searchModelVersionsApi({ run_id: runUuid }, this.searchModelVersionsRequestId);
  }

  handleSetRunTag = (tagName, value) => {
    const { runUuid } = this.props;
    return this.props
      .setTagApi(runUuid, tagName, value, this.setTagRequestId)
      .then(() => getRunApi(runUuid, this.getRunRequestId));
  };

  render() {
    return (
      <div className='App-content'>
        <RequestStateWrapper
          requestIds={[
            this.getRunRequestId,
            this.listArtifactRequestId,
            this.getExperimentRequestId,
          ]}
        >
          {(isLoading, shouldRenderError, requests) => {
            if (shouldRenderError) {
              const getRunRequest = Utils.getRequestWithId(requests, this.getRunRequestId);
              if (getRunRequest.error.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST) {
                return <RunNotFoundView runId={this.props.runUuid}/>;
              }
              return null;
            }
            return <RunView
              runUuid={this.props.runUuid}
              getMetricPagePath={(key) =>
                Routes.getMetricPageRoute([this.props.runUuid], key, this.props.experimentId)
              }
              experimentId={this.props.experimentId}
              modelVersions={this.props.modelVersions}
              handleSetRunTag={this.handleSetRunTag}
            />;
          }}
        </RequestStateWrapper>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { match } = ownProps;
  const runUuid = match.params.runUuid;
  const experimentId = parseInt(match.params.experimentId, 10);
  return {
    runUuid,
    experimentId,
    // so that we re-render the component when the route changes
    key: runUuid + experimentId,
  };
};

const mapDispatchToProps = {
  getRunApi,
  listArtifactsApi,
  getExperimentApi,
  searchModelVersionsApi,
  setTagApi,
};

export default connect(mapStateToProps, mapDispatchToProps)(RunPage);

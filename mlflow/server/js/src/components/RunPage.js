import React, { Component } from 'react';
import PropTypes from 'prop-types';
import RequestStateWrapper from './RequestStateWrapper';
import { getExperimentApi, getRunApi, getUUID, listArtifactsApi } from '../Actions';
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
    dispatch: PropTypes.func.isRequired,
  };

  state = {
    getRunRequestId: getUUID(),
    listArtifactRequestId: getUUID(),
    getExperimentRequestId: getUUID(),
  };


  componentWillMount() {
    this.props.dispatch(getRunApi(this.props.runUuid, this.state.getRunRequestId));
    this.props.dispatch(
      getExperimentApi(this.props.experimentId, this.state.getExperimentRequestId));
  }

  componentDidMount() {
    this.props.dispatch(
      listArtifactsApi(this.props.runUuid, undefined, this.state.listArtifactRequestId));
  }

  render() {
    return (
      <div className='App-content'>
        <RequestStateWrapper
          requestIds={[this.state.getRunRequestId,
            this.state.getExperimentRequestId]}
          asyncRequestIds={[this.state.listArtifactRequestId]}
        >
          {(isLoading, shouldRenderError, requests, asyncRequests) => {
            if (shouldRenderError) {
              const getRunRequest = Utils.getRequestWithId(requests, this.state.getRunRequestId);
              if (getRunRequest.error.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST) {
                return <RunNotFoundView runId={this.props.runUuid}/>;
              }
              return undefined;
            }
            const getArtifactsRequest = Utils.getRequestWithId(
              asyncRequests, this.state.listArtifactRequestId
            );
            const artifactsLoading = getArtifactsRequest === undefined ?
              true :
              getArtifactsRequest.active === true;
            return <RunView
              runUuid={this.props.runUuid}
              getMetricPagePath={(key) =>
                Routes.getMetricPageRoute([this.props.runUuid], key, this.props.experimentId)
              }
              experimentId={this.props.experimentId}
              artifactsAreLoading={artifactsLoading}
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
    key: runUuid + experimentId
  };
};

export default connect(mapStateToProps)(RunPage);

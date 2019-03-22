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
      listArtifactsApi(this.props.runUuid, undefined, this.state.listArtifactRequestId));
    this.props.dispatch(
      getExperimentApi(this.props.experimentId, this.state.getExperimentRequestId));
  }

  render() {
    return (
      <div className='App-content'>
        <RequestStateWrapper
          requestIds={[this.state.getRunRequestId,
            this.state.listArtifactRequestId,
            this.state.getExperimentRequestId]}
        >
          {(isLoading, shouldRenderError, requests) => {
            if (shouldRenderError) {
              const getRunRequest = Utils.getRequestWithId(requests, this.state.getRunRequestId);
              if (getRunRequest.error.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST) {
                return <RunNotFoundView runId={this.props.runUuid}/>;
              }
              return undefined;
            }
            return <RunView
              runUuid={this.props.runUuid}
              getMetricPagePath={(key) =>
                Routes.getMetricPageRoute([this.props.runUuid], key, this.props.experimentId)
              }
              experimentId={this.props.experimentId}
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

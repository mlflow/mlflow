import React, { Component } from 'react';
import PropTypes from 'prop-types';
import RequestStateWrapper from './RequestStateWrapper';
import { getExperimentApi, getRunApi, getUUID, listArtifactsApi } from '../Actions';
import { connect } from 'react-redux';
import RunView from './RunView';
import Routes from '../Routes';

class RunPage extends Component {
  static propTypes = {
    match: PropTypes.object.isRequired,
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
      <div>
        <RequestStateWrapper
          requestIds={[this.state.getRunRequestId,
            this.state.listArtifactRequestId,
            this.state.getExperimentRequestId]}
        >
          <RunView
            runUuid={this.props.runUuid}
            getMetricPagePath={
              (key) => Routes.getMetricPageRoute([this.props.runUuid], key, this.props.experimentId)
            }
            experimentId={this.props.experimentId}
          />
        </RequestStateWrapper>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { match } = ownProps;
  return {
    runUuid: match.params.runUuid,
    match,
    experimentId: parseInt(match.params.experimentId, 10)
  };
};

export default connect(mapStateToProps)(RunPage);

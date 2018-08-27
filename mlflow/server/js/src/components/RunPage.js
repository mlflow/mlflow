import React, { Component } from 'react';
import PropTypes from 'prop-types';
import RequestStateWrapper from './RequestStateWrapper';
import { getExperimentApi, getRunApi, getUUID, listArtifactsApi, setTagApi } from '../Actions';
import { connect } from 'react-redux';
import RunView from './RunView';
import Routes from '../Routes';

class RunPage extends Component {

  constructor(props) {
    super(props);
    this.onSetTag = this.onSetTag.bind(this);
  }

  static propTypes = {
    match: PropTypes.object.isRequired,
    runUuid: PropTypes.string.isRequired,
    experimentId: PropTypes.number.isRequired,
    dispatch: PropTypes.func.isRequired,
    dispatchSetTag: PropTypes.func.isRequired,
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
            onSetTag={this.onSetTag}
          />
        </RequestStateWrapper>
      </div>
    );
  }

  onSetTag(tagKey, tagValue) {
    const setTagRequestId = this.props.dispatchSetTag(
      this.props.runUuid, tagKey, tagValue);
    this.setState({ setTagRequestId });
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


// eslint-disable-next-line no-unused-vars
const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    dispatch,
    dispatchSetTag: (runUuid, tagKey, tagValue) => {
      const requestId = getUUID();
      dispatch(setTagApi(runUuid, tagKey, tagValue, requestId));
      return requestId;
    }
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(RunPage);

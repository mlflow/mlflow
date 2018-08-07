import React, { Component } from 'react';
import PropTypes from 'prop-types';
import qs from 'qs';
import { connect } from 'react-redux';
import {getExperimentApi, getRunApi, getUUID} from '../Actions';
import RequestStateWrapper from './RequestStateWrapper';
import CompareRunView from './CompareRunView';

class CompareRunPage extends Component {
  static propTypes = {
    experimentId: PropTypes.number, // Optional in case we allow comparison across experiments later
    runUuids: PropTypes.arrayOf(String).isRequired,
  };

  componentWillMount() {
    this.requestIds = [];
    if (this.props.experimentId !== null) {
      const experimentRequestId = getUUID();
      this.props.dispatch(getExperimentApi(this.props.experimentId, experimentRequestId));
      this.requestIds.push(experimentRequestId);
    }
    this.props.runUuids.forEach((runUuid) => {
      const requestId = getUUID();
      this.requestIds.push(requestId);
      this.props.dispatch(getRunApi(runUuid, requestId));
    });
  }

  render() {
    return (
      <RequestStateWrapper requestIds={this.requestIds}>
        <CompareRunView runUuids={this.props.runUuids} experimentId={this.props.experimentId}/>
      </RequestStateWrapper>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { location } = ownProps;
  const searchValues = qs.parse(location.search);
  const runUuids = JSON.parse(searchValues["?runs"]);
  let experimentId = null;
  if (searchValues.hasOwnProperty("experiment")) {
    experimentId = parseInt(searchValues["experiment"], 10);
  }
  return { experimentId, runUuids };
};

export default connect(mapStateToProps)(CompareRunPage);

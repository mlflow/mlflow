import React, { Component } from 'react';
import PropTypes from 'prop-types';
import qs from 'qs';
import { connect } from 'react-redux';
import { getExperimentApi, getRunApi, getUUID } from '../Actions';
import RequestStateWrapper from './RequestStateWrapper';
import CompareRunView from './CompareRunView';

class CompareRunPage extends Component {
  static propTypes = {
    experimentId: PropTypes.number.isRequired,
    runUuids: PropTypes.arrayOf(String).isRequired,
    dispatch: PropTypes.func.isRequired,
  };

  componentWillMount() {
    this.requestIds = [];
    const experimentRequestId = getUUID();
    this.props.dispatch(getExperimentApi(this.props.experimentId, experimentRequestId));
    this.requestIds.push(experimentRequestId);
    this.props.runUuids.forEach((runUuid) => {
      const requestId = getUUID();
      this.requestIds.push(requestId);
      this.props.dispatch(getRunApi(runUuid, requestId));
    });
  }

  render() {
    return (
      <div className='App-content'>
        <RequestStateWrapper requestIds={this.requestIds}>
          <CompareRunView runUuids={this.props.runUuids} experimentId={this.props.experimentId}/>
        </RequestStateWrapper>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { location } = ownProps;
  const searchValues = qs.parse(location.search);
  const runUuids = JSON.parse(searchValues["?runs"]);
  const experimentId = parseInt(searchValues["experiment"], 10);
  return { experimentId, runUuids };
};

export default connect(mapStateToProps)(CompareRunPage);

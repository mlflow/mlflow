import React, { Component } from 'react';
import PropTypes from 'prop-types';
import qs from 'qs';
import { connect } from 'react-redux';
import { getRunApi, getUUID } from '../../Actions';
import { getRegisteredModelApi, getModelVersionApi } from "../actions";
import RequestStateWrapper from '../../components/RequestStateWrapper';
import CompareModelVersionsView from './CompareModelVersionsView';


// TODO: Write integration tests for this component
class CompareModelVersionsPage extends Component {
  static propTypes = {
    modelName: PropTypes.string.isRequired,
    runsToVersions: PropTypes.object.isRequired,
    dispatch: PropTypes.func.isRequired,
  };

  componentWillMount() {
    this.requestIds = [];
    const registeredModelRequestId = getUUID();
    this.requestIds.push(registeredModelRequestId);
    this.props.dispatch(getRegisteredModelApi(this.props.modelName, registeredModelRequestId));
    console.log(this.props.runsToVersions);
    for (const runUuid in this.props.runsToVersions) {
      const runRequestId = getUUID();
      this.requestIds.push(runRequestId);
      this.props.dispatch(getRunApi(runUuid, runRequestId));
      const versionRequestId = getUUID();
      this.requestIds.push(versionRequestId);
      this.props.dispatch(getModelVersionApi(this.props.modelName, this.props.runsToVersions[runUuid], versionRequestId))
    }
  }

  render() {
    return (
      <div className='App-content'>
        <RequestStateWrapper requestIds={this.requestIds}>
          <CompareModelVersionsView modelName={this.props.modelName} runsToVersions={this.props.runsToVersions}/>
        </RequestStateWrapper>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { location } = ownProps;
  const searchValues = qs.parse(location.search);
  console.log(searchValues);
  const modelName = JSON.parse(searchValues["?name"]);
  const runsToVersions = JSON.parse(searchValues["runs"]);
  return { modelName, runsToVersions };
};

export default connect(mapStateToProps)(CompareModelVersionsPage);

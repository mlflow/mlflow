import React, { Component } from 'react';
import PropTypes from 'prop-types';
import qs from 'qs';
import { connect } from 'react-redux';
import { getRunApi } from '../../experiment-tracking/actions';
import { getUUID } from '../../common/utils/ActionUtils';
import { getRegisteredModelApi, getModelVersionApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import CompareModelVersionsView from './CompareModelVersionsView';
import _ from 'lodash';

// TODO: Write integration tests for this component
class CompareModelVersionsPage extends Component {
  static propTypes = {
    modelName: PropTypes.string.isRequired,
    versionsToRuns: PropTypes.object.isRequired,
    getRunApi: PropTypes.func.isRequired,
    getRegisteredModelApi: PropTypes.func.isRequired,
    getModelVersionApi: PropTypes.func.isRequired,
  };

  registeredModelRequestId = getUUID();
  versionRequestId = getUUID();
  runRequestId = getUUID();

  state = {
    requestIds: [
      // requests that must be fulfilled before rendering
      this.registeredModelRequestId,
      this.runRequestId,
      this.versionRequestId,
    ],
  };

  componentWillMount() {
    this.props.getRegisteredModelApi(this.props.modelName, this.registeredModelRequestId);
  }

  removeRunRequestId() {
    this.setState((prevState) => ({
      requestIds: _.without(prevState.requestIds, this.runRequestId),
    }));
  }

  componentDidMount() {
    for (const modelVersion in this.props.versionsToRuns) {
      if ({}.hasOwnProperty.call(this.props.versionsToRuns, modelVersion)) {
        const runID = this.props.versionsToRuns[modelVersion];
        if (runID) {
          this.props.getRunApi(runID, this.runRequestId).catch(() => {
            // Failure of this call should not block the page. Here we remove
            // `runRequestId` from `requestIds` to unblock RequestStateWrapper
            // from rendering its content
            this.removeRunRequestId();
          });
        } else {
          this.removeRunRequestId();
        }
        const modelName = this.props.modelName;
        this.props.getModelVersionApi(modelName, modelVersion, this.versionRequestId);
      }
    }
  }

  render() {
    return (
      <div className='App-content'>
        <RequestStateWrapper requestIds={this.state.requestIds}>
          <CompareModelVersionsView
            modelName={this.props.modelName}
            versionsToRuns={this.props.versionsToRuns}
          />
        </RequestStateWrapper>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { location } = ownProps;
  const searchValues = qs.parse(location.search);
  const modelName = JSON.parse(searchValues['?name']);
  const versionsToRuns = JSON.parse(searchValues['runs']);
  return { modelName, versionsToRuns };
};

const mapDispatchToProps = {
  getRunApi,
  getRegisteredModelApi,
  getModelVersionApi,
};

export default connect(mapStateToProps, mapDispatchToProps)(CompareModelVersionsPage);

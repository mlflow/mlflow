import React, { Component } from 'react';
import PropTypes from 'prop-types';
import qs from 'qs';
import { connect } from 'react-redux';
import { getRunApi } from '../../experiment-tracking/actions';
import { getUUID } from '../../common/utils/ActionUtils';
import {
  getRegisteredModelApi,
  getModelVersionApi,
  getModelVersionArtifactApi,
  parseMlModelFile,
} from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import { CompareModelVersionsView } from './CompareModelVersionsView';
import _ from 'lodash';

// TODO: Write integration tests for this component
class CompareModelVersionsPage extends Component {
  static propTypes = {
    modelName: PropTypes.string.isRequired,
    versionsToRuns: PropTypes.object.isRequired,
    getRunApi: PropTypes.func.isRequired,
    getRegisteredModelApi: PropTypes.func.isRequired,
    getModelVersionApi: PropTypes.func.isRequired,
    getModelVersionArtifactApi: PropTypes.func.isRequired,
    parseMlModelFile: PropTypes.func.isRequired,
  };

  registeredModelRequestId = getUUID();
  versionRequestId = getUUID();
  runRequestId = getUUID();
  getMlModelFileRequestId = getUUID();

  state = {
    requestIds: [
      // requests that must be fulfilled before rendering
      this.registeredModelRequestId,
      this.runRequestId,
      this.versionRequestId,
      this.getMlModelFileRequestId,
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
        const { modelName } = this.props;
        this.props.getModelVersionApi(modelName, modelVersion, this.versionRequestId);
        this.props
          .getModelVersionArtifactApi(modelName, modelVersion)
          .then((content) =>
            this.props.parseMlModelFile(
              modelName,
              modelVersion,
              content.value,
              this.getMlModelFileRequestId,
            ),
          )
          .catch(() => {
            // Failure of this call chain should not block the page. Here we remove
            // `getMlModelFileRequestId` from `requestIds` to unblock RequestStateWrapper
            // from rendering its content
            this.setState((prevState) => ({
              requestIds: _.without(prevState.requestIds, this.getMlModelFileRequestId),
            }));
          });
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
  const modelName = decodeURIComponent(JSON.parse(searchValues['?name']));
  const versionsToRuns = JSON.parse(searchValues['runs']);
  return { modelName, versionsToRuns };
};

const mapDispatchToProps = {
  getRunApi,
  getRegisteredModelApi,
  getModelVersionApi,
  getModelVersionArtifactApi,
  parseMlModelFile,
};

export default connect(mapStateToProps, mapDispatchToProps)(CompareModelVersionsPage);

import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { ArtifactView } from './ArtifactView';
import { listArtifactsApi } from '../actions';
import { searchModelVersionsApi } from '../../model-registry/actions';
import { connect } from 'react-redux';
import { getArtifactRootUri, getArtifacts } from '../reducers/Reducers';
import { getAllModelVersions } from '../../model-registry/reducers';
import { ArtifactNode } from '../utils/ArtifactUtils';
import {
  MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL,
} from '../../model-registry/constants';
import _ from 'lodash';
import Utils from '../../common/utils/Utils';
import { getUUID } from '../../common/utils/ActionUtils';

class ArtifactPage extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    modelVersionsBySource: PropTypes.object,
    // The root artifact node.
    artifactNode: PropTypes.instanceOf(ArtifactNode).isRequired,
    artifactRootUri: PropTypes.string.isRequired,
    apis: PropTypes.object.isRequired,
    listArtifactsApi: PropTypes.func.isRequired,
    searchModelVersionsApi: PropTypes.func.isRequired,
  };

  state = { activeNodeIsDirectory: false };

  searchRequestId = getUUID();

  pollModelVersionsForCurrentRun = () => {
    const { apis, runUuid } = this.props;
    const { activeNodeIsDirectory } = this.state;
    const searchRequest = apis[this.searchRequestId];
    if (activeNodeIsDirectory && !(searchRequest && searchRequest.active)) {
      this.props
        .searchModelVersionsApi({ run_id: runUuid }, this.searchRequestId)
        .catch(console.error);
    }
  };

  handleActiveNodeChange = (activeNodeIsDirectory) => {
    this.setState({ activeNodeIsDirectory });
  };

  componentDidMount() {
    if (Utils.isModelRegistryEnabled()) {
      this.pollModelVersionsForCurrentRun();
      this.pollIntervalId = setInterval(this.pollModelVersionsForCurrentRun, POLL_INTERVAL);
    }
  }

  componentWillUnmount() {
    if (Utils.isModelRegistryEnabled()) {
      clearInterval(this.pollIntervalId);
    }
  }

  render() {
    return <ArtifactView {...this.props} handleActiveNodeChange={this.handleActiveNodeChange}/>;
  }
}

const mapStateToProps = (state, ownProps) => {
  const { runUuid } = ownProps;
  const { apis } = state;
  const artifactNode = getArtifacts(runUuid, state);
  const artifactRootUri = getArtifactRootUri(runUuid, state);
  const modelVersionsBySource = _.groupBy(getAllModelVersions(state), 'source');
  return { artifactNode, artifactRootUri, modelVersionsBySource, apis };
};


const mapDispatchToProps = {
  listArtifactsApi,
  searchModelVersionsApi,
};

export default connect(mapStateToProps, mapDispatchToProps)(ArtifactPage);

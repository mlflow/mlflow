import React, { Component } from 'react';
import PropTypes from 'prop-types';
import ArtifactView from './ArtifactView';
import { listArtifactsApi } from '../Actions';
import { connect } from 'react-redux';

class ArtifactPage extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    fetchArtifacts: PropTypes.func.isRequired,
    // For now, assume isHydrated is always true.
    isHydrated: PropTypes.bool,
  };

  render() {
    // If not hydrated then try to get the data before rendering this view.
    return <ArtifactView runUuid={this.props.runUuid} fetchArtifacts={this.props.fetchArtifacts}/>;
  }
}

const mapDispatchToProps = (dispatch) => {
  const fetchArtifacts = (runUuid, path) => {
    dispatch(listArtifactsApi(runUuid, path));
  };
  return {
    fetchArtifacts
  };
};

export default connect(undefined, mapDispatchToProps)(ArtifactPage);

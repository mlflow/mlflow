import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './ShowArtifactImageView.css';
import { getSrc } from './ShowArtifactPage';

class ShowArtifactImageView extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
  };
  render() {
    const { path, runUuid } = this.props;
    return (
      <div className="image-outer-container">
        <div className="image-container"
             style={{ backgroundImage: `url(${getSrc(path, runUuid)})` }}/>
      </div>
    );
  }
}

export default ShowArtifactImageView;

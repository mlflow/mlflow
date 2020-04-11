import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getExtension,
    IMAGE_EXTENSIONS,
    TEXT_EXTENSIONS,
    MAP_EXTENSIONS,
    HTML_EXTENSIONS} from '../../../common/utils/FileUtils';
import ShowArtifactImageView from './ShowArtifactImageView';
import ShowArtifactTextView from './ShowArtifactTextView';
import ShowArtifactMapView from './ShowArtifactMapView';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';
import previewIcon from '../../../common/static/preview-icon.png';
import './ShowArtifactPage.css';

class ShowArtifactPage extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    // If path is not defined don't render anything
    path: PropTypes.string,
  };

  render() {
    if (this.props.path) {
      const normalizedExtension = getExtension(this.props.path);
      if (normalizedExtension) {
        if (IMAGE_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactImageView runUuid={this.props.runUuid} path={this.props.path}/>;
        } else if (TEXT_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactTextView runUuid={this.props.runUuid} path={this.props.path}/>;
        } else if (MAP_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactMapView runUuid={this.props.runUuid} path={this.props.path}/>;
        } else if (HTML_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactHtmlView runUuid={this.props.runUuid} path={this.props.path}/>;
        }
      }
    }
    return (
      <div className="select-preview-outer-container">
        <div className="select-preview-container">
          <div className="select-preview-image-container">
            <img className="select-preview-image" alt="Preview icon." src={previewIcon}/>
          </div>
          <div className="select-preview-text">
            <span className="select-preview-header">Select a file to preview</span>
            <span className="select-preview-supported-formats">
              Supported formats: image, text, html, geojson files
            </span>
          </div>
        </div>
      </div>
    );
  }
}

export const getSrc = (path, runUuid) => {
  const basePath = "get-artifact";
  return `${basePath}?path=${encodeURIComponent(path)}&run_uuid=${encodeURIComponent(runUuid)}`;
};


export default ShowArtifactPage;

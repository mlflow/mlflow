import React, { Component } from 'react';
import PropTypes from 'prop-types';
import {
  getExtension,
  IMAGE_EXTENSIONS,
  TEXT_EXTENSIONS,
  MAP_EXTENSIONS,
  HTML_EXTENSIONS,
  PDF_EXTENSIONS,
} from '../../../common/utils/FileUtils';
import { getLoggedModelPathsFromTags } from '../../../common/utils/TagUtils';
import ShowArtifactImageView from './ShowArtifactImageView';
import ShowArtifactTextView from './ShowArtifactTextView';
import ShowArtifactMapView from './ShowArtifactMapView';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';
import ShowArtifactPdfView from './ShowArtifactPdfView';
import ShowArtifactLoggedModelView from './ShowArtifactLoggedModelView';
import previewIcon from '../../../common/static/preview-icon.png';
import './ShowArtifactPage.css';
import { getModelVersionPageURL } from '../../../model-registry/routes';

class ShowArtifactPage extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    artifactRootUri: PropTypes.string.isRequired,
    // If path is not defined don't render anything
    path: PropTypes.string,
    runTags: PropTypes.object,
    modelVersions: PropTypes.arrayOf(PropTypes.object),
  };

  render() {
    if (this.props.path) {
      const normalizedExtension = getExtension(this.props.path);
      let registeredModelLink;
      const { modelVersions } = this.props;
      if (modelVersions) {
        const [registeredModel] = modelVersions.filter((model) =>
          model.source.endsWith(`artifacts/${normalizedExtension}`),
        );
        if (registeredModel) {
          const { name: registeredModelName, version } = registeredModel;
          registeredModelLink = getModelVersionPageURL(registeredModelName, version);
        }
      }
      if (normalizedExtension) {
        if (IMAGE_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactImageView runUuid={this.props.runUuid} path={this.props.path} />;
        } else if (TEXT_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactTextView runUuid={this.props.runUuid} path={this.props.path} />;
        } else if (MAP_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactMapView runUuid={this.props.runUuid} path={this.props.path} />;
        } else if (HTML_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactHtmlView runUuid={this.props.runUuid} path={this.props.path} />;
        } else if (PDF_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactPdfView runUuid={this.props.runUuid} path={this.props.path} />;
        } else if (
          this.props.runTags &&
          getLoggedModelPathsFromTags(this.props.runTags).includes(normalizedExtension)
        ) {
          return (
            <ShowArtifactLoggedModelView
              runUuid={this.props.runUuid}
              path={this.props.path}
              artifactRootUri={this.props.artifactRootUri}
              registeredModelLink={registeredModelLink}
            />
          );
        }
      }
    }
    return (
      <div className='select-preview-outer-container'>
        <div className='select-preview-container'>
          <div className='select-preview-image-container'>
            <img className='select-preview-image' alt='Preview icon.' src={previewIcon} />
          </div>
          <div className='select-preview-text'>
            <span className='select-preview-header'>Select a file to preview</span>
            <span className='select-preview-supported-formats'>
              Supported formats: image, text, html, pdf, geojson files
            </span>
          </div>
        </div>
      </div>
    );
  }
}

export const getSrc = (path, runUuid) => {
  const basePath = 'get-artifact';
  return `${basePath}?path=${encodeURIComponent(path)}&run_uuid=${encodeURIComponent(runUuid)}`;
};

export default ShowArtifactPage;

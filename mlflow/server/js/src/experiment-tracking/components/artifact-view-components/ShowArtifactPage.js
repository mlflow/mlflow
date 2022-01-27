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
import { ONE_MB } from '../../constants';
import ShowArtifactImageView from './ShowArtifactImageView';
import ShowArtifactTextView from './ShowArtifactTextView';
import ShowArtifactMapView from './ShowArtifactMapView';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';
import ShowArtifactPdfView from './ShowArtifactPdfView';
import ShowArtifactLoggedModelView from './ShowArtifactLoggedModelView';
import previewIcon from '../../../common/static/preview-icon.png';
import warningSvg from '../../../common/static/warning.svg';
import './ShowArtifactPage.css';
import { getModelVersionPageRoute } from '../../../model-registry/routes';
import Utils from '../../../common/utils/Utils';

import { FormattedMessage } from 'react-intl';

const MAX_PREVIEW_ARTIFACT_SIZE_MB = 50;

class ShowArtifactPage extends Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    artifactRootUri: PropTypes.string.isRequired,
    // If path is not defined don't render anything
    path: PropTypes.string,
    size: PropTypes.number,
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
          registeredModelLink = Utils.getIframeCorrectedRoute(
            getModelVersionPageRoute(registeredModelName, version),
          );
        }
      }
      if (this.props.size > MAX_PREVIEW_ARTIFACT_SIZE_MB * ONE_MB) {
        return getFileTooLargeView();
      } else if (normalizedExtension) {
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
    return getSelectFileView();
  }
}

const getSelectFileView = () => {
  return (
    <div className='preview-outer-container'>
      <div className='preview-container'>
        <div className='preview-image-container'>
          <img className='preview-image' alt='Preview icon.' src={previewIcon} />
        </div>
        <div className='preview-text'>
          <span className='preview-header'>
            <FormattedMessage
              defaultMessage='Select a file to preview'
              description='Label to suggests users to select a file to preview the output'
            />
          </span>
          <span className='preview-supported-formats'>
            <FormattedMessage
              defaultMessage='Supported formats: image, text, html, pdf, geojson files'
              // eslint-disable-next-line max-len
              description='Text to explain users which formats are supported to display the artifacts'
            />
          </span>
        </div>
      </div>
    </div>
  );
};

const getFileTooLargeView = () => {
  return (
    <div className='preview-outer-container'>
      <div className='preview-container'>
        <div className='preview-image-container'>
          <img className='preview-image' alt='Preview icon.' src={warningSvg} />
        </div>
        <div className='preview-text'>
          <span className='preview-header'>
            <FormattedMessage
              defaultMessage='File is too large to preview'
              description='Label to indicate that the file is too large to preview'
            />
          </span>
          <span className='preview-max-size'>
            <FormattedMessage
              defaultMessage={`Maximum file size for preview: ${MAX_PREVIEW_ARTIFACT_SIZE_MB}MB`}
              // eslint-disable-next-line max-len
              description='Text to notify users of the maximum file size for which artifact previews are displayed'
            />
          </span>
        </div>
      </div>
    </div>
  );
};

export const getSrc = (path, runUuid) => {
  const basePath = 'get-artifact';
  return `${basePath}?path=${encodeURIComponent(path)}&run_uuid=${encodeURIComponent(runUuid)}`;
};

export default ShowArtifactPage;

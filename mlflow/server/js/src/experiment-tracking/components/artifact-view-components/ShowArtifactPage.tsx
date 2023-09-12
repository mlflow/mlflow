/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import {
  getExtension,
  IMAGE_EXTENSIONS,
  TEXT_EXTENSIONS,
  MAP_EXTENSIONS,
  HTML_EXTENSIONS,
  PDF_EXTENSIONS,
  DATA_EXTENSIONS,
} from '../../../common/utils/FileUtils';
import { getLoggedModelPathsFromTags } from '../../../common/utils/TagUtils';
import { ONE_MB } from '../../constants';
import ShowArtifactImageView from './ShowArtifactImageView';
import ShowArtifactTextView from './ShowArtifactTextView';
import { LazyShowArtifactMapView } from './LazyShowArtifactMapView';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';
import { LazyShowArtifactPdfView } from './LazyShowArtifactPdfView';
import { LazyShowArtifactTableView } from './LazyShowArtifactTableView';
import ShowArtifactLoggedModelView from './ShowArtifactLoggedModelView';
// @ts-expect-error TS(2307): Cannot find module '../../../common/static/preview... Remove this comment to see the full error message
import previewIcon from '../../../common/static/preview-icon.png';
import warningSvg from '../../../common/static/warning.svg';
import './ShowArtifactPage.css';
import { getModelVersionPageRoute } from '../../../model-registry/routes';
import Utils from '../../../common/utils/Utils';

import { FormattedMessage } from 'react-intl';

const MAX_PREVIEW_ARTIFACT_SIZE_MB = 50;

type ShowArtifactPageProps = {
  runUuid: string;
  artifactRootUri: string;
  path?: string;
  isDirectory?: boolean;
  size?: number;
  runTags?: any;
  modelVersions?: any[];
};

class ShowArtifactPage extends Component<ShowArtifactPageProps> {
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
      // @ts-expect-error TS(2532): Object is possibly 'undefined'.
      if (this.props.size > MAX_PREVIEW_ARTIFACT_SIZE_MB * ONE_MB) {
        return getFileTooLargeView();
      } else if (this.props.isDirectory) {
        if (
          this.props.runTags &&
          getLoggedModelPathsFromTags(this.props.runTags).includes(this.props.path)
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
      } else if (normalizedExtension) {
        if (IMAGE_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactImageView runUuid={this.props.runUuid} path={this.props.path} />;
        } else if (DATA_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <LazyShowArtifactTableView runUuid={this.props.runUuid} path={this.props.path} />;
        } else if (TEXT_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return (
            <ShowArtifactTextView
              runUuid={this.props.runUuid}
              path={this.props.path}
              size={this.props.size}
            />
          );
        } else if (MAP_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <LazyShowArtifactMapView runUuid={this.props.runUuid} path={this.props.path} />;
        } else if (HTML_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactHtmlView runUuid={this.props.runUuid} path={this.props.path} />;
        } else if (PDF_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <LazyShowArtifactPdfView runUuid={this.props.runUuid} path={this.props.path} />;
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

export default ShowArtifactPage;

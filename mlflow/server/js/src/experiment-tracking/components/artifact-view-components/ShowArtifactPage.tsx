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
  AUDIO_EXTENSIONS,
  VIDEO_EXTENSIONS,
} from '../../../common/utils/FileUtils';
import { getLoggedModelPathsFromTags, getLoggedTablesFromTags } from '../../../common/utils/TagUtils';
import { ONE_MB } from '../../constants';
import ShowArtifactImageView from './ShowArtifactImageView';
import ShowArtifactTextView from './ShowArtifactTextView';
import { LazyShowArtifactMapView } from './LazyShowArtifactMapView';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';
import { LazyShowArtifactPdfView } from './LazyShowArtifactPdfView';
import { LazyShowArtifactTableView } from './LazyShowArtifactTableView';
import ShowArtifactLoggedModelView from './ShowArtifactLoggedModelView';
import previewIcon from '../../../common/static/preview-icon.png';
import warningSvg from '../../../common/static/warning.svg';
import { ModelRegistryRoutes } from '../../../model-registry/routes';
import Utils from '../../../common/utils/Utils';
import { FormattedMessage } from 'react-intl';
import { ShowArtifactLoggedTableView } from './ShowArtifactLoggedTableView';
import { Empty, Spacer } from '@databricks/design-system';
import { LazyShowArtifactAudioView } from './LazyShowArtifactAudioView';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { LazyShowArtifactVideoView } from './LazyShowArtifactVideoView';
import type { KeyValueEntity } from '../../../common/types';

const MAX_PREVIEW_ARTIFACT_SIZE_MB = 50;

type ShowArtifactPageProps = {
  runUuid: string;
  artifactRootUri: string;
  path?: string;
  isDirectory?: boolean;
  size?: number;
  runTags?: any;
  modelVersions?: any[];
  showArtifactLoggedTableView?: boolean;
  entityTags?: Partial<KeyValueEntity>[];
} & LoggedModelArtifactViewerProps;

class ShowArtifactPage extends Component<ShowArtifactPageProps> {
  render() {
    if (this.props.path) {
      const { loggedModelId, isLoggedModelsMode, path, runUuid, experimentId, entityTags } = this.props;
      const commonArtifactProps = {
        loggedModelId,
        isLoggedModelsMode,
        path,
        runUuid,
        experimentId,
        entityTags,
      };

      const normalizedExtension = getExtension(this.props.path);
      let registeredModelLink;
      const { modelVersions } = this.props;
      if (modelVersions) {
        const [registeredModel] = modelVersions.filter((model) =>
          model.source.endsWith(`artifacts/${normalizedExtension}`),
        );
        if (registeredModel) {
          const { name: registeredModelName, version } = registeredModel;
          registeredModelLink = ModelRegistryRoutes.getModelVersionPageRoute(registeredModelName, version);
        }
      }
      // @ts-expect-error TS(2532): Object is possibly 'undefined'.
      if (this.props.size > MAX_PREVIEW_ARTIFACT_SIZE_MB * ONE_MB) {
        return getFileTooLargeView();
      } else if (this.props.isDirectory) {
        if (this.props.runTags && getLoggedModelPathsFromTags(this.props.runTags).includes(this.props.path)) {
          return (
            // getArtifact has a default in the component
            // @ts-expect-error TS(2741): Property 'getArtifact' is missing
            <ShowArtifactLoggedModelView
              runUuid={this.props.runUuid}
              path={this.props.path}
              artifactRootUri={this.props.artifactRootUri}
              registeredModelLink={registeredModelLink}
              experimentId={this.props.experimentId}
            />
          );
        }
      } else if (this.props.showArtifactLoggedTableView) {
        return <ShowArtifactLoggedTableView {...commonArtifactProps} />;
      } else if (normalizedExtension) {
        if (IMAGE_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactImageView {...commonArtifactProps} />;
        } else if (DATA_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <LazyShowArtifactTableView {...commonArtifactProps} />;
        } else if (TEXT_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactTextView {...commonArtifactProps} size={this.props.size} />;
        } else if (MAP_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <LazyShowArtifactMapView {...commonArtifactProps} />;
        } else if (HTML_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <ShowArtifactHtmlView {...commonArtifactProps} />;
        } else if (PDF_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <LazyShowArtifactPdfView {...commonArtifactProps} />;
        } else if (AUDIO_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <LazyShowArtifactAudioView {...commonArtifactProps} />;
        } else if (VIDEO_EXTENSIONS.has(normalizedExtension.toLowerCase())) {
          return <LazyShowArtifactVideoView {...commonArtifactProps} />;
        }
      }
    }
    return getSelectFileView();
  }
}

const getSelectFileView = () => {
  return (
    <div css={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <Empty
        image={
          <>
            <img alt="Preview icon." src={previewIcon} css={{ width: 64, height: 64 }} />
            <Spacer size="sm" />
          </>
        }
        title={
          <FormattedMessage
            defaultMessage="Select a file to preview"
            description="Label to suggests users to select a file to preview the output"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="Supported formats: image, text, html, pdf, audio, video, geojson files"
            description="Text to explain users which formats are supported to display the artifacts"
          />
        }
      />
    </div>
  );
};

const getFileTooLargeView = () => {
  return (
    <div css={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <Empty
        image={
          <>
            <img alt="Preview icon." src={warningSvg} css={{ width: 64, height: 64 }} />
            <Spacer size="sm" />
          </>
        }
        title={
          <FormattedMessage
            defaultMessage="File is too large to preview"
            description="Label to indicate that the file is too large to preview"
          />
        }
        description={
          <FormattedMessage
            // eslint-disable-next-line formatjs/enforce-default-message
            defaultMessage={`Maximum file size for preview: ${MAX_PREVIEW_ARTIFACT_SIZE_MB}MiB`}
            description="Text to notify users of the maximum file size for which artifact previews are displayed"
          />
        }
      />
    </div>
  );
};

export default ShowArtifactPage;

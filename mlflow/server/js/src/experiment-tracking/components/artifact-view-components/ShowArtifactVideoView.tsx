import React, { useEffect, useState } from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import {
  getArtifactBlob,
  getArtifactLocationUrl,
  getLoggedModelArtifactLocationUrl,
} from '../../../common/utils/ArtifactUtils';
import { DownloadLink, exceedsRenderSizeLimit } from '../../../shared/web-shared/media-rendering-utils';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';

type Props = {
  runUuid: string;
  path: string;
  getArtifact?: (...args: any[]) => any;
} & LoggedModelArtifactViewerProps;

const ShowArtifactVideoView = ({
  runUuid,
  path,
  getArtifact = getArtifactBlob,
  isLoggedModelsMode,
  loggedModelId,
}: Props) => {
  const [videoUrl, setVideoUrl] = useState<string>();
  const [contentLength, setContentLength] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let objUrl: string | undefined;

    const artifactUrl =
      isLoggedModelsMode && loggedModelId
        ? getLoggedModelArtifactLocationUrl(path, loggedModelId)
        : getArtifactLocationUrl(path, runUuid);

    getArtifact(artifactUrl).then((blob: Blob) => {
      objUrl = URL.createObjectURL(blob);
      setContentLength(blob.size);
      setVideoUrl(objUrl);
      setLoading(false);
    });

    return () => {
      if (objUrl) URL.revokeObjectURL(objUrl);
    };
  }, [runUuid, path, isLoggedModelsMode, loggedModelId, getArtifact]);

  const classNames = {
    videoOuterContainer: {
      padding: 10,
      overflow: 'hidden',
      background: 'black',
      minHeight: '100%',
    },
    hidden: { display: 'none' },
    video: {
      maxWidth: '100%',
      maxHeight: '62.5vh',
      objectFit: 'fit',
      display: 'block',
    },
  };

  if (exceedsRenderSizeLimit('video/mp4', contentLength) && videoUrl) {
    return (
      <div css={{ padding: 10 }}>
        <DownloadLink
          url={videoUrl}
          contentType="video/mp4"
          contentLength={contentLength}
          filename={path.split('/').pop()}
        />
      </div>
    );
  }

  return (
    <div css={{ flex: 1 }}>
      <div css={classNames.videoOuterContainer}>
        {loading && <LegacySkeleton active />}
        {videoUrl && (
          <video
            css={loading ? classNames.hidden : classNames.video}
            src={videoUrl}
            controls
            preload="auto"
            aria-label="video"
          >
            <track kind="captions" srcLang="en" src="" default />
          </video>
        )}
      </div>
    </div>
  );
};

export default ShowArtifactVideoView;

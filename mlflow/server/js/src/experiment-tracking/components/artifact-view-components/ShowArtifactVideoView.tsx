import React, { useEffect, useState } from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import { getArtifactBlob } from '../../../common/utils/ArtifactUtils';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';

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
  artifactRootUri,
  experimentId,
  entityTags,
}: Props) => {
  const [videoUrl, setVideoUrl] = useState<string>();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let objUrl: string | undefined;

    fetchArtifactUnified(
      { runUuid, path, artifactRootUri, isLoggedModelsMode, loggedModelId, experimentId, entityTags },
      getArtifact,
    ).then((result) => {
      objUrl = URL.createObjectURL(result as Blob);
      setVideoUrl(objUrl);
      setLoading(false);
    });

    return () => {
      if (objUrl) URL.revokeObjectURL(objUrl);
    };
  }, [runUuid, path, isLoggedModelsMode, loggedModelId, getArtifact, artifactRootUri, experimentId, entityTags]);

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

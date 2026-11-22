/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { useState, useEffect } from 'react';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import { getArtifactBytesContent } from '../../../common/utils/ArtifactUtils';
import { ImagePreviewGroup, Image } from '../../../shared/building_blocks/Image';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';

type Props = {
  runUuid: string;
  path: string;
  getArtifact?: (...args: any[]) => any;
} & LoggedModelArtifactViewerProps;

const ShowArtifactImageView = ({
  artifactRootUri,
  runUuid,
  path,
  getArtifact = getArtifactBytesContent,
  isLoggedModelsMode,
  loggedModelId,
}: Props) => {
  const [isLoading, setIsLoading] = useState(true);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [imageUrl, setImageUrl] = useState<string>();
  const [error, setError] = useState<Error>();

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | undefined;
    setIsLoading(true);
    setError(undefined);
    setImageUrl(undefined);
    fetchArtifactUnified(
      {
        runUuid,
        path,
        artifactRootUri,
        isLoggedModelsMode,
        loggedModelId,
      },
      getArtifact,
    )
      .then((result: any) => {
        if (cancelled) return;
        const options = path.toLowerCase().endsWith('.svg') ? { type: 'image/svg+xml' } : undefined;
        objectUrl = URL.createObjectURL(new Blob([new Uint8Array(result)], options));
        setImageUrl(objectUrl);
        setIsLoading(false);
      })
      .catch((error: Error) => {
        if (cancelled) return;
        setError(error);
        setIsLoading(false);
      });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [runUuid, path, getArtifact, isLoggedModelsMode, loggedModelId, artifactRootUri]);

  return (
    <div css={{ flex: 1 }}>
      {isLoading && <ArtifactViewSkeleton />}
      {error && <ArtifactViewErrorState error={error} />}
      {imageUrl && (
        <div css={classNames.imageOuterContainer}>
          <div css={classNames.imageWrapper}>
            <img alt={path} css={classNames.image} src={imageUrl} onClick={() => setPreviewVisible(true)} />
          </div>
          <div css={[classNames.hidden]}>
            <ImagePreviewGroup visible={previewVisible} onVisibleChange={setPreviewVisible}>
              <Image src={imageUrl} />
            </ImagePreviewGroup>
          </div>
        </div>
      )}
    </div>
  );
};

const classNames = {
  imageOuterContainer: {
    padding: '10px',
    overflow: 'scroll',
    // Let's keep images (esp. transparent PNGs) on the white background regardless of the theme
    background: 'white',
    minHeight: '100%',
  },
  imageWrapper: { display: 'inline-block' },
  image: {
    maxWidth: '100%',
    height: 'auto',
    cursor: 'pointer',
    '&:hover': {
      boxShadow: '0 0 4px gray',
    },
  },
  hidden: { display: 'none' },
};

export default ShowArtifactImageView;

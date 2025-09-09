/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { useState, useEffect } from 'react';
import { LegacySkeleton } from '@databricks/design-system';
import {
  getArtifactBytesContent,
  getArtifactLocationUrl,
  getLoggedModelArtifactLocationUrl,
} from '../../../common/utils/ArtifactUtils';
import { ImagePreviewGroup, Image } from '../../../shared/building_blocks/Image';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';

type Props = {
  runUuid: string;
  path: string;
  getArtifact?: (...args: any[]) => any;
} & LoggedModelArtifactViewerProps;

const ShowArtifactImageView = ({
  experimentId,
  runUuid,
  path,
  getArtifact = getArtifactBytesContent,
  isLoggedModelsMode,
  loggedModelId,
  entityTags,
}: Props) => {
  const [isLoading, setIsLoading] = useState(true);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);

  useEffect(() => {
    setIsLoading(true);

    // Download image contents using XHR so all necessary
    // HTTP headers will be automatically added
    fetchArtifactUnified(
      {
        runUuid,
        path,
        isLoggedModelsMode,
        loggedModelId,
        experimentId,
        entityTags,
      },
      getArtifact,
    ).then((result: any) => {
      const options = path.toLowerCase().endsWith('.svg') ? { type: 'image/svg+xml' } : undefined;
      // @ts-expect-error TS(2345): Argument of type 'string' is not assignable to par... Remove this comment to see the full error message
      setImageUrl(URL.createObjectURL(new Blob([new Uint8Array(result)], options)));
      setIsLoading(false);
    });
  }, [runUuid, path, getArtifact, isLoggedModelsMode, loggedModelId, experimentId, entityTags]);

  return (
    imageUrl && (
      <div css={{ flex: 1 }}>
        <div css={classNames.imageOuterContainer}>
          {isLoading && <LegacySkeleton active />}
          <div css={isLoading ? classNames.hidden : classNames.imageWrapper}>
            <img
              alt={path}
              css={classNames.image}
              src={imageUrl}
              onLoad={() => setIsLoading(false)}
              onClick={() => setPreviewVisible(true)}
            />
          </div>
          <div css={[classNames.hidden]}>
            <ImagePreviewGroup visible={previewVisible} onVisibleChange={setPreviewVisible}>
              <Image src={imageUrl} />
            </ImagePreviewGroup>
          </div>
        </div>
      </div>
    )
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
    cursor: 'pointer',
    '&:hover': {
      boxShadow: '0 0 4px gray',
    },
  },
  hidden: { display: 'none' },
};

export default ShowArtifactImageView;

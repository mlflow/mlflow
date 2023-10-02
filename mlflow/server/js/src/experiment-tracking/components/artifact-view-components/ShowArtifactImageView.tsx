/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { useState, useEffect, useContext } from 'react';
import { Image } from 'antd';
import { DesignSystemContext, LegacySkeleton } from '@databricks/design-system';
import {
  getArtifactBytesContent,
  getArtifactLocationUrl,
} from '../../../common/utils/ArtifactUtils';

type Props = {
  runUuid: string;
  path: string;
  getArtifact?: (...args: any[]) => any;
};

const ShowArtifactImageView = ({ runUuid, path, getArtifact = getArtifactBytesContent }: Props) => {
  const [isLoading, setIsLoading] = useState(true);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);

  const { getPopupContainer } = useContext(DesignSystemContext);

  useEffect(() => {
    setIsLoading(true);

    // Download image contents using XHR so all necessary
    // HTTP headers will be automatically added
    getArtifact(getArtifactLocationUrl(path, runUuid)).then((result: any) => {
      const options = path.toLowerCase().endsWith('.svg') ? { type: 'image/svg+xml' } : undefined;
      // @ts-expect-error TS(2345): Argument of type 'string' is not assignable to par... Remove this comment to see the full error message
      setImageUrl(URL.createObjectURL(new Blob([new Uint8Array(result)], options)));
      setIsLoading(false);
    });
  }, [runUuid, path, getArtifact]);

  return (
    <div css={classNames.imageOuterContainer}>
      {isLoading && <LegacySkeleton active />}
      <div css={isLoading ? classNames.hidden : classNames.imageWrapper}>
        <img
          alt={path}
          css={classNames.image}
          // @ts-expect-error TS(2322): Type 'null' is not assignable to type 'string | un... Remove this comment to see the full error message
          src={imageUrl}
          onLoad={() => setIsLoading(false)}
          onClick={() => setPreviewVisible(true)}
        />
      </div>
      <div css={classNames.hidden}>
        <Image.PreviewGroup
          preview={{
            visible: previewVisible,
            getContainer: getPopupContainer,
            onVisibleChange: (visible) => setPreviewVisible(visible),
          }}
        >
          {/* @ts-expect-error TS(2322): Type 'null' is not assignable to type 'string | un... Remove this comment to see the full error message */}
          <Image src={imageUrl} />
        </Image.PreviewGroup>
      </div>
    </div>
  );
};

const classNames = {
  imageOuterContainer: {
    padding: '10px',
    overflow: 'scroll',
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

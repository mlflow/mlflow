import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { Image } from 'antd';
import { Skeleton } from '@databricks/design-system';

const ShowArtifactImageView = ({ runUuid, path }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [previewVisible, setPreviewVisible] = useState(false);

  useEffect(() => {
    setIsLoading(true);
  }, [runUuid, path]);

  const src = getSrc(path, runUuid);
  return (
    <div css={classNames.imageOuterContainer}>
      {isLoading && <Skeleton active />}
      <div css={isLoading ? classNames.hidden : classNames.imageWrapper}>
        <img
          alt={path}
          css={classNames.image}
          src={src}
          onLoad={() => setIsLoading(false)}
          onClick={() => setPreviewVisible(true)}
        />
      </div>
      <div css={classNames.hidden}>
        <Image.PreviewGroup
          preview={{
            visible: previewVisible,
            onVisibleChange: (visible) => setPreviewVisible(visible),
          }}
        >
          <Image src={src} />
        </Image.PreviewGroup>
      </div>
    </div>
  );
};

ShowArtifactImageView.propTypes = {
  runUuid: PropTypes.string.isRequired,
  path: PropTypes.string.isRequired,
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

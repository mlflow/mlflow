import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { Image } from 'antd';
import { Skeleton } from '@databricks/design-system';
import { css } from 'emotion';

const ShowArtifactImageView = ({ runUuid, path }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [previewVisible, setPreviewVisible] = useState(false);

  useEffect(() => {
    setIsLoading(true);
  }, [runUuid, path]);

  const src = getSrc(path, runUuid);
  return (
    <div className={classNames.imageOuterContainer}>
      {isLoading && <Skeleton active />}
      <div className={isLoading ? classNames.hidden : classNames.imageWrapper}>
        <img
          className={classNames.image}
          src={src}
          onLoad={() => setIsLoading(false)}
          onClick={() => setPreviewVisible(true)}
        />
      </div>
      <div className={classNames.hidden}>
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
  imageOuterContainer: css({
    minHeight: '500px',
    maxHeight: '1000px',
    padding: '10px',
    overflow: 'scroll',
  }),
  imageWrapper: css({ display: 'inline-block' }),
  image: css({
    cursor: 'pointer',
    '&:hover': {
      boxShadow: '0 0 4px gray',
    },
  }),
  hidden: css({ display: 'none' }),
};

export default ShowArtifactImageView;

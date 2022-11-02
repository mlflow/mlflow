import React, { useState, useEffect, useContext } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { Image } from 'antd';
import { DesignSystemContext, Skeleton } from '@databricks/design-system';
import { getArtifactBytesContent } from '../../../common/utils/ArtifactUtils';

const ShowArtifactImageView = ({ runUuid, path, getArtifact = getArtifactBytesContent }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);

  const { getPopupContainer } = useContext(DesignSystemContext);

  useEffect(() => {
    setIsLoading(true);

    // Download image contents using XHR so all necessary
    // HTTP headers will be automatically added
    getArtifact(getSrc(path, runUuid)).then((result) => {
      setImageUrl(URL.createObjectURL(new Blob([new Uint8Array(result)])));
      setIsLoading(false);
    });
  }, [runUuid, path, getArtifact]);

  return (
    <div css={classNames.imageOuterContainer}>
      {isLoading && <Skeleton active />}
      <div css={isLoading ? classNames.hidden : classNames.imageWrapper}>
        <img
          alt={path}
          css={classNames.image}
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
          <Image src={imageUrl} />
        </Image.PreviewGroup>
      </div>
    </div>
  );
};

ShowArtifactImageView.propTypes = {
  runUuid: PropTypes.string.isRequired,
  path: PropTypes.string.isRequired,
  getArtifact: PropTypes.func,
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

import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import './ShowArtifactImageView.css';
import { getSrc } from './ShowArtifactPage';
import { Image as ImageTag, Skeleton } from 'antd';

export const ShowArtifactImageView = ({ runUuid, path }) => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setIsLoading(true);
  }, [runUuid, path]);

  const src = getSrc(path, runUuid);
  const img = new Image();
  img.onload = () => {
    setIsLoading(false);
  };
  img.src = src;

  return (
    <div className='image-outer-container'>
      {isLoading ? <Skeleton active /> : <ImageTag src={src} />}
    </div>
  );
};

ShowArtifactImageView.propTypes = {
  runUuid: PropTypes.string.isRequired,
  path: PropTypes.string.isRequired,
};

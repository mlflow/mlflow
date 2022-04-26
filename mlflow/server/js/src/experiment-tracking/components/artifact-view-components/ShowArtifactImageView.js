import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import './ShowArtifactImageView.css';
import { getSrc } from './ShowArtifactPage';
import { Image, Skeleton } from 'antd';

const ShowArtifactImageView = ({ runUuid, path }) => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setIsLoading(true);
  }, [runUuid, path]);

  return (
    <div className='image-outer-container'>
      {isLoading && <Skeleton active />}
      <Image
        style={
          isLoading
            ? { display: 'none' }
            : // Workaround for https://github.com/ant-design/ant-design/discussions/28434#discussion-77663
              { display: 'inline' }
        }
        src={getSrc(path, runUuid)}
        onLoad={() => setIsLoading(false)}
      />
    </div>
  );
};

ShowArtifactImageView.propTypes = {
  runUuid: PropTypes.string.isRequired,
  path: PropTypes.string.isRequired,
};

export default ShowArtifactImageView;

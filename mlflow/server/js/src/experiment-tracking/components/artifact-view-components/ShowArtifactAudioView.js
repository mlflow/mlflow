import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import WaveformView from './WaveformView';

const ShowArtifactAudioView = ({ runUuid, path, getArtifact }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState();
  const [waveformData, setWaveformData] = useState();

  useEffect(() => {
    resetState();
    fetchArtifacts({ path, runUuid, getArtifact });
  }, [runUuid, path, getArtifact]);

  function resetState() {
    setLoading(true);
    setError();
  }

  function fetchArtifacts(artifactData) {
    const artifactLocation = getSrc(artifactData.path, artifactData.runUuid);
    artifactData
      .getArtifact(artifactLocation, true)
      .then((arrayBuffer) => {
        try {
          setWaveformData(arrayBuffer);
          setLoading(false);
        } catch (e) {
          setLoading(false);
        }
      })
      .catch((e) => {
        setError(e);
        setLoading(false);
      });
  }

  if (loading) {
    return <div className='artifact-audio-view-loading'>Loading artifact...</div>;
  }

  if (error) {
    return (
      <div className='artifact-audio-view-error'>
        Oops we couldn't load your audio file because of an error.
      </div>
    );
  }

  return <WaveformView data={waveformData} />;
};

ShowArtifactAudioView.propTypes = {
  runUuid: PropTypes.string.isRequired,
  path: PropTypes.string.isRequired,
  getArtifact: PropTypes.func,
};

ShowArtifactAudioView.defaultProps = {
  getArtifact: getArtifactContent,
};

export default ShowArtifactAudioView;

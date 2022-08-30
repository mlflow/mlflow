import React, { useEffect, useState, useRef } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import WaveSurfer from 'wavesurfer.js';

function formatTimecode(seconds) {
  return new Date(1000 * seconds).toISOString().substring(11, 19);
}

const ShowArtifactAudioView = ({ runUuid, path, getArtifact }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState();
  const [waveformPainting, setWaveformPainting] = useState();
  const [playing, setPlaying] = useState(false);
  const [audioDuration, setAudioDuration] = useState(0);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const waveformRef = useRef();
  const [waveform, setWaveform] = useState();

  useEffect(() => {
    resetState();
    fetchArtifacts({ path, runUuid, getArtifact });
    return () => {
      if (waveform) {
        waveform.destroy();
      }
    };
  }, [runUuid, path]);

  function resetState() {
    setPlaying(false);
    setAudioDuration(0);
    setTimeElapsed(0);
    setLoading(true);
    setError();
  }

  function loadAudio(artifact) {
    let newWaveform;
    try {
      newWaveform = WaveSurfer.create({
        waveColor: '#1890ff',
        progressColor: '#2374BB',
        cursorColor: '#333333',
        container: waveformRef.current,
        responsive: true,
        height: 592,
      });
    } catch (e) {
      throw Error(e);
    }
    const blob = new window.Blob([new Uint8Array(artifact)]);
    try {
      newWaveform.loadBlob(blob);
      setWaveformPainting(true);
    } catch (e) {
      throw Error(e);
    }

    newWaveform.on('ready', () => {
      setAudioDuration(newWaveform.getDuration());
      setWaveformPainting(false);
    });

    newWaveform.on('error', (e) => {
      setError(e);
    });

    newWaveform.on('seek', () => {
      setTimeElapsed(newWaveform.getCurrentTime());
    });

    newWaveform.on('audioprocess', () => {
      if (newWaveform.isPlaying()) {
        setTimeElapsed(newWaveform.getCurrentTime());
      }
    });

    newWaveform.on('finish', () => {
      setPlaying(false);
    });

    setWaveform(newWaveform);
  }

  function fetchArtifacts(artifactData) {
    const artifactLocation = getSrc(artifactData.path, artifactData.runUuid);
    artifactData
      .getArtifact(artifactLocation, true)
      .then((arrayBuffer) => {
        try {
          setLoading(false);
          loadAudio(arrayBuffer);
        } catch (e) {
          setLoading(false);
        }
      })
      .catch((e) => {
        setError(e);
        setLoading(false);
      });
  }

  function handlePlayPause() {
    setPlaying(!playing);
    waveform.playPause();
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
  } else {
    let audioInfo;
    if (waveformPainting) {
      audioInfo = <div className='artifact-audio-load-progress'>Generating waveform...</div>;
    } else {
      audioInfo = (
        <>
          <a title={playing ? 'Pause' : 'Play'} onClick={handlePlayPause}>
            <i className={playing ? 'fas fa-pause' : 'fas fa-play'} style={{ fontSize: 21 }}></i>
          </a>
          <div style={{ width: 15 }}></div>
          <div className='timecode'>
            {`${formatTimecode(timeElapsed)} / ${formatTimecode(audioDuration)}`}
          </div>
        </>
      );
    }

    return (
      <div className='text-area-border-box'>
        <div
          style={{
            display: 'flex',
            height: 42,
            padding: '8px 16px',
            whiteSpace: 'nowrap',
            textAlign: 'center',
            alignItems: 'center',
            backgroundColor: '#fafafa',
          }}
        >
          {audioInfo}
        </div>
        <div id='33333' ref={waveformRef}></div>
      </div>
    );
  }
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

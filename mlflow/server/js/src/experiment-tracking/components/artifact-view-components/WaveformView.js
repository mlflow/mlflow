import React, { useEffect, useState, useRef } from 'react';
import PropTypes from 'prop-types';
import WaveSurfer from 'wavesurfer.js';

function formatTimecode(seconds) {
  return new Date(1000 * seconds).toISOString().substring(11, 19);
}

const WaveformView = ({ data }) => {
  const containerRef = useRef();
  const [waveformPainting, setWaveformPainting] = useState();
  const [playing, setPlaying] = useState(false);
  const [audioDuration, setAudioDuration] = useState(0);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [error, setError] = useState();
  const [waveform, setWaveform] = useState();

  useEffect(() => {
    const waveformEl = WaveSurfer.create({
      waveColor: '#1890ff',
      progressColor: '#2374BB',
      cursorColor: '#333333',
      container: containerRef.current,
      responsive: true,
      height: 592,
    });

    loadAudio(data, waveformEl);

    setWaveform(waveformEl);

    return () => waveformEl.destroy();
  }, [data, containerRef]);

  function loadAudio(waveformData, waveformEl) {
    const blob = new window.Blob([new Uint8Array(waveformData)]);
    try {
      waveformEl.loadBlob(blob);
      setWaveformPainting(true);
    } catch (e) {
      throw Error(e);
    }

    waveformEl.on('ready', () => {
      setAudioDuration(waveformEl.getDuration());
      setWaveformPainting(false);
    });

    waveformEl.on('error', (e) => {
      setError(e);
    });

    waveformEl.on('seek', () => {
      setTimeElapsed(waveformEl.getCurrentTime());
    });

    waveformEl.on('audioprocess', () => {
      if (waveformEl.isPlaying()) {
        setTimeElapsed(waveformEl.getCurrentTime());
      }
    });

    waveformEl.on('finish', () => {
      setPlaying(false);
    });
  }

  if (error) {
    return (
      <div className='artifact-audio-view-error'>
        Oops we couldn't load your audio file because of an error.
      </div>
    );
  }

  const handlePlayPause = () => {
    setPlaying(!playing);
    waveform.playPause();
  };

  let audioInfo;
  if (waveformPainting) {
    audioInfo = <div className='artifact-audio-load-progress'>Generating waveform...</div>;
  } else {
    audioInfo = (
      <>
        {/* eslint-disable-next-line jsx-a11y/anchor-is-valid */}
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
      <div ref={containerRef}></div>
    </div>
  );
};

WaveformView.propTypes = {
  data: PropTypes.objectOf(ArrayBuffer).isRequired,
};

export default WaveformView;

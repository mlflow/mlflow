import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import WaveSurfer from 'wavesurfer.js';

class ShowArtifactAudioView extends Component {
  constructor(props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
  }

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
    getArtifact: PropTypes.func,
  };

  static defaultProps = {
    getArtifact: getArtifactContent,
  };

  state = {
    loading: true,
    error: undefined,
    waveform: undefined,
    playing: false,
    audioDuration: 0,
    timeElapsed: 0,
    waveformPainting: undefined,
  };

  waveform = undefined;

  componentDidMount() {
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      if (this.waveform) {
        this.waveform.destroy();
      }
      this.setState({
        playing: false,
        audioDuration: 0,
        timeElapsed: 0,
        loading: true,
        error: undefined,
      });
      this.fetchArtifacts();
    }
  }

  componentWillUnmount() {
    if (this.waveform) {
      this.waveform.destroy();
    }
  }

  render() {
    if (this.state.loading) {
      return <div className='artifact-audio-view-loading'>Loading artifact...</div>;
    }
    if (this.state.error) {
      return (
        <div className='artifact-audio-view-error'>
          Oops we couldn't load your audio file because of an error.
        </div>
      );
    } else {
      let audioInfo;
      if (this.state.waveformPainting) {
        audioInfo = <div className='artifact-audio-load-progress'>Generating waveform...</div>;
      } else {
        audioInfo = (
          <>
            <a title={this.state.playing ? 'Pause' : 'Play'} onClick={this.handlePlayPause}>
              <i
                className={this.state.playing ? 'fas fa-pause' : 'fas fa-play'}
                style={{ fontSize: 21 }}
              ></i>
            </a>
            <div style={{ width: 15 }}></div>
            <div className='timecode'>
              {`${ShowArtifactAudioView.formatTimecode(
                this.state.timeElapsed,
              )} / ${ShowArtifactAudioView.formatTimecode(this.state.audioDuration)}`}
            </div>
          </>
        );
      }

      return (
        <div className='text-area-border-box'>
          <div style={{
            display: 'flex',
            height: 42,
            padding: '8px 16px',
            whiteSpace: 'nowrap',
            textAlign: 'center',
            alignItems: 'center',
            backgroundColor: '#fafafa',
          }}>
            {audioInfo}
          </div>
          <div id='waveform'></div>
        </div>
      );
    }
  }

  static formatTimecode(seconds) {
    return new Date(1000 * seconds).toISOString().substring(11, 19);
  }

  loadAudio(artifact) {
    try {
      this.waveform = WaveSurfer.create({
        waveColor: '#1890ff',
        progressColor: '#2374BB',
        cursorColor: '#333333',
        container: '#waveform',
        responsive: true,
        height: 592,
      });
    } catch (e) {
      throw Error(e);
    }
    const blob = new window.Blob([new Uint8Array(artifact)]);
    try {
      this.waveform.loadBlob(blob);
      this.setState({ waveformPainting: true });
    } catch (e) {
      throw Error(e);
    }

    this.waveform.on('ready', () => {
      this.setState({ audioDuration: this.waveform.getDuration(), waveformPainting: false });
    });

    this.waveform.on('error', (error) => {
      this.setState({ error: error });
    });

    this.waveform.on('audioprocess', () => {
      if (this.waveform.isPlaying()) {
        this.setState({ timeElapsed: this.waveform.getCurrentTime() });
      }
    });

    this.waveform.on('finish', () => {
      this.setState({ playing: false });
    });
  }

  handlePlayPause = () => {
    this.setState({ playing: !this.state.playing });
    this.waveform.playPause();
  };

  fetchArtifacts() {
    const artifactLocation = getSrc(this.props.path, this.props.runUuid);
    this.props
      .getArtifact(artifactLocation, true)
      .then((arrayBuffer) => {
        try {
          this.setState({ waveform: arrayBuffer, loading: false });
          this.loadAudio(arrayBuffer);
        } catch (e) {
          this.setState({ waveform: arrayBuffer, loading: false });
        }
      })
      .catch((error) => {
        this.setState({ error: error, loading: false });
      });
  }
}

export default ShowArtifactAudioView;

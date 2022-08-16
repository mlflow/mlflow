import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';

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
  };

  waveform = WaveSurfer.create({
    barWidth: 3,
    cursorWidth: 1,
    container: '#waveform',
    height: 80,
    progressColor: '#2D5BFF',
    responsive: true,
    waveColor: '#EFEFEF',
    cursorColor: 'black',
  });

  componentDidMount() {
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.fetchArtifacts();
    }
  }

  render() {
    if (this.state.loading) {
      return <div className='artifact-text-view-loading'>Loading...</div>;
    }
    if (this.state.error) {
      return (
        <div className='artifact-text-view-error'>
          Oops we couldn't load your audio file because of an error.
        </div>
      );
    } else {
      const language = getLanguage(this.props.path);
      const overrideStyles = {
        fontFamily: 'Source Code Pro,Menlo,monospace',
        fontSize: '13px',
        overflow: 'auto',
        marginTop: '0',
        width: '100%',
        height: '100%',
      };
      return (
        <div className='ShowArtifactPage'>
          <div className='text-area-border-box'>
            <div className='WaveformContainer'>
              <div id="waveform" />
            </div>
          </div>
        </div>
      );
    }
  }

  loadAudio(artifact) {
    this.waveform.load(artifact);
  }

  handlePlayPause = () => {
    this.setState({ playing: !this.state.playing });
    this.waveform.playPause();
  };

  fetchArtifacts() {
    const artifactLocation = getSrc(this.props.path, this.props.runUuid);
    this.props
      .getArtifact(artifactLocation)
      .then((text) => {
        try {
          this.loadAudio();
        } catch (e) {
          this.setState({ text: text, loading: false }); // Note this needs a banner on it
        }
      })
      .catch((error) => {
        this.setState({ error: error, loading: false });
      });
  }
}
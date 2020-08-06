import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactHtmlView.css';
import Iframe from 'react-iframe';

class ShowArtifactHtmlView extends Component {
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
    html: undefined,
  };

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
      return <div className='artifact-html-view-loading'>Loading...</div>;
    }
    if (this.state.error) {
      console.error('Unable to load HTML artifact, got error ' + this.state.error);
      return (
        <div className='artifact-html-view-error'>
          Oops we couldn't load your file because of an error.
        </div>
      );
    } else {
      return (
        <Iframe
          url=''
          src={this.getBlobURL(this.state.html, 'text/html')}
          width='100%'
          height='500px'
          id='html'
          className='html-iframe'
          display='block'
          position='relative'
          sandbox='allow-scripts'
        />
      );
    }
  }

  getBlobURL = (code, type) => {
    const blob = new Blob([code], { type });
    return URL.createObjectURL(blob);
  };

  /** Fetches artifacts and updates component state with the result */
  fetchArtifacts() {
    const artifactLocation = getSrc(this.props.path, this.props.runUuid);
    this.props
      .getArtifact(artifactLocation)
      .then((html) => {
        this.setState({ html: html, loading: false });
      })
      .catch((error) => {
        this.setState({ error: error, loading: false });
      });
  }
}

export default ShowArtifactHtmlView;

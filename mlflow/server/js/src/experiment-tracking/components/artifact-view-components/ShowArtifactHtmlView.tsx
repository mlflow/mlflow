import React, { Component } from 'react';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import './ShowArtifactHtmlView.css';
import Iframe from 'react-iframe';

type OwnProps = {
  runUuid: string;
  path: string;
  getArtifact?: (...args: any[]) => any;
};

type State = any;

type Props = OwnProps & typeof ShowArtifactHtmlView.defaultProps;

class ShowArtifactHtmlView extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
  }

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

  componentDidUpdate(prevProps: Props) {
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
        <div className='artifact-html-view'>
          <Iframe
            url=''
            src={this.getBlobURL(this.state.html, 'text/html')}
            width='100%'
            height='100%'
            id='html'
            className='html-iframe'
            display='block'
            position='relative'
            sandbox='allow-scripts'
          />
        </div>
      );
    }
  }

  getBlobURL = (code: any, type: any) => {
    const blob = new Blob([code], { type });
    return URL.createObjectURL(blob);
  };

  /** Fetches artifacts and updates component state with the result */
  fetchArtifacts() {
    const artifactLocation = getSrc(this.props.path, this.props.runUuid);
    this.props
      .getArtifact(artifactLocation)
      .then((html: any) => {
        this.setState({ html: html, loading: false });
      })
      .catch((error: any) => {
        this.setState({ error: error, loading: false });
      });
  }
}

export default ShowArtifactHtmlView;

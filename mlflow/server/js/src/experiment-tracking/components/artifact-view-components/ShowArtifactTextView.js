import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from './ShowArtifactUtils';
import './ShowArtifactTextView.css';

class ShowArtifactTextView extends Component {
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
  }

  state = {
    loading: true,
    error: undefined,
    text: undefined,
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
      return (
        <div className='artifact-text-view-loading'>
          Loading...
        </div>
      );
    }
    if (this.state.error) {
      return (
        <div className='artifact-text-view-error'>
          Oops we couldn't load your file because of an error.
        </div>
      );
    } else {
      return (
        <div className="ShowArtifactPage">
          <div className="text-area-border-box">
            <textarea className={"text-area"} readOnly value={this.state.text}/>
          </div>
        </div>
      );
    }
  }

  /** Fetches artifacts and updates component state with the result */
  fetchArtifacts() {
    const artifactLocation = getSrc(this.props.path, this.props.runUuid);
    this.props.getArtifact(artifactLocation).then((text) => {
      this.setState({ text: text, loading: false });
    }).catch((error) => {
      this.setState({ error: error, loading: false });
    });
  }
}

export default ShowArtifactTextView;

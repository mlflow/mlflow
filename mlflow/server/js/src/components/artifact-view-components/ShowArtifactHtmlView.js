import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import './ShowArtifactHtmlView.css';
import { getRequestHeaders } from '../../setupAjaxHeaders';
import Iframe from 'react-iframe';

class ShowArtifactHtmlView extends Component {
  constructor(props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
  }

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
  };

  state = {
    loading: true,
    error: undefined,
    html: undefined,
  };

  componentWillMount() {
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
        <div>
          Loading...
        </div>
      );
    }
    if (this.state.error) {
      console.error("Unable to load HTML artifact, got error " + this.state.error);
      return (
        <div>
          Oops we couldn't load your file because of an error.
        </div>
      );
    } else {
      return (
        <Iframe url=""
                src={this.getBlobURL(this.state.html, 'text/html')}
                width="100%"
                height="500px"
                id="html"
                className="html-iframe"
                display="block"
                position="relative"
                sandbox="allow-scripts"/>

      );
    }
  }

  getBlobURL = (code, type) => {
    const blob = new Blob([code], { type });
    return URL.createObjectURL(blob);
  }

  fetchArtifacts() {
    const getArtifactRequest = new Request(getSrc(this.props.path, this.props.runUuid), {
      method: 'GET',
      redirect: 'follow',
      headers: new Headers(getRequestHeaders(document.cookie)),
    });
    fetch(getArtifactRequest).then((response) => {
      return response.blob();
    }).then((blob) => {
      const fileReader = new FileReader();
      fileReader.onload = (event) => {
        this.setState({ html: event.target.result, loading: false });
      };
      fileReader.readAsText(blob);
    }).catch(error => this.setState({ error: error, loading: false }));
  }
}

export default ShowArtifactHtmlView;

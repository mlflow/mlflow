import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import './ShowArtifactNbView.css';
import { getRequestHeaders } from '../../setupAjaxHeaders';
import { parseNotebook } from "@nteract/commutable";
import NotebookPreview from "@nteract/notebook-preview";


class ShowArtifactNbView extends Component {
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
    immutableNotebook: undefined
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
      return (
        <div>
          Oops we couldn't load your file because of an error.
        </div>
      );
    } else {
      return (
        <div className="notebook-container">
            <NotebookPreview notebook={this.state.immutableNotebook} />
        </div>
      );
    }
  }

  fetchArtifacts() {
    const getArtifactRequest = new Request(getSrc(this.props.path, this.props.runUuid), {
      method: 'GET',
      redirect: 'follow',
      headers: new Headers(getRequestHeaders(document.cookie))
    });
    fetch(getArtifactRequest).then((response) => {
      return response.blob();
    }).then((blob) => {
      const fileReader = new FileReader();
      fileReader.onload = (event) => {
        this.setState({ immutableNotebook: parseNotebook(event.target.result), loading: false });
      };
      fileReader.readAsText(blob);
    });
  }
}

export default ShowArtifactNbView;

import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import './ShowArtifactTextView.css';
import { CSRF_HEADER_NAME, getCsrfToken } from '../../setupCsrf';

class ShowArtifactTextView extends Component {
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
    text: undefined,
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
        <div className="ShowArtifactPage">
          <div className="text-area-border-box">
            <textarea className={"text-area"} readOnly value={this.state.text}/>
          </div>
        </div>
      );
    }
  }

  fetchArtifacts() {
    const getArtifactRequest = new Request(getSrc(this.props.path, this.props.runUuid), {
      method: 'GET',
      redirect: 'follow',
      headers: new Headers({ [CSRF_HEADER_NAME]: getCsrfToken() })
    });
    fetch(getArtifactRequest).then((response) => {
      return response.blob();
    }).then((blob) => {
      const fileReader = new FileReader();
      fileReader.onload = (event) => {
        this.setState({ text: event.target.result, loading: false });
      };
      fileReader.readAsText(blob);
    });
  }
}

export default ShowArtifactTextView;

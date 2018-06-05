import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import './ShowArtifactTextView.css';

class ShowArtifactTextView extends Component {
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
    fetch(getSrc(this.props.path, this.props.runUuid)).then((response) => {
      return response.blob();
    }).then((blob) => {
      const fileReader = new FileReader();
      fileReader.onload = (event) => {
        this.setState({ text: event.target.result, loading: false });
      };
      fileReader.readAsText(blob);
    });
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
      )
    } else {
      return (
        <div className="ShowArtifactPage">
          <div className="text-area-border-box">
            <textarea className={"text-area"} readOnly={true}>
              {this.state.text}
            </textarea>
          </div>
        </div>
      )
    }
  }
}

export default ShowArtifactTextView;

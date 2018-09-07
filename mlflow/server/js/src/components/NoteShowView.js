import React, { Component } from 'react';
import Markdown from 'react-markdown';
import PropTypes from 'prop-types';
import './NoteShowView.css';

class NoteShowView extends Component {
  static propTypes = {
    content: PropTypes.string.isRequired,
  };

  render() {
    return (
      <div className="note-view-outer-container">
        <div className="note-view-text-area">
            <div className="note-view-preview">
              <Markdown className="note-view-preview-content" source={this.props.content}/>
            </div>
        </div>
      </div>
    );
  }
}

export default NoteShowView;

import React, { Component } from 'react';
import { getConverter, sanitizeConvertedHtml } from "../utils/MarkdownUtils";
import PropTypes from 'prop-types';
import './NoteShowView.css';

class NoteShowView extends Component {
  constructor(props) {
    super(props);
    this.converter = getConverter();
  }

  static propTypes = {
    content: PropTypes.string.isRequired,
  };

  render() {
    const htmlContent = sanitizeConvertedHtml(this.converter.makeHtml(this.props.content));
    return (
      <div className="note-view-outer-container">
        <div className="note-view-text-area">
            <div className="note-view-preview note-editor-preview">
              <div className="note-editor-preview-content"
                   // eslint-disable-next-line react/no-danger
                   dangerouslySetInnerHTML={{ __html: htmlContent }}>
              </div>
            </div>
        </div>
      </div>
    );
  }
}

export default NoteShowView;

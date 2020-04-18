import React, { Component } from 'react';
import { getConverter, sanitizeConvertedHtml } from "../../common/utils/MarkdownUtils";
import PropTypes from 'prop-types';
import './NoteShowView.css';

class NoteShowView extends Component {
  constructor(props) {
    super(props);
    this.converter = getConverter();
    this.noteType = props.noteType;
  }

  static propTypes = {
    content: PropTypes.string.isRequired,
    noteType: PropTypes.string.isRequired,
  };

  render() {
    const htmlContent = sanitizeConvertedHtml(this.converter.makeHtml(this.props.content));
    if (this.noteType === "run") {
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
    } else if (this.noteType === "experiment") {
      return (
            <div className="note-view-outer-container">
                <div className="note-view-text-area">
                    <div className="note-editor-preview-content"
                        // eslint-disable-next-line react/no-danger
                         dangerouslySetInnerHTML={{ __html: htmlContent }}>
                    </div>
                </div>
            </div>
      );
    } else {
      return null;
    }
  }
}

export default NoteShowView;

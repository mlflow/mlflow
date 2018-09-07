import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Button, ButtonGroup, ButtonToolbar} from 'react-bootstrap';
import ReactMde from 'react-mde';
import { Converter } from "showdown";
import PropTypes from 'prop-types';
import { setTagApi, getUUID } from '../Actions';
import { NoteInfo, NOTE_TAG_PREFIX } from "../utils/NoteUtils";
import 'react-mde/lib/styles/css/react-mde-all.css';
import './NoteEditorView.css';

class NoteEditorView extends Component {
  constructor(props) {
    super(props);
    this.converter = new Converter();
    // Use Github-like Markdown (i.e. support for tasklists, strikethrough,
    // simple line breaks, code blocks, emojis)
    this.converter.setFlavor('github');
    this.handleMdeValueChange = this.handleMdeValueChange.bind(this);
    this.handleSubmitClick = this.handleSubmitClick.bind(this);
    this.handleCancelClick = this.handleCancelClick.bind(this);
    this.renderButtonToolbar = this.renderButtonToolbar.bind(this);
    this.uneditedContent = this.getUneditedContent();
    this.state.mdeState = { markdown: this.uneditedContent };
    this.state.mdSource = this.uneditedContent;
  }

  state = {
    mdeState: undefined,
    mdSource: undefined,
    isSubmitting: false,
  };

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    noteInfo: PropTypes.instanceOf(NoteInfo).isRequired,
    submitCallback: PropTypes.func.isRequired,
    cancelCallback: PropTypes.func.isRequired,
    dispatch: PropTypes.func.isRequired,
  };

  getUneditedContent() {
    return this.props.noteInfo === undefined ? '' : this.props.noteInfo.content;
  }

  handleMdeValueChange(mdeState) {
    this.setState({ mdeState: mdeState, mdSource: mdeState.markdown });
  }

  handleSubmitClick() {
    this.setState({ isSubmitting: true });
    const submittedContent = this.state.mdSource;
    const setTagRequestId = getUUID();
    return this.props.dispatch(
      setTagApi(this.props.runUuid, NOTE_TAG_PREFIX + 'content', submittedContent, setTagRequestId))
      .then(() => {
        this.setState({ isSubmitting: false });
        this.props.submitCallback(submittedContent, undefined);
      }).catch((err) => {
        this.setState({ isSubmitting: false });
        this.props.submitCallback(submittedContent, err);
      });
  }

  handleCancelClick() {
    this.props.cancelCallback();
  }

  contentHasChanged() {
    return this.state.mdSource !== this.uneditedContent;
  }

  renderButtonToolbar() {
    const canSubmit = this.contentHasChanged() && !this.state.loading && !this.state.isSubmitting;
    return (
      <ButtonToolbar>
        <ButtonGroup>
          <Button className="submit-button" bsStyle="primary" onClick={this.handleSubmitClick}
                  {...(canSubmit ? {active: true} : {disabled: true})}>
            Save
          </Button>
        </ButtonGroup>
        <ButtonGroup>
          <Button onClick={this.handleCancelClick}>
            Cancel
          </Button>
        </ButtonGroup>
      </ButtonToolbar>
    );
  }

  render() {
    return (
      <div className="note-view-outer-container">
        <div className="note-view-text-area">
          <ReactMde
            layout="tabbed"
            onChange={this.handleMdeValueChange}
            editorState={this.state.mdeState}
            generateMarkdownPreview={markdown =>
              Promise.resolve(this.converter.makeHtml(markdown))}
          />
        </div>
        <this.renderButtonToolbar/>
      </div>
    );
  }
}

// eslint-disable-next-line no-unused-vars
const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    dispatch,
  };
};

export default connect(null, mapDispatchToProps)(NoteEditorView);

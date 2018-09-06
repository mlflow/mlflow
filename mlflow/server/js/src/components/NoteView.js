import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Button,
  ButtonGroup, ButtonToolbar,
  ToggleButtonGroup, ToggleButton } from 'react-bootstrap';
import ReactMde from 'react-mde';
import Markdown from 'react-markdown';
import { Converter } from "showdown";
import PropTypes from 'prop-types';
import { setTagApi, getUUID } from '../Actions';
import { NoteInfo, NOTE_TAG_PREFIX } from "../utils/NoteUtils";
import 'react-mde/lib/styles/css/react-mde-all.css';
import './NoteView.css';

const EditPreviewToggle = {
  edit: 0,
  preview: 1,
};

class ShowNoteView extends Component {
  constructor(props) {
    super(props);
    this.converter = new Converter({tables: true, simplifiedAutoLink: true});
    this.handleMdeValueChange = this.handleMdeValueChange.bind(this);
    this.handleSubmitClick = this.handleSubmitClick.bind(this);
    this.handleEditPreviewChange = this.handleEditPreviewChange.bind(this);
    this.renderButtonToolbar = this.renderButtonToolbar.bind(this);
    this.uneditedContent = this.getUneditedContent();
    this.state.mdeState = { markdown: this.uneditedContent };
    this.state.mdSource = this.uneditedContent;
  }

  state = {
    mdeState: undefined,
    mdSource: undefined,
    isSubmitting: false,
    editPreviewToggle: EditPreviewToggle.preview,
  };

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    noteInfo: PropTypes.instanceOf(NoteInfo).isRequired,
    submitCallback: PropTypes.func.isRequired,
    dispatch: PropTypes.func.isRequired,
  };

  getUneditedContent() {
    return this.props.noteInfo.content === undefined ? '' : this.props.noteInfo.content;
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

  handleEditPreviewChange(e) {
    this.setState({ editPreviewToggle: e });
  }

  contentHasChanged() {
    return this.state.mdSource !== this.uneditedContent;
  }

  renderButtonToolbar() {
    const canSubmit = this.contentHasChanged() && !this.state.loading && !this.state.isSubmitting;
    const submitText = this.state.isSubmitting ? "Submitting..." : "Submit";
    return (
      <ButtonToolbar>
        <ToggleButtonGroup
          type="radio" name="options"
          value={this.state.editPreviewToggle} onChange={this.handleEditPreviewChange}
        >
          <ToggleButton value={EditPreviewToggle.preview}>Preview</ToggleButton>
          <ToggleButton value={EditPreviewToggle.edit}>Edit</ToggleButton>
        </ToggleButtonGroup>
        <ButtonGroup>
          <Button bsStyle="primary" onClick={this.handleSubmitClick}
                  {...(canSubmit ? {active: true} : {disabled: true})}>
            { submitText }
          </Button>
        </ButtonGroup>
      </ButtonToolbar>
    );
  }

  render() {
    const inEditMode = this.state.editPreviewToggle === EditPreviewToggle.edit;
    const noteExists = this.props.noteInfo.content !== undefined;
    return (
      <div className="note-view-outer-container">
        <div className="note-view-text-area">
          {inEditMode ?
            <ReactMde
              layout="tabbed"
              onChange={this.handleMdeValueChange}
              editorState={this.state.mdeState}
              generateMarkdownPreview={markdown =>
                Promise.resolve(this.converter.makeHtml(markdown))}
            />
            :
            (
              (noteExists || this.state.mdSource) ?
                <div className="note-view-preview">
                  <Markdown className="note-view-preview-content" source={this.state.mdSource}/>
                </div>
                :
                <div className="note-view-no-content">
                  <text>No note to show! Click <em>Edit</em> to create a new note.</text>
                </div>
            )
          }
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

export default connect(null, mapDispatchToProps)(ShowNoteView);

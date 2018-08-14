import React, { Component } from 'react';
import { Button,
  ButtonGroup, ButtonToolbar,
  ToggleButtonGroup, ToggleButton } from 'react-bootstrap';
import ReactMde from 'react-mde';
import Markdown from 'react-markdown';
import { Converter } from "showdown";
import PropTypes from 'prop-types';
import { setTagApi } from '../Actions';
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
    this.uneditedContent = this.getUneditedContent();
    this.handleMdeValueChange = this.handleMdeValueChange.bind(this);
    this.renderNote = this.renderNote.bind(this);
    this.handleSubmitClick = this.handleSubmitClick.bind(this);
    this.handleEditPreviewChange = this.handleEditPreviewChange.bind(this);
    this.renderButtonToolbar = this.renderButtonToolbar.bind(this);
  }

  state = {
    mdeState: undefined,
    mdSource: undefined,
    editPreviewToggle: EditPreviewToggle.preview,
  };

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    noteInfo: PropTypes.instanceOf(NoteInfo).isRequired,
    submitCallback: PropTypes.func.isRequired,
  };

  getUneditedContent() {
    return this.props.noteInfo.content === undefined ? '' : this.props.noteInfo.content;
  }

  componentWillMount() {
    this.renderNote();
  }

  componentDidUpdate(prevProps) {
    if (this.props.noteInfo !== prevProps.noteInfo) {
      this.renderNote();
    }
  }

  handleMdeValueChange(mdeState) {
    this.setState({ mdeState: mdeState, mdSource: mdeState.markdown });
  }

  handleSubmitClick() {
    const self = this;
    const submittedContent = this.state.mdSource;

    const action = setTagApi(this.props.runUuid, NOTE_TAG_PREFIX + 'content', submittedContent);
    action.payload.then(
      () => {
        self.setState({
          noteInfo: new NoteInfo(submittedContent),
        });
        self.props.submitCallback(submittedContent, undefined);
      },
      (error) => {
        self.props.submitCallback(submittedContent, error);
      }
    );
  }

  handleEditPreviewChange(e) {
    this.setState({ editPreviewToggle: e });
  }

  contentHasChanged() {
    return this.state.mdSource !== this.uneditedContent;
  }

  renderButtonToolbar() {
    const canSubmit = this.contentHasChanged() && !this.state.loading;
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
            Submit
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
              layout="noPreview"
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

  renderNote() {
    this.setState(
      {
        mdeState: {
          markdown: this.uneditedContent
        },
        mdSource: this.uneditedContent
      }
    );
  }
}

export default ShowNoteView;

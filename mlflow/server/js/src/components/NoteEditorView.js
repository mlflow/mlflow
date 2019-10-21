import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Alert, Button, ButtonToolbar} from 'react-bootstrap';
import { Prompt } from 'react-router';
import ReactMde from 'react-mde';
import { getConverter, sanitizeConvertedHtml } from "../utils/MarkdownUtils";
import PropTypes from 'prop-types';
import { setTagApi, setExperimentTagApi, getUUID } from '../Actions';
import { NoteInfo, NOTE_CONTENT_TAG } from "../utils/NoteUtils";
import 'react-mde/lib/styles/css/react-mde-all.css';
import './NoteEditorView.css';

class NoteEditorView extends Component {
  constructor(props) {
    super(props);
    this.converter = getConverter();
    this.handleMdeValueChange = this.handleMdeValueChange.bind(this);
    this.handleSubmitClick = this.handleSubmitClick.bind(this);
    this.handleCancelClick = this.handleCancelClick.bind(this);
    this.handleErrorAlertDismissed = this.handleErrorAlertDismissed.bind(this);
    this.renderButtonToolbar = this.renderButtonToolbar.bind(this);
    this.uneditedContent = this.getUneditedContent();
    this.state = {
      mdeState: {
        markdown: this.uneditedContent,
      },
      mdSource: this.uneditedContent
    };
  }

  state = {
    mdeState: undefined,
    mdSource: undefined,
    error: undefined,
    errorAlertDismissed: false,
    isSubmitting: false,
  };

  static propTypes = {
    runUuid: PropTypes.string,
    experimentId: PropTypes.string,
    type: PropTypes.string.isRequired,
    submitCallback: PropTypes.func.isRequired,
    cancelCallback: PropTypes.func.isRequired,
    dispatch: PropTypes.func.isRequired,
    noteInfo: PropTypes.instanceOf(NoteInfo),
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
    let id = '';
    let tagApiCall = '';
    if (this.props.type === "experiment") {
      id = this.props.experimentId;
      tagApiCall = setExperimentTagApi;
    } else if (this.props.type === "run") {
      id = this.props.runUuid;
      tagApiCall = setTagApi;
    } else {
      throw new Error("Cannot display a note editor for this type.");
    }
    return this.props.dispatch(
      tagApiCall(id, NOTE_CONTENT_TAG, submittedContent, setTagRequestId))
      .then(() => {
        this.setState({ isSubmitting: false, error: undefined });
        this.props.submitCallback(undefined);
      }).catch((err) => {
        this.setState({ isSubmitting: false, error: err, errorAlertDismissed: false });
        this.props.submitCallback(err);
      });
  }

  handleCancelClick() {
    this.props.cancelCallback();
  }

  handleErrorAlertDismissed() {
    this.setState({ errorAlertDismissed: true });
  }

  contentHasChanged() {
    return this.state.mdSource !== this.uneditedContent;
  }

  renderButtonToolbar() {
    const canSubmit = this.contentHasChanged() && !this.state.loading && !this.state.isSubmitting;
    return (
      <div className="note-editor-button-area">
        {this.state.error && !this.state.errorAlertDismissed ?
          <Alert bsStyle="danger" onDismiss={this.handleErrorAlertDismissed}>
            <h4>Failed to save content.</h4>
            <p>
            {this.state.error.getUserVisibleError()}
            </p>
          </Alert>
          :
          null
        }
        <ButtonToolbar>
          <Button className="mlflow-form-button mlflow-save-button"
                  bsStyle="primary"
                  type="submit"
                  onClick={this.handleSubmitClick}
                  {...(canSubmit ? {} : {disabled: true})}>
            Save
          </Button>
          <Button className="mlflow-form-button" onClick={this.handleCancelClick}>
            Cancel
          </Button>
        </ButtonToolbar>
      </div>
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
              Promise.resolve(sanitizeConvertedHtml(this.converter.makeHtml(markdown)))}
          />
        </div>
        <this.renderButtonToolbar/>
        <Prompt
          when={this.contentHasChanged()}
          message={"Are you sure you want to navigate away? " +
                   "Your changes to this run's note will be lost."}/>
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

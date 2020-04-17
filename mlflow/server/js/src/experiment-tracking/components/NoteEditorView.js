import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Alert, Button, ButtonToolbar} from 'react-bootstrap';
import { Tooltip } from 'antd';
import { Prompt } from 'react-router';
import ReactMde, { SvgIcon } from 'react-mde';
import { getConverter, sanitizeConvertedHtml } from "../../common/utils/MarkdownUtils";
import PropTypes from 'prop-types';
import { setTagApi, setExperimentTagApi} from '../actions';
import { NOTE_CONTENT_TAG } from "../utils/NoteUtils";
import 'react-mde/lib/styles/css/react-mde-all.css';
import './NoteEditorView.css';
import { getUUID } from '../../common/utils/ActionUtils';

class NoteEditorView extends Component {
  state = {
    markdown: this.props.defaultMarkdown,
    selectedTab: this.props.defaultSelectedTab,
    error: undefined,
    errorAlertDismissed: false,
    isSubmitting: false,
  };
  converter = getConverter();

  static propTypes = {
    runUuid: PropTypes.string,
    experimentId: PropTypes.string,
    type: PropTypes.string.isRequired,
    submitCallback: PropTypes.func.isRequired,
    cancelCallback: PropTypes.func.isRequired,
    dispatch: PropTypes.func.isRequired,
    defaultMarkdown: PropTypes.string,
    defaultSelectedTab: PropTypes.string,
  };

  static defaultProps = {
    defaultMarkdown: '',
    defaultSelectedTab: 'write',
  }

  handleMdeValueChange = (markdown) => {
    this.setState({ markdown });
  }

  handleSubmitClick = () => {
    this.setState({ isSubmitting: true });
    const submittedContent = this.state.markdown;
    const setTagRequestId = getUUID();
    let id;
    let tagApiCall;
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
      }
    );
  }

  handleCancelClick = () => {
    this.props.cancelCallback();
  }

  handleTabChange = (selectedTab) => {
    this.setState({ selectedTab });
  }

  handleErrorAlertDismissed = () => {
    this.setState({ errorAlertDismissed: true });
  }

  contentHasChanged = () => {
    return this.state.markdown !== this.props.defaultMarkdown;
  }

  renderButtonToolbar = () => {
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
          <Button
            className="mlflow-form-button mlflow-save-button"
            bsStyle="primary"
            type="submit"
            onClick={this.handleSubmitClick}
            disabled={!canSubmit}
          >
            Save
          </Button>
          <Button className="mlflow-form-button" onClick={this.handleCancelClick}>
            Cancel
          </Button>
        </ButtonToolbar>
      </div>
    );
  }

  getSanitizedHtmlContent = () => {
    const { markdown } = this.state;
    return markdown
      ? sanitizeConvertedHtml(this.converter.makeHtml(markdown))
      : null;
  }

  render() {
    const { markdown, selectedTab } = this.state;

    return (
      <div className="note-view-outer-container">
        <div className="note-view-text-area">
          <ReactMde
            value={markdown}
            onChange={this.handleMdeValueChange}
            selectedTab={selectedTab}
            onTabChange={this.handleTabChange}
            generateMarkdownPreview={() =>
              Promise.resolve(this.getSanitizedHtmlContent())
            }
            getIcon={(name) => <TooltipIcon name={name} />}
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

function TooltipIcon(props) {
  const { name } = props;
  return (
    <Tooltip position="top" title={name}>
      <span>
        <SvgIcon icon={name} />
      </span>
    </Tooltip>
  );
}

TooltipIcon.propTypes = { name: PropTypes.string };

// eslint-disable-next-line no-unused-vars
const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    dispatch,
  };
};

export default connect(null, mapDispatchToProps)(NoteEditorView);

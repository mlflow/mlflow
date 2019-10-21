import React, { Component } from 'react';
import { Alert, Button, Icon } from 'antd';
import { Prompt } from 'react-router';
import ReactMde from 'react-mde';
import { getConverter, sanitizeConvertedHtml } from '../../utils/MarkdownUtils';
import PropTypes from 'prop-types';

const PROMPT_MESSAGE =
  "Are you sure you want to navigate away? Your pending text changes will be lost.";

export class EditableNote extends Component {
  static propTypes = {
    defaultMarkdown: PropTypes.string,
    // Callback function handling the confirmed note. It should return a promise.
    onSubmit: PropTypes.func,
    onCancel: PropTypes.func,
    showEditor: PropTypes.bool,
  };

  static defaultProps = {
    defaultMarkdown: '',
    showEditor: false,
    confirmLoading: false,
  };

  state = {
    // Using mdeState is the react-mde@5.8 way of managing the state
    // Reference: https://github.com/andrerpena/react-mde/tree/5.5.0-alpha.4
    // TODO(Zangr) Upgrade react-mde to the latest version for more straightforward state handling
    mdeState: {
      markdown: this.props.defaultMarkdown,
    },
    error: null,
  };

  converter = getConverter();

  handleMdeValueChange = (mdeState) => {
    this.setState({ mdeState });
  };

  handleSubmitClick = () => {
    const { onSubmit } = this.props;
    const { mdeState } = this.state;
    this.setState({ confirmLoading: true });
    if (onSubmit) {
      Promise.resolve(onSubmit(mdeState.markdown))
        .then(() => {
          this.setState({ confirmLoading: false, error: null });
        })
        .catch((e) => {
          this.setState({
            confirmLoading: false,
            error: e.getMessageField ? e.getMessageField() : 'Failed to submit'
          });
        });
    }
  };

  handleCancelClick = () => {
    // Reset to the last defaultMarkdown passed in as props.
    this.setState({
      mdeState: {
        markdown: this.props.defaultMarkdown,
      },
    });
    const { onCancel } = this.props;
    if (onCancel) {
      onCancel();
    }
  };

  contentHasChanged() {
    return this.state.mdeState.markdown !== this.props.defaultMarkdown;
  }

  renderActions() {
    const { confirmLoading } = this.state;
    return (
      <div className='editable-note-actions'>
        <div>
          <Button
            htmlType='button'
            type='primary'
            onClick={this.handleSubmitClick}
            disabled={!this.contentHasChanged() || confirmLoading}
          >
            {confirmLoading && <Icon type='loading' />} Save
          </Button>
          <Button htmlType='button' onClick={this.handleCancelClick} disabled={confirmLoading}>
            Cancel
          </Button>
        </div>
      </div>
    );
  }

  getSanitizedHtmlContent() {
    const { mdeState } = this.state;
    return mdeState.markdown
      ? sanitizeConvertedHtml(this.converter.makeHtml(mdeState.markdown))
      : null;
  }

  render() {
    const { showEditor } = this.props;
    const { mdeState, error } = this.state;
    const htmlContent = this.getSanitizedHtmlContent();
    return (
      <div className='note-view-outer-container'>
        {showEditor ? (
          <React.Fragment>
            <div className='note-view-text-area'>
              <ReactMde
                layout='tabbed'
                editorState={mdeState}
                onChange={this.handleMdeValueChange}
                generateMarkdownPreview={(markdown) =>
                  Promise.resolve(this.getSanitizedHtmlContent(markdown))
                }
              />
            </div>
            {error && (
              <Alert
                type='error'
                message='There was an error submitting your note.'
                description={error}
                closable
              />
            )}
            {this.renderActions()}
            <Prompt when={this.contentHasChanged()} message={PROMPT_MESSAGE} />
          </React.Fragment>
        ) : (
          <HTMLNoteContent content={htmlContent}/>
        )}
      </div>
    );
  }
}

function HTMLNoteContent(props) {
  const { content } = props;
  return content ? (
    <div className="note-view-outer-container">
      <div className="note-view-text-area">
        <div className="note-view-preview note-editor-preview">
          <div className="note-editor-preview-content"
            // eslint-disable-next-line react/no-danger
             dangerouslySetInnerHTML={{ __html: props.content }}>
          </div>
      </div>
      </div>
    </div>
  ) : (
    <div>None</div>
  );
}

HTMLNoteContent.propTypes = { content: PropTypes.string };

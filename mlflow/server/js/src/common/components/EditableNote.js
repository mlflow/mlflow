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
    defaultSelectedTab: PropTypes.string,
    // Callback function handling the confirmed note. It should return a promise.
    onSubmit: PropTypes.func,
    onCancel: PropTypes.func,
    showEditor: PropTypes.bool,
  };

  static defaultProps = {
    defaultMarkdown: '',
    defaultSelectedTab: 'write',
    showEditor: false,
    confirmLoading: false,
  };

  state = {
    markdown: this.props.defaultMarkdown,
    selectedTab: this.props.defaultSelectedTab,
    error: null,
  };

  converter = getConverter();

  handleMdeValueChange = (markdown) => {
    this.setState({ markdown });
  };

  handleTabChange = (selectedTab) => {
    this.setState({ selectedTab });
  }

  handleSubmitClick = () => {
    const { onSubmit } = this.props;
    const { markdown } = this.state;
    this.setState({ confirmLoading: true });
    if (onSubmit) {
      Promise.resolve(onSubmit(markdown))
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
      markdown: this.props.defaultMarkdown,
      selectedTab: this.props.defaultSelectedTab
    });
    const { onCancel } = this.props;
    if (onCancel) {
      onCancel();
    }
  };

  contentHasChanged() {
    return this.state.markdown !== this.props.defaultMarkdown;
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
    const { markdown } = this.state;
    return markdown
      ? sanitizeConvertedHtml(this.converter.makeHtml(markdown))
      : null;
  }

  render() {
    const { showEditor } = this.props;
    const { markdown, selectedTab, error } = this.state;
    const htmlContent = this.getSanitizedHtmlContent();
    return (
      <div className='note-view-outer-container'>
        {showEditor ? (
          <React.Fragment>
            <div className='note-view-text-area'>
              <ReactMde
                value={markdown}
                onChange={this.handleMdeValueChange}
                selectedTab={selectedTab}
                onTabChange={this.handleTabChange}
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

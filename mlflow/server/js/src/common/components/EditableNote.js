import React, { Component } from 'react';
import { LoadingOutlined } from '@ant-design/icons';
import { Alert, Button, Tooltip } from 'antd';
import { Prompt } from 'react-router';
import ReactMde, { SvgIcon } from 'react-mde';
import { getConverter, sanitizeConvertedHtml } from '../utils/MarkdownUtils';
import PropTypes from 'prop-types';
import './EditableNote.css';
import { FormattedMessage, injectIntl } from 'react-intl';

export class EditableNoteImpl extends Component {
  static propTypes = {
    defaultMarkdown: PropTypes.string,
    defaultSelectedTab: PropTypes.string,
    // Callback function handling the confirmed note. It should return a promise.
    onSubmit: PropTypes.func,
    onCancel: PropTypes.func,
    showEditor: PropTypes.bool,
    saveText: PropTypes.object,
    // React-MDE props
    toolbarCommands: PropTypes.array,
    maxEditorHeight: PropTypes.number,
    minEditorHeight: PropTypes.number,
    childProps: PropTypes.object,
    intl: PropTypes.any,
  };

  static defaultProps = {
    defaultMarkdown: '',
    defaultSelectedTab: 'write',
    showEditor: false,
    saveText: (
      <FormattedMessage
        defaultMessage='Save'
        description='Default text for save button on editable notes in MLflow'
      />
    ),
    confirmLoading: false,
    toolbarCommands: [
      ['header', 'bold', 'italic', 'strikethrough'],
      ['link', 'quote', 'code', 'image'],
      ['unordered-list', 'ordered-list', 'checked-list'],
    ],
    maxEditorHeight: 500,
    minEditorHeight: 200,
    childProps: {},
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
  };

  handleSubmitClick = () => {
    const { onSubmit } = this.props;
    const { markdown } = this.state;
    this.setState({ confirmLoading: true });
    if (onSubmit) {
      return Promise.resolve(onSubmit(markdown))
        .then(() => {
          this.setState({ confirmLoading: false, error: null });
        })
        .catch((e) => {
          this.setState({
            confirmLoading: false,
            error:
              e && e.getMessageField
                ? e.getMessageField()
                : this.props.intl.formatMessage({
                    defaultMessage: 'Failed to submit',
                    description:
                      'Message text for failing to save changes in editable note in MLflow',
                  }),
          });
        });
    }
    return null;
  };

  handleCancelClick = () => {
    // Reset to the last defaultMarkdown passed in as props.
    this.setState({
      markdown: this.props.defaultMarkdown,
      selectedTab: this.props.defaultSelectedTab,
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
            className='editable-note-save-button'
            onClick={this.handleSubmitClick}
            disabled={!this.contentHasChanged() || confirmLoading}
          >
            {confirmLoading && <LoadingOutlined />} {this.props.saveText}
          </Button>
          <Button
            htmlType='button'
            className='editable-note-cancel-button'
            onClick={this.handleCancelClick}
            disabled={confirmLoading}
          >
            <FormattedMessage
              defaultMessage='Cancel'
              description='Text for the cancel button in an editable note in MLflow'
            />
          </Button>
        </div>
      </div>
    );
  }

  getSanitizedHtmlContent() {
    const { markdown } = this.state;
    return markdown ? sanitizeConvertedHtml(this.converter.makeHtml(markdown)) : null;
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
                minEditorHeight={this.props.minEditorHeight}
                maxEditorHeight={this.props.maxEditorHeight}
                minPreviewHeight={50}
                childProps={this.props.childProps}
                toolbarCommands={this.props.toolbarCommands}
                onChange={this.handleMdeValueChange}
                selectedTab={selectedTab}
                onTabChange={this.handleTabChange}
                generateMarkdownPreview={(md) => Promise.resolve(this.getSanitizedHtmlContent(md))}
                getIcon={(name) => <TooltipIcon name={name} />}
              />
            </div>
            {error && (
              <Alert
                type='error'
                message={this.props.intl.formatMessage({
                  defaultMessage: 'There was an error submitting your note.',
                  description: 'Error message text when saving an editable note in MLflow',
                })}
                description={error}
                closable
              />
            )}
            {this.renderActions()}
            <Prompt
              when={this.contentHasChanged()}
              message={this.props.intl.formatMessage({
                defaultMessage:
                  'Are you sure you want to navigate away? Your pending text changes will be lost.',
                description:
                  'Prompt text for navigating away before saving changes in editable note in' +
                  ' MLflow',
              })}
            />
          </React.Fragment>
        ) : (
          <HTMLNoteContent content={htmlContent} />
        )}
      </div>
    );
  }
}

function TooltipIcon(props) {
  const { name } = props;
  return (
    <Tooltip position='top' title={name}>
      <span>
        <SvgIcon icon={name} />
      </span>
    </Tooltip>
  );
}

function HTMLNoteContent(props) {
  const { content } = props;
  return content ? (
    <div className='note-view-outer-container'>
      <div className='note-view-text-area'>
        <div className='note-view-preview note-editor-preview'>
          <div
            className='note-editor-preview-content'
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{ __html: props.content }}
          ></div>
        </div>
      </div>
    </div>
  ) : (
    <div>
      <FormattedMessage
        defaultMessage='None'
        description='Default text for no content in an editable note in MLflow'
      />
    </div>
  );
}

TooltipIcon.propTypes = { name: PropTypes.string };
HTMLNoteContent.propTypes = { content: PropTypes.string };

export const EditableNote = injectIntl(EditableNoteImpl);

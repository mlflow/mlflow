/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { Alert, Button, LegacyTooltip, useDesignSystemTheme } from '@databricks/design-system';
import { Prompt } from './Prompt';
import 'react-mde/lib/styles/css/react-mde-all.css';
import ReactMde, { SvgIcon } from 'react-mde';
import { forceAnchorTagNewTab, getMarkdownConverter, sanitizeConvertedHtml } from '../utils/MarkdownUtils';
import './EditableNote.css';
import type { IntlShape } from 'react-intl';
import { FormattedMessage, injectIntl } from 'react-intl';

type EditableNoteImplProps = {
  defaultMarkdown?: string;
  defaultSelectedTab?: string;
  onSubmit?: (...args: any[]) => any;
  onCancel?: (...args: any[]) => any;
  showEditor?: boolean;
  saveText?: any;
  toolbarCommands?: any[];
  maxEditorHeight?: number;
  minEditorHeight?: number;
  childProps?: any;
  intl: IntlShape;
};

type EditableNoteImplState = any;

const getReactMdeIcon = (name: string) => <TooltipIcon name={name} />;

export class EditableNoteImpl extends Component<EditableNoteImplProps, EditableNoteImplState> {
  static defaultProps = {
    defaultMarkdown: '',
    defaultSelectedTab: 'write',
    showEditor: false,
    saveText: (
      <FormattedMessage defaultMessage="Save" description="Default text for save button on editable notes in MLflow" />
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

  converter = getMarkdownConverter();

  componentDidUpdate(prevProps: EditableNoteImplProps) {
    if (
      prevProps.defaultMarkdown !== this.props.defaultMarkdown ||
      prevProps.defaultSelectedTab !== this.props.defaultSelectedTab
    ) {
      this.setState({
        markdown: this.props.defaultMarkdown,
        selectedTab: this.props.defaultSelectedTab,
      });
    }
  }

  handleMdeValueChange = (markdown: any) => {
    this.setState({ markdown });
  };

  handleTabChange = (selectedTab: any) => {
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
                    description: 'Message text for failing to save changes in editable note in MLflow',
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
    // @ts-expect-error TS(2339): Property 'confirmLoading' does not exist on type '... Remove this comment to see the full error message
    const { confirmLoading } = this.state;
    return (
      <div className="mlflow-editable-note-actions" data-testid="editable-note-actions">
        <div>
          <Button
            componentId="codegen_mlflow_app_src_common_components_editablenote.tsx_114"
            type="primary"
            className="editable-note-save-button"
            onClick={this.handleSubmitClick}
            disabled={!this.contentHasChanged() || confirmLoading}
            loading={confirmLoading}
            data-testid="editable-note-save-button"
          >
            {this.props.saveText}
          </Button>
          <Button
            componentId="codegen_mlflow_app_src_common_components_editablenote.tsx_124"
            htmlType="button"
            className="editable-note-cancel-button"
            onClick={this.handleCancelClick}
            disabled={confirmLoading}
          >
            <FormattedMessage
              defaultMessage="Cancel"
              description="Text for the cancel button in an editable note in MLflow"
            />
          </Button>
        </div>
      </div>
    );
  }

  getSanitizedHtmlContent() {
    const { markdown } = this.state;
    if (markdown) {
      const sanitized = sanitizeConvertedHtml(this.converter.makeHtml(markdown));
      return forceAnchorTagNewTab(sanitized);
    }
    return null;
  }

  render() {
    const { showEditor } = this.props;
    const { markdown, selectedTab, error } = this.state;
    const htmlContent = this.getSanitizedHtmlContent();
    return (
      <div className="note-view-outer-container" data-testid="note-view-outer-container">
        {showEditor ? (
          <React.Fragment>
            <div className="note-view-text-area">
              <ReactMde
                value={markdown}
                minEditorHeight={this.props.minEditorHeight}
                maxEditorHeight={this.props.maxEditorHeight}
                minPreviewHeight={50}
                childProps={this.props.childProps}
                toolbarCommands={this.props.toolbarCommands}
                onChange={this.handleMdeValueChange}
                // @ts-expect-error TS(2322): Type 'string' is not assignable to type '"write" |... Remove this comment to see the full error message
                selectedTab={selectedTab}
                onTabChange={this.handleTabChange}
                // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
                generateMarkdownPreview={(md) => Promise.resolve(this.getSanitizedHtmlContent(md))}
                getIcon={getReactMdeIcon}
              />
            </div>
            {error && (
              <Alert
                componentId="codegen_mlflow_app_src_common_components_editablenote.tsx_178"
                type="error"
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
                defaultMessage: 'Are you sure you want to navigate away? Your pending text changes will be lost.',
                description: 'Prompt text for navigating away before saving changes in editable note in MLflow',
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

type TooltipIconProps = {
  name?: string;
};

function TooltipIcon(props: TooltipIconProps) {
  const { theme } = useDesignSystemTheme();
  const { name } = props;
  return (
    // @ts-expect-error TS(2322): Type '{ children: Element; position: string; title... Remove this comment to see the full error message
    <LegacyTooltip position="top" title={name}>
      <span css={{ color: theme.colors.textPrimary }}>
        {/* @ts-expect-error TS(2322): Type 'string | undefined' is not assignable to typ... Remove this comment to see the full error message */}
        <SvgIcon icon={name} />
      </span>
    </LegacyTooltip>
  );
}

type HTMLNoteContentProps = {
  content?: string;
};

function HTMLNoteContent(props: HTMLNoteContentProps) {
  const { content } = props;
  return content ? (
    <div className="note-view-outer-container" data-testid="note-view-outer-container">
      <div className="note-view-text-area">
        <div className="note-view-preview note-editor-preview">
          <div
            className="note-editor-preview-content"
            data-testid="note-editor-preview-content"
            // @ts-expect-error TS(2322): Type 'string | undefined' is not assignable to typ... Remove this comment to see the full error message
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{ __html: props.content }}
          />
        </div>
      </div>
    </div>
  ) : (
    <div>
      <FormattedMessage defaultMessage="None" description="Default text for no content in an editable note in MLflow" />
    </div>
  );
}

export const EditableNote = injectIntl(EditableNoteImpl);

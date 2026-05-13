import { useState } from 'react';
import { EditableNote } from '../../../../common/components/EditableNote';
import type { KeyValueEntity } from '../../../../common/types';
import { NOTE_CONTENT_TAG } from '../../../utils/NoteUtils';
import { Button, PencilIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useDispatch } from 'react-redux';
import type { ThunkDispatch } from '../../../../redux-types';
import { setTagApi } from '../../../actions';
import { FormattedMessage, useIntl } from 'react-intl';
import { CollapsibleContainer } from '@mlflow/mlflow/src/common/components/CollapsibleContainer';

/**
 * Displays editable description section in run detail overview.
 */
export const RunViewDescriptionBox = ({
  runUuid,
  tags,
  onDescriptionChanged,
  isFlatLayout = false,
}: {
  runUuid: string;
  tags: Record<string, KeyValueEntity>;
  onDescriptionChanged: () => void | Promise<void>;
  /**
   * When true, renders in a compact horizontal layout.
   * - Removes section title and bottom margin
   * - Places edit button inline with content
   * - Hides "No description" placeholder
   * When false (default), renders in standard vertical layout with full styling.
   */
  isFlatLayout?: boolean;
}) => {
  const noteContent = tags[NOTE_CONTENT_TAG]?.value || '';

  const [showNoteEditor, setShowNoteEditor] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const dispatch = useDispatch<ThunkDispatch>();

  const handleSubmitEditNote = (markdown: string) =>
    dispatch(setTagApi(runUuid, NOTE_CONTENT_TAG, markdown))
      .then(onDescriptionChanged)
      .then(() => setShowNoteEditor(false));
  const handleCancelEditNote = () => setShowNoteEditor(false);

  const isEmpty = !noteContent;

  const TypographyWrapper = isFlatLayout ? Typography.Text : Typography.Title;

  return (
    <div css={{ marginBottom: isFlatLayout ? 0 : theme.spacing.md }}>
      <TypographyWrapper
        level={4}
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
        }}
        withoutMargins={isFlatLayout}
      >
        {!isFlatLayout && (
          <FormattedMessage
            defaultMessage="Description"
            description="Run page > Overview > Description section > Section title"
          />
        )}
        {isFlatLayout && !isEmpty ? null : (
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewdescriptionbox.tsx_46"
            size="small"
            type="tertiary"
            aria-label={intl.formatMessage({
              defaultMessage: 'Edit description',
              description: 'Run page > Overview > Description section > Edit button label',
            })}
            onClick={() => setShowNoteEditor(true)}
            icon={<PencilIcon />}
            css={{
              ...(isFlatLayout && {
                '&&': {
                  padding: '0 !important',
                },
              }),
            }}
          >
            {isFlatLayout ? (
              <FormattedMessage
                defaultMessage="Add description"
                description="Run page > Overview > Description section > Add description button"
              />
            ) : null}
          </Button>
        )}
      </TypographyWrapper>
      {isEmpty && !showNoteEditor && !isFlatLayout && (
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="No description"
            description="Run page > Overview > Description section > Empty value placeholder"
          />
        </Typography.Hint>
      )}
      {(!isEmpty || showNoteEditor) &&
        (isFlatLayout ? (
          <div
            css={{
              display: 'flex',
              flexDirection: 'row',
              justifyContent: 'center',
              alignItems: 'center',
              gap: theme.spacing.sm,
            }}
          >
            <div
              css={{
                '&& p': {
                  margin: '0 !important',
                },
              }}
            >
              {showNoteEditor ? (
                <EditableNote
                  defaultMarkdown={noteContent}
                  onSubmit={handleSubmitEditNote}
                  onCancel={handleCancelEditNote}
                  showEditor={showNoteEditor}
                />
              ) : (
                <CollapsibleContainer isExpanded={isExpanded} setIsExpanded={setIsExpanded} maxHeight={100}>
                  <EditableNote
                    defaultMarkdown={noteContent}
                    onSubmit={handleSubmitEditNote}
                    onCancel={handleCancelEditNote}
                    showEditor={showNoteEditor}
                  />
                </CollapsibleContainer>
              )}
            </div>
            {!showNoteEditor && (
              <Button
                componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewdescriptionbox.tsx_46"
                size="small"
                type="tertiary"
                aria-label={intl.formatMessage({
                  defaultMessage: 'Edit description',
                  description: 'Run page > Overview > Description section > Edit button label',
                })}
                onClick={() => setShowNoteEditor(true)}
                icon={<PencilIcon />}
              />
            )}
          </div>
        ) : showNoteEditor ? (
          <EditableNote
            defaultMarkdown={noteContent}
            onSubmit={handleSubmitEditNote}
            onCancel={handleCancelEditNote}
            showEditor={showNoteEditor}
          />
        ) : (
          <CollapsibleContainer isExpanded={isExpanded} setIsExpanded={setIsExpanded} maxHeight={100}>
            <EditableNote
              defaultMarkdown={noteContent}
              onSubmit={handleSubmitEditNote}
              onCancel={handleCancelEditNote}
              showEditor={showNoteEditor}
            />
          </CollapsibleContainer>
        ))}
    </div>
  );
};

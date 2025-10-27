import { useState } from 'react';
import { EditableNote } from '../../../../common/components/EditableNote';
import type { KeyValueEntity } from '../../../../common/types';
import { NOTE_CONTENT_TAG } from '../../../utils/NoteUtils';
import { Button, PencilIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useDispatch } from 'react-redux';
import type { ThunkDispatch } from '../../../../redux-types';
import { setTagApi } from '../../../actions';
import { FormattedMessage, useIntl } from 'react-intl';

/**
 * Displays editable description section in run detail overview.
 */
export const RunViewDescriptionBox = ({
  runUuid,
  tags,
  onDescriptionChanged,
}: {
  runUuid: string;
  tags: Record<string, KeyValueEntity>;
  onDescriptionChanged: () => void | Promise<void>;
}) => {
  const noteContent = tags[NOTE_CONTENT_TAG]?.value || '';

  const [showNoteEditor, setShowNoteEditor] = useState(false);
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const dispatch = useDispatch<ThunkDispatch>();

  const handleSubmitEditNote = (markdown: string) =>
    dispatch(setTagApi(runUuid, NOTE_CONTENT_TAG, markdown))
      .then(onDescriptionChanged)
      .then(() => setShowNoteEditor(false));
  const handleCancelEditNote = () => setShowNoteEditor(false);

  const isEmpty = !noteContent;

  return (
    <div css={{ marginBottom: theme.spacing.md }}>
      <Typography.Title level={4} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <FormattedMessage
          defaultMessage="Description"
          description="Run page > Overview > Description section > Section title"
        />
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
      </Typography.Title>
      {isEmpty && !showNoteEditor && (
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="No description"
            description="Run page > Overview > Description section > Empty value placeholder"
          />
        </Typography.Hint>
      )}
      {(!isEmpty || showNoteEditor) && (
        <EditableNote
          defaultMarkdown={noteContent}
          onSubmit={handleSubmitEditNote}
          onCancel={handleCancelEditNote}
          showEditor={showNoteEditor}
        />
      )}
    </div>
  );
};

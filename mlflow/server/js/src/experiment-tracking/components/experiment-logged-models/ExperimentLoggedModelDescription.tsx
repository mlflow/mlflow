import { useState } from 'react';
import { EditableNote } from '../../../common/components/EditableNote';
import type { LoggedModelProto } from '../../types';
import { NOTE_CONTENT_TAG } from '../../utils/NoteUtils';
import { Button, PencilIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { usePatchLoggedModelsTags } from '../../hooks/logged-models/usePatchLoggedModelsTags';
import { useUserActionErrorHandler } from '@databricks/web-shared/metrics';

/**
 * Displays editable description section in logged model detail overview.
 */
export const ExperimentLoggedModelDescription = ({
  loggedModel,
  onDescriptionChanged,
}: {
  loggedModel?: LoggedModelProto;
  onDescriptionChanged: () => void | Promise<void>;
}) => {
  const descriptionContent = loggedModel?.info?.tags?.find((tag) => tag.key === NOTE_CONTENT_TAG)?.value ?? undefined;

  const [showNoteEditor, setShowDescriptionEditor] = useState(false);
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { patch } = usePatchLoggedModelsTags({ loggedModelId: loggedModel?.info?.model_id });
  const { handleError } = useUserActionErrorHandler();

  const handleSubmitEditDescription = async (markdown: string) => {
    try {
      await patch({ [NOTE_CONTENT_TAG]: markdown });
      await onDescriptionChanged();
      setShowDescriptionEditor(false);
    } catch (error: any) {
      handleError(error);
    }
  };

  const handleCancelEditDescription = () => setShowDescriptionEditor(false);

  const isEmpty = !descriptionContent;

  return (
    <div css={{ marginBottom: theme.spacing.md }}>
      <Typography.Title level={4} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <FormattedMessage
          defaultMessage="Description"
          description="Label for descriptions section on the logged models details page"
        />
        <Button
          componentId="mlflow.logged_models.details.description.edit"
          size="small"
          type="tertiary"
          aria-label={intl.formatMessage({
            defaultMessage: 'Edit description',
            description: 'Label for the edit description button on the logged models details page',
          })}
          onClick={() => setShowDescriptionEditor(true)}
          icon={<PencilIcon />}
        />
      </Typography.Title>
      {isEmpty && !showNoteEditor && (
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="No description"
            description="Placeholder text when no description is provided for the logged model displayed in the logged models details page"
          />
        </Typography.Hint>
      )}
      {(!isEmpty || showNoteEditor) && (
        <EditableNote
          defaultMarkdown={descriptionContent}
          onSubmit={handleSubmitEditDescription}
          onCancel={handleCancelEditDescription}
          showEditor={showNoteEditor}
        />
      )}
    </div>
  );
};

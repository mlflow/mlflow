import type { RegisteredPrompt } from '../types';
import { Button, PencilIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useUpdateRegisteredPromptTags } from '../hooks/useUpdateRegisteredPromptTags';
import { isUserFacingTag } from '../../../../common/utils/TagUtils';
import { KeyValueTag } from '../../../../common/components/KeyValueTag';

export const PromptsListTableTagsBox = ({
  promptEntity,
  onTagsUpdated,
}: {
  promptEntity?: RegisteredPrompt;
  onTagsUpdated?: () => void;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const { EditTagsModal, showEditPromptTagsModal } = useUpdateRegisteredPromptTags({ onSuccess: onTagsUpdated });

  const visibleTagList = promptEntity?.tags.filter((tag) => isUserFacingTag(tag.key)) || [];
  const containsTags = visibleTagList.length > 0;

  return (
    <div
      css={{
        paddingTop: theme.spacing.xs,
        paddingBottom: theme.spacing.xs,

        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        '> *': {
          marginRight: '0 !important',
        },
        gap: theme.spacing.xs,
      }}
    >
      {visibleTagList?.map((tag) => (
        <KeyValueTag key={tag.key} tag={tag} />
      ))}
      <Button
        componentId="mlflow.prompts.details.tags.edit"
        size="small"
        icon={!containsTags ? undefined : <PencilIcon />}
        onClick={() => promptEntity && showEditPromptTagsModal(promptEntity)}
        aria-label={intl.formatMessage({
          defaultMessage: 'Edit tags',
          description: 'Label for the edit tags button on the registered prompt details page"',
        })}
        children={
          !containsTags ? (
            <FormattedMessage
              defaultMessage="Add tags"
              description="Label for the add tags button on the registered prompt details page"
            />
          ) : undefined
        }
        type="tertiary"
      />
      {EditTagsModal}
    </div>
  );
};

import { RegisteredPrompt } from '../types';
import { Button, PencilIcon, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useUpdateModelVersionTracesTagsModal } from '../hooks/useUpdateRegisteredPromptTags';
import { isUserFacingTag } from '../../../../common/utils/TagUtils';

export const PromptsListTableTagsBox = ({
  promptEntity,
  onTagsUpdated,
}: {
  promptEntity?: RegisteredPrompt;
  onTagsUpdated?: () => void;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const { EditTagsModal, showEditPromptTagsModal } = useUpdateModelVersionTracesTagsModal({ onSuccess: onTagsUpdated });

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
        <Tag componentId="TODO" key={tag.key}>
          <Typography.Text>
            <Typography.Text bold>{tag.key}:</Typography.Text> {tag.value}
          </Typography.Text>
        </Tag>
      ))}
      <Button
        componentId="TODO"
        size="small"
        icon={!containsTags ? undefined : <PencilIcon />}
        onClick={() => promptEntity && showEditPromptTagsModal(promptEntity)}
        aria-label={intl.formatMessage({
          defaultMessage: 'Edit tags',
          description: 'TODO',
        })}
        children={!containsTags ? <FormattedMessage defaultMessage="Add tags" description="TODO" /> : undefined}
        type="tertiary"
      />
      {EditTagsModal}
    </div>
  );
};

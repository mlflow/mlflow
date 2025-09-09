import { useDesignSystemTheme } from '@databricks/design-system';
import { Tag, Button, PencilIcon } from '@databricks/design-system';
import { KeyValueEntity } from '../types';
import { FormattedMessage } from 'react-intl';

interface Props {
  tags: KeyValueEntity[];
  onEdit: () => void;
}

export const TagList = ({ tags, onEdit }: Props) => {
  const { theme } = useDesignSystemTheme();

  const hasTags = tags.length > 0;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        flexWrap: 'wrap',
        alignItems: 'center',
        gap: theme.spacing.xs,
      }}
    >
      {tags?.map((tag) => (
        <Tag key={tag.key} componentId={tag.key}>
          <strong>
            {tag.key}
            {tag.value ? ':' : null}
          </strong>
          &nbsp;
          {tag.value}
        </Tag>
      ))}
      <Button
        componentId="databricks-experiment-tracking-prompt-edit-tags-button"
        size="small"
        icon={hasTags ? <PencilIcon /> : undefined}
        onClick={onEdit}
      >
        {hasTags ? null : (
          <FormattedMessage defaultMessage="Add tags" description="Add new prompt version tags button" />
        )}
      </Button>
    </div>
  );
};

import { ColumnDef } from '@tanstack/react-table';
import { RegisteredPrompt } from '../types';
import { Button, PencilIcon, Tag, Typography } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { isUserFacingTag } from '../../../../common/utils/TagUtils';
import { PromptsTableMetadata } from '../utils';

export const PromptsListTableTagsCell: ColumnDef<RegisteredPrompt>['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const intl = useIntl();

  const { onEditTags } = meta as PromptsTableMetadata;

  const visibleTagList = original?.tags?.filter((tag) => isUserFacingTag(tag.key)) || [];
  const containsTags = visibleTagList.length > 0;

  return (
    <div css={{ display: 'flex' }}>
      <div css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {visibleTagList?.map((tag) => (
          <Tag componentId="TODO" key={tag.key}>
            <Typography.Text>
              <Typography.Text bold>{tag.key}:</Typography.Text> {tag.value}
            </Typography.Text>
          </Tag>
        ))}
      </div>
      <Button
        componentId="TODO"
        size="small"
        icon={!containsTags ? undefined : <PencilIcon />}
        onClick={() => onEditTags?.(original)}
        aria-label={intl.formatMessage({
          defaultMessage: 'Edit tags',
          description: 'TODO',
        })}
        children={!containsTags ? <FormattedMessage defaultMessage="Add tags" description="TODO" /> : undefined}
        css={{
          flexShrink: 0,
          opacity: 0,
          '[role=row]:hover &': {
            opacity: 1,
          },
          '[role=row]:focus-within &': {
            opacity: 1,
          },
        }}
        type="tertiary"
      />
    </div>
  );
};

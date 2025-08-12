import { Button, PencilIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { KeyValueTag } from './KeyValueTag';
import { MLFLOW_INTERNAL_PREFIX } from '../../utils/TraceUtils';

export const TagsCellRenderer = ({
  onAddEditTags,
  tags,
  baseComponentId,
}: {
  tags: { key: string; value: string }[];
  onAddEditTags?: () => void;
  baseComponentId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const visibleTagList = tags?.filter(({ key }) => key && !key.startsWith(MLFLOW_INTERNAL_PREFIX)) || [];
  const containsTags = visibleTagList.length > 0;
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        flexWrap: 'wrap',
        columnGap: theme.spacing.xs,
        rowGap: theme.spacing.xs,
      }}
    >
      {visibleTagList.map((tag) => (
        <KeyValueTag
          key={tag.key}
          tag={tag}
          css={{ marginRight: 0 }}
          charLimit={20}
          maxWidth={150}
          enableFullViewModal
        />
      ))}
      {onAddEditTags && (
        <Button
          componentId={`${baseComponentId}.traces_table.edit_tag`}
          size="small"
          icon={!containsTags ? undefined : <PencilIcon />}
          onClick={onAddEditTags}
          children={
            !containsTags ? (
              <FormattedMessage
                defaultMessage="Add tags"
                description="Button text to add tags to a trace in the experiment traces table"
              />
            ) : undefined
          }
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
      )}
    </div>
  );
};

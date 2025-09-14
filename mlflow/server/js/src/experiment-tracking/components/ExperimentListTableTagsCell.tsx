import { Button, PencilIcon } from '@databricks/design-system';
import 'react-virtualized/styles.css';
import { FormattedMessage, useIntl } from 'react-intl';
import { KeyValueTag } from '../../common/components/KeyValueTag';
import { isUserFacingTag } from '../../common/utils/TagUtils';
import type { ExperimentTableColumnDef, ExperimentTableMetadata } from './ExperimentListTable';

export const ExperimentListTableTagsCell: ExperimentTableColumnDef['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const intl = useIntl();

  const { onEditTags } = meta as ExperimentTableMetadata;

  const visibleTagList = original?.tags?.filter((tag) => isUserFacingTag(tag.key)) || [];
  const containsTags = visibleTagList.length > 0;

  return (
    <div css={{ display: 'flex' }}>
      <div css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'flex' }}>
        {visibleTagList?.map((tag) => (
          <KeyValueTag key={tag.key} tag={tag} />
        ))}
      </div>
      <Button
        componentId="mlflow.experiment.list.tag.add"
        size="small"
        icon={!containsTags ? undefined : <PencilIcon />}
        onClick={() => onEditTags?.(original)}
        aria-label={intl.formatMessage({
          defaultMessage: 'Edit tags',
          description: 'Label for the edit tags button in the experiment list table',
        })}
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
      >
        {!containsTags ? (
          <FormattedMessage
            defaultMessage="Add tags"
            description="Label for the add tags button in the experiment list table"
          />
        ) : undefined}
      </Button>
    </div>
  );
};

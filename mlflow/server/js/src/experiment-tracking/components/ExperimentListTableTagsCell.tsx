import { Button, ChevronDoubleDownIcon, ChevronDoubleUpIcon, PencilIcon, Tooltip } from '@databricks/design-system';
import 'react-virtualized/styles.css';
import { FormattedMessage, useIntl } from 'react-intl';
import { useState } from 'react';
import { KeyValueTag } from '../../common/components/KeyValueTag';
import { isUserFacingTag } from '../../common/utils/TagUtils';
import type { ExperimentTableColumnDef, ExperimentTableMetadata } from './ExperimentListTable';

const MAX_VISIBLE_TAGS = 3;

export const ExperimentListTableTagsCell: ExperimentTableColumnDef['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const intl = useIntl();
  const [showMore, setShowMore] = useState(false);

  const { onEditTags } = meta as ExperimentTableMetadata;

  const visibleTagList = original?.tags?.filter((tag) => isUserFacingTag(tag.key)) || [];
  const containsTags = visibleTagList.length > 0;
  const hasMoreTags = visibleTagList.length > MAX_VISIBLE_TAGS;

  const tagsToDisplay = showMore ? visibleTagList : visibleTagList.slice(0, MAX_VISIBLE_TAGS);
  const remainingTagsCount = visibleTagList.length - MAX_VISIBLE_TAGS;

  // Render tags for overflow tooltip
  const overflowTagsTooltipContent = hasMoreTags && !showMore
    ? visibleTagList.slice(MAX_VISIBLE_TAGS).map((tag) => `${tag.key}: ${tag.value || ''}`).join('\n')
    : null;

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <div css={{ display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center' }}>
        {tagsToDisplay?.map((tag) => (
          <KeyValueTag key={tag.key} tag={tag} />
        ))}
        {hasMoreTags && !showMore && (
          <Tooltip
            componentId="mlflow.experiment.list.tags.overflow_tooltip"
            content={overflowTagsTooltipContent}
            title={
              <FormattedMessage
                defaultMessage="Additional tags"
                description="Tooltip title for overflow tags in experiments table"
              />
            }
          >
            <Button
              componentId="mlflow.experiment.list.tags.show_more"
              size="small"
              onClick={() => setShowMore(true)}
              type="tertiary"
              css={{ flexShrink: 0 }}
            >
              <FormattedMessage
                defaultMessage="+{count} more"
                description="Show more tags button in experiments table"
                values={{ count: remainingTagsCount }}
              />
            </Button>
          </Tooltip>
        )}
        {showMore && hasMoreTags && (
          <Button
            componentId="mlflow.experiment.list.tags.show_less"
            size="small"
            onClick={() => setShowMore(false)}
            icon={<ChevronDoubleUpIcon />}
            type="tertiary"
            css={{ flexShrink: 0 }}
          >
            <FormattedMessage
              defaultMessage="Show less"
              description="Show less tags button in experiments table"
            />
          </Button>
        )}
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

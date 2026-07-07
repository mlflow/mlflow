import { Button, Overflow, PencilIcon, useDesignSystemTheme } from '@databricks/design-system';
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
  const { theme } = useDesignSystemTheme();

  const { onEditTags } = meta as ExperimentTableMetadata;

  const visibleTagList = original?.tags?.filter((tag) => isUserFacingTag(tag.key)) || [];
  const containsTags = visibleTagList.length > 0;

  return (
    <div css={{ display: 'flex', alignItems: 'center' }}>
      {containsTags && (
        <Overflow noMargin className="experiment-tags-overflow">
          {visibleTagList.map((tag) => (
            <KeyValueTag key={tag.key} tag={tag} />
          ))}
        </Overflow>
      )}
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
          marginLeft: containsTags ? theme.spacing.sm : 0,
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

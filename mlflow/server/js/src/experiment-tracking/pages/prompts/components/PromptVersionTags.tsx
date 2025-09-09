import { useState } from 'react';
import { Button, PencilIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { KeyValueTag } from '@mlflow/mlflow/src/common/components/KeyValueTag';
import type { KeyValueEntity } from '../../../../common/types';
import { isNil } from 'lodash';

export const PromptVersionTags = ({
  tags,
  onEditVersionMetadata,
}: {
  tags: KeyValueEntity[];
  onEditVersionMetadata?: () => void;
}) => {
  const [showAll, setShowAll] = useState(false);
  const { theme } = useDesignSystemTheme();

  const displayThreshold = 3;
  const visibleCount = showAll ? tags.length : Math.min(displayThreshold, tags.length || 0);
  const hasMore = tags.length > displayThreshold;
  const shouldAllowEditingMetadata = !isNil(onEditVersionMetadata);

  const editButton =
    tags.length > 0 ? (
      <Button
        componentId="mlflow.prompts.details.version.edit_tags"
        size="small"
        icon={<PencilIcon />}
        onClick={onEditVersionMetadata}
      />
    ) : (
      <Button
        componentId="mlflow.prompts.details.version.add_tags"
        size="small"
        type="link"
        onClick={onEditVersionMetadata}
      >
        <FormattedMessage
          defaultMessage="Add"
          description="Model registry > model version table > metadata column > 'add' button label"
        />
      </Button>
    );

  return (
    <>
      <Typography.Text bold>
        <FormattedMessage
          defaultMessage="Metadata:"
          description="A key-value pair for the metadata in the prompt details page"
        />
      </Typography.Text>
      <div>
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
            {tags.slice(0, visibleCount).map((tag) => (
              <KeyValueTag css={{ margin: 0 }} key={tag.key} tag={tag} />
            ))}
            {shouldAllowEditingMetadata && editButton}
            {!shouldAllowEditingMetadata && tags.length === 0 && <Typography.Hint>—</Typography.Hint>}
            {hasMore && (
              <Button
                componentId="mlflow.prompts.details.version.tags.show_more"
                size="small"
                type="link"
                onClick={() => setShowAll(!showAll)}
              >
                {showAll ? (
                  <FormattedMessage
                    defaultMessage="Show less"
                    description="Label for a link that shows less tags when clicked"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="{count} more..."
                    description="Label for a link that renders the remaining tags when clicked"
                    values={{ count: tags.length - visibleCount }}
                  />
                )}
              </Button>
            )}
          </div>
        </>
      </div>
    </>
  );
};

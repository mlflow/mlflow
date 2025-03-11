import { useState } from 'react';
import { Button, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { KeyValueTag } from '@mlflow/mlflow/src/common/components/KeyValueTag';
import { KeyValueEntity } from '../../../types';

export const PromptVersionTags = ({tags}: {tags: KeyValueEntity[]}) => {
  const [showAll, setShowAll] = useState(false);
  const { theme } = useDesignSystemTheme();

  const displayThreshold = 3;
  const visibleCount = showAll ? tags.length : Math.min(displayThreshold, tags.length || 0);
  const hasMore = tags.length > displayThreshold;

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
              {tags.slice(0, visibleCount).map((tag) => <KeyValueTag key={tag.key} tag={tag} />)}
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
                      defaultMessage={'{count} more...'}
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

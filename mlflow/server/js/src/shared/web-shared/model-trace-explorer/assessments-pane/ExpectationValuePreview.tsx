import { isString } from 'lodash';

import { Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';

export const ExpectationValuePreview = ({ objectKey, value }: { objectKey?: string; value: any }) => {
  const { theme } = useDesignSystemTheme();
  const displayValue = isString(value) ? value : JSON.stringify(value);

  return (
    <Tooltip content={displayValue} componentId="shared.model-trace-explorer.expectation-value-preview-tooltip">
      <Tag
        color="indigo"
        componentId="shared.model-trace-explorer.expectation-array-item-tag"
        css={{ width: 'min-content', maxWidth: '100%' }}
      >
        <Typography.Text css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {objectKey && (
            <Typography.Text bold css={{ marginRight: theme.spacing.xs }}>
              {objectKey}:
            </Typography.Text>
          )}
          {displayValue}
        </Typography.Text>
      </Tag>
    </Tooltip>
  );
};

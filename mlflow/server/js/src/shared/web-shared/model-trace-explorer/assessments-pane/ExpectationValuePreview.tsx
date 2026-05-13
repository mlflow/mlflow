import { isString, isObject, isNil } from 'lodash';

import { Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';

const SingleExpectationValuePreview = ({ objectKey, value }: { objectKey?: string; value: any }) => {
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

export const ExpectationValuePreview = ({
  parsedValue,
  singleLine = false,
}: {
  parsedValue: any;
  singleLine?: boolean;
}): React.ReactElement | null => {
  const { theme } = useDesignSystemTheme();

  if (isNil(parsedValue)) {
    return null;
  }

  if (Array.isArray(parsedValue)) {
    return singleLine ? (
      <SingleExpectationValuePreview value={parsedValue} />
    ) : (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
        }}
      >
        {parsedValue.map((item, index) => (
          <SingleExpectationValuePreview value={item} key={index} />
        ))}
      </div>
    );
  }

  if (isObject(parsedValue)) {
    return singleLine ? (
      <SingleExpectationValuePreview value={parsedValue} />
    ) : (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
        }}
      >
        {Object.entries(parsedValue).map(([key, value]) => (
          <SingleExpectationValuePreview key={key} objectKey={key} value={value} />
        ))}
      </div>
    );
  }

  return <SingleExpectationValuePreview value={parsedValue} />;
};

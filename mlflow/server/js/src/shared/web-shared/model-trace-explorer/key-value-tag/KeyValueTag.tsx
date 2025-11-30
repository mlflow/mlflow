import type { Interpolation, Theme } from '@emotion/react';

import { Tag, Tooltip, Typography } from '@databricks/design-system';

// max characters for key + value before truncation
const MAX_CHARS_LENGTH = 18;

const getTruncatedStyles = (shouldTruncate: boolean): Interpolation<Theme> =>
  shouldTruncate
    ? {
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }
    : { whiteSpace: 'nowrap' };

/**
 * A <Tag /> wrapper used for displaying key-value entity
 */
export const KeyValueTag = ({
  itemKey,
  itemValue,
  charLimit = MAX_CHARS_LENGTH,
  maxWidth = 150,
  className,
}: {
  itemKey: string;
  itemValue: string;
  charLimit?: number;
  maxWidth?: number;
  className?: string;
}) => {
  const { shouldTruncateKey, shouldTruncateValue } = getKeyAndValueComplexTruncation(itemKey, itemValue, charLimit);

  return (
    <Tooltip componentId="shared.model-trace-explorer.key-value-tag.hover-tooltip" content={`${itemKey}: ${itemValue}`}>
      <Tag componentId="shared.model-trace-explorer.key-value-tag" className={className}>
        <span css={{ maxWidth, display: 'inline-flex' }}>
          <Typography.Text bold css={getTruncatedStyles(shouldTruncateKey)} size="sm">
            {itemKey}
          </Typography.Text>
          :&nbsp;
          <Typography.Text css={getTruncatedStyles(shouldTruncateValue)} size="sm">
            {itemValue}
          </Typography.Text>
        </span>
      </Tag>
    </Tooltip>
  );
};

export function getKeyAndValueComplexTruncation(
  key: string,
  value: string,
  charLimit: number,
): { shouldTruncateKey: boolean; shouldTruncateValue: boolean } {
  const fullLength = key.length + value.length;
  const isKeyLonger = key.length > value.length;
  const shorterLength = isKeyLonger ? value.length : key.length;

  // No need to truncate if tag is short enough
  if (fullLength <= charLimit) return { shouldTruncateKey: false, shouldTruncateValue: false };
  // If the shorter string is too long, truncate both key and value.
  if (shorterLength > charLimit / 2) return { shouldTruncateKey: true, shouldTruncateValue: true };

  // Otherwise truncate the longer string
  return {
    shouldTruncateKey: isKeyLonger,
    shouldTruncateValue: !isKeyLonger,
  };
}

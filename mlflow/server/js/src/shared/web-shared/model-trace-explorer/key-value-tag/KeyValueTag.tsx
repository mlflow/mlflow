import type { Interpolation, Theme } from '@emotion/react';

import { Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';

// max characters for key + value before truncation
const MAX_CHARS_LENGTH = 18;

const URL_REGEX = /^"?(https?:\/\/[^\s"]+)"?$/;

function extractUrl(value: string): string | null {
  const match = value.match(URL_REGEX);
  return match ? match[1] : null;
}

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
  const { theme } = useDesignSystemTheme();
  const url = extractUrl(itemValue);

  return (
    <Tooltip componentId="shared.model-trace-explorer.key-value-tag.hover-tooltip" content={`${itemKey}: ${itemValue}`}>
      <Tag componentId="shared.model-trace-explorer.key-value-tag" className={className}>
        <span css={{ maxWidth, display: 'inline-flex' }}>
          <Typography.Text bold css={getTruncatedStyles(shouldTruncateKey)} size="sm">
            {itemKey}
          </Typography.Text>
          :&nbsp;
          {url ? (
            <Typography.Link
              componentId="shared.model-trace-explorer.key-value-tag.link"
              href={url}
              openInNewTab
              css={[getTruncatedStyles(shouldTruncateValue), { fontSize: theme.typography.fontSizeSm }]}
              onClick={(e: React.MouseEvent) => e.stopPropagation()}
            >
              {url}
            </Typography.Link>
          ) : (
            <Typography.Text css={getTruncatedStyles(shouldTruncateValue)} size="sm">
              {itemValue}
            </Typography.Text>
          )}
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

  // No need to truncate if tag is short enough
  if (fullLength <= charLimit) return { shouldTruncateKey: false, shouldTruncateValue: false };

  // If the shorter string exceeds half the limit, truncate both
  if (Math.min(key.length, value.length) > charLimit / 2) {
    return { shouldTruncateKey: true, shouldTruncateValue: true };
  }

  // Otherwise truncate only the longer string
  const isKeyLonger = key.length > value.length;
  return {
    shouldTruncateKey: isKeyLonger,
    shouldTruncateValue: !isKeyLonger,
  };
}

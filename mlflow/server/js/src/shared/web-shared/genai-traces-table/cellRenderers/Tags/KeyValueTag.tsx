import type { Interpolation, Theme } from '@emotion/react';
import React, { useState } from 'react';

import { Tag, Tooltip, Typography } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { KeyValueTagFullViewModal } from './KeyValueTagFullViewModal';

/**
 * An arbitrary number that is used to determine if a tag is too
 * long and should be truncated. We want to avoid short keys or values
 * in a long tag to be truncated
 * */
const TRUNCATE_ON_CHARS_LENGTH = 30;

function getTruncatedStyles(shouldTruncate = true): Interpolation<Theme> {
  return shouldTruncate
    ? {
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        textWrap: 'nowrap',
        whiteSpace: 'nowrap' as const,
      }
    : { whiteSpace: 'nowrap' as const };
}

interface KeyValueEntity {
  key: string;
  value: string;
}

/**
 * A <Tag /> wrapper used for displaying key-value entity
 */
export const KeyValueTag = ({
  isClosable = false,
  onClose,
  tag,
  enableFullViewModal = false,
  charLimit = TRUNCATE_ON_CHARS_LENGTH,
  maxWidth = 300,
  className,
}: {
  isClosable?: boolean;
  onClose?: () => void;
  tag: KeyValueEntity;
  enableFullViewModal?: boolean;
  charLimit?: number;
  maxWidth?: number;
  className?: string;
}) => {
  const intl = useIntl();

  const [isKeyValueTagFullViewModalVisible, setIsKeyValueTagFullViewModalVisible] = useState(false);

  const { shouldTruncateKey, shouldTruncateValue } = getKeyAndValueComplexTruncation(tag, charLimit);
  const allowFullViewModal = enableFullViewModal && (shouldTruncateKey || shouldTruncateValue);

  const fullViewModalLabel = intl.formatMessage({
    defaultMessage: 'Click to see more',
    description: 'Run page > Overview > Tags cell > Tag',
  });

  return (
    <div>
      <Tag
        componentId="codegen_mlflow_app_src_common_components_keyvaluetag.tsx_60"
        closable={isClosable}
        onClose={onClose}
        title={tag.key}
        className={className}
      >
        <Tooltip
          componentId="web-shared.genai-traces-table.key-value-tag.full-view-tooltip"
          content={allowFullViewModal ? <span>{fullViewModalLabel}</span> : undefined}
        >
          <span
            css={{ maxWidth, display: 'inline-flex' }}
            onClick={() => (allowFullViewModal ? setIsKeyValueTagFullViewModalVisible(true) : undefined)}
          >
            <Typography.Text bold title={tag.key} css={getTruncatedStyles(shouldTruncateKey)}>
              {tag.key}
            </Typography.Text>
            {tag.value && (
              <Typography.Text title={tag.value} css={getTruncatedStyles(shouldTruncateValue)}>
                : {tag.value}
              </Typography.Text>
            )}
          </span>
        </Tooltip>
      </Tag>
      <div>
        {isKeyValueTagFullViewModalVisible && (
          <KeyValueTagFullViewModal
            tagKey={tag.key}
            tagValue={tag.value}
            isKeyValueTagFullViewModalVisible={isKeyValueTagFullViewModalVisible}
            setIsKeyValueTagFullViewModalVisible={setIsKeyValueTagFullViewModalVisible}
          />
        )}
      </div>
    </div>
  );
};

export function getKeyAndValueComplexTruncation(
  tag: KeyValueEntity,
  charLimit = TRUNCATE_ON_CHARS_LENGTH,
): { shouldTruncateKey: boolean; shouldTruncateValue: boolean } {
  const { key, value } = tag;
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

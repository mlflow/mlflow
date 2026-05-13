import { useMemo } from 'react';

import { HoverCard, ListIcon, Overflow, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { MLFLOW_INTERNAL_PREFIX, parseJSONSafe, truncateToFirstLineWithMaxLength } from './TagUtils';

export const ModelTraceHeaderMetadataPill = ({
  traceMetadata,
  getTruncatedLabel = (label: string) => truncateToFirstLineWithMaxLength(label, 40),
}: {
  traceMetadata?: Record<string, unknown>;
  getTruncatedLabel?: (label: string) => string;
}) => {
  const { theme } = useDesignSystemTheme();

  const MAX_ITEMS = 100;

  const formatTraceMetadataValue = (value: unknown): string => {
    // If it's already an object/array, try to stringify directly
    if (value && typeof value === 'object') {
      try {
        return JSON.stringify(value);
      } catch {
        return String(value as any);
      }
    }
    // If it's a string, try to parse JSON and pretty return
    if (typeof value === 'string') {
      const parsed = parseJSONSafe(value);
      if (parsed && typeof parsed === 'object') {
        try {
          return JSON.stringify(parsed);
        } catch {
          return value;
        }
      }
      return value;
    }
    return String(value ?? '');
  };

  const metadataItems = useMemo(() => {
    const items = Object.entries(traceMetadata || {})
      .filter(([k, v]) => v !== null && v !== '' && !k.startsWith(MLFLOW_INTERNAL_PREFIX))
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => ({ key, value, displayValue: formatTraceMetadataValue(value) }));
    return items;
  }, [traceMetadata]);

  if (metadataItems.length === 0) {
    return null;
  }

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        gap: theme.spacing.sm,
      }}
    >
      <Typography.Text size="md" color="secondary">
        <FormattedMessage defaultMessage="Metadata" description="Label for the metadata section" />
      </Typography.Text>
      <Overflow noMargin>
        <HoverCard
          trigger={
            <Tag
              componentId="shared.model-trace-explorer.header-metadata-pill"
              color="default"
              css={{ cursor: 'default', width: 'fit-content', maxWidth: '100%' }}
            >
              <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <ListIcon css={{ fontSize: 12, color: theme.colors.textSecondary }} />
                <span>{metadataItems.length}</span>
              </span>
            </Tag>
          }
          content={
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.xs,
                maxWidth: 480,
                maxHeight: 320,
                overflowY: 'auto',
              }}
            >
              {metadataItems.slice(0, MAX_ITEMS).map(({ key, displayValue, value }) => (
                <div
                  key={`${key}:${value}`}
                  css={{ display: 'flex', flexDirection: 'row', alignItems: 'flex-start', gap: theme.spacing.xs }}
                >
                  <Typography.Text css={{ flexBasis: '45%', flexShrink: 0 }}>{getTruncatedLabel(key)}</Typography.Text>
                  <Typography.Text color="secondary" css={{ wordBreak: 'break-word' }}>
                    {displayValue}
                  </Typography.Text>
                </div>
              ))}
              {metadataItems.length > MAX_ITEMS && (
                <Typography.Text color="secondary">+{metadataItems.length - MAX_ITEMS} more</Typography.Text>
              )}
            </div>
          }
        />
      </Overflow>
    </div>
  );
};

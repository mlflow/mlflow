import { HoverCard, ListIcon, Overflow, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { parseJSONSafe, truncateToFirstLineWithMaxLength } from './TagUtils';

export const ModelTraceHeaderMetadataPill = ({
  traceMetadata,
  getTruncatedLabel = (label: string) => truncateToFirstLineWithMaxLength(label, 40),
  getComponentId,
}: {
  traceMetadata?: Record<string, string>;
  getTruncatedLabel?: (label: string) => string;
  getComponentId: (key: string) => string;
}) => {
  const { theme } = useDesignSystemTheme();

  const metadataItems = Object.entries(traceMetadata || {})
    .filter(([k, v]) => !!v && !k.startsWith('mlflow.'))
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => ({ key, value }));

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
              componentId={getComponentId('metadata-pill')}
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
              {metadataItems.map(({ key, value }) => (
                <div
                  key={`${key}:${value}`}
                  css={{ display: 'flex', flexDirection: 'row', alignItems: 'flex-start', gap: theme.spacing.xs }}
                >
                  <Typography.Text css={{ flexBasis: '45%', flexShrink: 0 }}>{getTruncatedLabel(key)}</Typography.Text>
                  <Typography.Text color="secondary" css={{ wordBreak: 'break-word' }}>
                    {(() => {
                      const parsed = parseJSONSafe(value);
                      if (parsed && typeof parsed === 'object') {
                        try {
                          return JSON.stringify(parsed);
                        } catch {
                          return String(value);
                        }
                      }
                      return String(value);
                    })()}
                  </Typography.Text>
                </div>
              ))}
            </div>
          }
        />
      </Overflow>
    </div>
  );
};

import { Typography, XCircleIcon, useDesignSystemTheme } from '@databricks/design-system';

import type { ModelTraceEvent, ModelTraceSpanNode } from '../ModelTrace.types';
import { getSpanExceptionEvents, isValidException } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';

type AttributeToRender = { key: string; value: any; renderMode?: 'text' | 'json' };

const EXCLUDED_KEYS = new Set(['exception.type', 'exception.message', 'exception.stacktrace']);

const getAttributesToRender = (event: ModelTraceEvent): AttributeToRender[] => {
  const attributes = event?.attributes ?? {};
  const stacktrace = attributes['exception.stacktrace'];

  const otherAttributes = Object.entries(attributes)
    .filter(([key]) => !EXCLUDED_KEYS.has(key))
    .map(([key, value]) => ({ key, value }));

  return stacktrace
    ? [{ key: 'exception.stacktrace', value: stacktrace, renderMode: 'text' as const }, ...otherAttributes]
    : otherAttributes;
};

export const ModelTraceExplorerSummaryViewExceptionsSection = ({ node }: { node: ModelTraceSpanNode }) => {
  const { theme } = useDesignSystemTheme();
  const exceptionEvents = getSpanExceptionEvents(node);
  const isRoot = !node.parentId;
  // to prevent excessive nesting, we only show the first exception.
  // it is likely that any given span only has one exception,
  // since execution usually stops after throwing.
  const firstException = exceptionEvents[0];

  if (!firstException || !isValidException(firstException)) {
    return null;
  }

  const exceptionType = firstException.attributes['exception.type'];
  const exceptionMessage = firstException.attributes['exception.message'];

  return (
    <ModelTraceExplorerCollapsibleSection
      withBorder
      isExceptionSection
      key={firstException.name}
      sectionKey={firstException.name}
      css={{ marginBottom: isRoot ? theme.spacing.sm : 0 }}
      title={
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            minWidth: 0,
            flex: 1,
          }}
        >
          <XCircleIcon color="danger" css={{ flexShrink: 0 }} />
          <div css={{ minWidth: 0, flex: 1 }}>
            <Typography.Text
              color="error"
              bold
              css={{
                display: 'block',
                width: `calc(100% - ${theme.spacing.lg}px)`,
              }}
            >
              {exceptionType}: {exceptionMessage}
            </Typography.Text>
          </div>
        </div>
      }
    >
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          paddingBottom: theme.spacing.sm,
          paddingLeft: isRoot ? 0 : theme.spacing.lg,
        }}
      >
        {getAttributesToRender(firstException).map(({ key, value, renderMode }) => (
          <ModelTraceExplorerFieldRenderer
            key={key}
            title={key}
            data={JSON.stringify(value, null, 2)}
            renderMode={renderMode ?? 'text'}
          />
        ))}
      </div>
    </ModelTraceExplorerCollapsibleSection>
  );
};

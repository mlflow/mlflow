import { Typography, XCircleIcon, useDesignSystemTheme } from '@databricks/design-system';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { getSpanExceptionEvents } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';

export const ModelTraceExplorerSummaryViewExceptionsSection = ({ node }: { node: ModelTraceSpanNode }) => {
  const { theme } = useDesignSystemTheme();
  const exceptionEvents = getSpanExceptionEvents(node);
  const isRoot = !node.parentId;
  // to prevent excessive nesting, we only show the first exception.
  // it is likely that any given span only has one exception,
  // since execution usually stops after throwing.
  const firstException = exceptionEvents[0];

  if (!firstException) {
    return null;
  }

  return (
    <ModelTraceExplorerCollapsibleSection
      css={{ marginBottom: isRoot ? theme.spacing.sm : 0 }}
      withBorder={isRoot}
      key={firstException.name}
      sectionKey={firstException.name}
      title={
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <XCircleIcon color="danger" />
          <Typography.Text color="error" bold>
            Exception
          </Typography.Text>
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
        {Object.entries(firstException.attributes ?? {}).map(([attribute, value]) => (
          <ModelTraceExplorerFieldRenderer
            key={attribute}
            title={attribute}
            data={JSON.stringify(value, null, 2)}
            renderMode="text"
          />
        ))}
      </div>
    </ModelTraceExplorerCollapsibleSection>
  );
};

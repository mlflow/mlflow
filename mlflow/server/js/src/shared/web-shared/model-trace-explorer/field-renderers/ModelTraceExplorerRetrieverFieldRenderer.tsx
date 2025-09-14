import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { RetrieverDocument } from '../ModelTrace.types';
import { ModelTraceExplorerRetrieverDocument } from '../right-pane/ModelTraceExplorerRetrieverDocument';

export const ModelTraceExplorerRetrieverFieldRenderer = ({
  title,
  documents,
}: {
  title: string;
  documents: RetrieverDocument[];
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        backgroundColor: theme.colors.backgroundPrimary,
        borderRadius: theme.borders.borderRadiusSm,
        border: `1px solid ${theme.colors.border}`,
      }}
    >
      {title && (
        <div
          css={{
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
            borderBottom: `1px solid ${theme.colors.border}`,
          }}
        >
          <Typography.Text bold>{title}</Typography.Text>
        </div>
      )}
      {documents.map((document, idx) => (
        <div key={idx} css={{ borderBottom: idx !== documents.length - 1 ? `1px solid ${theme.colors.border}` : '' }}>
          <ModelTraceExplorerRetrieverDocument key={idx} text={document.page_content} metadata={document.metadata} />
        </div>
      ))}
    </div>
  );
};

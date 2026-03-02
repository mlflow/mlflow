import { useMemo } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { Assessment, FeedbackAssessment, RetrieverDocument } from '../ModelTrace.types';
import { buildDocumentRelevanceAssessmentMap } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerRetrieverDocument } from '../right-pane/ModelTraceExplorerRetrieverDocument';

export const ModelTraceExplorerRetrieverFieldRenderer = ({
  title,
  documents,
  assessments,
}: {
  title: string;
  documents: RetrieverDocument[];
  assessments?: Assessment[];
}) => {
  const { theme } = useDesignSystemTheme();

  // Build a map from document index to relevance assessment
  const documentRelevanceMap = useMemo(() => buildDocumentRelevanceAssessmentMap(assessments ?? []), [assessments]);

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
          <ModelTraceExplorerRetrieverDocument
            key={idx}
            text={document.page_content}
            metadata={document.metadata}
            relevanceAssessment={documentRelevanceMap.get(idx) as FeedbackAssessment | undefined}
          />
        </div>
      ))}
    </div>
  );
};

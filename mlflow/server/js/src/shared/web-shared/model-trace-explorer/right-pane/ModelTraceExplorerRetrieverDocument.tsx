import { useCallback, useState } from 'react';

import { ModelTraceExplorerRetrieverDocumentFull } from './ModelTraceExplorerRetrieverDocumentFull';
import { ModelTraceExplorerRetrieverDocumentPreview } from './ModelTraceExplorerRetrieverDocumentPreview';
import type { FeedbackAssessment } from '../ModelTrace.types';
import { createListFromObject } from '../ModelTraceExplorer.utils';

export function ModelTraceExplorerRetrieverDocument({
  text,
  metadata,
  relevanceAssessment,
}: {
  text: string;
  metadata: { [key: string]: any };
  relevanceAssessment?: FeedbackAssessment;
}) {
  const [expanded, setExpanded] = useState(false);
  const metadataTags = createListFromObject(metadata);

  return expanded ? (
    <ModelTraceExplorerRetrieverDocumentFull
      // comment to prevent copybara formatting
      text={text}
      metadataTags={metadataTags}
      setExpanded={setExpanded}
      relevanceAssessment={relevanceAssessment}
    />
  ) : (
    <ModelTraceExplorerRetrieverDocumentPreview
      // comment to prevent copybara formatting
      text={text}
      metadataTags={metadataTags}
      setExpanded={setExpanded}
      relevanceAssessment={relevanceAssessment}
    />
  );
}

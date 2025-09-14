import { useCallback, useState } from 'react';

import { ModelTraceExplorerRetrieverDocumentFull } from './ModelTraceExplorerRetrieverDocumentFull';
import { ModelTraceExplorerRetrieverDocumentPreview } from './ModelTraceExplorerRetrieverDocumentPreview';
import { createListFromObject } from '../ModelTraceExplorer.utils';

export function ModelTraceExplorerRetrieverDocument({
  text,
  metadata,
}: {
  text: string;
  metadata: { [key: string]: any };
}) {
  const [expanded, setExpanded] = useState(false);
  const metadataTags = createListFromObject(metadata);

  return expanded ? (
    <ModelTraceExplorerRetrieverDocumentFull
      // comment to prevent copybara formatting
      text={text}
      metadataTags={metadataTags}
      setExpanded={setExpanded}
    />
  ) : (
    <ModelTraceExplorerRetrieverDocumentPreview
      // comment to prevent copybara formatting
      text={text}
      metadataTags={metadataTags}
      setExpanded={setExpanded}
    />
  );
}

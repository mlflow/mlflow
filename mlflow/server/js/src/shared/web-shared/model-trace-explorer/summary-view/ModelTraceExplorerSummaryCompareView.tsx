import { useDesignSystemTheme } from '@databricks/design-system';

import { ModelTraceExplorerSummarySpans } from './ModelTraceExplorerSummarySpans';
import type { ModelTrace } from '../ModelTrace.types';
import { getModelTraceId, useIntermediateNodes } from '../ModelTraceExplorer.utils';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
} from '../ModelTraceExplorerViewStateContext';

export const ModelTraceExplorerSummaryCompareView = ({ modelTraces }: { modelTraces: ModelTrace[] }) => {
  const context = useModelTraceExplorerViewState();
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flex: 1,
        overflow: 'hidden',
        '&>div:not(:first-child)': { borderLeft: `1px solid ${theme.colors.border}` },
      }}
    >
      {modelTraces.map((modelTrace) => (
        <ModelTraceExplorerViewStateProvider
          {...context}
          readOnly
          modelTrace={modelTrace}
          key={getModelTraceId(modelTrace)}
        >
          <ModelTraceExplorerSummaryCompareViewItem />
        </ModelTraceExplorerViewStateProvider>
      ))}
    </div>
  );
};

const ModelTraceExplorerSummaryCompareViewItem = () => {
  const { rootNode } = useModelTraceExplorerViewState();
  const intermediateNodes = useIntermediateNodes(rootNode);

  if (!rootNode) {
    return null;
  }

  return <ModelTraceExplorerSummarySpans rootNode={rootNode} intermediateNodes={intermediateNodes} />;
};

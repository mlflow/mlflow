import { useDesignSystemTheme } from '@databricks/design-system';

import type { ModelTrace } from './ModelTrace.types';
import { isV3ModelTraceInfo } from './ModelTraceExplorer.utils';
import { ModelTraceExplorerDetailView } from './ModelTraceExplorerDetailView';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
} from './ModelTraceExplorerViewStateContext';

export const ModelTraceExplorerDetailCompareView = ({ modelTraces }: { modelTraces: ModelTrace[] }) => {
  const context = useModelTraceExplorerViewState();
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        overflow: 'hidden',
        flex: 1,
        '&>div': { flex: 1, overflow: 'hidden' },
        '&>div:not(:first-child)': { borderLeft: `1px solid ${theme.colors.border}` },
      }}
    >
      {modelTraces?.map((trace, index) => {
        const traceInfo = trace.info;
        // const treeNode = getNode(index);
        if (!isV3ModelTraceInfo(traceInfo)) {
          return null;
        }
        return (
          <ModelTraceExplorerViewStateProvider {...context} readOnly modelTrace={trace} key={traceInfo.trace_id}>
            <ModelTraceExplorerDetailView key={traceInfo.trace_id} modelTraceInfo={traceInfo} />
          </ModelTraceExplorerViewStateProvider>
        );
      })}
    </div>
  );
};

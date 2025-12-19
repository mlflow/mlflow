import { isNil } from 'lodash';

import { useDesignSystemTheme } from '@databricks/design-system';

import { ModelTraceExplorerDefaultSpanView } from './ModelTraceExplorerDefaultSpanView';
import { ModelTraceExplorerRetrieverSpanView } from './ModelTraceExplorerRetrieverSpanView';
import type { ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { isRenderableRetrieverSpan } from '../ModelTraceExplorer.utils';

export function ModelTraceExplorerContentTab({
  activeSpan,
  className,
  searchFilter,
  activeMatch,
}: {
  activeSpan: ModelTraceSpanNode | undefined;
  className?: string;
  searchFilter: string;
  activeMatch: SearchMatch | null;
}) {
  const { theme } = useDesignSystemTheme();

  if (!isNil(activeSpan) && isRenderableRetrieverSpan(activeSpan)) {
    return (
      <div
        css={{
          overflowY: 'auto',
          padding: theme.spacing.md,
        }}
        className={className}
        data-testid="model-trace-explorer-content-tab"
      >
        <ModelTraceExplorerRetrieverSpanView
          activeSpan={activeSpan}
          className={className}
          searchFilter={searchFilter}
          activeMatch={activeMatch}
        />
      </div>
    );
  }

  return (
    <div
      css={{
        overflowY: 'auto',
        padding: theme.spacing.md,
      }}
      className={className}
      data-testid="model-trace-explorer-content-tab"
    >
      <ModelTraceExplorerDefaultSpanView
        activeSpan={activeSpan}
        className={className}
        searchFilter={searchFilter}
        activeMatch={activeMatch}
      />
    </div>
  );
}

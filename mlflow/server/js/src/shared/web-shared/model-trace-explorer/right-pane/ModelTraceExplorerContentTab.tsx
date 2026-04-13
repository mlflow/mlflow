import { useDesignSystemTheme } from '@databricks/design-system';

import { ModelTraceExplorerDefaultSpanView } from './ModelTraceExplorerDefaultSpanView';
import type { ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { useModelTraceExplorerPreferences } from '../ModelTraceExplorerPreferencesContext';

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
  const { renderMode } = useModelTraceExplorerPreferences();

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
        renderMode={renderMode}
      />
    </div>
  );
}

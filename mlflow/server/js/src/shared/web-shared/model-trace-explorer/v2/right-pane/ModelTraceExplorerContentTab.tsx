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
}): React.ReactElement | null {
  const { renderMode } = useModelTraceExplorerPreferences();

  return (
    <div
      css={{
        overflowY: 'auto',
      }}
      className={className}
      data-testid="model-trace-explorer-content-tab"
    >
      <ModelTraceExplorerDefaultSpanView
        key={renderMode}
        activeSpan={activeSpan}
        className={className}
        searchFilter={searchFilter}
        activeMatch={activeMatch}
        defaultRenderMode={renderMode}
      />
    </div>
  );
}

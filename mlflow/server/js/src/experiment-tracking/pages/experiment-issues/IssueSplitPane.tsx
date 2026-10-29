import type { ReactNode } from 'react';
import { useIntl } from 'react-intl';
import { useState } from 'react';
import { useLocalStorage } from '../../../shared/web-shared/hooks';
import { ModelTraceExplorerResizablePane } from '../../../shared/web-shared/model-trace-explorer';

/** One third of the container, matching the fixed 1:2 grid the split view used before it was resizable. */
const DEFAULT_ISSUE_LIST_RATIO = 1 / 3;
/** Below this the issue names in the list are unreadable. */
export const ISSUE_LIST_MIN_WIDTH = 280;
/**
 * Room for the detail content. Honoured only while the container is at least
 * ISSUE_LIST_MIN_WIDTH + ISSUE_DETAIL_MIN_WIDTH; below that the list keeps its own minimum and the
 * detail pane takes whatever is left. That is deliberately the same squeeze the fixed
 * `minmax(280px, 1fr) 2fr` grid produced here before -- its 280px floor took over at exactly this
 * width too -- so narrow windows are no worse than they were, and the divider stays usable rather
 * than the layout collapsing to one pane.
 */
const ISSUE_DETAIL_MIN_WIDTH = 480;

export const ISSUE_LIST_RATIO_STORAGE_KEY = 'mlflow.issues.list-ratio';
export const ISSUE_LIST_RATIO_STORAGE_VERSION = 1;

/**
 * A width read back from storage can be anything, and a ratio outside (0, 1) would leave one pane
 * pinned to its minimum width with no way back.
 */
const sanitizeRatio = (ratio: unknown): number =>
  typeof ratio === 'number' && Number.isFinite(ratio) && ratio > 0 && ratio < 1 ? ratio : DEFAULT_ISSUE_LIST_RATIO;

export interface IssueSplitPaneProps {
  /** Scrollable list of issues. Must stretch itself (`flex: 1`, `minWidth: 0`). */
  list: ReactNode;
  /** Detail pane for the selected issue. When absent the list gets the full width and no drag handle renders. */
  detail?: ReactNode;
}

/**
 * List/detail layout for the Issues tab, with a draggable divider between the two panes. The chosen
 * ratio is remembered per user across sessions, mirroring the traces table's persisted column layout.
 */
export const IssueSplitPane = ({ list, detail }: IssueSplitPaneProps) => {
  const intl = useIntl();
  const [storedRatio, setStoredRatio] = useLocalStorage<number>({
    key: ISSUE_LIST_RATIO_STORAGE_KEY,
    version: ISSUE_LIST_RATIO_STORAGE_VERSION,
    initialValue: DEFAULT_ISSUE_LIST_RATIO,
  });
  // Only a placeholder for the first paint: the pane recomputes the width from the ratio as soon as
  // it has measured its container.
  const [paneWidth, setPaneWidth] = useState(ISSUE_LIST_MIN_WIDTH);

  return (
    <div css={{ display: 'flex', flex: 1, width: '100%', minWidth: 0, minHeight: 0, overflow: 'hidden' }}>
      {detail ? (
        <ModelTraceExplorerResizablePane
          resizeLabel={intl.formatMessage({
            defaultMessage: 'Resize issue list',
            description: 'Experiment Issues list/detail separator accessible label',
          })}
          initialRatio={sanitizeRatio(storedRatio)}
          paneWidth={paneWidth}
          setPaneWidth={setPaneWidth}
          leftMinWidth={ISSUE_LIST_MIN_WIDTH}
          rightMinWidth={ISSUE_DETAIL_MIN_WIDTH}
          onRatioChange={setStoredRatio}
          leftChild={list}
          rightChild={detail}
        />
      ) : (
        list
      )}
    </div>
  );
};

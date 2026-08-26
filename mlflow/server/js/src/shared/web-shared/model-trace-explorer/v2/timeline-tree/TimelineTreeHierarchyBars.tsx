import { useDesignSystemTheme } from '@databricks/design-system';

import type { HierarchyBar } from './TimelineTree.types';
import { SPAN_INDENT_WIDTH, SPAN_ROW_HEIGHT, TimelineTreeZIndex } from './TimelineTree.utils';

const IconBottomConnector = ({ active, rowHeight }: { active: boolean; rowHeight: number }) => {
  const { theme } = useDesignSystemTheme();
  const borderColor = active ? theme.colors.blue500 : theme.colors.border;

  return (
    <div
      data-testid="icon-bottom-connector"
      data-active={active}
      data-row-center={rowHeight / 2}
      css={{
        position: 'absolute',
        left: '100%',
        top: rowHeight / 2,
        bottom: 0,
        // not sure why the +1 is necessary but
        // there is a 1 pixel misalignment with the
        // left connector otherwise
        width: SPAN_INDENT_WIDTH / 2 + 1,
        boxSizing: 'border-box',
        borderTopRightRadius: theme.borders.borderRadiusMd,
        borderTop: `1px solid ${borderColor}`,
        borderRight: `1px solid ${borderColor}`,
        zIndex: active ? TimelineTreeZIndex.NORMAL : TimelineTreeZIndex.LOW,
      }}
    />
  );
};

const IconLeftConnector = ({ active, rowHeight }: { active: boolean; rowHeight: number }) => {
  const { theme } = useDesignSystemTheme();
  const borderColor = active ? theme.colors.blue500 : theme.colors.border;

  return (
    <div
      data-testid="icon-left-connector"
      data-row-center={rowHeight / 2}
      css={{
        position: 'absolute',
        left: '50%',
        top: 0,
        width: SPAN_INDENT_WIDTH / 2,
        height: rowHeight / 2,
        boxSizing: 'border-box',
        borderBottomLeftRadius: theme.borders.borderRadiusMd,
        borderBottom: `1px solid ${borderColor}`,
        borderLeft: `1px solid ${borderColor}`,
        zIndex: active ? TimelineTreeZIndex.NORMAL : TimelineTreeZIndex.LOW,
      }}
    />
  );
};

const VerticalConnector = ({ active, rowHeight }: { active: boolean; rowHeight: number }) => {
  const { theme } = useDesignSystemTheme();
  const borderColor = active ? theme.colors.blue500 : theme.colors.border;

  return (
    <div
      css={{
        position: 'absolute',
        width: SPAN_INDENT_WIDTH / 2,
        left: '50%',
        height: rowHeight,
        borderLeft: `1px solid ${borderColor}`,
        boxSizing: 'border-box',
        zIndex: active ? TimelineTreeZIndex.NORMAL : TimelineTreeZIndex.LOW,
      }}
    />
  );
};

/**
 * This component renders the bars that represent the hierarchical
 * connections in the span tree.
 */
export const TimelineTreeHierarchyBars = ({
  isActiveSpan,
  isInActiveChain,
  linesToRender,
  hasChildren,
  isExpanded,
  rowHeight = SPAN_ROW_HEIGHT,
}: {
  // whether or not the current span is active
  isActiveSpan: boolean;
  // true if the span is either active or a parent of the active span
  isInActiveChain: boolean;
  // an array of bars to render to the left of the span icon / name
  linesToRender: Array<HierarchyBar>;
  hasChildren: boolean;
  isExpanded: boolean;
  rowHeight?: number;
}): React.ReactElement => {
  if (linesToRender.length === 0) {
    return (
      <div
        css={{
          width: 0,
          height: rowHeight,
          boxSizing: 'border-box',
          position: 'relative',
        }}
      >
        {hasChildren && <IconBottomConnector active={isInActiveChain && !isActiveSpan} rowHeight={rowHeight} />}
      </div>
    );
  }

  return (
    <>
      {linesToRender.map(({ shouldRender, isActive }, idx) => (
        // for each depth level, render a spacer. depending on the span's
        // position within the tree, the spacer might be empty or contain
        // a vertical bar
        <div
          key={idx}
          css={{
            width: SPAN_INDENT_WIDTH,
            height: rowHeight,
            boxSizing: 'border-box',
            position: 'relative',
          }}
        >
          {shouldRender && (
            // render a vertical bar in the middle of the spacer
            <VerticalConnector active={isActive} rowHeight={rowHeight} />
          )}
          {idx === linesToRender.length - 1 && (
            // at the last spacer, render a curved
            // line that connects up to the parent
            <>
              <IconLeftConnector active={isInActiveChain || isActiveSpan} rowHeight={rowHeight} />
              {hasChildren && isExpanded && (
                <IconBottomConnector active={isInActiveChain && !isActiveSpan} rowHeight={rowHeight} />
              )}
            </>
          )}
        </div>
      ))}
    </>
  );
};

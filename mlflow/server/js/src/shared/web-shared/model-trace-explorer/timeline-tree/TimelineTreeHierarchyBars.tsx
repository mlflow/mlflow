import { useDesignSystemTheme } from '@databricks/design-system';

import { getLineCenterX, hierarchyLineStyles } from '../hierarchyLineStyles';

import type { HierarchyBar } from './TimelineTree.types';
import { SPAN_INDENT_WIDTH, SPAN_ROW_HEIGHT, TimelineTreeZIndex } from './TimelineTree.utils';

const IconBottomConnector = ({ active, left }: { active: boolean; left: number }) => {
  const { theme } = useDesignSystemTheme();
  const borderColor = active ? theme.colors.blue500 : theme.colors.border;

  return (
    <div
      css={{
        position: 'absolute',
        left,
        bottom: 0,
        // not sure why the +1 is necessary but
        // there is a 1 pixel misalignment with the
        // left connector otherwise
        width: SPAN_INDENT_WIDTH / 2 + 1,
        height: theme.spacing.md,
        boxSizing: 'border-box',
        borderTopRightRadius: theme.borders.borderRadiusMd,
        borderTop: `1px solid ${borderColor}`,
        borderRight: `1px solid ${borderColor}`,
        zIndex: TimelineTreeZIndex.LOW, // render behind the span's icon
      }}
    />
  );
};

const IconLeftConnector = ({ active, left }: { active: boolean; left: number }) => {
  const { theme } = useDesignSystemTheme();
  const borderColor = active ? theme.colors.blue500 : theme.colors.border;

  return (
    <div
      css={{
        position: 'absolute',
        left,
        top: 0,
        width: SPAN_INDENT_WIDTH / 2,
        height: theme.spacing.md,
        boxSizing: 'border-box',
        borderBottomLeftRadius: theme.borders.borderRadiusMd,
        borderBottom: `1px solid ${borderColor}`,
        borderLeft: `1px solid ${borderColor}`,
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
}: {
  // whether or not the current span is active
  isActiveSpan: boolean;
  // true if the span is either active or a parent of the active span
  isInActiveChain: boolean;
  // an array of bars to render to the left of the span icon / name
  linesToRender: Array<HierarchyBar>;
  hasChildren: boolean;
  isExpanded: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const linesStyle = hierarchyLineStyles(linesToRender, theme.colors.blue500, theme.colors.border);
  if (linesToRender.length === 0) {
    return (
      <div
        css={{
          width: 0,
          height: SPAN_ROW_HEIGHT,
          boxSizing: 'border-box',
          position: 'relative',
        }}
      >
        {hasChildren && <IconBottomConnector active={isInActiveChain && !isActiveSpan} left={0} />}
      </div>
    );
  }

  return (
    <div
      css={{
        width: SPAN_INDENT_WIDTH * linesToRender.length,
        height: SPAN_ROW_HEIGHT,
        boxSizing: 'border-box',
        position: 'relative',
      }}
    >
      {linesStyle && <div style={linesStyle} />}
      <IconLeftConnector active={isInActiveChain} left={getLineCenterX(linesToRender.length - 1)} />
      {hasChildren && isExpanded && (
        <IconBottomConnector
          active={isInActiveChain && !isActiveSpan}
          left={linesToRender.length * SPAN_INDENT_WIDTH}
        />
      )}
    </div>
  );
};

import { useDesignSystemTheme } from '@databricks/design-system';
import React from 'react';
import { useIntl } from 'react-intl';

const HIT_AREA_WIDTH = 8;
const VISIBLE_BAR_WIDTH = 3;

/**
 * Drag handle rendered by `ResizableBox` on the west edge of the V2 dataset records side
 * panel. Wider invisible hit area for forgiving grab; a thin colored bar appears on hover so
 * the user knows the divider is interactive. Forwarded ref / spread props are required so
 * `react-resizable` can attach its own ref + className (`react-resizable-handle*`) via
 * `cloneElement`.
 */
export const SidePanelResizeHandle = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  function SidePanelResizeHandle(props, ref) {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    return (
      <div
        ref={ref}
        {...props}
        role="separator"
        aria-orientation="vertical"
        aria-label={intl.formatMessage({
          defaultMessage: 'Resize side panel',
          description: 'Aria label for the drag handle on the V2 dataset record side panel',
        })}
        css={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          left: -HIT_AREA_WIDTH / 2,
          width: HIT_AREA_WIDTH,
          cursor: 'ew-resize',
          zIndex: 10,
          // Thin colored strip appears on hover/drag — same actionDefault hover color the v1
          // resizer uses so the two pages feel consistent.
          '&:hover::after, &:active::after': {
            opacity: 1,
          },
          '&::after': {
            content: '""',
            position: 'absolute',
            left: HIT_AREA_WIDTH / 2 - VISIBLE_BAR_WIDTH / 2,
            top: 0,
            bottom: 0,
            width: VISIBLE_BAR_WIDTH,
            backgroundColor: theme.colors.actionDefaultBorderHover,
            opacity: 0,
            transition: 'opacity 0.15s',
          },
        }}
      />
    );
  },
);

import { Global } from '@emotion/react';
import { clamp } from 'lodash';
import React, { forwardRef, useCallback, useImperativeHandle, useLayoutEffect, useRef, useState } from 'react';
import { ResizableBox } from 'react-resizable';

import { useDesignSystemTheme } from '@databricks/design-system';

import { useResizeObserver } from '../hooks';

interface ModelTraceExplorerResizablePaneProps {
  initialRatio: number;
  paneWidth: number;
  setPaneWidth: (paneWidth: number) => void;
  leftChild: React.ReactNode;
  leftMinWidth: number;
  rightChild: React.ReactNode;
  rightMinWidth: number;
}

export interface ModelTraceExplorerResizablePaneRef {
  updateRatio: (newPaneWidth: number) => void;
}

/**
 * This component takes a left and right child, and adds
 * a draggable handle between them to resize. It handles
 * logic such as preserving the ratio of the pane width
 * when the container/window is resized, and also ensures
 * that the left and right panes conform to specified min
 * widths.
 */
const ModelTraceExplorerResizablePane = forwardRef<
  ModelTraceExplorerResizablePaneRef,
  ModelTraceExplorerResizablePaneProps
>(({ initialRatio, paneWidth, setPaneWidth, leftChild, leftMinWidth, rightChild, rightMinWidth }, ref) => {
  const [isResizing, setIsResizing] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const containerWidth = useResizeObserver({ ref: containerRef })?.width;
  // if container width is not available, don't set a max width
  const maxWidth = (containerWidth ?? Infinity) - rightMinWidth;

  const ratio = useRef(initialRatio);
  const { theme } = useDesignSystemTheme();

  const updateRatio = useCallback(
    // used by the parent component to update the ratio when
    // the pane is resized via the show/hide gantt button
    (newPaneWidth: number) => {
      if (containerWidth) {
        ratio.current = newPaneWidth / containerWidth;
      }
    },
    [containerWidth],
  );

  useImperativeHandle(ref, () => ({
    updateRatio,
  }));

  useLayoutEffect(() => {
    // preserve the ratio of the pane width when the container is resized
    if (containerWidth) {
      setPaneWidth(clamp(containerWidth * ratio.current, leftMinWidth, maxWidth));
    }
  }, [containerWidth, maxWidth, leftMinWidth, rightMinWidth, setPaneWidth]);

  return (
    <div
      ref={containerRef}
      css={{
        display: 'flex',
        flex: 1,
        overflow: 'hidden',
        flexDirection: 'row',
      }}
    >
      {isResizing && (
        <Global
          styles={{
            'body, :host': {
              userSelect: 'none',
            },
          }}
        />
      )}
      <ResizableBox
        axis="x"
        width={paneWidth}
        css={{ display: 'flex', flex: `0 0 ${paneWidth}px` }}
        handle={
          <div css={{ width: 0, position: 'relative' }}>
            <div
              css={{
                position: 'relative',
                width: theme.spacing.sm,
                marginLeft: -theme.spacing.xs,
                minHeight: '100%',
                cursor: 'ew-resize',
                backgroundColor: `rgba(0,0,0,0)`,
                zIndex: 1,
                ':hover': {
                  backgroundColor: `rgba(0,0,0,0.1)`,
                },
              }}
            />
          </div>
        }
        onResize={(e, { size }) => {
          const clampedSize = clamp(size.width, leftMinWidth, maxWidth);
          setPaneWidth(clampedSize);
          if (containerWidth) {
            ratio.current = clampedSize / containerWidth;
          }
        }}
        onResizeStart={() => setIsResizing(true)}
        onResizeStop={() => setIsResizing(false)}
        minConstraints={[leftMinWidth, Infinity]}
        maxConstraints={[maxWidth, Infinity]}
      >
        {leftChild}
      </ResizableBox>
      {rightChild}
    </div>
  );
});

export default ModelTraceExplorerResizablePane;

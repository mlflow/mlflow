import { Button, ChevronLeftIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';
import React, { useState } from 'react';
import { ResizableBox } from 'react-resizable';
import { useUpdateExperimentViewUIState } from '../../contexts/ExperimentPageUIStateContext';
import { Global } from '@emotion/react';

const RESIZE_BAR_WIDTH = 3;

/**
 * A component wrapping experiment runs table and providing a resizer
 * to adjust its width when displayed in a split view.
 */
export const ExperimentViewRunsTableResizer = ({
  runListHidden,
  width,
  onResize,
  children,
  onHiddenChange,
  maxWidth,
}: React.PropsWithChildren<{
  runListHidden: boolean;
  width: number;
  onResize: React.Dispatch<React.SetStateAction<number>>;
  onHiddenChange?: (isHidden: boolean) => void;
  maxWidth: number | undefined;
}>) => {
  const updateUIState = useUpdateExperimentViewUIState();
  const [dragging, setDragging] = useState(false);

  return (
    <>
      <ResizableBox
        css={{ display: 'flex', position: 'relative' }}
        style={{ flex: `0 0 ${runListHidden ? 0 : width}px` }}
        width={width}
        axis="x"
        resizeHandles={['e']}
        minConstraints={[250, 0]}
        maxConstraints={maxWidth === undefined ? undefined : [maxWidth, 0]}
        handle={
          <ExperimentViewRunsTableResizerHandle
            runListHidden={runListHidden}
            updateRunListHidden={(value) => {
              if (onHiddenChange) {
                onHiddenChange(value);
                return;
              }
              updateUIState((state) => ({ ...state, runListHidden: value }));
            }}
          />
        }
        onResize={(event, { size }) => {
          if (runListHidden) {
            return;
          }
          onResize(size.width);
        }}
        onResizeStart={() => !runListHidden && setDragging(true)}
        onResizeStop={() => setDragging(false)}
      >
        {children}
      </ResizableBox>
      {dragging && (
        <Global
          styles={{
            'body, :host': {
              userSelect: 'none',
            },
          }}
        />
      )}
    </>
  );
};

export const ExperimentViewRunsTableResizerHandle = React.forwardRef<
  HTMLDivElement,
  {
    updateRunListHidden: (newValue: boolean) => void;
    runListHidden: boolean;
  }
>(({ updateRunListHidden, runListHidden, ...props }, ref) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      ref={ref}
      {...props}
      css={{
        transition: 'opacity 0.2s',
        width: 0,
        overflow: 'visible',
        height: '100%',
        position: 'relative',
        zIndex: 10,
        display: 'flex',
        opacity: runListHidden ? 1 : 0,
        '&:hover': {
          opacity: 1,
          '.bar': { opacity: 1 },
          '.button': {
            border: `2px solid ${theme.colors.actionDefaultBorderHover}`,
          },
        },
      }}
    >
      <div
        css={{
          position: 'absolute',
          // For the resizing area, use the icon size which is
          // the same as "collapse" button
          left: -theme.general.iconSize / 2,
          width: theme.general.iconSize,
          cursor: runListHidden ? undefined : 'ew-resize',
          height: '100%',
          top: 0,
          bottom: 0,
        }}
      >
        <div
          className="button"
          css={{
            top: '50%',
            transition: 'border-color 0.2s',
            position: 'absolute',
            width: theme.general.iconSize,
            height: theme.general.iconSize,
            backgroundColor: theme.colors.backgroundPrimary,
            borderRadius: theme.general.iconSize,
            overflow: 'hidden',
            border: `1px solid ${theme.colors.border}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 11,
          }}
        >
          <Button
            componentId="mlflow.experiment_page.table_resizer.collapse"
            onClick={() => updateRunListHidden(!runListHidden)}
            icon={runListHidden ? <ChevronRightIcon /> : <ChevronLeftIcon />}
            size="small"
          />
        </div>
      </div>
      <div
        className="bar"
        css={{
          position: 'absolute',
          opacity: 0,
          left: -RESIZE_BAR_WIDTH / 2,
          width: RESIZE_BAR_WIDTH,
          height: '100%',
          top: 0,
          bottom: 0,
          backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
        }}
      />
    </div>
  );
});

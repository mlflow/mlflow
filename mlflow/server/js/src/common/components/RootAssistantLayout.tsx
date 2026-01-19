/**
 * Root-level Assistant Layout - sits at MlflowRootRoute level
 * Provides split view: Main content | Gap | Assistant Panel
 */

import { useDesignSystemTheme } from '@databricks/design-system';
import { useCallback, useRef, useState, type ReactNode } from 'react';
import { useAssistant } from '../../assistant/AssistantContext';
import { AssistantChatPanel } from '../../assistant/AssistantChatPanel';

const MIN_PANEL_WIDTH = 300;
const MAX_PANEL_WIDTH_PERCENT = 60;
const DEFAULT_PANEL_WIDTH_PERCENT = 25;

export const RootAssistantLayout = ({ children }: { children: ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  const { isPanelOpen } = useAssistant();
  const [panelWidthPercent, setPanelWidthPercent] = useState(DEFAULT_PANEL_WIDTH_PERCENT);
  const isDraggingRef = useRef(false);

  const showPanel = isPanelOpen;

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDraggingRef.current = true;

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;

      const windowWidth = window.innerWidth;
      const distanceFromRight = windowWidth - e.clientX - theme.spacing.md;
      const newWidthPercent = (distanceFromRight / windowWidth) * 100;

      const minPercent = (MIN_PANEL_WIDTH / windowWidth) * 100;
      const clampedWidth = Math.max(minPercent, Math.min(MAX_PANEL_WIDTH_PERCENT, newWidthPercent));
      setPanelWidthPercent(clampedWidth);
    };

    const handleMouseUp = () => {
      isDraggingRef.current = false;
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, []);

  return (
    <>
      <div
        css={{
          display: 'flex',
          flexGrow: 1,
          minHeight: 0,
          backgroundColor: theme.colors.backgroundSecondary,
        }}
      >
        <div css={{ flex: 1, minWidth: 0, display: 'flex' }}>{children}</div>

        {showPanel && (
          <div
            data-assistant-ui="true"
            css={{
              position: 'relative',
              width: `${panelWidthPercent}%`,
              flexShrink: 0,
              backgroundColor: theme.colors.backgroundPrimary,
              margin: `${theme.spacing.sm}px ${theme.spacing.sm}px ${theme.spacing.sm}px 0`,
              borderRadius: theme.borders.borderRadiusMd,
              boxShadow: theme.shadows.md,
            }}
          >
            <div
              onMouseDown={handleMouseDown}
              css={{
                position: 'absolute',
                left: -theme.spacing.md / 2,
                top: 0,
                bottom: 0,
                width: 4,
                cursor: 'col-resize',
                '&:hover': {
                  backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                },
                zIndex: 1,
              }}
            />
            <AssistantChatPanel />
          </div>
        )}
      </div>
    </>
  );
};

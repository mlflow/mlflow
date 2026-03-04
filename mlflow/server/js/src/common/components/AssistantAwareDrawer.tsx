/**
 * AssistantAwareDrawer: A drawer component that automatically positions itself
 * based on the assistant panel state.
 *
 * When the assistant panel is closed, this renders the standard design system Drawer
 * from the right side. When the assistant panel is open, it renders the drawer
 * from the left side to avoid overlapping with the assistant.
 *
 * This component is designed to be a drop-in replacement for Drawer.Root + Drawer.Content
 * with automatic assistant-awareness.
 */

import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { Global } from '@emotion/react';
import { Drawer, useDesignSystemTheme } from '@databricks/design-system';
import { useAssistant } from '../../assistant/AssistantContext';

const ASSISTANT_OPEN_DRAWER_WIDTH = '70vw';
const MIN_DRAWER_WIDTH = 400;
const MAX_DRAWER_WIDTH_VW = 90;

// Context to pass modal state from Root to Content
const AssistantDrawerModalContext = createContext(false);

function resolveWidthToPixels(w: number | string): number {
  if (typeof w === 'number') return w;
  if (typeof w === 'string' && w.endsWith('vw')) {
    return (parseFloat(w) / 100) * window.innerWidth;
  }
  if (typeof w === 'string' && w.endsWith('px')) {
    return parseFloat(w);
  }
  return 320;
}

// Keeping modal prop for compatibility with Drawer.Root props.
function Root({
  open,
  onOpenChange,
  modal = false,
  children,
}: {
  // Matches with Drawer.Root props
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modal?: boolean;
  children: ReactNode;
}) {
  return (
    // NB: Modal defaults to false to allow clicks on the assistant chat panel to pass through,
    // but can be overridden when needed (e.g., for comparison drawers that need proper modal behavior)
    <AssistantDrawerModalContext.Provider value={modal}>
      <Drawer.Root modal={modal} open={open} onOpenChange={onOpenChange}>
        {children}
      </Drawer.Root>
    </AssistantDrawerModalContext.Provider>
  );
}

function Content({
  children,
  title,
  width = 320,
  componentId,
  footer,
  expandContentToFullHeight,
  useCustomScrollBehavior,
  disableOpenAutoFocus,
  hideClose,
  size,
  css: cssProp,
}: Drawer.DrawerContentProps) {
  const { isPanelOpen } = useAssistant();
  const isModal = useContext(AssistantDrawerModalContext);
  const { theme } = useDesignSystemTheme();

  // When assistant is open, position on left and use fixed width
  // When assistant is closed, position on right with user-specified width
  const position = isPanelOpen ? 'left' : 'right';
  const baseWidth = isPanelOpen ? ASSISTANT_OPEN_DRAWER_WIDTH : width;

  // Resizable width state
  const [drawerWidth, setDrawerWidth] = useState(() => resolveWidthToPixels(baseWidth));
  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);

  // Reset width when assistant panel state or initial width changes
  useEffect(() => {
    setDrawerWidth(resolveWidthToPixels(isPanelOpen ? ASSISTANT_OPEN_DRAWER_WIDTH : width));
  }, [isPanelOpen, width]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      isDraggingRef.current = true;
      setIsDragging(true);

      const handleMouseMove = (moveEvent: MouseEvent) => {
        if (!isDraggingRef.current) return;

        const windowWidth = window.innerWidth;
        const maxWidth = windowWidth * (MAX_DRAWER_WIDTH_VW / 100);
        let newWidth: number;

        if (position === 'right') {
          newWidth = windowWidth - moveEvent.clientX;
        } else {
          newWidth = moveEvent.clientX;
        }

        setDrawerWidth(Math.max(MIN_DRAWER_WIDTH, Math.min(maxWidth, newWidth)));
      };

      const handleMouseUp = () => {
        isDraggingRef.current = false;
        setIsDragging(false);
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [position],
  );

  // Override the design system's dark shadow with a subtler one
  const shadowOverride = {
    boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04)',
  };
  const mergedCss = cssProp ? [shadowOverride, ...(Array.isArray(cssProp) ? cssProp : [cssProp])] : shadowOverride;

  return (
    <>
      {/* Hide the dark overlay for non-modal drawers to prevent it from tinting the sidenav */}
      {!isModal && (
        <Global
          styles={{
            '[data-testid="drawer-overlay"]': {
              backgroundColor: 'transparent !important',
            },
          }}
        />
      )}
      {isDragging && <Global styles={{ 'body, body *': { userSelect: 'none', cursor: 'col-resize !important' } }} />}
      <Drawer.Content
        componentId={componentId}
        width={drawerWidth}
        title={title}
        position={position}
        footer={footer}
        expandContentToFullHeight={expandContentToFullHeight}
        useCustomScrollBehavior={useCustomScrollBehavior}
        disableOpenAutoFocus={disableOpenAutoFocus}
        hideClose={hideClose}
        size={size}
        css={mergedCss}
        onInteractOutside={(event) => {
          // Prevent drawer from closing when clicking on assistant UI elements.
          // We use a data attribute selector instead of React patterns (e.g., stopPropagation)
          // because Radix's DismissableLayer uses document-level listeners that bypass
          // React's event propagation. The onInteractOutside handler is the correct
          // interception point, and we identify assistant elements via data-assistant-ui
          // attribute set on AssistantIconButton and RootAssistantLayout.
          const target = event.target as HTMLElement;
          const isAssistantClick = target?.closest('[data-assistant-ui="true"]');
          const isResizeHandle = target?.closest('[data-drawer-resize-handle="true"]');
          if (isAssistantClick || isResizeHandle) {
            event.preventDefault();
          }
        }}
      >
        {children}
      </Drawer.Content>
      {/* Resize handle rendered via portal to share z-index context with the portaled drawer.
          The outer div is a wide (12px) invisible hit area for easy grabbing.
          The inner ::after pseudo-element is the thin (2px) visible line that appears on hover. */}
      {createPortal(
        <div
          data-drawer-resize-handle="true"
          onMouseDown={handleMouseDown}
          css={{
            position: 'fixed',
            top: 0,
            bottom: 0,
            ...(position === 'right' ? { right: drawerWidth - 6 } : { left: drawerWidth - 6 }),
            width: 12,
            cursor: 'col-resize',
            // Above drawer content (design system uses zIndexBase + 2 for content)
            zIndex: theme.options.zIndexBase + 3,
            // Thin centered line shown on hover/active
            '&::after': {
              content: '""',
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: '50%',
              width: 2,
              transform: 'translateX(-50%)',
              backgroundColor: 'transparent',
              borderRadius: 1,
              transition: 'background-color 0.15s',
            },
            '&:hover::after': {
              backgroundColor: theme.colors.borderDecorative,
            },
            '&:active::after': {
              backgroundColor: theme.colors.border,
            },
          }}
        />,
        document.body,
      )}
    </>
  );
}

export const AssistantAwareDrawer = {
  Root,
  Content,
};

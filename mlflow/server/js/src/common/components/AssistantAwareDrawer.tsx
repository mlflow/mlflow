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

import type { ReactNode } from 'react';
import { Drawer } from '@databricks/design-system';
import { useAssistant } from '../../assistant/AssistantContext';

const ASSISTANT_OPEN_DRAWER_WIDTH = '70vw';

// Keeping modal prop for compatibility with Drawer.Root props.
function Root({
  open,
  onOpenChange,
  modal,
  children,
}: {
  // Matches with Drawer.Root props
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modal?: boolean;
  children: ReactNode;
}) {
  return (
    // NB: Modal is set to false to allow clicks on the assistant chat panel to pass through
    <Drawer.Root modal={false} open={open} onOpenChange={onOpenChange}>
      {children}
    </Drawer.Root>
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

  // When assistant is open, position on left and use fixed width
  // When assistant is closed, position on right with user-specified width
  const position = isPanelOpen ? 'left' : 'right';
  const effectiveWidth = isPanelOpen ? ASSISTANT_OPEN_DRAWER_WIDTH : width;

  return (
    <Drawer.Content
      componentId={componentId}
      width={effectiveWidth}
      title={title}
      position={position}
      footer={footer}
      expandContentToFullHeight={expandContentToFullHeight}
      useCustomScrollBehavior={useCustomScrollBehavior}
      disableOpenAutoFocus={disableOpenAutoFocus}
      hideClose={hideClose}
      size={size}
      css={cssProp}
      onInteractOutside={(event) => {
        // Prevent closing when clicking on assistant chat panel
        const target = event.detail.originalEvent.target as HTMLElement;
        if (target?.closest?.('[data-assistant-ui="true"]')) {
          event.preventDefault();
        }
      }}
    >
      {children}
    </Drawer.Content>
  );
}

export const AssistantAwareDrawer = {
  Root,
  Content,
};

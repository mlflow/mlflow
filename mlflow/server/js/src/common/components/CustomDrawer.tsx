/**
 * Custom Drawer component with flexible positioning and width.
 * Unlike the design system Drawer, this supports:
 * - Opening from left or right
 * - Dynamic width based on layout state
 * - Modal behavior with backdrop that respects assistant UI
 */

import { useEffect, useRef, type ReactNode } from 'react';
import { CloseIcon, useDesignSystemTheme } from '@databricks/design-system';

// Delay before enabling click-outside listener to prevent immediate close on open
const CLICK_OUTSIDE_DELAY_MS = 100;

export interface CustomDrawerProps {
  open: boolean;
  onClose: () => void;
  width?: string;
  position?: 'left' | 'right';
  title?: ReactNode;
  children: ReactNode;
  componentId?: string;
}

export const CustomDrawer = ({
  open,
  onClose,
  width = '70vw',
  position = 'right',
  title,
  children,
  componentId,
}: CustomDrawerProps) => {
  const { theme } = useDesignSystemTheme();
  const drawerRef = useRef<HTMLDivElement>(null);

  // Handle escape key
  useEffect(() => {
    if (!open) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [open, onClose]);

  // Handle click outside to close
  useEffect(() => {
    if (!open) return;

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;

      // Don't close if clicking inside the drawer
      if (drawerRef.current && drawerRef.current.contains(target)) {
        return;
      }

      // Don't close if clicking on assistant panel or button
      const isAssistantUI = target.closest('[data-assistant-ui="true"]');
      if (isAssistantUI) {
        return;
      }

      // Close the drawer
      onClose();
    };

    // Add a small delay to prevent immediate closing when opening
    const timeoutId = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside);
    }, CLICK_OUTSIDE_DELAY_MS);

    return () => {
      clearTimeout(timeoutId);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [open, onClose]);

  if (!open) return null;

  const isRight = position === 'right';

  return (
    <>
      {/* Backdrop overlay */}
      <div
        css={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.45)',
          zIndex: 999,
          animation: 'fadeIn 0.3s ease-out',
          '@keyframes fadeIn': {
            from: { opacity: 0 },
            to: { opacity: 1 },
          },
        }}
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        ref={drawerRef}
        css={{
          position: 'fixed',
          top: 0,
          bottom: 0,
          [position]: 0,
          width,
          backgroundColor: theme.colors.backgroundPrimary,
          boxShadow: theme.shadows.xl,
          zIndex: 1000,
          display: 'flex',
          flexDirection: 'column',
          animation: `slideIn${isRight ? 'Right' : 'Left'} 0.3s ease-out`,
          '@keyframes slideInRight': {
            from: { transform: 'translateX(100%)' },
            to: { transform: 'translateX(0)' },
          },
          '@keyframes slideInLeft': {
            from: { transform: 'translateX(-100%)' },
            to: { transform: 'translateX(0)' },
          },
        }}
        data-component-id={componentId}
      >
        {/* Header */}
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: theme.spacing.lg,
            borderBottom: `1px solid ${theme.colors.border}`,
            flexShrink: 0,
          }}
        >
          <div
            css={{
              fontSize: theme.typography.fontSizeLg,
              fontWeight: theme.typography.typographyBoldFontWeight,
              color: theme.colors.textPrimary,
            }}
          >
            {title}
          </div>
          <button
            onClick={onClose}
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 32,
              height: 32,
              border: 'none',
              background: 'transparent',
              cursor: 'pointer',
              borderRadius: theme.borders.borderRadiusSm,
              color: theme.colors.textSecondary,
              transition: 'all 0.2s ease',
              '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                color: theme.colors.textPrimary,
              },
              '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
              },
            }}
            aria-label="Close drawer"
          >
            <CloseIcon />
          </button>
        </div>

        {/* Content */}
        <div
          css={{
            flex: 1,
            overflowY: 'auto',
            padding: theme.spacing.lg,
          }}
        >
          {children}
        </div>
      </div>
    </>
  );
};

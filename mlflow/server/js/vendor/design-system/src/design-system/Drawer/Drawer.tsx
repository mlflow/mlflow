import { css, keyframes } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import React, { useCallback, useMemo, useRef, useState } from 'react';

import { Button } from '../Button';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider';
import { ApplyDesignSystemContextOverrides } from '../DesignSystemProvider';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { CloseIcon } from '../Icon';
import { Spacer } from '../Spacer';
import { Typography } from '../Typography';
import type { AnalyticsEventProps } from '../types';
import { getShadowScrollStyles, useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface DrawerContentProps extends AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnView> {
  /** Contents displayed in the drawer */
  children: React.ReactNode;

  /**
   * Drawer header to be announced when the dialog is opened
   * If passing in a string styling will be provided, otherwise caller is responsible for proper title styling
   **/
  title: React.ReactNode;

  /**
   * The content width with a min width of 320 and a max width of 90vw
   * @default 320
   */
  width?: number | string;

  /**
   * The layout direction, on which side the drawer will appear
   * @default 'right'
   */
  position?: 'left' | 'right';

  /**
   * Provide a footer; using this property will ensure the correct scrolling behavior
   * @default 'undefined'
   */
  footer?: React.ReactNode;

  /**
   * Delegates all content scroll behavior to the caller if true
   *    Disable the default scroll drop shadow
   *    Hide the vertical content overflow
   *    Sets content right padding to 0 to leave room for caller to do so for proper scrollbar placement
   * @default false
   */
  useCustomScrollBehavior?: boolean;

  /**
   * If true the content of the Drawer will take up all available vertical space.
   * This is to keep the footer at the bottom of the drawer
   * @default false
   */
  expandContentToFullHeight?: boolean;

  /**
   * Disable auto focus on open
   * @default false
   */
  disableOpenAutoFocus?: boolean;

  /**
   * If true, the drawer and the backdrop will both be hidden. They will remain mounted, but not visible.
   * @default false
   */
  seeThrough?: boolean;

  /**
   * Event handler called when an interaction (pointer or focus event) happens outside the bounds of the component.
   * It can be prevented by calling event.preventDefault.
   */
  onInteractOutside?: DialogPrimitive.DialogContentProps['onInteractOutside'];

  /**
   * If true, the "x" icon in the header will be hidden
   * @default false
   */
  hideClose?: boolean;

  /**
   * Event handler called when the close button is clicked.
   * The default behavior of closing the drawer can be prevented by calling event.preventDefault.
   */
  onCloseClick?: React.MouseEventHandler<HTMLButtonElement>;

  /**
   * Drawer size. When set to small will reduce the padding around the content, title and buttons.
   * @default "default"
   */
  size?: 'default' | 'small';
}

const DEFAULT_WIDTH = 320;
const MIN_WIDTH = 320;
const MAX_WIDTH = '90vw';
const DEFAULT_POSITION = 'right';
const ZINDEX_OVERLAY = 1;
const ZINDEX_CONTENT = ZINDEX_OVERLAY + 1;

export const Content = ({
  children,
  footer,
  title,
  width,
  position: positionOverride,
  useCustomScrollBehavior,
  expandContentToFullHeight,
  disableOpenAutoFocus,
  onInteractOutside,
  seeThrough,
  hideClose,
  onCloseClick,
  componentId = 'design_system.drawer.content',
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView],
  size = 'default',
  ...props
}: DrawerContentProps) => {
  const { getPopupContainer } = useDesignSystemContext();
  const { theme } = useDesignSystemTheme();
  const horizontalContentPadding = size === 'small' ? theme.spacing.md : theme.spacing.lg;
  const [shouldContentBeFocusable, setShouldContentBeFocusable] = useState<boolean>(false);
  const contentContainerRef = useRef<HTMLDivElement>(null);
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Drawer,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
  });
  const { useNewShadows } = useDesignSystemSafexFlags();

  const { elementRef: onViewRef } = useNotifyOnFirstView<HTMLDivElement>({ onView: eventContext.onView });

  const contentRef = useCallback((node: HTMLDivElement) => {
    if (!node || !node.clientHeight) return;
    setShouldContentBeFocusable(node.scrollHeight > node.clientHeight);
  }, []);

  const mergedContentRef = useMergeRefs([contentRef, onViewRef]);

  const position = positionOverride ?? DEFAULT_POSITION;
  const overlayShow =
    position === 'right'
      ? keyframes({
          '0%': { transform: 'translate(100%, 0)' },
          '100%': { transform: 'translate(0, 0)' },
        })
      : keyframes({
          '0%': { transform: 'translate(-100%, 0)' },
          '100%': { transform: 'translate(0, 0)' },
        });

  const dialogPrimitiveContentStyle = css({
    color: theme.colors.textPrimary,
    backgroundColor: theme.colors.backgroundPrimary,
    boxShadow: useNewShadows
      ? theme.shadows.xl
      : 'hsl(206 22% 7% / 35%) 0px 10px 38px -10px, hsl(206 22% 7% / 20%) 0px 10px 20px -15px',
    position: 'fixed',
    top: 0,
    left: position === 'left' ? 0 : undefined,
    right: position === 'right' ? 0 : undefined,
    boxSizing: 'border-box',
    width: width ?? DEFAULT_WIDTH,
    minWidth: MIN_WIDTH,
    maxWidth: MAX_WIDTH,
    zIndex: theme.options.zIndexBase + ZINDEX_CONTENT,
    height: '100vh',
    paddingTop: size === 'small' ? theme.spacing.sm : theme.spacing.md,
    paddingLeft: 0,
    paddingBottom: 0,
    paddingRight: 0,
    overflow: 'hidden',
    '&:focus': { outline: 'none' },
    '@media (prefers-reduced-motion: no-preference)': {
      animation: `${overlayShow} 350ms cubic-bezier(0.16, 1, 0.3, 1)`,
    },
  });

  return (
    <DialogPrimitive.Portal container={getPopupContainer && getPopupContainer()}>
      <DialogPrimitive.Overlay
        css={{
          backgroundColor: theme.colors.overlayOverlay,
          position: 'fixed',
          inset: 0,
          // needed so that it covers the PersonaNavSidebar
          zIndex: theme.options.zIndexBase + ZINDEX_OVERLAY,
          opacity: seeThrough ? 0 : 1,
        }}
      />
      <DialogPrimitive.DialogContent
        {...addDebugOutlineIfEnabled()}
        css={dialogPrimitiveContentStyle}
        style={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'flex-start',
          opacity: seeThrough ? 0 : 1,
          ...(theme.isDarkMode && {
            borderLeft: `1px solid ${theme.colors.borderDecorative}`,
            ...(!useNewShadows && {
              boxShadow: 'none',
            }),
          }),
        }}
        aria-hidden={seeThrough}
        ref={contentContainerRef}
        onOpenAutoFocus={(event) => {
          if (disableOpenAutoFocus) {
            event.preventDefault();
          }
        }}
        onInteractOutside={onInteractOutside}
        {...props}
        {...eventContext.dataComponentProps}
      >
        <ApplyDesignSystemContextOverrides getPopupContainer={() => contentContainerRef.current ?? document.body}>
          {(title || !hideClose) && (
            <div
              css={{
                flexGrow: 0,
                flexShrink: 1,
                display: 'flex',
                flexDirection: 'row',
                justifyContent: 'space-between',
                alignItems: 'center',
                paddingRight: horizontalContentPadding,
                paddingLeft: horizontalContentPadding,
                marginBottom: theme.spacing.sm,
              }}
            >
              <DialogPrimitive.Title
                title={typeof title === 'string' ? title : undefined}
                asChild={typeof title === 'string'}
                css={{
                  flexGrow: 1,
                  marginBottom: 0,
                  marginTop: 0,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                }}
              >
                {typeof title === 'string' ? (
                  <Typography.Title elementLevel={2} level={size === 'small' ? 3 : 2} withoutMargins ellipsis>
                    {title}
                  </Typography.Title>
                ) : (
                  title
                )}
              </DialogPrimitive.Title>
              {!hideClose && (
                <DialogPrimitive.Close
                  asChild
                  css={{ flexShrink: 1, marginLeft: theme.spacing.xs }}
                  onClick={onCloseClick}
                >
                  <Button
                    componentId={`${componentId}.close`}
                    aria-label="Close"
                    icon={<CloseIcon />}
                    size={size === 'small' ? 'small' : undefined}
                  />
                </DialogPrimitive.Close>
              )}
            </div>
          )}
          <div
            ref={mergedContentRef}
            // Needed to make drawer content focusable when scrollable for keyboard-only users to be able to focus & scroll
            // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
            tabIndex={shouldContentBeFocusable ? 0 : -1}
            css={{
              // in order to have specific content in the drawer scroll with fixed title
              // hide overflow here and remove padding on the right side; content will be responsible for setting right padding
              // so that the scrollbar will appear in the padding right gutter
              paddingRight: useCustomScrollBehavior ? 0 : horizontalContentPadding,
              paddingLeft: horizontalContentPadding,
              overflowY: useCustomScrollBehavior ? 'hidden' : 'auto',
              height: expandContentToFullHeight ? '100%' : undefined,
              ...(!useCustomScrollBehavior ? getShadowScrollStyles(theme) : {}),
            }}
          >
            {children}
            {!footer && <Spacer size={size === 'small' ? 'md' : 'lg'} />}
          </div>

          {footer && (
            <div
              style={{
                paddingTop: theme.spacing.md,
                paddingRight: horizontalContentPadding,
                paddingLeft: horizontalContentPadding,
                paddingBottom: size === 'small' ? theme.spacing.md : theme.spacing.lg,
                flexGrow: 0,
                flexShrink: 1,
              }}
            >
              {footer}
            </div>
          )}
        </ApplyDesignSystemContextOverrides>
      </DialogPrimitive.DialogContent>
    </DialogPrimitive.Portal>
  );
};

export function Root(props: Pick<DialogPrimitive.DialogProps, 'onOpenChange' | 'children' | 'open' | 'modal'>) {
  return <DialogPrimitive.Root {...props} />;
}

export function Trigger(props: Omit<DialogPrimitive.DialogTriggerProps, 'asChild'>) {
  return <DialogPrimitive.Trigger asChild {...props} />;
}

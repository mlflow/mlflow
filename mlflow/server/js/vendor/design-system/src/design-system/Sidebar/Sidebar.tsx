import type { Interpolation, Theme as EmotionTheme } from '@emotion/react';
import { Global, css, keyframes } from '@emotion/react';
import type { ComponentPropsWithoutRef, CSSProperties } from 'react';
import React, { createContext, forwardRef, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { ResizableBox } from 'react-resizable';

import { Button, type ButtonProps } from '../Button';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { ChevronLeftIcon, ChevronRightIcon, CloseIcon } from '../Icon';
import { Typography } from '../Typography';
import type { AnalyticsEventProps } from '../types';
import { importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { useMediaQuery } from '../utils/useMediaQuery';

export interface SidebarProps {
  /** The layout direction */
  position?: 'left' | 'right';

  /** Contents displayed in the sidebar */
  children?: React.ReactNode;

  /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
  dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}

export interface NavProps {
  /** Contents displayed in the nav bar */
  children?: React.ReactNode;

  /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
  dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}

/**
 * `ResizableBox` passes `handleAxis` to the element used as handle. We need to wrap the handle to prevent
 * `handleAxis becoming an attribute on the div element.
 */
const ResizablePanelHandle = forwardRef<
  HTMLDivElement,
  {
    handleAxis?: string;
  } & ComponentPropsWithoutRef<'div'>
>(function ResizablePanelHandle({ handleAxis, children, ...otherProps }, ref) {
  return (
    <div ref={ref} {...otherProps}>
      {children}
    </div>
  );
});

export interface NavButtonProps extends ButtonProps {
  /** Check if the currrent button in nav bar is being selected */
  active?: boolean;

  /** Check if the currrent button in nav bar is being disabled */
  disabled?: boolean;

  /** The icon on the button */
  icon?: React.ReactNode;

  /** Contents displayed in the nav bar */
  children?: React.ReactNode;

  /** The callback function when nav button is clicked */
  onClick?: () => void;

  'aria-label'?: string;

  /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
  dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}

export interface ContentProps extends AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
  /** The open panel id */
  openPanelId?: number;

  /** The content width, default is 200px */
  width?: number;

  /** The minimum content width */
  minWidth?: number;

  /** The maximum content width */
  maxWidth?: number;

  /** Whether or not to make the component resizable */
  disableResize?: boolean;

  /** Whether or not to show a close button which can close the panel */
  closable?: boolean;

  /** Whether to destroy inactive panels and their state when initializing the component or switching the active panel */
  destroyInactivePanels?: boolean;

  /** The callback function when close button is clicked */
  onClose?: () => void;

  /** This callback function is called when the content resize is started */
  onResizeStart?: (size: number) => void;

  /** This callback function is called when the content resize is completed */
  onResizeStop?: (size: number) => void;

  /** Contents displayed in the content */
  children?: React.ReactNode;

  /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
  dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;

  /** For migration purposes. Enables updates where panel is overlay and toggleable for non closable panel in compact mode */
  enableCompact?: boolean;

  /** Applies styles to the react-resizable container */
  resizeBoxStyle?: CSSProperties;

  /** Removes side border for cases where Navbar is not used */
  noSideBorder?: boolean;
}

export interface PanelProps {
  /** The panel id */
  panelId: number;

  /** Contents displayed in the the panel */
  children?: React.ReactNode;

  /** Forced render of content in the panel, not lazy render after clicking */
  forceRender?: boolean;

  /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
  dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}

export interface PanelHeaderProps extends AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
  /** Contents displayed in the header section of the panel */
  children?: React.ReactNode;

  /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
  dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}

export interface PanelHeaderTitleProps {
  /** Text displayed in the header section of the panel */
  title: string;

  /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
  dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}

export interface PanelHeaderButtonProps {
  /** Optional buttons displayed in the panel header */
  children?: React.ReactNode;

  /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
  dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}

export interface PanelBodyProps {
  /** Contents displayed in the body of the panel */
  children?: React.ReactNode;

  /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
  dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}

interface SidebarContextType {
  /** The layout direction */
  position?: 'left' | 'right';
}

interface ContentContextType {
  openPanelId?: number;
  closable?: boolean;
  destroyInactivePanels?: boolean;
  setIsClosed: () => void;
}

interface ToggleButtonProps extends AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
  isExpanded: boolean;
  position: 'left' | 'right';
  toggleIsExpanded: () => void;
}

const DEFAULT_WIDTH = 200;

const ContentContextDefaults: ContentContextType = {
  openPanelId: undefined,
  closable: true,
  destroyInactivePanels: false,
  setIsClosed: () => {},
};

const SidebarContextDefaults: SidebarContextType = {
  position: 'left',
};

const ContentContext = createContext<ContentContextType>(ContentContextDefaults);

const SidebarContext = createContext<SidebarContextType>(SidebarContextDefaults);

export function Nav({ children, dangerouslyAppendEmotionCSS }: NavProps): JSX.Element {
  const { theme } = useDesignSystemTheme();
  return (
    <nav
      css={[
        {
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
          padding: theme.spacing.xs,
        },
        dangerouslyAppendEmotionCSS,
      ]}
    >
      {children}
    </nav>
  );
}

export const NavButton = React.forwardRef<HTMLButtonElement, NavButtonProps>(
  (
    { active, disabled, icon, onClick, children, dangerouslyAppendEmotionCSS, 'aria-label': ariaLabel, ...restProps },
    ref,
  ) => {
    const { theme } = useDesignSystemTheme();
    return (
      <div
        css={[
          active
            ? importantify({
                borderRadius: theme.legacyBorders.borderRadiusMd,
                background: theme.colors.actionDefaultBackgroundPress,
                button: {
                  '&:enabled:not(:hover):not(:active) > .anticon': { color: theme.colors.actionTertiaryTextPress },
                },
              })
            : undefined,
          dangerouslyAppendEmotionCSS,
        ]}
      >
        <Button ref={ref} icon={icon} onClick={onClick} disabled={disabled} aria-label={ariaLabel} {...restProps}>
          {children}
        </Button>
      </div>
    );
  },
);

const TOGGLE_BUTTON_Z_INDEX = 100;
const COMPACT_CONTENT_Z_INDEX = 50;

const ToggleButton = ({ isExpanded, position, toggleIsExpanded, componentId }: ToggleButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const positionStyle = useMemo(() => {
    if (position === 'right') {
      return isExpanded
        ? { right: DEFAULT_WIDTH, transform: 'translateX(+50%)' }
        : { left: 0, transform: 'translateX(-50%)' };
    } else {
      return isExpanded
        ? { left: DEFAULT_WIDTH, transform: 'translateX(-50%)' }
        : { right: 0, transform: 'translateX(+50%)' };
    }
  }, [isExpanded, position]);
  const ToggleIcon = useMemo(() => {
    if (position === 'right') {
      return isExpanded ? ChevronRightIcon : ChevronLeftIcon;
    } else {
      return isExpanded ? ChevronLeftIcon : ChevronRightIcon;
    }
  }, [isExpanded, position]);

  return (
    <div
      css={{
        position: 'absolute',
        top: 0,
        height: 46,
        display: 'flex',
        alignItems: 'center',
        zIndex: TOGGLE_BUTTON_Z_INDEX,
        ...positionStyle,
      }}
    >
      {/* Phantom element with white background to avoid button being overlapped by underlying component due to transparent background */}
      <div
        css={{
          borderRadius: '100%',
          width: theme.spacing.lg,
          height: theme.spacing.lg,
          backgroundColor: theme.colors.backgroundPrimary,
          position: 'absolute',
        }}
      />
      <Button
        componentId={componentId}
        css={{
          borderRadius: '100%',
          '&&': {
            padding: '0px !important',
            width: `${theme.spacing.lg}px !important`,
          },
        }}
        onClick={toggleIsExpanded}
        size="small"
        aria-label={isExpanded ? 'hide sidebar' : 'expand sidebar'}
        aria-expanded={isExpanded}
      >
        <ToggleIcon />
      </Button>
    </div>
  );
};

const getContentAnimation = (width: number) => {
  const showAnimation = keyframes`
  from { opacity: 0 }
  80%  { opacity: 0 }
  to   { opacity: 1 }`;
  const openAnimation = keyframes`
  from { width: 50px }
  to   { width: ${width}px }`;

  return {
    open: `${openAnimation} .2s cubic-bezier(0, 0, 0.2, 1)`,
    show: `${showAnimation} .25s linear`,
  };
};

export function Content({
  disableResize,
  openPanelId,
  closable = true,
  onClose,
  onResizeStart,
  onResizeStop,
  width,
  minWidth,
  maxWidth,
  destroyInactivePanels = false,
  children,
  dangerouslyAppendEmotionCSS,
  enableCompact,
  resizeBoxStyle,
  noSideBorder,
  componentId,
}: ContentProps): JSX.Element {
  const { theme } = useDesignSystemTheme();
  const isCompact = useMediaQuery({ query: `not (min-width: ${theme.responsive.breakpoints.sm}px)` }) && enableCompact;
  const defaultAnimation = useMemo(
    () => getContentAnimation(isCompact ? DEFAULT_WIDTH : width || DEFAULT_WIDTH),
    [isCompact, width],
  );
  // specifically for non closable panel in compact mode
  const [isExpanded, setIsExpanded] = useState(true);
  // hide the panel in compact mode when the panel is not closable and collapsed
  const isNotExpandedStyle = css(isCompact && !closable && !isExpanded && { display: 'none' });

  const sidebarContext = useContext(SidebarContext);
  const onCloseRef = useRef(onClose);
  const resizeHandleStyle = sidebarContext.position === 'right' ? { left: 0 } : { right: 0 };
  const [dragging, setDragging] = useState(false);
  const isPanelClosed = openPanelId == null;
  const [animation, setAnimation] = useState(isPanelClosed ? defaultAnimation : undefined);
  const compactStyle = css(
    isCompact && {
      position: 'absolute',
      zIndex: COMPACT_CONTENT_Z_INDEX,
      left: sidebarContext.position === 'left' && closable ? '100%' : undefined,
      right: sidebarContext.position === 'right' && closable ? '100%' : undefined,
      backgroundColor: theme.colors.backgroundPrimary,
      width: DEFAULT_WIDTH,
      // shift to the top due to border
      top: -1,
    },
  );
  const hiddenPanelStyle = css(isPanelClosed && { display: 'none' });
  const containerStyle = css({
    animation: animation?.open,
    direction: sidebarContext.position === 'right' ? 'rtl' : 'ltr',
    position: 'relative',
    borderWidth:
      sidebarContext.position === 'right'
        ? `0 ${noSideBorder ? 0 : theme.general.borderWidth}px 0 0 `
        : `0 0 0 ${noSideBorder ? 0 : theme.general.borderWidth}px`,
    borderStyle: 'inherit',
    borderColor: 'inherit',
    boxSizing: 'content-box',
  });
  const highlightedBorderStyle =
    sidebarContext.position === 'right'
      ? css({ borderLeft: `2px solid ${theme.colors.actionDefaultBorderHover}` })
      : css({ borderRight: `2px solid ${theme.colors.actionDefaultBorderHover}` });

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  // For non closable panel, reset expanded state to true so that the panel stays open
  // the next time the screen goes into compact mode.
  useEffect(() => {
    if (!closable && enableCompact && !isCompact) {
      setIsExpanded(true);
    }
  }, [isCompact, closable, defaultAnimation, enableCompact]);

  const value = useMemo(
    () => ({
      openPanelId,
      closable,
      destroyInactivePanels,
      setIsClosed: () => {
        onCloseRef.current?.();
        if (!animation) {
          setAnimation(defaultAnimation);
        }
      },
    }),
    [openPanelId, closable, defaultAnimation, animation, destroyInactivePanels],
  );

  return (
    <ContentContext.Provider value={value}>
      {disableResize || isCompact ? (
        <>
          <div
            css={[
              css({ width: width || '100%', height: '100%', overflow: 'hidden' }, containerStyle, compactStyle),
              dangerouslyAppendEmotionCSS,
              hiddenPanelStyle,
              isNotExpandedStyle,
            ]}
            aria-hidden={isPanelClosed}
          >
            <div css={{ opacity: 1, height: '100%', animation: animation?.show, direction: 'ltr' }}>{children}</div>
          </div>
          {/* only shows the toggle button for non closable panel in compact mode */}
          {!closable && isCompact && (
            <div
              css={{
                width: !isExpanded ? theme.spacing.md : undefined,
                marginRight: isExpanded ? theme.spacing.md : undefined,
                position: 'relative',
              }}
            >
              <ToggleButton
                componentId={componentId ? `${componentId}.toggle` : 'sidebar-toggle'}
                isExpanded={isExpanded}
                position={sidebarContext.position || 'left'}
                toggleIsExpanded={() => setIsExpanded((prev) => !prev)}
              />
            </div>
          )}
        </>
      ) : (
        <>
          {/* Avoids selecting text in the page during dragging */}
          {dragging && (
            <Global
              styles={{
                'body, :host': {
                  userSelect: 'none',
                },
              }}
            />
          )}
          <ResizableBox
            style={resizeBoxStyle}
            width={width || DEFAULT_WIDTH}
            height={undefined}
            axis="x"
            resizeHandles={sidebarContext.position === 'right' ? ['w'] : ['e']}
            minConstraints={[minWidth ?? DEFAULT_WIDTH, 150]}
            maxConstraints={[maxWidth ?? 800, 150]}
            onResizeStart={(_, { size }) => {
              onResizeStart?.(size.width);
              setDragging(true);
            }}
            onResizeStop={(_, { size }) => {
              onResizeStop?.(size.width);
              setDragging(false);
            }}
            handle={
              <ResizablePanelHandle
                css={css(
                  {
                    width: 10,
                    height: '100%',
                    position: 'absolute',
                    top: 0,
                    cursor: sidebarContext.position === 'right' ? 'w-resize' : 'e-resize',
                    '&:hover': highlightedBorderStyle,
                    ...resizeHandleStyle,
                  },
                  dragging && highlightedBorderStyle,
                )}
              />
            }
            css={[containerStyle, hiddenPanelStyle]}
            aria-hidden={isPanelClosed}
          >
            <div
              css={[
                {
                  opacity: 1,
                  animation: animation?.show,
                  direction: 'ltr',
                  height: '100%',
                },
                dangerouslyAppendEmotionCSS,
              ]}
            >
              {children}
            </div>
          </ResizableBox>
        </>
      )}
    </ContentContext.Provider>
  );
}

export function Panel({
  panelId,
  children,
  forceRender = false,
  dangerouslyAppendEmotionCSS,
  ...delegated
}: PanelProps): JSX.Element | null {
  const { openPanelId, destroyInactivePanels } = useContext(ContentContext);
  const hasOpenedPanelRef = useRef(false);
  const isPanelOpen = openPanelId === panelId;

  if (isPanelOpen && !hasOpenedPanelRef.current) {
    hasOpenedPanelRef.current = true;
  }

  if ((destroyInactivePanels || !hasOpenedPanelRef.current) && !isPanelOpen && !forceRender) return null;

  return (
    <div
      css={[
        { display: 'flex', height: '100%', flexDirection: 'column' },
        dangerouslyAppendEmotionCSS,
        !isPanelOpen && { display: 'none' },
      ]}
      aria-hidden={!isPanelOpen}
      {...delegated}
    >
      {children}
    </div>
  );
}

export function PanelHeader({ children, dangerouslyAppendEmotionCSS, componentId }: PanelHeaderProps): JSX.Element {
  const { theme } = useDesignSystemTheme();
  const contentContext = useContext(ContentContext);
  return (
    <div
      css={[
        {
          display: 'flex',
          paddingLeft: 8,
          paddingRight: 4,
          alignItems: 'center',
          minHeight: theme.general.heightSm,
          justifyContent: 'space-between',
          fontWeight: theme.typography.typographyBoldFontWeight,
          color: theme.colors.textPrimary,
        },
        dangerouslyAppendEmotionCSS,
      ]}
    >
      <div css={{ width: contentContext.closable ? `calc(100% - ${theme.spacing.lg}px)` : '100%' }}>
        <Typography.Title
          level={4}
          css={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',

            '&&': {
              margin: 0,
            },
          }}
        >
          {children}
        </Typography.Title>
      </div>
      {contentContext.closable ? (
        <div>
          <Button
            componentId={
              componentId ? `${componentId}.close` : 'codegen_design-system_src_design-system_sidebar_sidebar.tsx_427'
            }
            size="small"
            icon={<CloseIcon />}
            aria-label="Close"
            onClick={() => {
              contentContext.setIsClosed();
            }}
          />
        </div>
      ) : null}
    </div>
  );
}

export function PanelHeaderTitle({ title, dangerouslyAppendEmotionCSS }: PanelHeaderTitleProps) {
  return (
    <div
      title={title}
      css={[
        {
          alignSelf: 'center',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        },
        dangerouslyAppendEmotionCSS,
      ]}
    >
      {title}
    </div>
  );
}

export function PanelHeaderButtons({ children, dangerouslyAppendEmotionCSS }: PanelHeaderButtonProps) {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={[
        { display: 'flex', alignItems: 'center', gap: theme.spacing.xs, paddingRight: theme.spacing.xs },
        dangerouslyAppendEmotionCSS,
      ]}
    >
      {children}
    </div>
  );
}

export function PanelBody({ children, dangerouslyAppendEmotionCSS }: PanelBodyProps): JSX.Element {
  const { theme } = useDesignSystemTheme();
  const [shouldBeFocusable, setShouldBeFocusable] = useState<boolean>(false);
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ref = bodyRef.current;
    if (ref) {
      if (ref.scrollHeight > ref.clientHeight) {
        setShouldBeFocusable(true);
      } else {
        setShouldBeFocusable(false);
      }
    }
  }, []);

  return (
    <div
      ref={bodyRef}
      // Needed to make panel body content focusable when scrollable for keyboard-only users to be able to focus & scroll
      // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
      tabIndex={shouldBeFocusable ? 0 : -1}
      css={[
        {
          height: '100%',
          overflowX: 'hidden',
          overflowY: 'auto',
          padding: '0 8px',
          colorScheme: theme.isDarkMode ? 'dark' : 'light',
        },
        dangerouslyAppendEmotionCSS,
      ]}
    >
      {children}
    </div>
  );
}

export const Sidebar = /* #__PURE__ */ (() => {
  function Sidebar({ position, children, dangerouslyAppendEmotionCSS }: SidebarProps): JSX.Element {
    const { theme } = useDesignSystemTheme();
    const value = useMemo(() => {
      return {
        position: position || 'left',
      };
    }, [position]);
    return (
      <SidebarContext.Provider value={value}>
        <div
          {...addDebugOutlineIfEnabled()}
          css={[
            {
              display: 'flex',
              height: '100%',
              backgroundColor: theme.colors.backgroundPrimary,
              flexDirection: position === 'right' ? 'row-reverse' : 'row',
              borderStyle: 'solid',
              borderColor: theme.colors.borderDecorative,
              borderWidth:
                position === 'right'
                  ? `0 0 0 ${theme.general.borderWidth}px`
                  : `0px ${theme.general.borderWidth}px 0 0`,
              boxSizing: 'content-box',
              position: 'relative',
            },
            dangerouslyAppendEmotionCSS,
          ]}
        >
          {children}
        </div>
      </SidebarContext.Provider>
    );
  }

  Sidebar.Content = Content;
  Sidebar.Nav = Nav;
  Sidebar.NavButton = NavButton;
  Sidebar.Panel = Panel;
  Sidebar.PanelHeader = PanelHeader;
  Sidebar.PanelHeaderTitle = PanelHeaderTitle;
  Sidebar.PanelHeaderButtons = PanelHeaderButtons;
  Sidebar.PanelBody = PanelBody;

  return Sidebar;
})();

import type { CSSObject, Interpolation } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import * as ScrollArea from '@radix-ui/react-scroll-area';
import * as RadixTabs from '@radix-ui/react-tabs';
import { debounce } from 'lodash';
import React, { useMemo } from 'react';

import type { ButtonProps, DataComponentProps } from '..';
import {
  Button,
  CloseSmallIcon,
  PlusIcon,
  getShadowScrollStyles,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
  useDesignSystemTheme,
} from '..';
import type { Theme } from '../../theme';
import { getCommonTabsListStyles, getCommonTabsTriggerStyles } from '../_shared_';
import type { AnalyticsEventValueChangeNoPiiFlagProps, DangerousGeneralProps } from '../types';

interface TabsRootContextType {
  activeValue?: string;
  dataComponentProps: DataComponentProps;
}

interface TabsListContextType {
  viewportRef: React.RefObject<HTMLDivElement>;
}

const TabsRootContext = React.createContext<TabsRootContextType>({
  activeValue: undefined,
  dataComponentProps: {
    'data-component-id': 'design_system.tabs.default_component_id',
    'data-component-type': DesignSystemEventProviderComponentTypes.Tabs,
  },
});
const TabsListContext = React.createContext<TabsListContextType>({ viewportRef: { current: null } });

interface RootProps
  extends Omit<RadixTabs.TabsProps, 'asChild' | 'orientation' | 'dir'>,
    AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {}

export const Root = React.forwardRef<HTMLDivElement, RootProps>(
  (
    {
      value,
      defaultValue,
      onValueChange,
      componentId,
      analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
      valueHasNoPii,
      ...props
    },
    forwardedRef,
  ) => {
    const isControlled = value !== undefined;
    const [uncontrolledActiveValue, setUncontrolledActiveValue] = React.useState<string | undefined>(defaultValue);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
      componentType: DesignSystemEventProviderComponentTypes.Tabs,
      componentId,
      analyticsEvents: memoizedAnalyticsEvents,
      valueHasNoPii,
      shouldStartInteraction: true,
    });

    const onValueChangeWrapper = (value: string) => {
      eventContext.onValueChange(value);
      if (onValueChange) {
        onValueChange(value);
      }
      if (!isControlled) {
        setUncontrolledActiveValue(value);
      }
    };

    return (
      <TabsRootContext.Provider
        value={{
          activeValue: isControlled ? value : uncontrolledActiveValue,
          dataComponentProps: eventContext.dataComponentProps,
        }}
      >
        <RadixTabs.Root
          value={value}
          defaultValue={defaultValue}
          onValueChange={onValueChangeWrapper}
          {...props}
          ref={forwardedRef}
        />
      </TabsRootContext.Provider>
    );
  },
);

interface AddButtonProps extends DangerousGeneralProps, Pick<ButtonProps, 'onClick' | 'className'> {}

interface ListProps extends DangerousGeneralProps, Omit<RadixTabs.TabsListProps, 'asChild' | 'loop'> {
  /** The add tab button is only displayed when this prop is passed. Include the `onClick` handler for the button's action */
  addButtonProps?: AddButtonProps;
  /** Styling for the list's scrollable area viewport */
  scrollAreaViewportCss?: Interpolation<Theme>;

  /** If using a background color other than primary, pass this color along so the shadow scroll styling can use this color*/
  shadowScrollStylesBackgroundColor?: string;

  /** For customizing the height of the list's scrollbar. The default height is 3px.
   *  Customizing the scrollbar height is not recommended. Ask in #dubois before using.
   */
  scrollbarHeight?: number;

  /** Optional callback to get access to the viewport element. */
  getScrollAreaViewportRef?: (element: HTMLDivElement | null) => void;

  /** Optional CSS to append to the tab list container */
  tabListCss?: Interpolation<Theme>;
}

export const List = React.forwardRef<HTMLDivElement, ListProps>(
  (
    {
      addButtonProps,
      scrollAreaViewportCss,
      tabListCss,
      children,
      dangerouslyAppendEmotionCSS,
      shadowScrollStylesBackgroundColor,
      scrollbarHeight,
      getScrollAreaViewportRef,
      ...props
    },
    forwardedRef,
  ) => {
    const viewportRef = React.useRef<HTMLDivElement>(null);
    const { dataComponentProps } = React.useContext(TabsRootContext);
    const css = useListStyles(shadowScrollStylesBackgroundColor, scrollbarHeight);

    React.useEffect(() => {
      if (getScrollAreaViewportRef) {
        getScrollAreaViewportRef(viewportRef.current);
      }
    }, [getScrollAreaViewportRef]);

    return (
      <TabsListContext.Provider value={{ viewportRef }}>
        <div css={[css['container'], dangerouslyAppendEmotionCSS]}>
          <ScrollArea.Root type="hover" css={[css['root']]}>
            <ScrollArea.Viewport css={[css['viewport'], scrollAreaViewportCss]} ref={viewportRef}>
              <RadixTabs.List css={[css['list'], tabListCss]} {...props} ref={forwardedRef} {...dataComponentProps}>
                {children}
              </RadixTabs.List>
            </ScrollArea.Viewport>
            <ScrollArea.Scrollbar orientation="horizontal" css={css['scrollbar']}>
              <ScrollArea.Thumb css={css['thumb']} />
            </ScrollArea.Scrollbar>
          </ScrollArea.Root>
          {addButtonProps && (
            <div css={[css['addButtonContainer'], addButtonProps.dangerouslyAppendEmotionCSS]}>
              <Button
                icon={<PlusIcon />}
                size="small"
                aria-label="Add tab"
                css={css['addButton']}
                onClick={addButtonProps.onClick}
                componentId={`${dataComponentProps['data-component-id']}.add_tab`}
                className={addButtonProps.className}
              />
            </div>
          )}
        </div>
      </TabsListContext.Provider>
    );
  },
);

export interface TriggerProps extends Omit<RadixTabs.TabsTriggerProps, 'asChild'> {
  /** Called when the close tab icon is clicked. The close icon is only displayed when this prop is passed */
  onClose?: (value: string) => void;
}

export const Trigger = React.forwardRef<HTMLButtonElement, TriggerProps>(
  ({ onClose, value, disabled, children, ...props }, forwardedRef) => {
    const triggerRef = React.useRef<HTMLButtonElement>(null);
    const mergedRef = useMergeRefs([forwardedRef, triggerRef]);
    const { activeValue, dataComponentProps } = React.useContext(TabsRootContext);
    const componentId = dataComponentProps['data-component-id'];
    const { viewportRef } = React.useContext(TabsListContext);
    const isClosable = onClose !== undefined && !disabled;
    const css = useTriggerStyles(isClosable);
    const eventContext = useDesignSystemEventComponentCallbacks({
      componentType: DesignSystemEventProviderComponentTypes.Button,
      componentId: `${componentId}.close_tab`,
      analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
    });

    const scrollActiveTabIntoView = React.useCallback(() => {
      if (triggerRef.current && viewportRef.current && activeValue === value) {
        const viewportPosition = viewportRef.current.getBoundingClientRect();
        const triggerPosition = triggerRef.current.getBoundingClientRect();
        if (triggerPosition.left < viewportPosition.left) {
          viewportRef.current.scrollLeft -= viewportPosition.left - triggerPosition.left;
        } else if (triggerPosition.right > viewportPosition.right) {
          viewportRef.current.scrollLeft += triggerPosition.right - viewportPosition.right;
        }
      }
    }, [viewportRef, activeValue, value]);

    const debouncedScrollActiveTabIntoView = React.useMemo(
      () => debounce(scrollActiveTabIntoView, 10),
      [scrollActiveTabIntoView],
    );

    React.useEffect(() => {
      scrollActiveTabIntoView();
    }, [scrollActiveTabIntoView]);

    React.useEffect(() => {
      if (!viewportRef.current || !triggerRef.current) {
        return;
      }

      const resizeObserver = new ResizeObserver(debouncedScrollActiveTabIntoView);
      resizeObserver.observe(viewportRef.current);
      resizeObserver.observe(triggerRef.current);

      return () => {
        resizeObserver.disconnect();
        debouncedScrollActiveTabIntoView.cancel();
      };
    }, [debouncedScrollActiveTabIntoView, viewportRef]);

    return (
      <RadixTabs.Trigger
        css={css['trigger']}
        value={value}
        disabled={disabled}
        // The close icon cannot be focused within the trigger button
        // Instead, we close the tab when the Delete key is pressed
        onKeyDown={(e) => {
          if (isClosable && e.key === 'Delete') {
            eventContext.onClick(e);
            e.stopPropagation();
            e.preventDefault();
            onClose(value);
          }
        }}
        // Middle click also closes the tab
        // The Radix Tabs implementation uses onMouseDown for handling clicking tabs so we use it here as well
        onMouseDown={(e) => {
          if (isClosable && e.button === 1) {
            eventContext.onClick(e);
            e.stopPropagation();
            e.preventDefault();
            onClose(value);
          }
        }}
        {...props}
        ref={mergedRef}
      >
        {children}
        {isClosable && (
          // An icon is used instead of a button to prevent nesting a button within a button
          <CloseSmallIcon
            onMouseDown={(e) => {
              // The Radix Tabs implementation only allows the trigger to be selected when the left mouse
              // button is clicked and not when the control key is pressed (to avoid MacOS right click).
              // Reimplementing the same behavior for the close icon in the trigger
              if (!disabled && e.button === 0 && e.ctrlKey === false) {
                eventContext.onClick(e);
                // Clicking the close icon should not select the tab
                e.stopPropagation();
                e.preventDefault();
                onClose(value);
              }
            }}
            css={css['closeSmallIcon']}
            aria-hidden="false"
            aria-label="Press delete to close the tab"
          />
        )}
      </RadixTabs.Trigger>
    );
  },
);

export const Content = React.forwardRef<HTMLDivElement, Omit<RadixTabs.TabsContentProps, 'asChild'>>(
  ({ ...props }, forwardedRef) => {
    const css = useContentStyles();
    return <RadixTabs.Content css={css} {...props} ref={forwardedRef} />;
  },
);

const useListStyles = (
  shadowScrollStylesBackgroundColor?: string,
  scrollbarHeight?: number,
): Record<string, CSSObject> => {
  const { theme } = useDesignSystemTheme();
  const containerStyles = getCommonTabsListStyles(theme);
  return {
    container: containerStyles,
    root: { overflow: 'hidden' },
    viewport: {
      ...getShadowScrollStyles(theme, {
        orientation: 'horizontal',
        backgroundColor: shadowScrollStylesBackgroundColor,
      }),
    },
    list: { display: 'flex', alignItems: 'center' },
    scrollbar: {
      display: 'flex',
      flexDirection: 'column',
      userSelect: 'none',
      /* Disable browser handling of all panning and zooming gestures on touch devices */
      touchAction: 'none',
      height: scrollbarHeight ?? 3,
    },
    thumb: {
      flex: 1,
      background: theme.isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(17, 23, 28, 0.2)',
      '&:hover': {
        background: theme.isDarkMode ? 'rgba(255, 255, 255, 0.3)' : 'rgba(17, 23, 28, 0.3)',
      },
      borderRadius: theme.legacyBorders.borderRadiusMd,
      position: 'relative',
    },
    addButtonContainer: { flex: 1 },
    addButton: { margin: '2px 0 6px 0' },
  };
};

const useTriggerStyles = (isClosable: boolean): Record<string, CSSObject> => {
  const { theme } = useDesignSystemTheme();
  const commonTriggerStyles = getCommonTabsTriggerStyles(theme);
  return {
    trigger: {
      ...commonTriggerStyles,
      alignItems: 'center',
      justifyContent: isClosable ? 'space-between' : 'center',
      minWidth: isClosable ? theme.spacing.lg + theme.spacing.md : theme.spacing.lg,
      color: theme.colors.textSecondary,
      lineHeight: theme.typography.lineHeightBase,
      whiteSpace: 'nowrap',
      border: 'none',
      padding: `${theme.spacing.xs}px 0 ${theme.spacing.sm}px 0`,

      // The close icon is hidden on inactive tabs until the tab is hovered
      // Checking for the last icon to handle cases where the tab name includes an icon
      [`& > .anticon:last-of-type`]: {
        visibility: 'hidden',
      },

      '&:hover': {
        cursor: 'pointer',
        color: theme.colors.actionDefaultTextHover,
        [`& > .anticon:last-of-type`]: {
          visibility: 'visible',
        },
      },
      '&:active': {
        color: theme.colors.actionDefaultTextPress,
      },

      outlineStyle: 'none',
      outlineColor: theme.colors.actionDefaultBorderFocus,
      '&:focus-visible': {
        outlineStyle: 'auto',
      },

      '&[data-state="active"]': {
        color: theme.colors.textPrimary,
        // Use box-shadow instead of border to prevent it from affecting the size of the element, which results in visual
        // jumping when switching tabs.
        boxShadow: `inset 0 -4px 0 ${theme.colors.actionPrimaryBackgroundDefault}`,

        // The close icon is always visible on active tabs
        [`& > .anticon:last-of-type`]: {
          visibility: 'visible',
        },
      },

      '&[data-disabled]': {
        color: theme.colors.actionDisabledText,
        '&:hover': {
          cursor: 'not-allowed',
        },
      },
    },
    closeSmallIcon: {
      marginLeft: theme.spacing.xs,
      color: theme.colors.textSecondary,
      '&:hover': {
        color: theme.colors.actionDefaultTextHover,
      },
      '&:active': {
        color: theme.colors.actionDefaultTextPress,
      },
    },
  };
};

const useContentStyles = (): CSSObject => {
  // This is needed so force mounted content is not displayed when the tab is inactive
  return {
    '&[data-state="inactive"]': {
      display: 'none',
    },
  };
};

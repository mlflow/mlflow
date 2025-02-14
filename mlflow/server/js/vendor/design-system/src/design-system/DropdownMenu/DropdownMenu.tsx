import { type CSSObject, type Interpolation } from '@emotion/react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import type { ComponentPropsWithRef, ReactElement } from 'react';
import React, { createContext, forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef } from 'react';

import { handleKeyboardNavigation } from './utils';
import type { Theme } from '../../theme';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { CheckIcon, ChevronRightIcon } from '../Icon';
import { useModalContext } from '../Modal';
import { getNewChildren } from '../_shared_/Menu';
import type { AnalyticsEventProps, AnalyticsEventValueChangeNoPiiFlagProps } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { getDarkModePortalStyles, importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

interface DropdownContextProps {
  isOpen?: boolean;
  setIsOpen?: (isOpen: boolean) => void;
}

const DropdownContext = createContext<DropdownContextProps>({ isOpen: false, setIsOpen: (isOpen: boolean) => {} });
const useDropdownContext = () => React.useContext(DropdownContext);

export const Root = ({ children, ...props }: DropdownMenu.DropdownMenuProps): ReactElement => {
  const [isOpen, setIsOpen] = React.useState(Boolean(props.defaultOpen || props.open));

  const useExternalState = useRef(props.open !== undefined || props.onOpenChange !== undefined).current;

  useEffect(() => {
    if (useExternalState) {
      setIsOpen(Boolean(props.open));
    }
  }, [useExternalState, props.open]);

  const handleOpenChange = (isOpen: boolean) => {
    if (!useExternalState) {
      setIsOpen(isOpen);
    }

    // In case the consumer doesn't manage open state but wants to listen to the callback
    if (props.onOpenChange) {
      props.onOpenChange(isOpen);
    }
  };

  return (
    <DropdownMenu.Root
      {...props}
      {...(!useExternalState && {
        open: isOpen,
        onOpenChange: handleOpenChange,
      })}
    >
      <DropdownContext.Provider
        value={{
          isOpen: useExternalState ? props.open : isOpen,
          setIsOpen: useExternalState ? props.onOpenChange : handleOpenChange,
        }}
      >
        {children}
      </DropdownContext.Provider>
    </DropdownMenu.Root>
  );
};

export interface DropdownMenuProps extends DropdownMenu.MenuContentProps {
  minWidth?: number;
  forceCloseOnEscape?: boolean;
}

export interface DropdownMenuItemProps
  extends DropdownMenu.DropdownMenuItemProps,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
  danger?: boolean;
  disabledReason?: React.ReactNode;
}

export interface DropdownMenuSubTriggerProps extends DropdownMenu.DropdownMenuSubTriggerProps {
  disabledReason?: React.ReactNode;
}

export interface DropdownMenuCheckboxItemProps
  extends DropdownMenu.DropdownMenuCheckboxItemProps,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  disabledReason?: React.ReactNode;
}

export interface DropdownMenuRadioGroupProps
  extends DropdownMenu.DropdownMenuRadioGroupProps,
    AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {}

export interface DropdownMenuRadioItemProps extends DropdownMenu.DropdownMenuRadioItemProps {
  disabledReason?: React.ReactNode;
}

export const Content = forwardRef<HTMLDivElement, DropdownMenuProps>(function Content(
  { children, minWidth = 220, forceCloseOnEscape, onEscapeKeyDown, onKeyDown, ...props },
  ref,
): ReactElement {
  const { getPopupContainer } = useDesignSystemContext();
  const { theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();
  const { setIsOpen } = useDropdownContext();
  const { isInsideModal } = useModalContext();

  return (
    <DropdownMenu.Portal container={getPopupContainer && getPopupContainer()}>
      <DropdownMenu.Content
        {...addDebugOutlineIfEnabled()}
        ref={ref}
        loop={true}
        css={[contentStyles(theme, useNewShadows), { minWidth }]}
        sideOffset={4}
        align="start"
        onKeyDown={(e) => {
          // This is a workaround for Radix's DropdownMenu.Content not receiving Escape key events
          // when nested inside a modal. We need to stop propagation of the event so that the modal
          // doesn't close when the DropdownMenu should.
          if (e.key === 'Escape') {
            if (isInsideModal || forceCloseOnEscape) {
              e.stopPropagation();
              setIsOpen?.(false);
            }
            onEscapeKeyDown?.(e.nativeEvent);
          }
          if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            handleKeyboardNavigation(e);
          }
          onKeyDown?.(e);
        }}
        {...props}
        onWheel={(e) => {
          e.stopPropagation();
          props?.onWheel?.(e);
        }}
        onTouchMove={(e) => {
          e.stopPropagation();
          props?.onTouchMove?.(e);
        }}
      >
        {children}
      </DropdownMenu.Content>
    </DropdownMenu.Portal>
  );
});

export const SubContent = forwardRef<HTMLDivElement, Omit<DropdownMenuProps, 'forceCloseOnEscape'>>(function Content(
  { children, minWidth = 220, onKeyDown, ...props },
  ref,
): ReactElement {
  const { getPopupContainer } = useDesignSystemContext();
  const { theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();
  const [contentFitsInViewport, setContentFitsInViewport] = React.useState(true);
  const [dataSide, setDataSide] = React.useState<string | null>(null);
  const { isOpen } = useSubContext();

  const elemRef = useRef<HTMLDivElement>(null);
  useImperativeHandle(ref, () => elemRef.current as HTMLDivElement);

  const checkAvailableWidth = useCallback(() => {
    if (elemRef.current) {
      const elemStyle = getComputedStyle(elemRef.current as Element);
      const availableWidth = parseFloat(elemStyle.getPropertyValue('--radix-dropdown-menu-content-available-width'));
      const elemWidth = elemRef.current.offsetWidth;
      const openOnSide = elemRef.current.getAttribute('data-side');

      if (openOnSide === 'left' || openOnSide === 'right') {
        setDataSide(openOnSide);
      } else {
        setDataSide(null);
      }

      if (availableWidth < elemWidth) {
        setContentFitsInViewport(false);
      } else {
        setContentFitsInViewport(true);
      }
    }
  }, []);

  useEffect(() => {
    window.addEventListener('resize', checkAvailableWidth);
    checkAvailableWidth();

    return () => {
      window.removeEventListener('resize', checkAvailableWidth);
    };
  }, [checkAvailableWidth]);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => {
        checkAvailableWidth();
      }, 25);
    }
  }, [isOpen, checkAvailableWidth]);

  let transformCalc = `calc(var(--radix-dropdown-menu-content-available-width) + var(--radix-dropdown-menu-trigger-width) * -1)`;

  if (dataSide === 'left') {
    transformCalc = `calc(var(--radix-dropdown-menu-trigger-width) - var(--radix-dropdown-menu-content-available-width))`;
  }

  const responsiveCss = `
    transform-origin: var(--radix-dropdown-menu-content-transform-origin) !important;
    transform: translateX(${transformCalc}) !important;
`;

  return (
    <DropdownMenu.Portal container={getPopupContainer && getPopupContainer()}>
      <DropdownMenu.SubContent
        {...addDebugOutlineIfEnabled()}
        ref={elemRef}
        loop={true}
        css={[contentStyles(theme, useNewShadows), { minWidth }, contentFitsInViewport ? '' : responsiveCss]}
        sideOffset={-2}
        alignOffset={-5}
        onKeyDown={(e) => {
          if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.stopPropagation();
            handleKeyboardNavigation(e);
          }
          onKeyDown?.(e);
        }}
        {...props}
      >
        {children}
      </DropdownMenu.SubContent>
    </DropdownMenu.Portal>
  );
});

export const Trigger = forwardRef<HTMLButtonElement, DropdownMenu.DropdownMenuTriggerProps>(function Trigger(
  { children, ...props },
  ref,
): ReactElement {
  return (
    <DropdownMenu.Trigger {...addDebugOutlineIfEnabled()} ref={ref} {...props}>
      {children}
    </DropdownMenu.Trigger>
  );
});

export const Item = forwardRef<HTMLDivElement, DropdownMenuItemProps>(function Item(
  {
    children,
    disabledReason,
    danger,
    onClick,
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
    ...props
  },
  ref,
): ReactElement {
  const itemRef = useRef<HTMLDivElement>(null);
  useImperativeHandle<HTMLDivElement | null, HTMLDivElement | null>(ref, () => itemRef.current);

  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.DropdownMenuItem,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
  });

  return (
    <DropdownMenu.Item
      css={(theme) => [dropdownItemStyles, danger && dangerItemStyles(theme)]}
      ref={itemRef}
      onClick={(e) => {
        if (props.disabled) {
          e.preventDefault();
        } else {
          if (!props.asChild) {
            eventContext.onClick(e);
          }
          onClick?.(e);
        }
      }}
      onKeyDown={(e) => {
        if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault();
        }
        props.onKeyDown?.(e);
      }}
      {...props}
      {...eventContext.dataComponentProps}
    >
      {getNewChildren(children, props, disabledReason, itemRef)}
    </DropdownMenu.Item>
  );
});

export const Label = forwardRef<HTMLDivElement, DropdownMenu.DropdownMenuLabelProps>(function Label(
  { children, ...props },
  ref,
): ReactElement {
  return (
    <DropdownMenu.Label
      ref={ref}
      css={[
        dropdownItemStyles,
        (theme) => ({
          color: theme.colors.textSecondary,
          '&:hover': {
            cursor: 'default',
          },
        }),
      ]}
      {...props}
    >
      {children}
    </DropdownMenu.Label>
  );
});

export const Separator = forwardRef<HTMLDivElement, DropdownMenu.DropdownMenuSeparatorProps>(function Separator(
  { children, ...props },
  ref,
): ReactElement {
  return (
    <DropdownMenu.Separator ref={ref} css={dropdownSeparatorStyles} {...props}>
      {children}
    </DropdownMenu.Separator>
  );
});

export const SubTrigger = forwardRef<HTMLDivElement, DropdownMenuSubTriggerProps>(function TriggerItem(
  { children, disabledReason, ...props },
  ref,
): ReactElement {
  const subTriggerRef = useRef<HTMLDivElement>(null);
  useImperativeHandle<HTMLDivElement | null, HTMLDivElement | null>(ref, () => subTriggerRef.current);

  return (
    <DropdownMenu.SubTrigger
      ref={subTriggerRef}
      css={[
        dropdownItemStyles,
        (theme) => ({
          '&[data-state="open"]': {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
          },
        }),
      ]}
      onKeyDown={(e) => {
        if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault();
        }
        props.onKeyDown?.(e);
      }}
      {...props}
    >
      {getNewChildren(children, props, disabledReason, subTriggerRef)}
      <HintColumn
        css={(theme) => ({
          margin: CONSTANTS.subMenuIconMargin(theme),
          display: 'flex',
          alignSelf: 'stretch',
          alignItems: 'center',
        })}
      >
        <ChevronRightIcon css={(theme) => ({ fontSize: CONSTANTS.subMenuIconSize(theme) })} />
      </HintColumn>
    </DropdownMenu.SubTrigger>
  );
});

/**
 * Deprecated. Use `SubTrigger` instead.
 * @deprecated
 */
export const TriggerItem = SubTrigger;

export const CheckboxItem = forwardRef<HTMLDivElement, DropdownMenuCheckboxItemProps>(function CheckboxItem(
  {
    children,
    disabledReason,
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    onCheckedChange,
    ...props
  },
  ref,
): ReactElement {
  const checkboxItemRef = useRef<HTMLDivElement>(null);
  useImperativeHandle<HTMLDivElement | null, HTMLDivElement | null>(ref, () => checkboxItemRef.current);

  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.DropdownMenuCheckboxItem,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii: true,
  });

  const onCheckedChangeWrapper = useCallback(
    (checked: boolean) => {
      eventContext.onValueChange(checked);
      onCheckedChange?.(checked);
    },
    [eventContext, onCheckedChange],
  );

  return (
    <DropdownMenu.CheckboxItem
      ref={checkboxItemRef}
      css={(theme) => [dropdownItemStyles, checkboxItemStyles(theme)]}
      onCheckedChange={onCheckedChangeWrapper}
      onKeyDown={(e) => {
        if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault();
        }
        props.onKeyDown?.(e);
      }}
      {...props}
      {...eventContext.dataComponentProps}
    >
      {getNewChildren(children, props, disabledReason, checkboxItemRef)}
    </DropdownMenu.CheckboxItem>
  );
});

export const RadioGroup = forwardRef<HTMLDivElement, DropdownMenuRadioGroupProps>(function RadioGroup(
  {
    children,
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    onValueChange,
    valueHasNoPii,
    ...props
  },
  ref,
): ReactElement {
  const radioGroupItemRef = useRef<HTMLDivElement>(null);
  useImperativeHandle<HTMLDivElement | null, HTMLDivElement | null>(ref, () => radioGroupItemRef.current);

  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.DropdownMenuRadioGroup,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii,
  });

  const onValueChangeWrapper = useCallback(
    (value: string) => {
      eventContext.onValueChange(value);
      onValueChange?.(value);
    },
    [eventContext, onValueChange],
  );

  return (
    <DropdownMenu.RadioGroup
      ref={radioGroupItemRef}
      onValueChange={onValueChangeWrapper}
      {...props}
      {...eventContext.dataComponentProps}
    >
      {children}
    </DropdownMenu.RadioGroup>
  );
});

export const ItemIndicator = forwardRef<HTMLDivElement, DropdownMenu.DropdownMenuItemIndicatorProps>(
  function ItemIndicator({ children, ...props }, ref): ReactElement {
    return (
      <DropdownMenu.ItemIndicator
        ref={ref}
        css={(theme) => ({
          marginLeft: -(CONSTANTS.checkboxIconWidth(theme) + CONSTANTS.checkboxPaddingRight(theme)),
          position: 'absolute',
          fontSize: theme.general.iconFontSize,
        })}
        {...props}
      >
        {children ?? (
          <CheckIcon
            css={(theme) => ({
              color: theme.colors.textSecondary,
            })}
          />
        )}
      </DropdownMenu.ItemIndicator>
    );
  },
);

export const Arrow = forwardRef<SVGSVGElement, DropdownMenu.DropdownMenuArrowProps>(function Arrow(
  { children, ...props },
  ref,
): ReactElement {
  const { theme } = useDesignSystemTheme();
  return (
    <DropdownMenu.Arrow
      css={{
        fill: theme.colors.backgroundPrimary,
        stroke: theme.colors.borderDecorative,
        strokeDashoffset: -CONSTANTS.arrowBottomLength(),
        strokeDasharray: CONSTANTS.arrowBottomLength() + 2 * CONSTANTS.arrowSide(),
        strokeWidth: CONSTANTS.arrowStrokeWidth(),
        // TODO: This is a temporary fix for the alignment of the Arrow;
        // Radix has changed the implementation for v1.0.0 (uses floating-ui)
        // which has new behaviors for alignment that we don't want. Generally
        // we need to fix the arrow to always be aligned to the left of the menu (with
        // offset equal to border radius)
        position: 'relative',
        top: -1,
      }}
      ref={ref}
      width={12}
      height={6}
      {...props}
    >
      {children}
    </DropdownMenu.Arrow>
  );
});

export const RadioItem = forwardRef<HTMLDivElement, DropdownMenuRadioItemProps>(function RadioItem(
  { children, disabledReason, ...props },
  ref,
): ReactElement {
  const radioItemRef = useRef<HTMLDivElement>(null);
  useImperativeHandle<HTMLDivElement | null, HTMLDivElement | null>(ref, () => radioItemRef.current);

  return (
    <DropdownMenu.RadioItem
      ref={radioItemRef}
      css={(theme) => [dropdownItemStyles, checkboxItemStyles(theme)]}
      {...props}
    >
      {getNewChildren(children, props, disabledReason, radioItemRef)}
    </DropdownMenu.RadioItem>
  );
});

const SubContext = createContext({ isOpen: false });
const useSubContext = () => React.useContext(SubContext);

export const Sub = ({ children, onOpenChange, ...props }: DropdownMenu.DropdownMenuSubProps) => {
  const [isOpen, setIsOpen] = React.useState(props.defaultOpen ?? false);

  const handleOpenChange = (isOpen: boolean) => {
    onOpenChange?.(isOpen);
    setIsOpen(isOpen);
  };

  return (
    <DropdownMenu.Sub onOpenChange={handleOpenChange} {...props}>
      <SubContext.Provider value={{ isOpen }}>{children}</SubContext.Provider>
    </DropdownMenu.Sub>
  );
};

// UNWRAPPED RADIX-UI-COMPONENTS
export const Group = DropdownMenu.Group;

// EXTRA COMPONENTS
export const HintColumn = forwardRef<HTMLDivElement, ComponentPropsWithRef<'div'>>(function HintColumn(
  { children, ...props },
  ref,
): ReactElement {
  return (
    <div
      ref={ref}
      css={[
        metaTextStyles,
        {
          marginLeft: 'auto',
        },
      ]}
      {...props}
    >
      {children}
    </div>
  );
});

export const HintRow = forwardRef<HTMLDivElement, ComponentPropsWithRef<'div'>>(function HintRow(
  { children, ...props },
  ref,
): ReactElement {
  return (
    <div
      ref={ref}
      css={[
        metaTextStyles,
        {
          minWidth: '100%',
        },
      ]}
      {...props}
    >
      {children}
    </div>
  );
});

export const IconWrapper = forwardRef<HTMLDivElement, ComponentPropsWithRef<'div'>>(function IconWrapper(
  { children, ...props },
  ref,
): ReactElement {
  return (
    <div
      ref={ref}
      css={(theme) => ({
        fontSize: 16,
        color: theme.colors.textSecondary,
        paddingRight: theme.spacing.sm,
      })}
      {...props}
    >
      {children}
    </div>
  );
});

// CONSTANTS
const CONSTANTS = {
  itemPaddingVertical(theme: Theme) {
    // The number from the mocks is the midpoint between constants
    return 0.5 * theme.spacing.xs + 0.5 * theme.spacing.sm;
  },
  itemPaddingHorizontal(theme: Theme) {
    return theme.spacing.sm;
  },
  checkboxIconWidth(theme: Theme) {
    return theme.general.iconFontSize;
  },
  checkboxPaddingLeft(theme: Theme) {
    return theme.spacing.sm + theme.spacing.xs;
  },
  checkboxPaddingRight(theme: Theme) {
    return theme.spacing.sm;
  },
  subMenuIconMargin(theme: Theme) {
    // Negative margin so the icons can be larger without increasing the overall item height
    const iconMarginVertical = this.itemPaddingVertical(theme) / 2;
    const iconMarginRight = -this.itemPaddingVertical(theme) + theme.spacing.sm * 1.5;
    return `${-iconMarginVertical}px ${-iconMarginRight}px ${-iconMarginVertical}px auto`;
  },
  subMenuIconSize(theme: Theme) {
    return theme.spacing.lg;
  },
  arrowBottomLength() {
    // The built in arrow is a polygon: 0,0 30,0 15,10
    return 30;
  },
  arrowHeight() {
    return 10;
  },
  arrowSide() {
    return 2 * (this.arrowHeight() ** 2 * 2) ** 0.5;
  },
  arrowStrokeWidth() {
    // This is eyeballed b/c relative to the svg viewbox coordinate system
    return 2;
  },
};

export const dropdownContentStyles = (theme: Theme, useNewShadows: boolean): CSSObject => ({
  backgroundColor: theme.colors.backgroundPrimary,
  color: theme.colors.textPrimary,
  lineHeight: theme.typography.lineHeightBase,
  border: `1px solid ${theme.colors.borderDecorative}`,
  borderRadius: theme.legacyBorders.borderRadiusMd,
  padding: `${theme.spacing.xs}px 0`,
  boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowLow,
  userSelect: 'none',

  // Allow for scrolling within the dropdown when viewport is too small
  overflowY: 'auto',
  maxHeight: 'var(--radix-dropdown-menu-content-available-height)',
  ...getDarkModePortalStyles(theme, useNewShadows),
  // Ant Design uses 1000s for their zIndex space; this ensures Radix works with that, but
  // we'll likely need to be sure that all Radix components are using the same zIndex going forward.
  //
  // Additionally, there is an issue where macOS overlay scrollbars in Chrome and Safari (sometimes!)
  // overlap other elements with higher zIndex, because the scrollbars themselves have zIndex 9999,
  // so we have to use a higher value than that: https://github.com/databricks/universe/pull/232825
  zIndex: 10000,
  a: importantify({
    color: theme.colors.textPrimary,
    '&:hover, &:focus': {
      color: theme.colors.textPrimary,
      textDecoration: 'none',
    },
  }),
});

const contentStyles = (theme: Theme, useNewShadows: boolean): Interpolation<Theme> => ({
  ...dropdownContentStyles(theme, useNewShadows),
});

export const dropdownItemStyles: (theme: Theme) => Interpolation<Theme> = (theme) => ({
  padding: `${CONSTANTS.itemPaddingVertical(theme)}px ${CONSTANTS.itemPaddingHorizontal(theme)}px`,
  display: 'flex',
  flexWrap: 'wrap',
  alignItems: 'center',
  outline: 'unset',
  '&:hover': {
    cursor: 'pointer',
  },
  '&:focus': {
    backgroundColor: theme.colors.actionTertiaryBackgroundHover,

    '&:not(:hover)': {
      outline: `2px auto ${theme.colors.actionDefaultBorderFocus}`,
      outlineOffset: '-1px',
    },
  },
  '&[data-disabled]': {
    pointerEvents: 'none',
    color: theme.colors.actionDisabledText,
  },
});

const dangerItemStyles = (theme: Theme): Interpolation<Theme> => ({
  color: theme.colors.textValidationDanger,
  '&:hover, &:focus': {
    backgroundColor: theme.colors.actionDangerDefaultBackgroundHover,
  },
});

const checkboxItemStyles = (theme: Theme): Interpolation<Theme> => ({
  position: 'relative',
  paddingLeft:
    CONSTANTS.checkboxIconWidth(theme) + CONSTANTS.checkboxPaddingLeft(theme) + CONSTANTS.checkboxPaddingRight(theme),
});

const metaTextStyles = (theme: Theme): Interpolation<Theme> => ({
  color: theme.colors.textSecondary,
  fontSize: theme.typography.fontSizeSm,
  '[data-disabled] &': {
    color: theme.colors.actionDisabledText,
  },
});

export const dropdownSeparatorStyles = (theme: Theme) => ({
  height: 1,
  margin: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
  backgroundColor: theme.colors.borderDecorative,
});

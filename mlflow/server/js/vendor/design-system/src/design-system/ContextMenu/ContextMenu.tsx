import type { Theme } from '@emotion/react';
import { css } from '@emotion/react';
import type {
  ContextMenuCheckboxItemProps as RadixContextMenuCheckboxItemProps,
  ContextMenuContentProps as RadixContextMenuContentProps,
  ContextMenuItemProps as RadixContextMenuItemProps,
  ContextMenuLabelProps as RadixContextMenuLabelProps,
  ContextMenuRadioGroupProps as RadixContextMenuRadioGroupProps,
  ContextMenuRadioItemProps as RadixContextMenuRadioItemProps,
  ContextMenuSubContentProps as RadixContextMenuSubContentProps,
  ContextMenuSubTriggerProps as RadixContextMenuSubTriggerProps,
  ContextMenuProps as RadixContextMenuProps,
} from '@radix-ui/react-context-menu';
import {
  ContextMenuArrow,
  ContextMenuCheckboxItem,
  ContextMenuContent,
  ContextMenuGroup,
  ContextMenuItem,
  ContextMenuItemIndicator,
  ContextMenuLabel,
  ContextMenuPortal,
  ContextMenuRadioGroup,
  ContextMenuRadioItem,
  ContextMenuSeparator,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger,
  ContextMenu as RadixContextMenu,
} from '@radix-ui/react-context-menu';
import type { ReactElement } from 'react';
import React, { createContext, useCallback, useMemo, useRef } from 'react';

import {
  CheckIcon,
  ChevronRightIcon,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
  useDesignSystemSafexFlags,
  useDesignSystemTheme,
  useModalContext,
} from '..';
import { dropdownContentStyles, dropdownItemStyles, dropdownSeparatorStyles } from '../DropdownMenu/DropdownMenu';
import { handleKeyboardNavigation } from '../DropdownMenu/utils';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
import { getNewChildren } from '../_shared_/Menu';
import type { AnalyticsEventProps, AnalyticsEventValueChangeNoPiiFlagProps } from '../types';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export const Trigger = ContextMenuTrigger;
export const ItemIndicator = ContextMenuItemIndicator;
export const Group = ContextMenuGroup;
export const Arrow = ContextMenuArrow;
export const Sub = ContextMenuSub;

const ContextMenuProps = createContext({ isOpen: false, setIsOpen: (isOpen: boolean) => {} });
const useContextMenuProps = () => React.useContext(ContextMenuProps);

export const Root = ({ children, onOpenChange, ...props }: RadixContextMenuProps): ReactElement => {
  const [isOpen, setIsOpen] = React.useState(false);

  const handleChange = (isOpen: boolean) => {
    setIsOpen(isOpen);
    onOpenChange?.(isOpen);
  };

  return (
    <RadixContextMenu onOpenChange={handleChange} {...props}>
      <ContextMenuProps.Provider value={{ isOpen, setIsOpen }}>{children}</ContextMenuProps.Provider>
    </RadixContextMenu>
  );
};
export interface ContextMenuSubTriggerProps extends RadixContextMenuSubTriggerProps {
  disabledReason?: React.ReactNode;
  withChevron?: boolean;
}

export const SubTrigger = ({ children, disabledReason, withChevron, ...props }: ContextMenuSubTriggerProps) => {
  const { theme } = useDesignSystemTheme();
  const ref = useRef<HTMLDivElement>(null);

  return (
    <ContextMenuSubTrigger
      {...props}
      css={dropdownItemStyles(theme)}
      ref={ref}
      onKeyDown={(e) => {
        if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault();
        }
        props.onKeyDown?.(e);
      }}
    >
      {getNewChildren(children, props, disabledReason, ref)}
      {withChevron && (
        <ContextMenu.Hint>
          <ChevronRightIcon />
        </ContextMenu.Hint>
      )}
    </ContextMenuSubTrigger>
  );
};

export interface ContextMenuContentProps extends RadixContextMenuContentProps {
  minWidth?: number;
  forceCloseOnEscape?: boolean;
}

export const Content = ({
  children,
  minWidth,
  forceCloseOnEscape,
  onEscapeKeyDown,
  onKeyDown,
  ...childrenProps
}: ContextMenuContentProps) => {
  const { getPopupContainer } = useDesignSystemContext();
  const { theme } = useDesignSystemTheme();
  const { isInsideModal } = useModalContext();
  const { isOpen, setIsOpen } = useContextMenuProps();
  const { useNewShadows } = useDesignSystemSafexFlags();

  return (
    <ContextMenuPortal container={getPopupContainer && getPopupContainer()}>
      {isOpen && (
        <ContextMenuContent
          {...addDebugOutlineIfEnabled()}
          onKeyDown={(e) => {
            // This is a workaround for Radix's ContextMenu.Content not receiving Escape key events
            // when nested inside a modal. We need to stop propagation of the event so that the modal
            // doesn't close when the DropdownMenu should.
            if (e.key === 'Escape') {
              if (isInsideModal || forceCloseOnEscape) {
                e.stopPropagation();
                setIsOpen(false);
              }
              onEscapeKeyDown?.(e.nativeEvent);
            } else if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
              e.stopPropagation();
              handleKeyboardNavigation(e);
            }
            onKeyDown?.(e);
          }}
          {...childrenProps}
          css={[dropdownContentStyles(theme, useNewShadows), { minWidth }]}
        >
          {children}
        </ContextMenuContent>
      )}
    </ContextMenuPortal>
  );
};

export interface ContextMenuSubContentProps extends RadixContextMenuSubContentProps {
  minWidth?: number;
}

export const SubContent = ({ children, minWidth, ...childrenProps }: ContextMenuSubContentProps) => {
  const { getPopupContainer } = useDesignSystemContext();
  const { theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();

  return (
    <ContextMenuPortal container={getPopupContainer && getPopupContainer()}>
      <ContextMenuSubContent
        {...addDebugOutlineIfEnabled()}
        {...childrenProps}
        css={[dropdownContentStyles(theme, useNewShadows), { minWidth }]}
        onKeyDown={(e) => {
          if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.stopPropagation();
            handleKeyboardNavigation(e);
          }
          childrenProps.onKeyDown?.(e);
        }}
      >
        {children}
      </ContextMenuSubContent>
    </ContextMenuPortal>
  );
};

export interface ContextMenuItemProps
  extends RadixContextMenuItemProps,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
  disabledReason?: React.ReactNode;
}

export const Item = ({
  children,
  disabledReason,
  onClick,
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
  asChild,
  ...props
}: ContextMenuItemProps) => {
  const { theme } = useDesignSystemTheme();
  const ref = useRef<HTMLDivElement>(null);
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.ContextMenuItem,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
  });

  const onClickWrapper = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!asChild) {
        eventContext.onClick(e);
      }
      onClick?.(e);
    },
    [asChild, eventContext, onClick],
  );

  return (
    <ContextMenuItem
      {...props}
      asChild={asChild}
      onClick={onClickWrapper}
      css={dropdownItemStyles(theme)}
      ref={ref}
      onKeyDown={(e) => {
        if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault();
        }
        props.onKeyDown?.(e);
      }}
      {...eventContext.dataComponentProps}
    >
      {getNewChildren(children, props, disabledReason, ref)}
    </ContextMenuItem>
  );
};

export interface ContextMenuCheckboxItemProps
  extends RadixContextMenuCheckboxItemProps,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  disabledReason?: React.ReactNode;
}

export const CheckboxItem = ({
  children,
  disabledReason,
  onCheckedChange,
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
  ...props
}: ContextMenuCheckboxItemProps) => {
  const { theme } = useDesignSystemTheme();
  const ref = useRef<HTMLDivElement>(null);
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.ContextMenuCheckboxItem,
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
    <ContextMenuCheckboxItem
      {...props}
      onCheckedChange={onCheckedChangeWrapper}
      css={dropdownItemStyles(theme)}
      ref={ref}
      onKeyDown={(e) => {
        if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault();
        }
        props.onKeyDown?.(e);
      }}
      {...eventContext.dataComponentProps}
    >
      <ContextMenuItemIndicator css={itemIndicatorStyles(theme)}>
        <CheckIcon />
      </ContextMenuItemIndicator>
      {!props.checked && <div style={{ width: theme.general.iconFontSize + theme.spacing.xs }} />}
      {getNewChildren(children, props, disabledReason, ref)}
    </ContextMenuCheckboxItem>
  );
};

export interface ContextMenuRadioGroupProps
  extends RadixContextMenuRadioGroupProps,
    AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {}

export const RadioGroup = ({
  onValueChange,
  componentId,
  analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
  valueHasNoPii,
  ...props
}: ContextMenuRadioGroupProps) => {
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.ContextMenuRadioGroup,
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

  return <ContextMenuRadioGroup {...props} onValueChange={onValueChangeWrapper} {...eventContext.dataComponentProps} />;
};

export interface ContextMenuRadioItemProps extends RadixContextMenuRadioItemProps {
  disabledReason?: React.ReactNode;
}

export const RadioItem = ({ children, disabledReason, ...props }: ContextMenuRadioItemProps) => {
  const { theme } = useDesignSystemTheme();
  const ref = useRef<HTMLDivElement>(null);

  return (
    <ContextMenuRadioItem
      {...props}
      css={[
        dropdownItemStyles(theme),
        {
          '&[data-state="unchecked"]': {
            paddingLeft: theme.general.iconFontSize + theme.spacing.xs + theme.spacing.sm,
          },
        },
      ]}
      ref={ref}
    >
      <ContextMenuItemIndicator css={itemIndicatorStyles(theme)}>
        <CheckIcon />
      </ContextMenuItemIndicator>
      {getNewChildren(children, props, disabledReason, ref)}
    </ContextMenuRadioItem>
  );
};

export interface ContextMenuLabelProps extends RadixContextMenuLabelProps {}

export const Label = ({ children, ...props }: ContextMenuLabelProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <ContextMenuLabel
      {...props}
      css={{ color: theme.colors.textSecondary, padding: `${theme.spacing.sm - 2}px ${theme.spacing.sm}px` }}
    >
      {children}
    </ContextMenuLabel>
  );
};

export const Hint = ({ children }: { children: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();

  return <span css={{ display: 'inline-flex', marginLeft: 'auto', paddingLeft: theme.spacing.sm }}>{children}</span>;
};

export const Separator = () => {
  const { theme } = useDesignSystemTheme();
  return <ContextMenuSeparator css={dropdownSeparatorStyles(theme)} />;
};

export const itemIndicatorStyles = (theme: Theme) => css({ display: 'inline-flex', paddingRight: theme.spacing.xs });

export const ContextMenu = {
  Root,
  Trigger,
  Label,
  Item,
  Group,
  RadioGroup,
  CheckboxItem,
  RadioItem,
  Arrow,
  Separator,
  Sub,
  SubTrigger,
  SubContent,
  Content,
  Hint,
};

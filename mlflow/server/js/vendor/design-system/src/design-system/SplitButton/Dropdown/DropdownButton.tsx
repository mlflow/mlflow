import EllipsisOutlined from '@ant-design/icons';
import type { DropdownMenuProps } from '@radix-ui/react-dropdown-menu';
import { Button as AntDButton, Dropdown as AntDDropdown } from 'antd';
import classNames from 'classnames';
import * as React from 'react';

import { Button } from '../../Button';
import type { DesignSystemEventProviderAnalyticsEventTypes } from '../../DesignSystemEventProvider/DesignSystemEventProvider';
import { DropdownMenu } from '../../DropdownMenu';
import { useDesignSystemContext } from '../../Hooks/useDesignSystemContext';
import { ChevronDownIcon } from '../../Icon';
import type { AnalyticsEventProps } from '../../types';

// All code below is from ant-design source code (https://github.com/ant-design/ant-design/blob/master/components/dropdown/dropdown-button.tsx)
// with a few small modifications and addition of new prop `menuButtonLabel` which sets the `aria-label` attribute on the second button.

type SizeType = 'small' | 'middle' | undefined;

interface ButtonGroupProps {
  size?: SizeType;
  style?: React.CSSProperties;
  className?: string;
  prefixCls?: string;
  children?: React.ReactNode;
}

type Placement = 'topLeft' | 'topCenter' | 'topRight' | 'bottomLeft' | 'bottomCenter' | 'bottomRight';

type OverlayFunc = () => React.ReactElement;

type Align = {
  points?: [string, string];
  offset?: [number, number];
  targetOffset?: [number, number];
  overflow?: {
    adjustX?: boolean;
    adjustY?: boolean;
  };
  useCssRight?: boolean;
  useCssBottom?: boolean;
  useCssTransform?: boolean;
};
interface DropdownProps {
  autoFocus?: boolean;
  arrow?: boolean;
  trigger?: ('click' | 'hover' | 'contextMenu')[];
  overlay?: React.ReactElement | OverlayFunc;
  onOpenChange?: (open: boolean) => void;
  open?: boolean;
  disabled?: boolean;
  destroyPopupOnHide?: boolean;
  align?: Align;
  getPopupContainer?: (triggerNode: HTMLElement) => HTMLElement;
  prefixCls?: string;
  className?: string;
  transitionName?: string;
  placement?: Placement;
  overlayClassName?: string;
  overlayStyle?: React.CSSProperties;
  forceRender?: boolean;
  mouseEnterDelay?: number;
  mouseLeaveDelay?: number;
  openClassName?: string;
  children?: React.ReactNode;
  leftButtonIcon?: React.ReactNode;
}

export interface DropdownButtonProps
  extends ButtonGroupProps,
    DropdownProps,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnClick> {
  type?: 'primary';
  htmlType?: 'submit' | 'reset' | 'button';
  danger?: boolean;
  disabled?: boolean;
  loading?: boolean | { delay?: number };
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
  icon?: React.ReactNode;
  href?: string;
  children?: React.ReactNode;
  title?: string;
  buttonsRender?: (buttons: [React.ReactNode, React.ReactNode]) => [React.ReactNode, React.ReactNode];
  menuButtonLabel?: string;
  menu?: React.ReactElement;
  dropdownMenuRootProps?: DropdownMenuProps;
  'aria-label'?: string;
}

const ButtonGroup = AntDButton.Group;

export const DropdownButton: React.FC<DropdownButtonProps> = (props) => {
  const { getPopupContainer: getContextPopupContainer, getPrefixCls } = useDesignSystemContext();

  const {
    type,
    danger,
    disabled,
    loading,
    onClick,
    htmlType,
    children,
    className,
    overlay,
    trigger,
    align,
    open,
    onOpenChange,
    placement,
    getPopupContainer,
    href,
    icon = <EllipsisOutlined />,
    title,
    buttonsRender = (buttons: React.ReactNode[]) => buttons,
    mouseEnterDelay,
    mouseLeaveDelay,
    overlayClassName,
    overlayStyle,
    destroyPopupOnHide,
    menuButtonLabel = 'Open dropdown',
    menu,
    leftButtonIcon,
    dropdownMenuRootProps,
    'aria-label': ariaLabel,
    componentId,
    analyticsEvents,
    ...restProps
  } = props;

  const prefixCls = getPrefixCls('dropdown-button');
  const dropdownProps = {
    align,
    overlay,
    disabled,
    trigger: disabled ? [] : trigger,
    onOpenChange,
    getPopupContainer: getPopupContainer || getContextPopupContainer,
    mouseEnterDelay,
    mouseLeaveDelay,
    overlayClassName,
    overlayStyle,
    destroyPopupOnHide,
  } as DropdownProps;

  if ('open' in props) {
    dropdownProps.open = open;
  }

  if ('placement' in props) {
    dropdownProps.placement = placement;
  } else {
    dropdownProps.placement = 'bottomRight';
  }

  const leftButton = (
    <Button
      componentId={
        componentId
          ? `${componentId}.primary_button`
          : 'codegen_design-system_src_design-system_splitbutton_dropdown_dropdownbutton.tsx_148'
      }
      type={type}
      danger={danger}
      disabled={disabled}
      loading={loading}
      onClick={onClick}
      htmlType={htmlType}
      href={href}
      title={title}
      icon={children && leftButtonIcon ? leftButtonIcon : undefined}
      aria-label={ariaLabel}
    >
      {leftButtonIcon && !children ? leftButtonIcon : undefined}
      {children}
    </Button>
  );

  const rightButton = (
    <Button
      componentId={
        componentId
          ? `${componentId}.dropdown_button`
          : 'codegen_design-system_src_design-system_splitbutton_dropdown_dropdownbutton.tsx_166'
      }
      type={type}
      danger={danger}
      disabled={disabled}
      aria-label={menuButtonLabel}
    >
      {icon ? icon : <ChevronDownIcon />}
    </Button>
  );

  const [leftButtonToRender, rightButtonToRender] = buttonsRender([leftButton, rightButton]);

  return (
    <ButtonGroup {...restProps} className={classNames(prefixCls, className)}>
      {leftButtonToRender}
      {overlay !== undefined ? (
        <AntDDropdown {...dropdownProps} overlay={overlay}>
          {rightButtonToRender}
        </AntDDropdown>
      ) : (
        <DropdownMenu.Root {...dropdownMenuRootProps}>
          <DropdownMenu.Trigger disabled={disabled} asChild>
            {rightButtonToRender}
          </DropdownMenu.Trigger>
          {menu && React.cloneElement(menu, { align: menu.props.align || 'end' })}
        </DropdownMenu.Root>
      )}
    </ButtonGroup>
  );
};

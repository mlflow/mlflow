import type { TooltipProps as AntDTooltipProps } from 'antd';
import { Tooltip as AntDTooltip } from 'antd';
import { isNil } from 'lodash';
import type { HTMLAttributes } from 'react';
import React, { useRef } from 'react';

import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { HTMLDataAttributes } from '../types';
import { getDarkModePortalStyles, useDesignSystemSafexFlags } from '../utils';
import { useUniqueId } from '../utils/useUniqueId';

/**
 * `LegacyTooltip` is deprecated in favor of the new `Tooltip` component
 * @deprecated
 */
export interface LegacyTooltipProps extends HTMLDataAttributes {
  /**
   * The element with which the tooltip should be associated.
   */
  children: React.ReactNode;
  /**
   * Plain text that will appear within the tooltip. Links and formatted content should not be use. However, we allow
   * any React element to be passed in here, rather than just a string, to allow for i18n formatting components.
   */
  title: React.ReactNode;
  /**
   * Value that determines where the tooltip will be positioned relative to the associated element.
   */
  placement?: AntDTooltipProps['placement'];
  /**
   * Escape hatch to allow passing props directly to the underlying Ant `Tooltip` component.
   */
  dangerouslySetAntdProps?: Partial<AntDTooltipProps>;
  /**
   * ID used to refer to this element in unit tests.
   */
  dataTestId?: string;
  /**
   * Prop that forces the tooltip's arrow to be centered on the target element
   */
  arrowPointAtCenter?: boolean;
  /**
   * Toggle wrapper live region off
   */
  silenceScreenReader?: boolean;
  /**
   * Toggles screen readers reading the tooltip content as the label for the hovered/focused element
   */
  useAsLabel?: boolean;
}

/**
 * `LegacyTooltip` is deprecated in favor of the new `Tooltip` component
 * @deprecated
 */
export const LegacyTooltip: React.FC<LegacyTooltipProps> = ({
  children,
  title,
  placement = 'top',
  dataTestId,
  dangerouslySetAntdProps,
  silenceScreenReader = false,
  useAsLabel = false,
  ...props
}) => {
  const { theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();
  const tooltipRef = useRef<any>(null);
  const duboisId = useUniqueId('dubois-tooltip-component-');
  const id = dangerouslySetAntdProps?.id ? dangerouslySetAntdProps?.id : duboisId;

  if (!title) {
    return <React.Fragment>{children}</React.Fragment>;
  }

  const titleProps: HTMLAttributes<HTMLSpanElement> & { 'data-testid'?: string } = silenceScreenReader
    ? {}
    : ({ 'aria-live': 'polite', 'aria-relevant': 'additions' } as const);
  if (dataTestId) {
    titleProps['data-testid'] = dataTestId;
  }
  const liveTitle =
    title && React.isValidElement(title) ? React.cloneElement(title, titleProps) : <span {...titleProps}>{title}</span>;
  const ariaProps = { 'aria-hidden': false };

  const addAriaProps = (e: React.MouseEvent | React.FocusEvent) => {
    if (
      !tooltipRef.current ||
      e.currentTarget.hasAttribute('aria-describedby') ||
      e.currentTarget.hasAttribute('aria-labelledby')
    ) {
      return;
    }

    if (id) {
      e.currentTarget.setAttribute('aria-live', 'polite');
      if (useAsLabel) {
        e.currentTarget.setAttribute('aria-labelledby', id);
      } else {
        e.currentTarget.setAttribute('aria-describedby', id);
      }
    }
  };

  const removeAriaProps = (e: React.MouseEvent) => {
    if (
      !tooltipRef ||
      (!e.currentTarget.hasAttribute('aria-describedby') && !e.currentTarget.hasAttribute('aria-labelledby'))
    ) {
      return;
    }

    if (useAsLabel) {
      e.currentTarget.removeAttribute('aria-labelledby');
    } else {
      e.currentTarget.removeAttribute('aria-describedby');
    }
    e.currentTarget.removeAttribute('aria-live');
  };

  const interactionProps = {
    onMouseEnter: (e: any) => {
      addAriaProps(e);
    },
    onMouseLeave: (e: any) => {
      removeAriaProps(e);
    },
    onFocus: (e: any) => {
      addAriaProps(e);
    },
    onBlur: (e: any) => {
      removeAriaProps(e);
    },
  };

  const childWithProps = React.isValidElement(children) ? (
    React.cloneElement<any>(children, { ...ariaProps, ...interactionProps, ...children.props })
  ) : isNil(children) ? (
    children
  ) : (
    <span {...ariaProps} {...interactionProps}>
      {children}
    </span>
  );

  const { overlayInnerStyle, overlayStyle, ...delegatedDangerouslySetAntdProps } = dangerouslySetAntdProps || {};
  return (
    <DesignSystemAntDConfigProvider>
      <AntDTooltip
        id={id}
        ref={tooltipRef}
        title={liveTitle}
        placement={placement}
        // Always trigger on hover and focus
        trigger={['hover', 'focus']}
        overlayInnerStyle={{
          backgroundColor: '#2F3941',
          lineHeight: '22px',
          padding: '4px 8px',
          boxShadow: theme.general.shadowLow,
          ...overlayInnerStyle,
          ...getDarkModePortalStyles(theme, useNewShadows),
        }}
        overlayStyle={{
          zIndex: theme.options.zIndexBase + 70,
          ...overlayStyle,
        }}
        css={{
          ...getAnimationCss(theme.options.enableAnimation),
        }}
        {...delegatedDangerouslySetAntdProps}
        {...props}
      >
        {childWithProps}
      </AntDTooltip>
    </DesignSystemAntDConfigProvider>
  );
};

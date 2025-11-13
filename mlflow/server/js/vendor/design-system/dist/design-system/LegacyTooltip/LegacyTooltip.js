import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { Tooltip as AntDTooltip } from 'antd';
import { isNil } from 'lodash';
import React, { useRef } from 'react';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { getDarkModePortalStyles, useDesignSystemSafexFlags } from '../utils';
import { useUniqueId } from '../utils/useUniqueId';
/**
 * `LegacyTooltip` is deprecated in favor of the new `Tooltip` component
 * @deprecated
 */
export const LegacyTooltip = ({ children, title, placement = 'top', dataTestId, dangerouslySetAntdProps, silenceScreenReader = false, useAsLabel = false, ...props }) => {
    const { theme } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderColors } = useDesignSystemSafexFlags();
    const tooltipRef = useRef(null);
    const duboisId = useUniqueId('dubois-tooltip-component-');
    const id = dangerouslySetAntdProps?.id ? dangerouslySetAntdProps?.id : duboisId;
    if (!title) {
        return _jsx(React.Fragment, { children: children });
    }
    const titleProps = silenceScreenReader
        ? {}
        : { 'aria-live': 'polite', 'aria-relevant': 'additions' };
    if (dataTestId) {
        titleProps['data-testid'] = dataTestId;
    }
    const liveTitle = title && React.isValidElement(title) ? React.cloneElement(title, titleProps) : _jsx("span", { ...titleProps, children: title });
    const ariaProps = { 'aria-hidden': false };
    const addAriaProps = (e) => {
        if (!tooltipRef.current ||
            e.currentTarget.hasAttribute('aria-describedby') ||
            e.currentTarget.hasAttribute('aria-labelledby')) {
            return;
        }
        if (id) {
            e.currentTarget.setAttribute('aria-live', 'polite');
            if (useAsLabel) {
                e.currentTarget.setAttribute('aria-labelledby', id);
            }
            else {
                e.currentTarget.setAttribute('aria-describedby', id);
            }
        }
    };
    const removeAriaProps = (e) => {
        if (!tooltipRef ||
            (!e.currentTarget.hasAttribute('aria-describedby') && !e.currentTarget.hasAttribute('aria-labelledby'))) {
            return;
        }
        if (useAsLabel) {
            e.currentTarget.removeAttribute('aria-labelledby');
        }
        else {
            e.currentTarget.removeAttribute('aria-describedby');
        }
        e.currentTarget.removeAttribute('aria-live');
    };
    const interactionProps = {
        onMouseEnter: (e) => {
            addAriaProps(e);
        },
        onMouseLeave: (e) => {
            removeAriaProps(e);
        },
        onFocus: (e) => {
            addAriaProps(e);
        },
        onBlur: (e) => {
            removeAriaProps(e);
        },
    };
    const childWithProps = React.isValidElement(children) ? (React.cloneElement(children, { ...ariaProps, ...interactionProps, ...children.props })) : isNil(children) ? (children) : (_jsx("span", { ...ariaProps, ...interactionProps, children: children }));
    const { overlayInnerStyle, overlayStyle, ...delegatedDangerouslySetAntdProps } = dangerouslySetAntdProps || {};
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDTooltip, { id: id, ref: tooltipRef, title: liveTitle, placement: placement, 
            // Always trigger on hover and focus
            trigger: ['hover', 'focus'], overlayInnerStyle: {
                backgroundColor: '#2F3941',
                lineHeight: '22px',
                padding: '4px 8px',
                boxShadow: theme.general.shadowLow,
                ...overlayInnerStyle,
                ...getDarkModePortalStyles(theme, useNewShadows, useNewBorderColors),
            }, overlayStyle: {
                zIndex: theme.options.zIndexBase + 70,
                ...overlayStyle,
            }, css: {
                ...getAnimationCss(theme.options.enableAnimation),
            }, ...delegatedDangerouslySetAntdProps, ...props, children: childWithProps }) }));
};
//# sourceMappingURL=LegacyTooltip.js.map
import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { useMergeRefs } from '@floating-ui/react';
import { Typography as AntDTypography } from 'antd';
import { forwardRef, useCallback, useMemo } from 'react';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { NewWindowIcon } from '../Icon';
import { useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
const getLinkStyles = (theme, clsPrefix) => {
    const classTypography = `.${clsPrefix}-typography`;
    const styles = {
        [`&${classTypography}, &${classTypography}:focus`]: {
            color: theme.colors.actionTertiaryTextDefault,
        },
        [`&${classTypography}:hover, &${classTypography}:hover .anticon`]: {
            color: theme.colors.actionTertiaryTextHover,
            textDecoration: 'underline',
        },
        [`&${classTypography}:active, &${classTypography}:active .anticon`]: {
            color: theme.colors.actionTertiaryTextPress,
            textDecoration: 'underline',
        },
        [`&${classTypography}:focus-visible`]: {
            textDecoration: 'underline',
        },
        '.anticon': {
            fontSize: 12,
            verticalAlign: 'baseline',
        },
        // manually update color for link within a LegacyTooltip since tooltip always has an inverted background color for light/dark mode
        // this is required for accessibility compliance
        [`.${clsPrefix}-tooltip-inner a&${classTypography}`]: {
            [`&, :focus`]: {
                color: theme.colors.blue500,
                '.anticon': { color: theme.colors.blue500 },
            },
            ':active': {
                color: theme.colors.blue500,
                '.anticon': { color: theme.colors.blue500 },
            },
            ':hover': {
                color: theme.colors.blue400,
                '.anticon': { color: theme.colors.blue400 },
            },
        },
    };
    return css(styles);
};
const getEllipsisNewTabLinkStyles = () => {
    const styles = {
        paddingRight: 'calc(2px + 1em)', // 1em for icon
        position: 'relative',
    };
    return css(styles);
};
const getIconStyles = (theme) => {
    const styles = {
        marginLeft: 4,
        color: theme.colors.actionTertiaryTextDefault,
        position: 'relative',
        top: '1px',
    };
    return css(styles);
};
const getEllipsisIconStyles = (useNewIcons) => {
    const styles = {
        position: 'absolute',
        right: 0,
        bottom: 0,
        top: 0,
        display: 'flex',
        alignItems: 'center',
        ...(useNewIcons && {
            fontSize: 12,
        }),
    };
    return css(styles);
};
export const Link = forwardRef(function Link({ dangerouslySetAntdProps, componentId, analyticsEvents, onClick, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.typographyLink', false);
    const { children, openInNewTab, ...restProps } = props;
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [DesignSystemEventProviderAnalyticsEventTypes.OnClick, DesignSystemEventProviderAnalyticsEventTypes.OnView]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnClick]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.TypographyLink,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        shouldStartInteraction: false,
    });
    const { elementRef: linkRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
    });
    const mergedRef = useMergeRefs([ref, linkRef]);
    const onClickHandler = useCallback((e) => {
        eventContext.onClick(e);
        onClick?.(e);
    }, [eventContext, onClick]);
    const newTabProps = {
        rel: 'noopener noreferrer',
        target: '_blank',
    };
    const linkProps = openInNewTab ? { ...restProps, ...newTabProps } : { ...restProps };
    const linkStyles = props.ellipsis && openInNewTab
        ? [getLinkStyles(theme, classNamePrefix), getEllipsisNewTabLinkStyles()]
        : getLinkStyles(theme, classNamePrefix);
    const iconStyles = props.ellipsis ? [getIconStyles(theme), getEllipsisIconStyles()] : getIconStyles(theme);
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsxs(AntDTypography.Link, { ...addDebugOutlineIfEnabled(), "aria-disabled": linkProps.disabled, css: linkStyles, ref: mergedRef, onClick: onClickHandler, ...linkProps, ...dangerouslySetAntdProps, ...eventContext.dataComponentProps, children: [children, openInNewTab ? _jsx(NewWindowIcon, { css: iconStyles, ...newTabProps }) : null] }) }));
});
//# sourceMappingURL=Link.js.map
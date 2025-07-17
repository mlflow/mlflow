import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Alert as AntDAlert } from 'antd';
import cx from 'classnames';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '../Button';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, DesignSystemEventProviderComponentSubTypeMap, } from '../DesignSystemEventProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { CloseIcon, CloseSmallIcon } from '../Icon';
import { SeverityIcon } from '../Icon/iconMap';
import { Modal } from '../Modal';
import { Typography } from '../Typography';
import { useDesignSystemSafexFlags, useNotifyOnFirstView } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
export const Alert = ({ componentId, analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView], dangerouslySetAntdProps, closable = true, closeIconLabel = 'Close alert', onClose, actions, showMoreContent, showMoreText = 'Show details', showMoreModalTitle = 'Details', forceVerticalActionsPlacement = false, size = 'large', ...props }) => {
    const { theme, getPrefixedClassName } = useDesignSystemTheme();
    const { useNewBorderRadii, useNewLargeAlertSizing, useNewBorderColors } = useDesignSystemSafexFlags();
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Alert,
        componentId,
        componentSubType: DesignSystemEventProviderComponentSubTypeMap[props.type],
        analyticsEvents: memoizedAnalyticsEvents,
    });
    const closeButtonEventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentId: componentId ? `${componentId}.close` : 'codegen_design_system_src_design_system_alert_alert.tsx_50',
        analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
    });
    const { elementRef } = useNotifyOnFirstView({ onView: eventContext.onView });
    const clsPrefix = getPrefixedClassName('alert');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const mergedProps = {
        ...props,
        type: props.type || 'error',
        showIcon: true,
        closable,
    };
    const closeIconRef = useRef(null);
    useEffect(() => {
        if (closeIconRef.current) {
            closeIconRef.current.removeAttribute('aria-label');
            closeIconRef.current.closest('button')?.setAttribute('aria-label', closeIconLabel);
        }
    }, [mergedProps.closable, closeIconLabel, closeIconRef]);
    const onCloseWrapper = (e) => {
        closeButtonEventContext.onClick(e);
        onClose?.(e);
    };
    const memoizedActions = useMemo(() => {
        if (!actions?.length)
            return null;
        return actions.map((action, index) => _jsx(Button, { size: "small", ...action }, index));
    }, [actions]);
    // Determine action placement based on number of actions
    const actionsPlacement = actions?.length === 1 && !forceVerticalActionsPlacement ? 'horizontal' : 'vertical';
    const description = (_jsx("div", { css: {
            ...(useNewLargeAlertSizing
                ? {
                    display: 'flex',
                    flexDirection: 'row',
                    gap: theme.spacing.sm,
                    justifyContent: 'space-between',
                }
                : {}),
        }, children: _jsxs("div", { css: {
                ...(actionsPlacement === 'horizontal' && {
                    display: 'flex',
                    flexDirection: 'row',
                    justifyContent: 'space-between',
                    gap: theme.spacing.sm,
                    alignItems: 'flex-start',
                }),
            }, children: [_jsxs("div", { children: [props.description, ' ', showMoreContent && (_jsx(Typography.Link, { href: "#", componentId: componentId ? `${componentId}.show_more` : 'alert.show_more', onClick: (e) => {
                                e.preventDefault();
                                setIsModalOpen(true);
                            }, children: showMoreText }))] }), !useNewLargeAlertSizing && memoizedActions && actionsPlacement === 'horizontal' && (_jsx("div", { css: {
                        display: 'flex',
                        gap: theme.spacing.sm,
                        ...(useNewLargeAlertSizing
                            ? {
                                alignSelf: 'center',
                            }
                            : {}),
                    }, children: memoizedActions }))] }) }));
    // Create a separate section for vertical actions if needed
    const verticalActions = memoizedActions && actionsPlacement === 'vertical' && (_jsx("div", { css: {
            display: 'flex',
            gap: theme.spacing.sm,
            marginTop: useNewLargeAlertSizing ? theme.spacing.xs : theme.spacing.sm,
            marginBottom: theme.spacing.xs,
        }, children: memoizedActions }));
    // Create the final description that includes both the description and vertical actions if present
    const finalDescription = (_jsxs(_Fragment, { children: [description, verticalActions] }));
    return (_jsxs(DesignSystemAntDConfigProvider, { children: [_jsx(AntDAlert, { ...addDebugOutlineIfEnabled(), ...mergedProps, onClose: onCloseWrapper, className: cx(mergedProps.className), css: getAlertEmotionStyles(clsPrefix, theme, mergedProps, size, Boolean(actions?.length), forceVerticalActionsPlacement, useNewBorderRadii, useNewLargeAlertSizing, useNewBorderColors), icon: _jsx(SeverityIcon, { severity: mergedProps.type, ref: elementRef }), 
                // Antd calls this prop `closeText` but we can use it to set any React element to replace the close icon.
                closeText: mergedProps.closable &&
                    (useNewLargeAlertSizing ? (_jsx("div", { css: {
                            marginTop: actionsPlacement === 'horizontal' ? theme.spacing.xs : 0,
                        }, children: _jsx(CloseSmallIcon, { ref: closeIconRef, "aria-label": closeIconLabel, css: { alignSelf: 'center' } }) })) : (_jsx(CloseIcon, { ref: closeIconRef, "aria-label": closeIconLabel, css: { fontSize: theme.general.iconSize } }))), 
                // Always set a description for consistent styling (e.g. icon size)
                description: finalDescription, action: mergedProps.action
                    ? mergedProps.action
                    : useNewLargeAlertSizing && actionsPlacement === 'horizontal'
                        ? memoizedActions
                        : undefined, ...dangerouslySetAntdProps, ...eventContext.dataComponentProps }), showMoreContent && (_jsx(Modal, { title: showMoreModalTitle, visible: isModalOpen, onCancel: () => setIsModalOpen(false), componentId: componentId ? `${componentId}.show_more_modal` : 'alert.show_more_modal', footer: null, size: "wide", children: showMoreContent }))] }));
};
const getAlertEmotionStyles = (clsPrefix, theme, props, size, hasActions, isVertical, useNewBorderRadii, useNewLargeAlertSizing, useNewBorderColors) => {
    const isSmall = size === 'small';
    const classContent = `.${clsPrefix}-content`;
    const classCloseIcon = `.${clsPrefix}-close-icon`;
    const classCloseButton = `.${clsPrefix}-close-button`;
    const classCloseText = `.${clsPrefix}-close-text`;
    const classDescription = `.${clsPrefix}-description`;
    const classMessage = `.${clsPrefix}-message`;
    const classWithDescription = `.${clsPrefix}-with-description`;
    const classWithIcon = `.${clsPrefix}-icon`;
    const classAction = `.${clsPrefix}-action`;
    const ALERT_ICON_HEIGHT = 16;
    const ALERT_ICON_FONT_SIZE = 16;
    const BORDER_SIZE = theme.general.borderWidth;
    const LARGE_SIZE_PADDING = theme.spacing.xs * 3;
    const SMALL_SIZE_PADDING = theme.spacing.sm;
    const styles = {
        // General
        padding: theme.spacing.sm,
        ...(useNewLargeAlertSizing && {
            padding: `${LARGE_SIZE_PADDING - BORDER_SIZE}px ${LARGE_SIZE_PADDING}px`,
            boxSizing: 'border-box',
            ...(isSmall && {
                padding: `${theme.spacing.xs + BORDER_SIZE}px ${SMALL_SIZE_PADDING}px`,
            }),
            [classAction]: {
                alignSelf: 'center',
            },
        }),
        ...(useNewBorderRadii && {
            borderRadius: theme.borders.borderRadiusSm,
        }),
        ...(useNewBorderColors && {
            borderColor: theme.colors.border,
        }),
        [`${classMessage}, &${classWithDescription} ${classMessage}`]: {
            fontSize: theme.typography.fontSizeBase,
            fontWeight: theme.typography.typographyBoldFontWeight,
            lineHeight: theme.typography.lineHeightBase,
            ...(useNewLargeAlertSizing && {
                marginBottom: 0,
            }),
        },
        [`${classDescription}`]: {
            lineHeight: theme.typography.lineHeightBase,
        },
        // Icons
        [classCloseButton]: {
            fontSize: ALERT_ICON_FONT_SIZE,
            marginRight: 12,
        },
        [classCloseIcon]: {
            '&:focus-visible': {
                outlineStyle: 'auto',
                outlineColor: theme.colors.actionDefaultBorderFocus,
            },
        },
        [`${classCloseIcon}, ${classCloseButton}`]: {
            lineHeight: theme.typography.lineHeightBase,
            height: ALERT_ICON_HEIGHT,
            width: ALERT_ICON_HEIGHT,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
        },
        [classWithIcon]: {
            fontSize: ALERT_ICON_FONT_SIZE,
            marginTop: 2,
            ...(useNewLargeAlertSizing && {
                alignSelf: 'flex-start',
                marginTop: isSmall ? 0 : 2,
            }),
        },
        [`${classCloseIcon}, ${classCloseButton}, ${classCloseText} > span`]: {
            lineHeight: theme.typography.lineHeightBase,
            height: ALERT_ICON_HEIGHT,
            width: ALERT_ICON_HEIGHT,
            fontSize: ALERT_ICON_FONT_SIZE,
            marginTop: 2,
            '& > span, & > span > span': {
                lineHeight: theme.typography.lineHeightBase,
                display: 'inline-flex',
                alignItems: 'center',
            },
        },
        // No description
        ...(!props.description && {
            display: 'flex',
            alignItems: 'center',
            [classWithIcon]: {
                fontSize: ALERT_ICON_FONT_SIZE,
                marginTop: 0,
                ...(useNewLargeAlertSizing && {
                    alignSelf: 'flex-start',
                    marginTop: 2,
                }),
            },
            [classMessage]: {
                margin: 0,
            },
            [classDescription]: {
                display: 'none',
            },
            [classCloseIcon]: {
                alignSelf: 'baseline',
            },
        }),
        // No description with icons
        ...(!props.description &&
            hasActions && {
            ...(!isVertical && {
                [classContent]: {
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                },
            }),
            [classDescription]: {
                display: 'flex',
            },
        }),
        // Warning
        ...(props.type === 'warning' && {
            color: theme.colors.textValidationWarning,
            borderColor: theme.colors.yellow300,
        }),
        // Error
        ...(props.type === 'error' && {
            color: theme.colors.textValidationDanger,
            borderColor: theme.colors.red300,
        }),
        // Banner
        ...(props.banner && {
            borderStyle: 'solid',
            borderWidth: `${theme.general.borderWidth}px 0`,
        }),
        // After closed
        '&[data-show="false"]': {
            borderWidth: 0,
            padding: 0,
            width: 0,
            height: 0,
        },
        ...getAnimationCss(theme.options.enableAnimation),
    };
    return css(styles);
};
//# sourceMappingURL=Alert.js.map
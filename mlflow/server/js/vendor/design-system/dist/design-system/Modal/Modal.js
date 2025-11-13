import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import { Modal as AntDModal } from 'antd';
import { createContext, useContext, useEffect, useMemo, useRef } from 'react';
import { Button } from '../Button';
import { augmentWithDataComponentProps, useDesignSystemEventComponentCallbacks, DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemEventSuppressInteractionProviderContext, DesignSystemEventSuppressInteractionTrueContextValue, } from '../DesignSystemEventProvider/DesignSystemEventSuppressInteractionProvider';
import { DesignSystemAntDConfigProvider, getAnimationCss, RestoreAntDDefaultClsPrefix } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { CloseIcon, DangerIcon } from '../Icon';
import { getDarkModePortalStyles, getShadowScrollStyles, useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled, addDebugOutlineStylesIfEnabled } from '../utils/debug';
import { safex } from '../utils/safex';
import { useNotifyOnFirstView } from '../utils/useNotifyOnFirstView';
const ModalContext = createContext({
    isInsideModal: true,
});
export const useModalContext = () => useContext(ModalContext);
const SIZE_PRESETS = {
    normal: 640,
    wide: 880,
};
const getModalEmotionStyles = (args) => {
    const { theme, clsPrefix, hasFooter = true, maxedOutHeight, useNewShadows, useNewBorderRadii, useNewBorderColors, } = args;
    const classNameClose = `.${clsPrefix}-modal-close`;
    const classNameCloseX = `.${clsPrefix}-modal-close-x`;
    const classNameTitle = `.${clsPrefix}-modal-title`;
    const classNameContent = `.${clsPrefix}-modal-content`;
    const classNameBody = `.${clsPrefix}-modal-body`;
    const classNameHeader = `.${clsPrefix}-modal-header`;
    const classNameFooter = `.${clsPrefix}-modal-footer`;
    const classNameButton = `.${clsPrefix}-btn`;
    const classNameDropdownTrigger = `.${clsPrefix}-dropdown-button`;
    const MODAL_PADDING = theme.spacing.lg;
    const BUTTON_SIZE = theme.general.heightSm;
    // Needed for moving some of the padding from the header and footer to the content to avoid a scrollbar from appearing
    // when the content has some interior components that reach the limits of the content div
    // 8px is an arbitrary value, it still leaves enough padding for the header and footer too to avoid the same problem
    // from occurring there too
    const CONTENT_BUFFER = 8;
    const modalMaxHeight = '90vh';
    const headerHeight = 64;
    const footerHeight = hasFooter ? 52 : 0;
    const bodyMaxHeight = `calc(${modalMaxHeight} - ${headerHeight}px - ${footerHeight}px - ${MODAL_PADDING}px)`;
    return css({
        '&&': {
            ...addDebugOutlineStylesIfEnabled(theme),
        },
        [classNameHeader]: {
            background: 'transparent',
            paddingTop: theme.spacing.md,
            paddingLeft: theme.spacing.lg,
            paddingRight: theme.spacing.md,
            paddingBottom: theme.spacing.md,
        },
        [classNameFooter]: {
            height: footerHeight,
            paddingTop: theme.spacing.lg - CONTENT_BUFFER,
            paddingLeft: MODAL_PADDING,
            paddingRight: MODAL_PADDING,
            marginTop: 'auto',
            [`${classNameButton} + ${classNameButton}`]: {
                marginLeft: theme.spacing.sm,
            },
            // Needed to override AntD style for the SplitButton's dropdown button back to its original value
            [`${classNameDropdownTrigger} > ${classNameButton}:nth-of-type(2)`]: {
                marginLeft: -1,
            },
        },
        [classNameCloseX]: {
            fontSize: theme.general.iconSize,
            height: BUTTON_SIZE,
            width: BUTTON_SIZE,
            lineHeight: 'normal',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: theme.colors.textSecondary,
        },
        [classNameClose]: {
            height: BUTTON_SIZE,
            width: BUTTON_SIZE,
            // Note: Ant has the close button absolutely positioned, rather than in a flex container with the title.
            // This magic number is eyeballed to get the close X to align with the title text.
            margin: '16px 16px 0 0',
            borderRadius: useNewBorderRadii ? theme.borders.borderRadiusSm : theme.legacyBorders.borderRadiusMd,
            backgroundColor: theme.colors.actionDefaultBackgroundDefault,
            borderColor: theme.colors.actionDefaultBackgroundDefault,
            color: theme.colors.actionDefaultTextDefault,
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                borderColor: theme.colors.actionDefaultBackgroundHover,
                color: theme.colors.actionDefaultTextHover,
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                borderColor: theme.colors.actionDefaultBackgroundPress,
                color: theme.colors.actionDefaultTextPress,
            },
            '&:focus-visible': {
                outlineStyle: 'solid',
                outlineWidth: '2px',
                outlineOffset: '1px',
                outlineColor: theme.colors.actionDefaultBorderFocus,
            },
        },
        [classNameTitle]: {
            fontSize: theme.typography.fontSizeXl,
            lineHeight: theme.typography.lineHeightXl,
            fontWeight: theme.typography.typographyBoldFontWeight,
            paddingRight: MODAL_PADDING,
            minHeight: headerHeight / 2,
            display: 'flex',
            alignItems: 'center',
            overflowWrap: 'anywhere',
        },
        [classNameContent]: {
            backgroundColor: theme.colors.backgroundPrimary,
            maxHeight: modalMaxHeight,
            height: maxedOutHeight ? modalMaxHeight : '',
            overflow: 'hidden',
            paddingBottom: MODAL_PADDING,
            display: 'flex',
            flexDirection: 'column',
            boxShadow: useNewShadows ? theme.shadows.xl : theme.general.shadowHigh,
            ...(useNewBorderRadii && {
                borderRadius: theme.borders.borderRadiusLg,
            }),
            ...getDarkModePortalStyles(theme, useNewShadows, useNewBorderColors),
        },
        [classNameBody]: {
            overflowY: 'auto',
            maxHeight: bodyMaxHeight,
            paddingLeft: MODAL_PADDING,
            paddingRight: MODAL_PADDING,
            paddingTop: CONTENT_BUFFER,
            paddingBottom: CONTENT_BUFFER,
            ...getShadowScrollStyles(theme),
        },
        ...getAnimationCss(theme.options.enableAnimation),
    });
};
function closeButtonComponentId(componentId) {
    return componentId ? `${componentId}.footer.cancel` : 'codegen_design-system_src_design-system_modal_modal.tsx_260';
}
/**
 * Render default footer with our buttons. Copied from AntD.
 */
function DefaultFooter({ componentId, onOk, onCancel, confirmLoading, okText, cancelText, okButtonProps, cancelButtonProps, autoFocusButton, shouldStartInteraction, }) {
    const handleCancel = (e) => {
        onCancel?.(e);
    };
    const handleOk = (e) => {
        onOk?.(e);
    };
    return (_jsxs(_Fragment, { children: [cancelText && (_jsx(Button, { componentId: closeButtonComponentId(componentId), onClick: handleCancel, autoFocus: autoFocusButton === 'cancel', dangerouslyUseFocusPseudoClass: true, shouldStartInteraction: shouldStartInteraction, ...cancelButtonProps, children: cancelText })), okText && (_jsx(Button, { componentId: componentId ? `${componentId}.footer.ok` : 'codegen_design-system_src_design-system_modal_modal.tsx_271', loading: confirmLoading, onClick: handleOk, type: "primary", autoFocus: autoFocusButton === 'ok', dangerouslyUseFocusPseudoClass: true, shouldStartInteraction: shouldStartInteraction, ...okButtonProps, children: okText }))] }));
}
export function Modal(props) {
    return (_jsx(DesignSystemEventSuppressInteractionProviderContext.Provider, { value: DesignSystemEventSuppressInteractionTrueContextValue, children: _jsx(ModalInternal, { ...props }) }));
}
function ModalInternal({ componentId, analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView], okButtonProps, cancelButtonProps, dangerouslySetAntdProps, children, title, footer, size = 'normal', verticalSizing = 'dynamic', autoFocusButton, truncateTitle, shouldStartInteraction, ...props }) {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { useNewShadows, useNewBorderRadii, useNewBorderColors } = useDesignSystemSafexFlags();
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Modal,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        shouldStartInteraction,
    });
    const { elementRef } = useNotifyOnFirstView({ onView: eventContext.onView });
    const emitViewEventThroughVisibleEffect = dangerouslySetAntdProps?.closable === false;
    const isViewedViaVisibleEffectRef = useRef(false);
    useEffect(() => {
        if (emitViewEventThroughVisibleEffect && !isViewedViaVisibleEffectRef.current && props.visible === true) {
            isViewedViaVisibleEffectRef.current = true;
            eventContext.onView();
        }
    }, [props.visible, emitViewEventThroughVisibleEffect, eventContext]);
    // Need to simulate the close button being closed if the user clicks outside of the modal or clicks on the dismiss button
    // This should only be applied to the modal prop and not to the footer component
    const closeButtonEventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentId: closeButtonComponentId(componentId),
        analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
        shouldStartInteraction,
    });
    const onCancelWrapper = (e) => {
        closeButtonEventContext.onClick(e);
        props.onCancel?.(e);
    };
    // add data-component-* props to make modal discoverable by go/component-finder
    const augmentedChildren = safex('databricks.fe.observability.enableModalDataComponentProps', false)
        ? augmentWithDataComponentProps(children, eventContext.dataComponentProps)
        : children;
    return (_jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDModal, { ...addDebugOutlineIfEnabled(), css: getModalEmotionStyles({
                theme,
                clsPrefix: classNamePrefix,
                hasFooter: footer !== null,
                maxedOutHeight: verticalSizing === 'maxed_out',
                useNewShadows,
                useNewBorderRadii,
                useNewBorderColors,
            }), title: _jsx(RestoreAntDDefaultClsPrefix, { children: truncateTitle ? (_jsx("div", { css: {
                        textOverflow: 'ellipsis',
                        marginRight: theme.spacing.md,
                        overflow: 'hidden',
                        whiteSpace: 'nowrap',
                    }, title: typeof title === 'string' ? title : undefined, children: title })) : (title) }), footer: footer === null ? null : (_jsx(RestoreAntDDefaultClsPrefix, { children: footer === undefined ? (_jsx(DefaultFooter, { componentId: componentId, onOk: props.onOk, onCancel: props.onCancel, confirmLoading: props.confirmLoading, okText: props.okText, cancelText: props.cancelText, okButtonProps: okButtonProps, cancelButtonProps: cancelButtonProps, autoFocusButton: autoFocusButton, shouldStartInteraction: shouldStartInteraction })) : (footer) })), width: size ? SIZE_PRESETS[size] : undefined, closeIcon: _jsx(CloseIcon, { ref: elementRef }), centered: true, zIndex: theme.options.zIndexBase, maskStyle: {
                backgroundColor: theme.colors.overlayOverlay,
            }, ...props, onCancel: onCancelWrapper, ...dangerouslySetAntdProps, children: _jsx(RestoreAntDDefaultClsPrefix, { children: _jsx(ModalContext.Provider, { value: { isInsideModal: true }, children: augmentedChildren }) }) }) }));
}
export function DangerModal(props) {
    const { theme } = useDesignSystemTheme();
    const { title, onCancel, onOk, cancelText, okText, okButtonProps, cancelButtonProps, ...restProps } = props;
    const iconSize = 18;
    const iconFontSize = 18;
    const titleComp = (_jsxs("div", { css: { position: 'relative', display: 'inline-flex', alignItems: 'center' }, children: [_jsx(DangerIcon, { css: {
                    color: theme.colors.textValidationDanger,
                    left: 2,
                    height: iconSize,
                    width: iconSize,
                    fontSize: iconFontSize,
                } }), _jsx("div", { css: { paddingLeft: 6 }, children: title })] }));
    return (_jsx(Modal, { shouldStartInteraction: props.shouldStartInteraction, title: titleComp, footer: [
            _jsx(Button, { componentId: props.componentId
                    ? `${props.componentId}.danger.footer.cancel`
                    : 'codegen_design-system_src_design-system_modal_modal.tsx_386', onClick: onCancel, shouldStartInteraction: props.shouldStartInteraction, ...cancelButtonProps, children: cancelText || 'Cancel' }, "cancel"),
            _jsx(Button, { componentId: props.componentId
                    ? `${props.componentId}.danger.footer.ok`
                    : 'codegen_design-system_src_design-system_modal_modal.tsx_395', type: "primary", danger: true, onClick: onOk, loading: props.confirmLoading, shouldStartInteraction: props.shouldStartInteraction, ...okButtonProps, children: okText || 'Delete' }, "discard"),
        ], onOk: onOk, onCancel: onCancel, ...restProps }));
}
//# sourceMappingURL=Modal.js.map
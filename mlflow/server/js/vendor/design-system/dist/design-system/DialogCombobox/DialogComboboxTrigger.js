import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "@emotion/react/jsx-runtime";
import { css } from '@emotion/react';
import * as Popover from '@radix-ui/react-popover';
import React, { forwardRef, useEffect } from 'react';
import { DialogComboboxCountBadge } from './DialogComboboxCountBadge';
import { useDialogComboboxContext } from './hooks/useDialogComboboxContext';
import { Button } from '../Button';
import { DesignSystemEventProviderComponentTypes, useComponentFinderContext } from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';
import { ChevronDownIcon, CloseIcon } from '../Icon';
import { LegacyTooltip } from '../LegacyTooltip';
import { useSelectContext } from '../Select/hooks/useSelectContext';
import { ClearSelectionButton } from '../_shared_/Combobox/ClearSelectionButton';
import { useDesignSystemSafexFlags } from '../utils';
import { getValidationStateColor, importantify } from '../utils/css-utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
const getTriggerWrapperStyles = (theme, clsPrefix, removable, width, useNewFormUISpacing) => css(importantify({
    display: 'inline-flex',
    alignItems: 'center',
    ...(useNewFormUISpacing && {
        [`& + .${clsPrefix}-form-message`]: {
            marginTop: theme.spacing.sm,
        },
    }),
    ...(width && {
        width: width,
    }),
    ...(removable && {
        '& > button:last-of-type': importantify({
            borderBottomLeftRadius: 0,
            borderTopLeftRadius: 0,
            marginLeft: -1,
        }),
    }),
}));
const getTriggerStyles = (theme, disabled = false, maxWidth, minWidth, removable, width, validationState, isBare, isSelect, useNewShadows) => {
    const removeButtonInteractionStyles = {
        ...(removable && {
            zIndex: theme.options.zIndexBase + 2,
            '&& + button': {
                marginLeft: -1,
                zIndex: theme.options.zIndexBase + 1,
            },
        }),
    };
    const validationColor = getValidationStateColor(theme, validationState);
    return css(importantify({
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        maxWidth,
        minWidth,
        justifyContent: 'flex-start',
        background: 'transparent',
        padding: isBare ? 0 : '6px 8px 6px 12px',
        boxSizing: 'border-box',
        height: isBare ? theme.typography.lineHeightBase : theme.general.heightSm,
        border: isBare ? 'none' : `1px solid ${theme.colors.actionDefaultBorderDefault}`,
        ...(useNewShadows && {
            boxShadow: theme.shadows.xs,
        }),
        borderRadius: theme.borders.borderRadiusSm,
        color: theme.colors.textPrimary,
        lineHeight: theme.typography.lineHeightBase,
        fontSize: theme.typography.fontSizeBase,
        cursor: 'pointer',
        ...(width && {
            width: width,
            // Only set flex: 1 to items with width, otherwise in flex containers the trigger will take up all the space and break current usages that depend on content for width
            flex: 1,
        }),
        ...(removable && {
            borderBottomRightRadius: 0,
            borderTopRightRadius: 0,
            borderRightColor: 'transparent',
        }),
        '&:hover': {
            background: isBare ? 'transparent' : theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionDefaultBorderHover,
            ...removeButtonInteractionStyles,
        },
        '&:focus': {
            borderColor: theme.colors.actionDefaultBorderFocus,
            ...removeButtonInteractionStyles,
        },
        ...(validationState && {
            borderColor: validationColor,
            '&:hover': {
                borderColor: validationColor,
            },
            '&:focus': {
                outlineColor: validationColor,
                outlineOffset: -2,
            },
        }),
        ...(isSelect &&
            !disabled && {
            '&&, &&:hover, &&:focus': {
                background: 'transparent',
            },
            '&&:hover': {
                borderColor: theme.colors.actionDefaultBorderHover,
            },
            '&&:focus, &[data-state="open"]': {
                outlineColor: theme.colors.actionDefaultBorderFocus,
                outlineWidth: 2,
                outlineOffset: -2,
                outlineStyle: 'solid',
                borderColor: 'transparent',
                ...(!useNewShadows && {
                    boxShadow: 'none',
                }),
            },
        }),
        [`&[disabled]`]: {
            background: theme.colors.actionDisabledBackground,
            color: theme.colors.actionDisabledText,
            pointerEvents: 'none',
            userSelect: 'none',
            borderColor: theme.colors.actionDisabledBorder,
        },
    }));
};
export const DialogComboboxTrigger = forwardRef(({ removable = false, onRemove, children, minWidth = 0, maxWidth = 9999, showTagAfterValueCount = 3, allowClear = true, controlled, onClear, wrapperProps, width, withChevronIcon = true, validationState, withInlineLabel = true, placeholder, id: legacyId, isBare = false, renderDisplayedValue: formatDisplayedValue = (value) => value, ...restProps }, forwardedRef) => {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { label, id: topLevelId, value, isInsideDialogCombobox, multiSelect, setValue } = useDialogComboboxContext();
    const { isSelect, placeholder: selectPlaceholder } = useSelectContext();
    const { useNewShadows, useNewFormUISpacing } = useDesignSystemSafexFlags();
    const id = topLevelId ?? legacyId;
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxTrigger` must be used within `DialogCombobox`');
    }
    const handleRemove = () => {
        if (!onRemove) {
            // eslint-disable-next-line no-console -- TODO(FEINF-3587)
            console.warn('DialogCombobox.Trigger: Attempted remove without providing onRemove handler');
        }
        else {
            onRemove();
        }
    };
    const handleClear = (e) => {
        e.stopPropagation();
        if (controlled) {
            setValue([]);
            onClear?.();
        }
        else if (!onClear) {
            // eslint-disable-next-line no-console -- TODO(FEINF-3587)
            console.warn('DialogCombobox.Trigger: Attempted clear without providing onClear handler');
        }
        else {
            onClear();
        }
    };
    const [showTooltip, setShowTooltip] = React.useState();
    const triggerContentRef = React.useRef(null);
    useEffect(() => {
        if (value?.length > showTagAfterValueCount) {
            setShowTooltip(true);
        }
        else if (triggerContentRef.current) {
            const { clientWidth, scrollWidth } = triggerContentRef.current;
            setShowTooltip(clientWidth < scrollWidth);
        }
    }, [showTagAfterValueCount, value]);
    const renderFormattedValue = (v, index) => {
        const formattedValue = formatDisplayedValue(v);
        return (_jsxs(React.Fragment, { children: [index > 0 && ', ', typeof formattedValue === 'string' ? formattedValue : _jsx("span", { children: formattedValue })] }, index));
    };
    const getStringValue = (v) => {
        const formattedValue = formatDisplayedValue(v);
        return typeof formattedValue === 'string' ? formattedValue : v;
    };
    const numValues = Array.isArray(value) ? value.length : 1;
    const concatenatedValues = Array.isArray(value) ? (_jsxs(_Fragment, { children: [value.slice(0, numValues > 10 ? 10 : undefined).map(renderFormattedValue), numValues > 10 && ` + ${numValues - 10}`] })) : (renderFormattedValue(value, 0));
    const displayedValues = _jsx("span", { children: concatenatedValues });
    const valuesBeforeBadge = Array.isArray(value) ? (_jsx(_Fragment, { children: value.slice(0, showTagAfterValueCount).map(renderFormattedValue) })) : (renderFormattedValue(value, 0));
    let ariaLabel = '';
    if (!isSelect && !id && label) {
        ariaLabel = React.isValidElement(label) ? 'Dialog Combobox' : `${label}`;
        if (value?.length) {
            const stringValues = Array.isArray(value) ? value.map(getStringValue).join(', ') : getStringValue(value);
            ariaLabel += multiSelect
                ? `, multiselectable, ${value.length} options selected: ${stringValues}`
                : `, selected option: ${stringValues}`;
        }
        else {
            ariaLabel += multiSelect ? ', multiselectable, 0 options selected' : ', no option selected';
        }
    }
    else if (isSelect) {
        ariaLabel = ((typeof label === 'string' ? label : '') || restProps['aria-label']) ?? '';
    }
    const customSelectContent = isSelect && children ? children : null;
    const dialogComboboxClassname = !isSelect ? `${classNamePrefix}-dialogcombobox` : '';
    const selectV2Classname = isSelect ? `${classNamePrefix}-selectv2` : '';
    const triggerContent = isSelect ? (_jsxs(Popover.Trigger, { ...(ariaLabel && { 'aria-label': ariaLabel }), ref: forwardedRef, role: "combobox", "aria-haspopup": "listbox", "aria-invalid": validationState === 'error', id: id, ...restProps, css: getTriggerStyles(theme, restProps.disabled, maxWidth, minWidth, removable, width, validationState, isBare, isSelect, useNewShadows), children: [_jsx("span", { css: {
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    height: theme.typography.lineHeightBase,
                    marginRight: 'auto',
                }, ref: triggerContentRef, children: value?.length ? (customSelectContent ?? displayedValues) : (_jsx("span", { css: { color: theme.colors.textPlaceholder }, children: selectPlaceholder })) }), allowClear && value?.length ? _jsx(ClearSelectionButton, { onClick: handleClear }) : null, _jsx(ChevronDownIcon, { css: { color: theme.colors.textSecondary, marginLeft: theme.spacing.xs } })] })) : (_jsxs(Popover.Trigger, { id: id, ...(ariaLabel && { 'aria-label': ariaLabel }), ref: forwardedRef, role: "combobox", "aria-haspopup": "listbox", "aria-invalid": validationState === 'error', ...restProps, css: getTriggerStyles(theme, restProps.disabled, maxWidth, minWidth, removable, width, validationState, isBare, isSelect, useNewShadows), children: [_jsxs("span", { css: {
                    display: 'flex',
                    alignItems: 'center',
                    height: theme.typography.lineHeightBase,
                    marginRight: 'auto',
                    '&, & > *': {
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                    },
                }, ref: triggerContentRef, children: [withInlineLabel ? (_jsxs("span", { css: {
                            height: theme.typography.lineHeightBase,
                            marginRight: theme.spacing.xs,
                            whiteSpace: 'unset',
                            overflow: 'unset',
                            textOverflow: 'unset',
                        }, children: [label, value?.length ? ':' : null] })) : (!value?.length && _jsx("span", { css: { color: theme.colors.textPlaceholder }, children: placeholder })), value?.length > showTagAfterValueCount ? (_jsxs(_Fragment, { children: [_jsx("span", { style: { marginRight: theme.spacing.xs }, children: valuesBeforeBadge }), _jsx(DialogComboboxCountBadge, { countStartAt: showTagAfterValueCount, role: "status", "aria-label": "Selected options count" })] })) : (displayedValues)] }), allowClear && value?.length ? _jsx(ClearSelectionButton, { onClick: handleClear }) : null, withChevronIcon ? (_jsx(ChevronDownIcon, { css: {
                    color: theme.colors.textSecondary,
                    justifySelf: 'flex-end',
                    marginLeft: theme.spacing.xs,
                } })) : null] }));
    const dataComponentProps = useComponentFinderContext(DesignSystemEventProviderComponentTypes.DialogCombobox);
    return (_jsxs("div", { ...wrapperProps, className: `${restProps?.className ?? ''} ${dialogComboboxClassname} ${selectV2Classname}`.trim(), css: [
            getTriggerWrapperStyles(theme, classNamePrefix, removable, width, useNewFormUISpacing),
            wrapperProps?.css,
        ], ...addDebugOutlineIfEnabled(), ...dataComponentProps, children: [showTooltip && value?.length ? (_jsx(LegacyTooltip, { title: customSelectContent ?? displayedValues, children: triggerContent })) : (triggerContent), removable && (_jsx(Button, { componentId: "codegen_design-system_src_design-system_dialogcombobox_dialogcomboboxtrigger.tsx_355", "aria-label": `Remove ${label}`, onClick: handleRemove, dangerouslySetForceIconStyles: true, children: _jsx(CloseIcon, { "aria-label": `Remove ${label}`, "aria-hidden": "false" }) }))] }));
});
/**
 * A custom button trigger that can be wrapped around any button.
 */
export const DialogComboboxCustomButtonTriggerWrapper = ({ children }) => {
    return _jsx(Popover.Trigger, { asChild: true, children: children });
};
//# sourceMappingURL=DialogComboboxTrigger.js.map
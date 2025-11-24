import { E as useDesignSystemContext, u as useDesignSystemTheme, I as Icon, S as Spinner, J as DesignSystemEventProvider, D as DesignSystemAntDConfigProvider, G as RestoreAntDDefaultClsPrefix, s as safex, d as DesignSystemEventProviderAnalyticsEventTypes, e as useDesignSystemEventComponentCallbacks, f as DesignSystemEventProviderComponentTypes, j as useNotifyOnFirstView, y as addDebugOutlineStylesIfEnabled, b as getAnimationCss, h as DesignSystemEventProviderComponentSubTypeMap, B as Button, T as Typography, a as addDebugOutlineIfEnabled, C as CloseIcon, z as getDarkModePortalStyles, K as useUniqueId, R as Root$8, l as Trigger$4, m as Content$5, M as Arrow$2, o as ChevronRightIcon, N as LoadingState, O as visuallyHidden, v as DesignSystemEventSuppressInteractionProviderContext, w as DesignSystemEventSuppressInteractionTrueContextValue, i as importantify, c as useFormContext, P as ComponentFinderContext, r as getComboboxOptionItemWrapperStyles, Q as getComboboxContentWrapperStyles, U as getFooterStyles, V as highlightFirstNonDisabledOption, X as getInfoIconStyles, Y as getKeyboardNavigationFunctions, Z as dialogComboboxLookAheadKeyDown, _ as getCheckboxStyles, $ as getDialogComboboxOptionLabelWidth, q as generateUuidV4, a0 as getContentOptions, a1 as findHighlightedOption, a2 as highlightOption, a3 as findClosestOptionSibling, a4 as getComboboxOptionLabelStyles, a5 as getSelectItemWithHintColumnStyles, a6 as getHintColumnStyles, a7 as useComponentFinderContext, g as getValidationStateColor, a8 as SMALL_BUTTON_HEIGHT$2, a9 as ApplyDesignSystemContextOverrides, A as getShadowScrollStyles, k as DangerIcon, W as WarningIcon, L as LoadingIcon, aa as Title$2, ab as AccessibleContainer, n as ChevronLeftIcon, ac as DU_BOIS_ENABLE_ANIMATION_CLASSNAME, ad as getDefaultStyles, ae as getPrimaryStyles, af as getDisabledSplitButtonStyles, ag as getDisabledPrimarySplitButtonStyles, ah as getHorizontalTabShadowStyles } from './Popover-C9XzCfd9.js';
export { aq as ApplyDesignSystemFlags, aB as ColorVars, an as DesignSystemContext, ak as DesignSystemEventProviderComponentSubTypes, ap as DesignSystemProvider, am as DesignSystemThemeContext, ao as DesignSystemThemeProvider, aw as LoadingStateContext, av as NewWindowIcon, ax as Popover, au as WithDesignSystemThemeHoc, x as augmentWithDataComponentProps, az as getBottomOnlyShadowScrollStyles, aj as getButtonEmotionStyles, ai as getMemoizedButtonEmotionStyles, ay as getTypographyColor, aA as getVirtualizedComboboxMenuItemStyles, as as isOptionDisabled, at as resetTabIndexToFocusedElement, ar as useAntDConfigProviderContext, al as useDesignSystemEventSuppressInteractionContext } from './Popover-C9XzCfd9.js';
import { jsx, jsxs, Fragment } from '@emotion/react/jsx-runtime';
import * as React from 'react';
import React__default, { useRef, useMemo, forwardRef, createContext, useContext, useCallback, useState, useEffect, useImperativeHandle, Children, Fragment as Fragment$1, useLayoutEffect } from 'react';
import { d as Modal, e as useModalContext, I as InfoSmallIcon, C as CheckIcon, T as Tooltip$1, S as Spacer, L as ListIcon } from './WizardStepContentWrapper-Crl3e_yT.js';
export { i as Content, g as DangerModal, D as DocumentationSidebar, F as FIXED_VERTICAL_STEPPER_WIDTH, M as MAX_VERTICAL_WIZARD_CONTENT_WIDTH, N as Nav, h as NavButton, P as Panel, m as PanelBody, j as PanelHeader, l as PanelHeaderButtons, k as PanelHeaderTitle, n as Sidebar, o as Stepper, W as Wizard, a as WizardControlled, b as WizardModal, c as WizardStepContentWrapper, f as useRadixModalContext, u as useWizardCurrentStep } from './WizardStepContentWrapper-Crl3e_yT.js';
import { X as XCircleFillIcon, I as Input, S as SearchIcon } from './index-284YKJ8q.js';
export { C as ClockIcon, L as LockIcon, M as MegaphoneIcon, g as getInputStyles, a as useCallbackOnEnter } from './index-284YKJ8q.js';
import { css, Global, keyframes, ClassNames, createElement } from '@emotion/react';
import { Collapse, Alert as Alert$1, AutoComplete as AutoComplete$1, Tooltip, Breadcrumb as Breadcrumb$1, Checkbox as Checkbox$1, DatePicker, Dropdown as Dropdown$1, Select as Select$1, Radio as Radio$1, Switch as Switch$1, Col as Col$1, Row as Row$1, Space as Space$1, Layout as Layout$1, Form, notification, Popover as Popover$1, Skeleton, Pagination as Pagination$1, Table as Table$1, Tabs as Tabs$1, Menu as Menu$1, Button as Button$1, Steps as Steps$1, Tree as Tree$1 } from 'antd';
import classnames from 'classnames';
import isNil from 'lodash/isNil';
import { useMergeRefs, useFloating, autoUpdate, offset, flip, shift } from '@floating-ui/react';
import isUndefined from 'lodash/isUndefined';
import { ContextMenuTrigger, ContextMenuItemIndicator, ContextMenuGroup, ContextMenuArrow, ContextMenuSub, ContextMenu as ContextMenu$2, ContextMenuSubTrigger, ContextMenuPortal, ContextMenuContent, ContextMenuSubContent, ContextMenuItem, ContextMenuCheckboxItem, ContextMenuRadioGroup, ContextMenuRadioItem, ContextMenuLabel, ContextMenuSeparator } from '@radix-ui/react-context-menu';
import * as DropdownMenu$1 from '@radix-ui/react-dropdown-menu';
import * as Popover from '@radix-ui/react-popover';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { useController } from 'react-hook-form';
import uniqueId from 'lodash/uniqueId';
import { useCombobox, useMultipleSelection } from 'downshift';
import { computePosition, flip as flip$1, size } from '@floating-ui/dom';
import { createPortal } from 'react-dom';
import * as RadixHoverCard from '@radix-ui/react-hover-card';
import AntDIcon, { InfoCircleOutlined } from '@ant-design/icons';
import * as RadixNavigationMenu from '@radix-ui/react-navigation-menu';
import * as Toast from '@radix-ui/react-toast';
import random from 'lodash/random';
import times from 'lodash/times';
import * as RadixSlider from '@radix-ui/react-slider';
import * as ScrollArea from '@radix-ui/react-scroll-area';
import * as RadixTabs from '@radix-ui/react-tabs';
import debounce from 'lodash/debounce';
import * as Toggle from '@radix-ui/react-toggle';
import chroma from 'chroma-js';
import isEqual from 'lodash/isEqual';
import '@radix-ui/react-tooltip';
import '@radix-ui/react-tooltip-patch';
import 'lodash/memoize';
import '@emotion/unitless';
import 'lodash/endsWith';
import 'lodash/isBoolean';
import 'lodash/isNumber';
import 'lodash/isString';
import 'lodash/mapValues';
import 'lodash/noop';
import 'react-resizable';
import 'lodash/pick';
import 'lodash/compact';

function useDesignSystemFlags() {
    const context = useDesignSystemContext();
    return context.flags;
}

/**
 * A helper hook that allows quick creation of theme-dependent styles.
 * Results in more compact code than using useMemo and
 * useDesignSystemTheme separately.
 *
 * @example
 * const styles = useThemedStyles((theme) => ({
 *   overlay: {
 *     backgroundColor: theme.colors.backgroundPrimary,
 *     borderRadius: theme.borders.borderRadiusMd,
 *   },
 *   wrapper: {
 *     display: 'flex',
 *     gap: theme.spacing.md,
 *   },
 * }));

 * <div css={styles.overlay}>...</div>
 *
 * @param styleFactory Factory function that accepts theme object as a parameter and returns
 *     the style object. **Note**: factory function body is being memoized internally and is intended
 *     to be used only for simple style objects that depend solely on the theme. If you want to use
 *     styles that change depending on external values (state, props etc.) you should use
 *     `useDesignSystemTheme` directly with  your own reaction mechanism.
 * @returns The constructed style object
 */ const useThemedStyles = (styleFactory)=>{
    const { theme } = useDesignSystemTheme();
    // We can assume that the factory function won't change and we're
    // observing theme changes only.
    const styleFactoryRef = useRef(styleFactory);
    return useMemo(()=>styleFactoryRef.current(theme), [
        theme
    ]);
};

function SvgAzHorizontalIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M4.346 4.5a.75.75 0 0 0-.695.468L1 11.5h1.619l.406-1h2.643l.406 1h1.619L5.04 4.968a.75.75 0 0 0-.695-.468M5.06 9H3.634l.712-1.756zM12.667 6H9V4.5h5.25a.75.75 0 0 1 .58 1.225L11.333 10H15v1.5H9.75a.75.75 0 0 1-.58-1.225z",
            clipRule: "evenodd"
        })
    });
}
const AzHorizontalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgAzHorizontalIcon
    });
});
AzHorizontalIcon.displayName = 'AzHorizontalIcon';

function SvgAzVerticalIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                fill: "currentColor",
                fillRule: "evenodd",
                clipPath: "url(#AZVerticalIcon_svg__a)",
                clipRule: "evenodd",
                children: /*#__PURE__*/ jsx("path", {
                    d: "M7.996 0a.75.75 0 0 1 .695.468L11.343 7h-1.62l-.405-1H6.675l-.406 1H4.65L7.301.468A.75.75 0 0 1 7.996 0m-.712 4.5h1.425l-.713-1.756zM8.664 9.5H4.996V8h5.25a.75.75 0 0 1 .58 1.225L7.33 13.5h3.667V15h-5.25a.75.75 0 0 1-.58-1.225z"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const AzVerticalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgAzVerticalIcon
    });
});
AzVerticalIcon.displayName = 'AzVerticalIcon';

function SvgAlignCenterIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M1 2.5h14V1H1zM11.5 5.75h-7v-1.5h7zM15 8.75H1v-1.5h14zM15 15H1v-1.5h14zM4.5 11.75h7v-1.5h-7z"
        })
    });
}
const AlignCenterIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgAlignCenterIcon
    });
});
AlignCenterIcon.displayName = 'AlignCenterIcon';

function SvgAlignJustifyIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M1 2.5h14V1H1zm14 3.25H1v-1.5h14zm-14 3v-1.5h14v1.5zM1 15v-1.5h14V15zm0-3.25h14v-1.5H1z"
        })
    });
}
const AlignJustifyIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgAlignJustifyIcon
    });
});
AlignJustifyIcon.displayName = 'AlignJustifyIcon';

function SvgAlignLeftIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M1 2.5h14V1H1zM8 5.75H1v-1.5h7zM1 8.75v-1.5h14v1.5zM1 15v-1.5h14V15zM1 11.75h7v-1.5H1z"
        })
    });
}
const AlignLeftIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgAlignLeftIcon
    });
});
AlignLeftIcon.displayName = 'AlignLeftIcon';

function SvgAlignRightIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M1 2.5h14V1H1zM15 5.75H8v-1.5h7zM1 8.75v-1.5h14v1.5zM1 15v-1.5h14V15zM8 11.75h7v-1.5H8z"
        })
    });
}
const AlignRightIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgAlignRightIcon
    });
});
AlignRightIcon.displayName = 'AlignRightIcon';

function SvgAppIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2.75 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M8 1a1.75 1.75 0 1 0 0 3.5A1.75 1.75 0 0 0 8 1m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M2.75 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M2.75 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5m5.25 0A1.75 1.75 0 1 0 8 15a1.75 1.75 0 0 0 0-3.5m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5",
            clipRule: "evenodd"
        })
    });
}
const AppIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgAppIcon
    });
});
AppIcon.displayName = 'AppIcon';

function SvgArrowDownDotIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8 15a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3M3.47 6.53 8 11.06l4.53-4.53-1.06-1.06-2.72 2.72V1h-1.5v7.19L4.53 5.47z"
        })
    });
}
const ArrowDownDotIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowDownDotIcon
    });
});
ArrowDownDotIcon.displayName = 'ArrowDownDotIcon';

function SvgArrowDownFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M15 8.06H9.5V1h-3v7.06H1l7 7z"
        })
    });
}
const ArrowDownFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowDownFillIcon
    });
});
ArrowDownFillIcon.displayName = 'ArrowDownFillIcon';

function SvgArrowDownIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8.03 15.06 1 8.03l1.06-1.06 5.22 5.22V1h1.5v11.19L14 6.97l1.06 1.06z",
            clipRule: "evenodd"
        })
    });
}
const ArrowDownIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowDownIcon
    });
});
ArrowDownIcon.displayName = 'ArrowDownIcon';

function SvgArrowInIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M4.5 2.5h9v11h-9V11H3v3.25c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H3.75a.75.75 0 0 0-.75.75V5h1.5z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M12.06 8 8.03 3.97 6.97 5.03l2.22 2.22H1v1.5h8.19l-2.22 2.22 1.06 1.06z"
            })
        ]
    });
}
const ArrowInIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowInIcon
    });
});
ArrowInIcon.displayName = 'ArrowInIcon';

function SvgArrowLeftIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 8.03 8.03 1l1.061 1.06-5.22 5.22h11.19v1.5H3.87L9.091 14l-1.06 1.06z",
            clipRule: "evenodd"
        })
    });
}
const ArrowLeftIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowLeftIcon
    });
});
ArrowLeftIcon.displayName = 'ArrowLeftIcon';

function SvgArrowOverIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8 2.5a5.48 5.48 0 0 1 3.817 1.54l.009.009.5.451H11V6h4V2h-1.5v1.539l-.651-.588A7.003 7.003 0 0 0 1.367 5.76l1.42.48A5.5 5.5 0 0 1 8 2.5M8 11a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3"
        })
    });
}
const ArrowOverIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowOverIcon
    });
});
ArrowOverIcon.displayName = 'ArrowOverIcon';

function SvgArrowRightIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m15.06 8.03-7.03 7.03L6.97 14l5.22-5.22H1v-1.5h11.19L6.97 2.06 8.03 1z",
            clipRule: "evenodd"
        })
    });
}
const ArrowRightIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowRightIcon
    });
});
ArrowRightIcon.displayName = 'ArrowRightIcon';

function SvgArrowUpDotIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8 1a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3M12.53 9.47 8 4.94 3.47 9.47l1.06 1.06 2.72-2.72V15h1.5V7.81l2.72 2.72z"
        })
    });
}
const ArrowUpDotIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowUpDotIcon
    });
});
ArrowUpDotIcon.displayName = 'ArrowUpDotIcon';

function SvgArrowUpFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M15 8H9.5v7.06h-3V8H1l7-7z"
        })
    });
}
const ArrowUpFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowUpFillIcon
    });
});
ArrowUpFillIcon.displayName = 'ArrowUpFillIcon';

function SvgArrowUpIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m8.03 1 7.03 7.03L14 9.091l-5.22-5.22v11.19h-1.5V3.87l-5.22 5.22L1 8.031z",
            clipRule: "evenodd"
        })
    });
}
const ArrowUpIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowUpIcon
    });
});
ArrowUpIcon.displayName = 'ArrowUpIcon';

function SvgArrowsConnectIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M4.889 10.836h-1.5V9.3l-.918 1.026a1.25 1.25 0 0 0-.318.833v4.416h-1.5l-.001-4.416c0-.677.25-1.33.7-1.834L2.457 8.09l-1.811.001v-1.5h4.244zm10.587-4.244v1.5h-1.784l1.09 1.24c.442.501.686 1.147.686 1.816v4.426h-1.5v-4.426c0-.304-.112-.598-.312-.826l-.924-1.052v1.566h-1.5V6.592zm-4.475 4.645-1.06 1.06-1.19-1.19v4.415h-1.5v-4.415l-1.191 1.19L5 11.237l3-3.001zm-3-9.752a3 3 0 1 1-.001 6 3 3 0 0 1 0-6m0 1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3"
        })
    });
}
const ArrowsConnectIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowsConnectIcon
    });
});
ArrowsConnectIcon.displayName = 'ArrowsConnectIcon';

function SvgArrowsUpDownIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M5.03 1 1 5.03l1.06 1.061 2.22-2.22v6.19h1.5V3.87L8 6.091l1.06-1.06zM11.03 15.121l4.03-4.03-1.06-1.06-2.22 2.219V6.06h-1.5v6.19l-2.22-2.22L7 11.091z"
        })
    });
}
const ArrowsUpDownIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgArrowsUpDownIcon
    });
});
ArrowsUpDownIcon.displayName = 'ArrowsUpDownIcon';

function SvgAssistantIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M11.28 11.03H4.73v-1.7h6.55zm-2.8-4.7H4.73v1.7h3.75zM15.79 8h-1.7a6.09 6.09 0 0 1-6.08 6.08H3.12l.58-.58c.33-.33.33-.87 0-1.2A6.04 6.04 0 0 1 1.92 8 6.09 6.09 0 0 1 8 1.92V.22C3.71.22.22 3.71.22 8c0 1.79.6 3.49 1.71 4.87L.47 14.33c-.24.24-.32.61-.18.93.13.32.44.52.79.52h6.93c4.29 0 7.78-3.49 7.78-7.78m-.62-3.47c.4-.15.4-.72 0-.88l-1.02-.38c-.73-.28-1.31-.85-1.58-1.58L12.19.67c-.08-.2-.26-.3-.44-.3s-.36.1-.44.3l-.38 1.02c-.28.73-.85 1.31-1.58 1.58l-1.02.38c-.4.15-.4.72 0 .88l1.02.38c.73.28 1.31.85 1.58 1.58l.38 1.02c.08.2.26.3.44.3s.36-.1.44-.3l.38-1.02c.28-.73.85-1.31 1.58-1.58z"
        })
    });
}
const AssistantIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgAssistantIcon
    });
});
AssistantIcon.displayName = 'AssistantIcon';

function SvgAtIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2.5 8a5.5 5.5 0 1 1 11 0l-.002 1.08a.973.973 0 0 1-1.946-.002V4.984h-1.5v.194A3.52 3.52 0 0 0 8 4.5C6.22 4.5 4.5 5.949 4.5 8s1.72 3.5 3.5 3.5c.917 0 1.817-.384 2.475-1.037a2.473 2.473 0 0 0 4.523-1.38L15 8a7 7 0 1 0-3.137 5.839l-.83-1.25A5.5 5.5 0 0 1 2.5 8M6 8c0-1.153.976-2 2-2s2 .847 2 2-.976 2-2 2-2-.847-2-2",
            clipRule: "evenodd"
        })
    });
}
const AtIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgAtIcon
    });
});
AtIcon.displayName = 'AtIcon';

function SvgBackupIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M15.812 3.892v7.74a2.5 2.5 0 0 1-2.5 2.5h-9.1l-.255-.012a2.5 2.5 0 0 1-2.244-2.487V3.892l2-2.636h10.099zm-12.6 7.74a1 1 0 0 0 1 1h9.1a1 1 0 0 0 1-1V4.969h-11.1zm6.3-2.277 1.19-1.19 1.06 1.06-3 3.002L5.76 9.226l1.06-1.061 1.19 1.19v-3.86h1.5zM3.917 3.468h9.69l-.54-.712H4.458z"
        })
    });
}
const BackupIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBackupIcon
    });
});
BackupIcon.displayName = 'BackupIcon';

function SvgBadgeCodeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m5.56 8.53 1.97 1.97-1.06 1.06-3.03-3.03L6.47 5.5l1.06 1.06zM10.49 8.53 8.52 6.56 9.58 5.5l3.03 3.03-3.03 3.03-1.06-1.06z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8 0a3.25 3.25 0 0 0-3 2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H11a3.25 3.25 0 0 0-3-2M6.285 2.9a1.75 1.75 0 0 1 3.43 0c.07.349.378.6.735.6h3.05v10h-11v-10h3.05a.75.75 0 0 0 .735-.6",
                clipRule: "evenodd"
            })
        ]
    });
}
const BadgeCodeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBadgeCodeIcon
    });
});
BadgeCodeIcon.displayName = 'BadgeCodeIcon';

function SvgBadgeCodeOffIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 17",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M16 2.75v11.19l-1.5-1.5V3.5h-3.05a.75.75 0 0 1-.735-.6 1.75 1.75 0 0 0-3.43 0 .75.75 0 0 1-.735.6h-.99L4.06 2H6a3.25 3.25 0 0 1 6 0h3.25a.75.75 0 0 1 .75.75"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m12.1 10.04-1.06-1.06.48-.48-1.97-1.97 1.06-1.06 3.031 3.03z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "m12.94 15 1.03 1.03 1.06-1.06-13-13L.97 3.03 2 4.06v10.19c0 .414.336.75.75.75zm-4.455-4.454L7.47 11.56 4.44 8.53l1.015-1.016L3.5 5.561V13.5h7.94z",
                clipRule: "evenodd"
            })
        ]
    });
}
const BadgeCodeOffIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBadgeCodeOffIcon
    });
});
BadgeCodeOffIcon.displayName = 'BadgeCodeOffIcon';

function SvgBarChartIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M1 1v13.25c0 .414.336.75.75.75H15v-1.5H2.5V1z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7 1v11h1.5V1zM10 5v7h1.5V5zM4 5v7h1.5V5zM13 12V8h1.5v4z"
            })
        ]
    });
}
const BarChartIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBarChartIcon
    });
});
BarChartIcon.displayName = 'BarChartIcon';

function SvgBarGroupedIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M12.25 2a.75.75 0 0 0-.75.75V7H9.25a.75.75 0 0 0-.75.75v5.5c0 .414.336.75.75.75h6a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75zm-.75 10.5v-4H10v4zm1.5 0h1.5v-9H13zM3.75 5a.75.75 0 0 0-.75.75V9H.75a.75.75 0 0 0-.75.75v3.5c0 .414.336.75.75.75h6a.75.75 0 0 0 .75-.75v-7.5A.75.75 0 0 0 6.75 5zM3 12.5v-2H1.5v2zm1.5 0H6v-6H4.5z",
            clipRule: "evenodd"
        })
    });
}
const BarGroupedIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBarGroupedIcon
    });
});
BarGroupedIcon.displayName = 'BarGroupedIcon';

function SvgBarStackedIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M6.25 1a.75.75 0 0 0-.75.75V7H2.75a.75.75 0 0 0-.75.75v6.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-9.5a.75.75 0 0 0-.75-.75H10.5V1.75A.75.75 0 0 0 9.75 1zM9 8.5v5H7v-5zM9 7V2.5H7V7zm3.5 6.5h-2v-1.75h2zm-2-8v4.75h2V5.5zm-5 4.75V8.5h-2v1.75zm0 3.25v-1.75h-2v1.75z",
            clipRule: "evenodd"
        })
    });
}
const BarStackedIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBarStackedIcon
    });
});
BarStackedIcon.displayName = 'BarStackedIcon';

function SvgBarStackedPercentageIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zM9 8.5v5H7v-5zM9 7V2.5H7V7zm3.5 6.5h-2v-1.75h2zm-2-11v7.75h2V2.5zm-5 0h-2v7.75h2zm0 11v-1.75h-2v1.75z",
            clipRule: "evenodd"
        })
    });
}
const BarStackedPercentageIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBarStackedPercentageIcon
    });
});
BarStackedPercentageIcon.displayName = 'BarStackedPercentageIcon';

function SvgBarsAscendingHorizontalIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M3.25 9v6h1.5V9zM11.25 1v14h1.5V1zM8.75 15V5h-1.5v10z"
        })
    });
}
const BarsAscendingHorizontalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBarsAscendingHorizontalIcon
    });
});
BarsAscendingHorizontalIcon.displayName = 'BarsAscendingHorizontalIcon';

function SvgBarsAscendingVerticalIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M7 3.25H1v1.5h6zM15 11.25H1v1.5h14zM1 8.75h10v-1.5H1z"
        })
    });
}
const BarsAscendingVerticalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBarsAscendingVerticalIcon
    });
});
BarsAscendingVerticalIcon.displayName = 'BarsAscendingVerticalIcon';

function SvgBarsDescendingHorizontalIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M12.75 9v6h-1.5V9zM4.75 1v14h-1.5V1zM7.25 15V5h1.5v10z"
        })
    });
}
const BarsDescendingHorizontalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBarsDescendingHorizontalIcon
    });
});
BarsDescendingHorizontalIcon.displayName = 'BarsDescendingHorizontalIcon';

function SvgBarsDescendingVerticalIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M7 12.75H1v-1.5h6zM15 4.75H1v-1.5h14zM1 7.25h10v1.5H1z"
        })
    });
}
const BarsDescendingVerticalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBarsDescendingVerticalIcon
    });
});
BarsDescendingVerticalIcon.displayName = 'BarsDescendingVerticalIcon';

function SvgBeakerIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M5.75 1a.75.75 0 0 0-.75.75v6.089c0 .38-.173.739-.47.976l-2.678 2.143A2.27 2.27 0 0 0 3.27 15h9.46a2.27 2.27 0 0 0 1.418-4.042L11.47 8.815A1.25 1.25 0 0 1 11 7.839V1.75a.75.75 0 0 0-.75-.75zm.75 6.839V2.5h3v5.339c0 .606.2 1.188.559 1.661H5.942A2.75 2.75 0 0 0 6.5 7.839M4.2 11 2.79 12.13a.77.77 0 0 0 .48 1.37h9.461a.77.77 0 0 0 .481-1.37L11.8 11z",
            clipRule: "evenodd"
        })
    });
}
const BeakerIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBeakerIcon
    });
});
BeakerIcon.displayName = 'BeakerIcon';

function SvgBinaryIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 3a2 2 0 1 1 4 0v2a2 2 0 1 1-4 0zm2-.5a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0V3a.5.5 0 0 0-.5-.5m3.378-.628c.482 0 .872-.39.872-.872h1.5v4.25H10v1.5H6v-1.5h1.25V3.206c-.27.107-.564.166-.872.166H6v-1.5zm5 0c.482 0 .872-.39.872-.872h1.5v4.25H15v1.5h-4v-1.5h1.25V3.206c-.27.107-.564.166-.872.166H11v-1.5zM6 11a2 2 0 1 1 4 0v2a2 2 0 1 1-4 0zm2-.5a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0v-2a.5.5 0 0 0-.5-.5m-6.622-.378c.482 0 .872-.39.872-.872h1.5v4.25H5V15H1v-1.5h1.25v-2.044c-.27.107-.564.166-.872.166H1v-1.5zm10 0c.482 0 .872-.39.872-.872h1.5v4.25H15V15h-4v-1.5h1.25v-2.044c-.27.107-.564.166-.872.166H11v-1.5z",
            clipRule: "evenodd"
        })
    });
}
const BinaryIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBinaryIcon
    });
});
BinaryIcon.displayName = 'BinaryIcon';

function SvgBlockQuoteIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M16 2H0v1.5h16zM16 5.5H8V7h8zM16 9H8v1.5h8zM0 12.5V14h16v-1.5zM1.5 7.25A.25.25 0 0 1 1.75 7h.75V5.5h-.75A1.75 1.75 0 0 0 0 7.25v2.5c0 .414.336.75.75.75h1.5A.75.75 0 0 0 3 9.75v-1.5a.75.75 0 0 0-.75-.75H1.5zM5.5 7.5h.75a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-.75.75h-1.5A.75.75 0 0 1 4 9.75v-2.5c0-.966.784-1.75 1.75-1.75h.75V7h-.75a.25.25 0 0 0-.25.25z"
        })
    });
}
const BlockQuoteIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBlockQuoteIcon
    });
});
BlockQuoteIcon.displayName = 'BlockQuoteIcon';

function SvgBoldIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M4.75 3a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h4.375a2.875 2.875 0 0 0 1.496-5.33A2.875 2.875 0 0 0 8.375 3zm.75 5.75v2.75h3.625a1.375 1.375 0 0 0 0-2.75zm2.877-1.5a1.375 1.375 0 0 0-.002-2.75H5.5v2.75z",
            clipRule: "evenodd"
        })
    });
}
const BoldIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBoldIcon
    });
});
BoldIcon.displayName = 'BoldIcon';

function SvgBookIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2.75 1a.75.75 0 0 0-.75.75v13.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zM7.5 2.5h-4v6.055l1.495-1.36a.75.75 0 0 1 1.01 0L7.5 8.555zm-4 8.082 2-1.818 2.246 2.041A.75.75 0 0 0 9 10.25V2.5h3.5v12h-9z",
            clipRule: "evenodd"
        })
    });
}
const BookIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBookIcon
    });
});
BookIcon.displayName = 'BookIcon';

function SvgBookmarkFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M2.75 0A.75.75 0 0 0 2 .75v14.5a.75.75 0 0 0 1.28.53L8 11.06l4.72 4.72a.75.75 0 0 0 1.28-.53V.75a.75.75 0 0 0-.75-.75z"
        })
    });
}
const BookmarkFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBookmarkFillIcon
    });
});
BookmarkFillIcon.displayName = 'BookmarkFillIcon';

function SvgBookmarkIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2 .75A.75.75 0 0 1 2.75 0h10.5a.75.75 0 0 1 .75.75v14.5a.75.75 0 0 1-1.28.53L8 11.06l-4.72 4.72A.75.75 0 0 1 2 15.25zm1.5.75v11.94l3.97-3.97a.75.75 0 0 1 1.06 0l3.97 3.97V1.5z",
            clipRule: "evenodd"
        })
    });
}
const BookmarkIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBookmarkIcon
    });
});
BookmarkIcon.displayName = 'BookmarkIcon';

function SvgBooksIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 17",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.5 4.5v10h1v-10zM1 3a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1zM6.5 1.5v13h2v-13zM6 0a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "m11.63 7.74 1.773 6.773.967-.254-1.773-6.771zm-.864-1.324a1 1 0 0 0-.714 1.221l2.026 7.74a1 1 0 0 0 1.22.713l1.936-.506a1 1 0 0 0 .714-1.22l-2.026-7.74a1 1 0 0 0-1.22-.714z",
                clipRule: "evenodd"
            })
        ]
    });
}
const BooksIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBooksIcon
    });
});
BooksIcon.displayName = 'BooksIcon';

function SvgBracketsCheckIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M6 2.5h-.5c-.69 0-1.25.56-1.25 1.25v1c0 .931-.464 1.753-1.173 2.25A2.74 2.74 0 0 1 4.25 9.25v1c0 .69.56 1.25 1.25 1.25H6V13h-.5a2.75 2.75 0 0 1-2.75-2.75v-1C2.75 8.56 2.19 8 1.5 8H1V6h.5c.69 0 1.25-.56 1.25-1.25v-1A2.75 2.75 0 0 1 5.5 1H6zM10.5 1a2.75 2.75 0 0 1 2.75 2.75v1c0 .69.56 1.25 1.25 1.25h.5v1.691a5.2 5.2 0 0 0-2.339-.898 2.74 2.74 0 0 1-.911-2.043v-1c0-.69-.56-1.25-1.25-1.25H10V1z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12 8a4 4 0 1 1 0 8 4 4 0 0 1 0-8m-.5 4.19-.97-.97-1.06 1.06 2.03 2.03 3.28-3.28-1.06-1.06z",
                clipRule: "evenodd"
            })
        ]
    });
}
const BracketsCheckIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBracketsCheckIcon
    });
});
BracketsCheckIcon.displayName = 'BracketsCheckIcon';

function SvgBracketsCurlyIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M5.5 2a2.75 2.75 0 0 0-2.75 2.75v1C2.75 6.44 2.19 7 1.5 7H1v2h.5c.69 0 1.25.56 1.25 1.25v1A2.75 2.75 0 0 0 5.5 14H6v-1.5h-.5c-.69 0-1.25-.56-1.25-1.25v-1c0-.93-.462-1.752-1.168-2.25A2.75 2.75 0 0 0 4.25 5.75v-1c0-.69.56-1.25 1.25-1.25H6V2zM13.25 4.75A2.75 2.75 0 0 0 10.5 2H10v1.5h.5c.69 0 1.25.56 1.25 1.25v1c0 .93.462 1.752 1.168 2.25a2.75 2.75 0 0 0-1.168 2.25v1c0 .69-.56 1.25-1.25 1.25H10V14h.5a2.75 2.75 0 0 0 2.75-2.75v-1c0-.69.56-1.25 1.25-1.25h.5V7h-.5c-.69 0-1.25-.56-1.25-1.25z"
        })
    });
}
const BracketsCurlyIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBracketsCurlyIcon
    });
});
BracketsCurlyIcon.displayName = 'BracketsCurlyIcon';

function SvgBracketsErrorIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12 8a4 4 0 1 1 0 8 4 4 0 0 1 0-8m0 2.94-1.22-1.22-1.06 1.06L10.94 12l-1.22 1.22 1.06 1.06L12 13.06l1.22 1.22 1.06-1.06L13.06 12l1.22-1.22-1.06-1.06z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M6 2.5h-.5c-.69 0-1.25.56-1.25 1.25v1c0 .931-.464 1.753-1.173 2.25A2.74 2.74 0 0 1 4.25 9.25v1c0 .69.56 1.25 1.25 1.25H6V13h-.5a2.75 2.75 0 0 1-2.75-2.75v-1C2.75 8.56 2.19 8 1.5 8H1V6h.5c.69 0 1.25-.56 1.25-1.25v-1A2.75 2.75 0 0 1 5.5 1H6zM10.5 1a2.75 2.75 0 0 1 2.75 2.75v1c0 .69.56 1.25 1.25 1.25h.5v1.691a5.2 5.2 0 0 0-2.339-.898 2.74 2.74 0 0 1-.911-2.043v-1c0-.69-.56-1.25-1.25-1.25H10V1z"
            })
        ]
    });
}
const BracketsErrorIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBracketsErrorIcon
    });
});
BracketsErrorIcon.displayName = 'BracketsErrorIcon';

function SvgBracketsSquareIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 1.75A.75.75 0 0 1 1.75 1H5v1.5H2.5v11H5V15H1.75a.75.75 0 0 1-.75-.75zm12.5.75H11V1h3.25a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H11v-1.5h2.5z",
            clipRule: "evenodd"
        })
    });
}
const BracketsSquareIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBracketsSquareIcon
    });
});
BracketsSquareIcon.displayName = 'BracketsSquareIcon';

function SvgBracketsXIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#BracketsXIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M1.75 4.75A2.75 2.75 0 0 1 4.5 2H5v1.5h-.5c-.69 0-1.25.56-1.25 1.25v1c0 .93-.462 1.752-1.168 2.25a2.75 2.75 0 0 1 1.168 2.25v1c0 .69.56 1.25 1.25 1.25H5V14h-.5a2.75 2.75 0 0 1-2.75-2.75v-1C1.75 9.56 1.19 9 .5 9H0V7h.5c.69 0 1.25-.56 1.25-1.25zM11.5 2a2.75 2.75 0 0 1 2.75 2.75v1c0 .69.56 1.25 1.25 1.25h.5v2h-.5c-.69 0-1.25.56-1.25 1.25v1A2.75 2.75 0 0 1 11.5 14H11v-1.5h.5c.69 0 1.25-.56 1.25-1.25v-1c0-.93.462-1.752 1.168-2.25a2.75 2.75 0 0 1-1.168-2.25v-1c0-.69-.56-1.25-1.25-1.25H11V2z"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M4.97 6.03 6.94 8 4.97 9.97l1.06 1.06L8 9.06l1.97 1.97 1.06-1.06L9.06 8l1.97-1.97-1.06-1.06L8 6.94 6.03 4.97z"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const BracketsXIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBracketsXIcon
    });
});
BracketsXIcon.displayName = 'BracketsXIcon';

function SvgBranchCheckIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M3.74 1a2.998 2.998 0 0 1 .75 5.901v2.187a2.999 2.999 0 0 1-.75 5.902 2.998 2.998 0 0 1-.75-5.902V6.9A2.998 2.998 0 0 1 3.74 1m0 9.49a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3m8.817-4.243-3.53 3.53-2.03-2.03 1.06-1.06.97.97 2.47-2.47zM3.739 2.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3"
        })
    });
}
const BranchCheckIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBranchCheckIcon
    });
});
BranchCheckIcon.displayName = 'BranchCheckIcon';

function SvgBranchIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 4a3 3 0 1 1 5.186 2.055 3.23 3.23 0 0 0 2 1.155 3.001 3.001 0 1 1-.152 1.494A4.73 4.73 0 0 1 4.911 6.86a3 3 0 0 1-.161.046v2.19a3.001 3.001 0 1 1-1.5 0v-2.19A3 3 0 0 1 1 4m3-1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3M2.5 12a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0m7-3.75a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0",
            clipRule: "evenodd"
        })
    });
}
const BranchIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBranchIcon
    });
});
BranchIcon.displayName = 'BranchIcon';

function SvgBranchResetIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M3.99 1a2.998 2.998 0 0 1 .75 5.901v2.187a2.999 2.999 0 0 1-.75 5.902 2.998 2.998 0 0 1-.75-5.902V6.9A2.998 2.998 0 0 1 3.99 1m8.883 0a2.998 2.998 0 0 1 .75 5.901v2.187a2.999 2.999 0 1 1-3.75 2.902c0-1.397.956-2.569 2.25-2.902V6.9A2.997 2.997 0 0 1 12.873 1m-8.884 9.49a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3m8.884 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3M11.6 8.083l-3.001 3.001-1.06-1.06 1.19-1.191H5.4v-1.5H8.73l-1.19-1.19 1.06-1.061zM3.99 2.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3m8.883 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3"
        })
    });
}
const BranchResetIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBranchResetIcon
    });
});
BranchResetIcon.displayName = 'BranchResetIcon';

function SvgBriefcaseFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M5 4V2.75C5 1.784 5.784 1 6.75 1h2.5c.966 0 1.75.784 1.75 1.75V4h3.25a.75.75 0 0 1 .75.75v9.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75v-9.5A.75.75 0 0 1 1.75 4zm1.5-1.25a.25.25 0 0 1 .25-.25h2.5a.25.25 0 0 1 .25.25V4h-3zm-4 5.423V6.195A7.72 7.72 0 0 0 8 8.485c2.15 0 4.095-.875 5.5-2.29v1.978A9.2 9.2 0 0 1 8 9.985a9.2 9.2 0 0 1-5.5-1.812",
            clipRule: "evenodd"
        })
    });
}
const BriefcaseFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBriefcaseFillIcon
    });
});
BriefcaseFillIcon.displayName = 'BriefcaseFillIcon';

function SvgBriefcaseIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.75 4H5V2.75C5 1.784 5.784 1 6.75 1h2.5c.966 0 1.75.784 1.75 1.75V4h3.25a.75.75 0 0 1 .75.75v9.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75v-9.5A.75.75 0 0 1 1.75 4m5-1.5a.25.25 0 0 0-.25.25V4h3V2.75a.25.25 0 0 0-.25-.25zM2.5 8.173V13.5h11V8.173A9.2 9.2 0 0 1 8 9.985a9.2 9.2 0 0 1-5.5-1.812m0-1.978A7.72 7.72 0 0 0 8 8.485c2.15 0 4.095-.875 5.5-2.29V5.5h-11z",
            clipRule: "evenodd"
        })
    });
}
const BriefcaseIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBriefcaseIcon
    });
});
BriefcaseIcon.displayName = 'BriefcaseIcon';

function SvgBrushIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M11.563 1.377a1.75 1.75 0 0 1 2.474 0l.586.586a1.75 1.75 0 0 1 0 2.475l-6.875 6.874A3.75 3.75 0 0 1 4 15H.75a.751.751 0 0 1-.61-1.185l.668-.936c.287-.402.442-.885.442-1.38A3.25 3.25 0 0 1 4.5 8.25h.19zM4.499 9.75A1.75 1.75 0 0 0 2.75 11.5c0 .706-.193 1.398-.557 2H4a2.25 2.25 0 0 0 2.246-2.193L4.69 9.75zm8.478-7.312a.25.25 0 0 0-.354 0L6.061 9l.94.94 6.562-6.563a.25.25 0 0 0 0-.353z"
        })
    });
}
const BrushIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBrushIcon
    });
});
BrushIcon.displayName = 'BrushIcon';

function SvgBugIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M5.5 5a2.5 2.5 0 1 1 4.792 1H5.708A2.5 2.5 0 0 1 5.5 5M4.13 6.017a4 4 0 1 1 7.74 0l.047.065L14 4l1.06 1.06-2.41 2.412q.178.493.268 1.028H16V10h-3.02a6 6 0 0 1-.33 1.528l2.41 2.412L14 15l-2.082-2.082C11.002 14.187 9.588 15 8 15c-1.587 0-3.002-.813-3.918-2.082L2 15 .94 13.94l2.41-2.412A6 6 0 0 1 3.02 10H0V8.5h3.082q.09-.535.269-1.028L.939 5.061 2 4l2.082 2.081zm.812 1.538A4.4 4.4 0 0 0 4.5 9.5c0 2.347 1.698 4 3.5 4s3.5-1.653 3.5-4c0-.713-.163-1.375-.442-1.945z",
            clipRule: "evenodd"
        })
    });
}
const BugIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgBugIcon
    });
});
BugIcon.displayName = 'BugIcon';

function SvgCalendarClockIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M4.5 0v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75H6v-1.5H2.5V7H15V2.75a.75.75 0 0 0-.75-.75H11.5V0H10v2H6V0zm9 5.5v-2h-11v2z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M10.25 10.5V12c0 .199.079.39.22.53l1 1 1.06-1.06-.78-.78V10.5z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M7 12a4 4 0 1 1 8 0 4 4 0 0 1-8 0m4-2.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5",
                clipRule: "evenodd"
            })
        ]
    });
}
const CalendarClockIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCalendarClockIcon
    });
});
CalendarClockIcon.displayName = 'CalendarClockIcon';

function SvgCalendarEventIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8.5 10.25a1.75 1.75 0 1 1 3.5 0 1.75 1.75 0 0 1-3.5 0"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M10 2H6V0H4.5v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H11.5V0H10zM2.5 3.5v2h11v-2zm0 10V7h11v6.5z",
                clipRule: "evenodd"
            })
        ]
    });
}
const CalendarEventIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCalendarEventIcon
    });
});
CalendarEventIcon.displayName = 'CalendarEventIcon';

function SvgCalendarIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M4.5 0v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H11.5V0H10v2H6V0zm9 3.5v2h-11v-2zM2.5 7v6.5h11V7z",
            clipRule: "evenodd"
        })
    });
}
const CalendarIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCalendarIcon
    });
});
CalendarIcon.displayName = 'CalendarIcon';

function SvgCalendarRangeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#CalendarRangeIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M6 2h4V0h1.5v2h2.75a.75.75 0 0 1 .75.75V8.5h-1.5V7h-11v6.5H8V15H1.75a.75.75 0 0 1-.75-.75V2.75A.75.75 0 0 1 1.75 2H4.5V0H6zM2.5 5.5h11v-2h-11z",
                        clipRule: "evenodd"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M10.47 9.47 7.94 12l2.53 2.53 1.06-1.06-.72-.72h2.38l-.72.72 1.06 1.06L16.06 12l-2.53-2.53-1.06 1.06.72.72h-2.38l.72-.72z"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const CalendarRangeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCalendarRangeIcon
    });
});
CalendarRangeIcon.displayName = 'CalendarRangeIcon';

function SvgCalendarSyncIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M15.59 11.991c0 1.58-1.046 2.878-2.385 3.358-1.125.403-2.441.232-3.56-.732V16h-1.18v-3.638h3.792v1.218H10.29c.84.793 1.774.887 2.525.618.923-.33 1.594-1.207 1.594-2.207zM6.75 2h4V0h1.5v2H15a.75.75 0 0 1 .75.75v4.24h-8V7h-4.5v6.5h4.5V15H2.5a.75.75 0 0 1-.75-.75V2.75A.75.75 0 0 1 2.5 2h2.75V0h1.5zm8.84 9.138H11.8V9.92h1.967c-.84-.793-1.776-.887-2.527-.618-.923.33-1.593 1.207-1.593 2.207H8.465c0-1.58 1.048-2.878 2.387-3.358 1.124-.403 2.441-.232 3.56.732V7.5h1.179zM3.25 5.5h11v-2h-11z"
        })
    });
}
const CalendarSyncIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCalendarSyncIcon
    });
});
CalendarSyncIcon.displayName = 'CalendarSyncIcon';

function SvgCameraIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M10.255 1.878c.42.052.803.28 1.049.633l.263.38h1.495a2.5 2.5 0 0 1 2.5 2.5v6.24a2.5 2.5 0 0 1-2.5 2.5h-9.1l-.255-.012a2.5 2.5 0 0 1-2.244-2.487V5.39a2.5 2.5 0 0 1 2.5-2.5H5.48l.378-.468a1.5 1.5 0 0 1 1.166-.556h3.047zM6.646 3.835l-.45.556H3.963a1 1 0 0 0-1 1v6.24a1 1 0 0 0 1 1h9.099a1 1 0 0 0 1-1v-6.24a1 1 0 0 0-1-1h-2.279l-.447-.644-.264-.38H7.025zM8.5 4.894a3.5 3.5 0 1 1 0 7 3.5 3.5 0 0 1 0-7m0 1.5a2 2 0 1 0 0 4 2 2 0 0 0 0-4"
        })
    });
}
const CameraIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCameraIcon
    });
});
CameraIcon.displayName = 'CameraIcon';

function SvgCaretDownSquareIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 10a.75.75 0 0 1-.59-.286l-2.164-2.75a.75.75 0 0 1 .589-1.214h4.33a.75.75 0 0 1 .59 1.214l-2.166 2.75A.75.75 0 0 1 8 10"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z",
                clipRule: "evenodd"
            })
        ]
    });
}
const CaretDownSquareIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCaretDownSquareIcon
    });
});
CaretDownSquareIcon.displayName = 'CaretDownSquareIcon';

function SvgCaretUpSquareIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 5.75a.75.75 0 0 1 .59.286l2.164 2.75A.75.75 0 0 1 10.165 10h-4.33a.75.75 0 0 1-.59-1.214l2.166-2.75A.75.75 0 0 1 8 5.75"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z",
                clipRule: "evenodd"
            })
        ]
    });
}
const CaretUpSquareIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCaretUpSquareIcon
    });
});
CaretUpSquareIcon.displayName = 'CaretUpSquareIcon';

function SvgCatalogCloudIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M2.5 13.25V4.792c.306.134.644.208 1 .208h8v1H13V.75a.75.75 0 0 0-.75-.75H3.5A2.5 2.5 0 0 0 1 2.5v10.75A2.75 2.75 0 0 0 3.75 16H4v-1.5h-.25c-.69 0-1.25-.56-1.25-1.25m9-9.75h-8a1 1 0 0 1 0-2h8z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M10.179 7a3.61 3.61 0 0 0-3.464 2.595 3.251 3.251 0 0 0 .443 6.387.8.8 0 0 0 .163.018h5.821C14.758 16 16 14.688 16 13.107c0-1.368-.931-2.535-2.229-2.824A3.61 3.61 0 0 0 10.18 7m-2.805 7.496q.023 0 .044.004h5.555a1 1 0 0 1 .1-.002l.07.002c.753 0 1.357-.607 1.357-1.393s-.604-1.393-1.357-1.393h-.107a.75.75 0 0 1-.75-.75v-.357a2.107 2.107 0 0 0-4.199-.26.75.75 0 0 1-.698.656 1.75 1.75 0 0 0-.015 3.493",
                clipRule: "evenodd"
            })
        ]
    });
}
const CatalogCloudIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCatalogCloudIcon
    });
});
CatalogCloudIcon.displayName = 'CatalogCloudIcon';

function SvgCatalogGearIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                fillRule: "evenodd",
                clipPath: "url(#CatalogGearIcon_svg__a)",
                clipRule: "evenodd",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M14 7.5V.75a.75.75 0 0 0-.75-.75H4.5A2.5 2.5 0 0 0 2 2.5v10.75A2.75 2.75 0 0 0 4.75 16H8v-1.5H4.75c-.69 0-1.25-.56-1.25-1.25V4.792c.306.134.644.208 1 .208h8v2.5zm-9.5-4a1 1 0 0 1 0-2h8v2z"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M13.125 10.081q.364.114.673.325l.88-.703.936 1.173-.88.702q.136.344.166.729l1.098.25-.334 1.463-1.098-.25a2.6 2.6 0 0 1-.466.584l.49 1.014-1.352.651-.489-1.014a2.6 2.6 0 0 1-.748 0l-.488 1.014-1.351-.65.488-1.015a2.6 2.6 0 0 1-.466-.584l-1.098.25-.334-1.462 1.098-.25q.031-.385.166-.73l-.88-.702.935-1.172.88.702q.31-.211.674-.325V8.955h1.5zm.263 2.42a1.013 1.013 0 1 1-2.026 0 1.013 1.013 0 0 1 2.026 0"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const CatalogGearIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCatalogGearIcon
    });
});
CatalogGearIcon.displayName = 'CatalogGearIcon';

function SvgCatalogHomeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M14 6.5V.75a.75.75 0 0 0-.75-.75H4.5A2.5 2.5 0 0 0 2 2.5v10.75A2.75 2.75 0 0 0 4.75 16H6.5v-1.5H4.75c-.69 0-1.25-.56-1.25-1.25V4.792c.306.134.644.208 1 .208h8v1.5zm-9.5-3a1 1 0 0 1 0-2h8v2z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12.457 7.906a.75.75 0 0 0-.914 0l-3.25 2.5A.75.75 0 0 0 8 11v4.25c0 .414.336.75.75.75h6.5a.75.75 0 0 0 .75-.75V11a.75.75 0 0 0-.293-.594zM9.5 14.5v-3.13L12 9.445l2.5 1.923V14.5h-1.75V12h-1.5v2.5z",
                clipRule: "evenodd"
            })
        ]
    });
}
const CatalogHomeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCatalogHomeIcon
    });
});
CatalogHomeIcon.displayName = 'CatalogHomeIcon';

function SvgCatalogIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M14 .75a.75.75 0 0 0-.75-.75H4.5A2.5 2.5 0 0 0 2 2.5v10.75A2.75 2.75 0 0 0 4.75 16h8.5a.75.75 0 0 0 .75-.75zM3.5 4.792v8.458c0 .69.56 1.25 1.25 1.25h7.75V5h-8c-.356 0-.694-.074-1-.208m9-1.292v-2h-8a1 1 0 0 0 0 2z",
            clipRule: "evenodd"
        })
    });
}
const CatalogIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCatalogIcon
    });
});
CatalogIcon.displayName = 'CatalogIcon';

function SvgCatalogOffIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#CatalogOffIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "m14 11.94-1.5-1.5V5H7.061l-1.5-1.5h6.94v-2h-8c-.261 0-.499.1-.677.263L2.764.703A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75z"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M2 4.06.47 2.53l1.06-1.06 13.5 13.5-1.06 1.06-.03-.03H4.75A2.75 2.75 0 0 1 2 13.25zm1.5 1.5v7.69c0 .69.56 1.25 1.25 1.25h7.69z",
                        clipRule: "evenodd"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const CatalogOffIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCatalogOffIcon
    });
});
CatalogOffIcon.displayName = 'CatalogOffIcon';

function SvgCatalogSharedIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M4.5 5c-.356 0-.694-.074-1-.208v8.458c0 .69.56 1.25 1.25 1.25H10V16H4.75A2.75 2.75 0 0 1 2 13.25V2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75zm0-1.5a1 1 0 0 1 0-2h8v2z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M14 6.5a2 2 0 0 0-1.953 2.433l-.944.648a2 2 0 1 0 .105 3.262l.858.644a2 2 0 1 0 .9-1.2l-.988-.74a2 2 0 0 0-.025-.73l.944-.649A2 2 0 1 0 14 6.5m-.5 2a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0m-4 2.75a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0M14 13.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1",
                clipRule: "evenodd"
            })
        ]
    });
}
const CatalogSharedIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCatalogSharedIcon
    });
});
CatalogSharedIcon.displayName = 'CatalogSharedIcon';

function SvgCellsSquareIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75v4.75h4.75V2.5zm6.25 0v4.75h4.75V2.5zm-1.5 6.25H2.5v4.75h4.75zm1.5 4.75V8.75h4.75v4.75z",
            clipRule: "evenodd"
        })
    });
}
const CellsSquareIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCellsSquareIcon
    });
});
CellsSquareIcon.displayName = 'CellsSquareIcon';

function SvgCertifiedFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8 1c.682 0 1.283.342 1.644.862a1.998 1.998 0 0 1 2.848 1.645 1.997 1.997 0 0 1 1.644 2.847 1.997 1.997 0 0 1 .001 3.29 1.997 1.997 0 0 1-1.645 2.848 1.997 1.997 0 0 1-2.848 1.645 1.996 1.996 0 0 1-3.288 0 1.997 1.997 0 0 1-2.85-1.645 1.997 1.997 0 0 1-1.643-2.848 1.996 1.996 0 0 1 0-3.289 1.997 1.997 0 0 1 1.644-2.848 1.998 1.998 0 0 1 2.849-1.645C6.716 1.342 7.319 1 8 1m-.81 7.252L6.146 7.206 5 8.351l2.19 2.19L11 6.731 9.856 5.587z"
        })
    });
}
const CertifiedFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCertifiedFillIcon
    });
});
CertifiedFillIcon.displayName = 'CertifiedFillIcon';

function SvgCertifiedFillSmallIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8 3c.71 0 1.33.37 1.686.928a1.997 1.997 0 0 1 2.385 2.385 1.996 1.996 0 0 1 0 3.373 1.996 1.996 0 0 1-2.385 2.385 1.996 1.996 0 0 1-3.373 0 1.996 1.996 0 0 1-2.385-2.385 1.996 1.996 0 0 1 0-3.373 1.997 1.997 0 0 1 2.385-2.385A2 2 0 0 1 8 3m-.675 5.22-.87-.871-.955.954 1.825 1.825L10.5 6.954 9.546 6z"
        })
    });
}
const CertifiedFillSmallIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCertifiedFillSmallIcon
    });
});
CertifiedFillSmallIcon.displayName = 'CertifiedFillSmallIcon';

function SvgCertifiedIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#CertifiedIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M14.5 8a1.5 1.5 0 0 0-.8-1.327l-1.096-.58.364-1.186c.14-.455.06-.956-.233-1.344l-.138-.16a1.5 1.5 0 0 0-1.505-.372l-1.185.365-.58-1.096a1.499 1.499 0 0 0-2.654 0l-.58 1.096-1.186-.365a1.5 1.5 0 0 0-1.344.234l-.16.138a1.5 1.5 0 0 0-.372 1.505l.365 1.185-1.096.58a1.5 1.5 0 0 0-.785 1.116L1.5 8c0 .572.32 1.072.8 1.326l1.096.581-.365 1.185a1.5 1.5 0 0 0 .372 1.505l.16.138c.388.293.89.373 1.344.233l1.186-.364.58 1.095c.254.48.755.801 1.327.801V16a3 3 0 0 1-2.652-1.599 2.998 2.998 0 0 1-3.75-3.75 2.999 2.999 0 0 1 0-5.302 3 3 0 0 1 3.75-3.751 3 3 0 0 1 5.303 0 3 3 0 0 1 3.75 3.75 2.999 2.999 0 0 1 0 5.303 2.999 2.999 0 0 1-3.75 3.75A3 3 0 0 1 8 16v-1.5c.57 0 1.07-.32 1.325-.8l.581-1.098 1.186.366a1.5 1.5 0 0 0 1.505-.371l.138-.16c.293-.388.373-.89.233-1.344l-.366-1.187 1.097-.58c.48-.254.801-.754.801-1.326"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M9.856 5.587 11 6.732 7.19 10.54 5 8.35l1.144-1.145 1.047 1.046z"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const CertifiedIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCertifiedIcon
    });
});
CertifiedIcon.displayName = 'CertifiedIcon';

function SvgChainIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m6.144 12.331.972-.972 1.06 1.06-.971.973a3.625 3.625 0 1 1-5.127-5.127l2.121-2.121A3.625 3.625 0 0 1 10.32 8H8.766a2.125 2.125 0 0 0-3.507-.795l-2.121 2.12a2.125 2.125 0 0 0 3.005 3.006"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m9.856 3.669-.972.972-1.06-1.06.971-.973a3.625 3.625 0 1 1 5.127 5.127l-2.121 2.121A3.625 3.625 0 0 1 5.68 8h1.552a2.125 2.125 0 0 0 3.507.795l2.121-2.12a2.125 2.125 0 0 0-3.005-3.006"
            })
        ]
    });
}
const ChainIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChainIcon
    });
});
ChainIcon.displayName = 'ChainIcon';

function SvgChartLineIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M1 1v13.25c0 .414.336.75.75.75H15v-1.5H2.5V1z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m15.03 5.03-1.06-1.06L9.5 8.44 7 5.94 3.47 9.47l1.06 1.06L7 8.06l2.5 2.5z"
            })
        ]
    });
}
const ChartLineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChartLineIcon
    });
});
ChartLineIcon.displayName = 'ChartLineIcon';

function SvgCheckCircleBadgeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m10.47 5.47 1.06 1.06L7 11.06 4.47 8.53l1.06-1.06L7 8.94zM16 12.5a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M1.5 8a6.5 6.5 0 0 1 13-.084c.54.236 1.031.565 1.452.967Q16 8.448 16 8a8 8 0 1 0-7.117 7.952 5 5 0 0 1-.967-1.453A6.5 6.5 0 0 1 1.5 8"
            })
        ]
    });
}
const CheckCircleBadgeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCheckCircleBadgeIcon
    });
});
CheckCircleBadgeIcon.displayName = 'CheckCircleBadgeIcon';

function SvgCheckCircleFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m11.53-1.47-1.06-1.06L7 8.94 5.53 7.47 4.47 8.53l2 2 .53.53.53-.53z",
            clipRule: "evenodd"
        })
    });
}
const CheckCircleFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCheckCircleFillIcon
    });
});
CheckCircleFillIcon.displayName = 'CheckCircleFillIcon';

function SvgCheckCircleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M11.53 6.53 7 11.06 4.47 8.53l1.06-1.06L7 8.94l3.47-3.47z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13",
                clipRule: "evenodd"
            })
        ]
    });
}
const CheckCircleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCheckCircleIcon
    });
});
CheckCircleIcon.displayName = 'CheckCircleIcon';

function SvgCheckLineIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M15.06 2.06 14 1 5.53 9.47 2.06 6 1 7.06l4.53 4.531zM1.03 15.03h14v-1.5h-14z",
            clipRule: "evenodd"
        })
    });
}
const CheckLineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCheckLineIcon
    });
});
CheckLineIcon.displayName = 'CheckLineIcon';

function SvgCheckSmallIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M12.03 6.03 7 11.06 3.97 8.03l1.06-1.06L7 8.94l3.97-3.97z",
            clipRule: "evenodd"
        })
    });
}
const CheckSmallIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCheckSmallIcon
    });
});
CheckSmallIcon.displayName = 'CheckSmallIcon';

function SvgCheckboxIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M1.75 2a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h11.5a.75.75 0 0 0 .75-.75V9h-1.5v4.5h-10v-10H10V2z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m15.03 4.03-1.06-1.06L7.5 9.44 5.53 7.47 4.47 8.53l3.03 3.03z"
            })
        ]
    });
}
const CheckboxIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCheckboxIcon
    });
});
CheckboxIcon.displayName = 'CheckboxIcon';

function SvgChecklistIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "m5.5 2 1.06 1.06-3.53 3.531L1 4.561 2.06 3.5l.97.97zM15.03 4.53h-7v-1.5h7zM1.03 14.53v-1.5h14v1.5zM8.03 9.53h7v-1.5h-7zM6.56 8.06 5.5 7 3.03 9.47l-.97-.97L1 9.56l2.03 2.031z"
        })
    });
}
const ChecklistIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChecklistIcon
    });
});
ChecklistIcon.displayName = 'ChecklistIcon';

function SvgChevronDoubleDownIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M10.947 7.954 8 10.891 5.056 7.954 3.997 9.016l4.004 3.993 4.005-3.993z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M10.947 3.994 8 6.931 5.056 3.994 3.997 5.056 8.001 9.05l4.005-3.993z"
            })
        ]
    });
}
const ChevronDoubleDownIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChevronDoubleDownIcon
    });
});
ChevronDoubleDownIcon.displayName = 'ChevronDoubleDownIcon';

function SvgChevronDoubleLeftIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8.047 10.944 5.11 8l2.937-2.944-1.062-1.06L2.991 8l3.994 4.003z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M12.008 10.944 9.07 8l2.938-2.944-1.062-1.06L6.952 8l3.994 4.003z"
            })
        ]
    });
}
const ChevronDoubleLeftIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChevronDoubleLeftIcon
    });
});
ChevronDoubleLeftIcon.displayName = 'ChevronDoubleLeftIcon';

function SvgChevronDoubleLeftOffIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "m2.5 1.5 12 12-1 1-7.47-7.47-.94.94 2.97 2.97L7 12 2.97 7.97l2-2L1.5 2.5zM12.06 5l-1.97 1.97-1.06-1.06L11 3.94z"
        })
    });
}
const ChevronDoubleLeftOffIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChevronDoubleLeftOffIcon
    });
});
ChevronDoubleLeftOffIcon.displayName = 'ChevronDoubleLeftOffIcon';

function SvgChevronDoubleRightIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m7.954 5.056 2.937 2.946-2.937 2.945 1.062 1.059 3.993-4.004-3.993-4.005z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m3.994 5.056 2.937 2.946-2.937 2.945 1.062 1.059L9.05 8.002 5.056 3.997z"
            })
        ]
    });
}
const ChevronDoubleRightIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChevronDoubleRightIcon
    });
});
ChevronDoubleRightIcon.displayName = 'ChevronDoubleRightIcon';

function SvgChevronDoubleRightOffIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "m2.5 1.5 12 12-1 1-3.47-3.47-.97.97L8 10.94l.97-.97-.94-.94L5.06 12 4 10.94l2.97-2.97L1.5 2.5zM13.09 7.97l-1 1-4.03-4.03 1-1z"
        })
    });
}
const ChevronDoubleRightOffIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChevronDoubleRightOffIcon
    });
});
ChevronDoubleRightOffIcon.displayName = 'ChevronDoubleRightOffIcon';

function SvgChevronDoubleUpIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M5.056 8.047 8 5.11l2.944 2.937 1.06-1.062L8 2.991 3.997 6.985z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M5.056 12.008 8 9.07l2.944 2.937 1.06-1.062L8 6.952l-4.003 3.994z"
            })
        ]
    });
}
const ChevronDoubleUpIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChevronDoubleUpIcon
    });
});
ChevronDoubleUpIcon.displayName = 'ChevronDoubleUpIcon';

function SvgChevronDownIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 8.917 10.947 6 12 7.042 8 11 4 7.042 5.053 6z",
            clipRule: "evenodd"
        })
    });
}
const ChevronDownIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChevronDownIcon
    });
});
ChevronDownIcon.displayName = 'ChevronDownIcon';

function SvgChevronUpIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 7.083 5.053 10 4 8.958 8 5l4 3.958L10.947 10z",
            clipRule: "evenodd"
        })
    });
}
const ChevronUpIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChevronUpIcon
    });
});
ChevronUpIcon.displayName = 'ChevronUpIcon';

function SvgChipIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8.006 3h1.488V1.5h1.5V3h1.956l.082.004a.8.8 0 0 1 .718.796v1.956h1.483v1.5H13.75v1.488h1.483v1.5H13.75V12.2l-.004.082a.8.8 0 0 1-.714.714L12.95 13h-1.956v1.5h-1.5V13H8.006v1.5h-1.5V13H4.55l-.082-.004a.8.8 0 0 1-.714-.714L3.75 12.2v-1.956H2.268v-1.5H3.75V7.256H2.268v-1.5H3.75V3.8a.8.8 0 0 1 .8-.8h1.956V1.5h1.5zM5.25 11.5h7v-7h-7zm5.498-1.5h-4V6h4z"
        })
    });
}
const ChipIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgChipIcon
    });
});
ChipIcon.displayName = 'ChipIcon';

function SvgCircleIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M12.5 8a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0"
        })
    });
}
const CircleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCircleIcon
    });
});
CircleIcon.displayName = 'CircleIcon';

function SvgCircleOffIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m11.667 5.392 1.362-1.363-1.06-1.06-9 9 1.06 1.06 1.363-1.362a4.5 4.5 0 0 0 6.276-6.276m-1.083 1.083-4.11 4.109a3 3 0 0 0 4.11-4.11M8 3.5q.606.002 1.164.152L7.811 5.006A3 3 0 0 0 5.006 7.81L3.652 9.164A4.5 4.5 0 0 1 8 3.5",
            clipRule: "evenodd"
        })
    });
}
const CircleOffIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCircleOffIcon
    });
});
CircleOffIcon.displayName = 'CircleOffIcon';

function SvgCircleOutlineIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M5 8a3 3 0 1 0 6 0 3 3 0 0 0-6 0m3-4.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9",
            clipRule: "evenodd"
        })
    });
}
const CircleOutlineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCircleOutlineIcon
    });
});
CircleOutlineIcon.displayName = 'CircleOutlineIcon';

function SvgClipboardIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M5.5 0a.75.75 0 0 0-.75.75V1h-2a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75h-2V.75A.75.75 0 0 0 10.5 0zm5.75 2.5v.75a.75.75 0 0 1-.75.75h-5a.75.75 0 0 1-.75-.75V2.5H3.5v11h9v-11zm-5 0v-1h3.5v1z",
            clipRule: "evenodd"
        })
    });
}
const ClipboardIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgClipboardIcon
    });
});
ClipboardIcon.displayName = 'ClipboardIcon';

function SvgClockKeyIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 1.5a6.5 6.5 0 0 0-5.07 10.57l-1.065 1.065A8 8 0 1 1 15.418 11h-1.65A6.5 6.5 0 0 0 8 1.5"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.25 8V4h1.5v3.25H11v1.5H8A.75.75 0 0 1 7.25 8"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M4 13a3 3 0 0 1 5.959-.5h4.291a.75.75 0 0 1 .75.75V16h-1.5v-2h-1v2H11v-2H9.83A3.001 3.001 0 0 1 4 13m3-1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3",
                clipRule: "evenodd"
            })
        ]
    });
}
const ClockKeyIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgClockKeyIcon
    });
});
ClockKeyIcon.displayName = 'ClockKeyIcon';

function SvgClockOffIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 0a8 8 0 0 1 7.944 8.929 5.005 5.005 0 0 0-1.454-1.265A6.5 6.5 0 0 0 1.5 8a6.5 6.5 0 0 0 6.164 6.49 5 5 0 0 0 1.265 1.454A8 8 0 1 1 8 0"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8.75 8a.75.75 0 0 1-.22.53l-2 2-1.06-1.06 1.78-1.78V3h1.5z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12 8a4 4 0 1 1 0 8 4 4 0 0 1 0-8m0 2.94-1.125-1.126-1.06 1.061L10.94 12l-1.126 1.125 1.061 1.06L12 13.06l1.125 1.125 1.06-1.06L13.06 12l1.125-1.125-1.06-1.06z",
                clipRule: "evenodd"
            })
        ]
    });
}
const ClockOffIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgClockOffIcon
    });
});
ClockOffIcon.displayName = 'ClockOffIcon';

function SvgCloseSmallIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M7.064 8 4 4.936 4.936 4 8 7.064 11.063 4l.937.936L8.937 8 12 11.063l-.937.937L8 8.937 4.936 12 4 11.063z",
            clipRule: "evenodd"
        })
    });
}
const CloseSmallIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCloseSmallIcon
    });
});
CloseSmallIcon.displayName = 'CloseSmallIcon';

function SvgCloudDatabaseIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M3.394 4.586a4.752 4.752 0 0 1 9.351.946A3.75 3.75 0 0 1 15.787 8H14.12a2.25 2.25 0 0 0-1.871-1H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.475-.402.75.75 0 0 1-.698.657A2.75 2.75 0 0 0 4 11.49V13a.8.8 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M6.25 10.5c0-.851.67-1.42 1.293-1.731C8.211 8.435 9.08 8.25 10 8.25s1.79.185 2.457.519c.622.31 1.293.88 1.293 1.731v2.277c-.014.836-.677 1.397-1.293 1.705-.668.333-1.537.518-2.457.518s-1.79-.185-2.457-.518c-.616-.308-1.279-.869-1.293-1.705V10.5m1.964 2.64c.418.209 1.049.36 1.786.36s1.368-.151 1.786-.36c.209-.105.337-.21.406-.29a.3.3 0 0 0 .057-.096l.001-.004v-.423c-.636.273-1.423.423-2.25.423s-1.614-.15-2.25-.423v.427l.005.014a.3.3 0 0 0 .053.082c.069.08.197.185.406.29M7.75 10.5v-.004l.005-.014a.3.3 0 0 1 .053-.082c.069-.08.197-.185.406-.29.418-.209 1.049-.36 1.786-.36s1.368.151 1.786.36c.209.105.337.21.406.29a.3.3 0 0 1 .057.096l.001.004v.004l-.005.014a.3.3 0 0 1-.053.082c-.069.08-.197.185-.406.29-.418.209-1.049.36-1.786.36s-1.368-.151-1.786-.36a1.3 1.3 0 0 1-.406-.29.3.3 0 0 1-.058-.096z",
                clipRule: "evenodd"
            })
        ]
    });
}
const CloudDatabaseIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCloudDatabaseIcon
    });
});
CloudDatabaseIcon.displayName = 'CloudDatabaseIcon';

function SvgCloudDownloadIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 2a4.75 4.75 0 0 0-4.606 3.586 4.251 4.251 0 0 0 .427 8.393A.8.8 0 0 0 4 14v-1.511a2.75 2.75 0 0 1 .077-5.484.75.75 0 0 0 .697-.657 3.25 3.25 0 0 1 6.476.402v.5c0 .414.336.75.75.75h.25a2.25 2.25 0 1 1-.188 4.492L12 12.49V14l.077-.004q.086.004.173.004a3.75 3.75 0 0 0 .495-7.468A4.75 4.75 0 0 0 8 2"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.25 11.19 5.03 8.97l-1.06 1.06L8 14.06l4.03-4.03-1.06-1.06-2.22 2.22V6h-1.5z"
            })
        ]
    });
}
const CloudDownloadIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCloudDownloadIcon
    });
});
CloudDownloadIcon.displayName = 'CloudDownloadIcon';

function SvgCloudIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M3.394 5.586a4.752 4.752 0 0 1 9.351.946 3.75 3.75 0 0 1-.668 7.464L12 14H4a.8.8 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393m.72 6.914h7.762a.8.8 0 0 1 .186-.008q.092.008.188.008a2.25 2.25 0 0 0 0-4.5H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.475-.402.75.75 0 0 1-.698.657 2.75 2.75 0 0 0-.024 5.488z",
            clipRule: "evenodd"
        })
    });
}
const CloudIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCloudIcon
    });
});
CloudIcon.displayName = 'CloudIcon';

function SvgCloudKeyIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M3.394 5.586a4.752 4.752 0 0 1 9.351.946A3.75 3.75 0 0 1 15.787 9H14.12a2.25 2.25 0 0 0-1.871-1H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.475-.402.75.75 0 0 1-.698.657A2.75 2.75 0 0 0 4 12.49V14a.8.8 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393M15.25 10.5h-4.291a3 3 0 1 0-.13 1.5H12v2h1.5v-2h1v2H16v-2.75a.75.75 0 0 0-.75-.75M8 9.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3",
            clipRule: "evenodd"
        })
    });
}
const CloudKeyIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCloudKeyIcon
    });
});
CloudKeyIcon.displayName = 'CloudKeyIcon';

function SvgCloudModelIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M3.394 5.586a4.752 4.752 0 0 1 9.351.946A3.75 3.75 0 0 1 15.787 9H14.12a2.25 2.25 0 0 0-1.871-1H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.475-.402.75.75 0 0 1-.698.657A2.75 2.75 0 0 0 4 12.49V14a.8.8 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8 7a2.25 2.25 0 0 1 2.03 3.22l.5.5a2.25 2.25 0 1 1-1.06 1.06l-.5-.5A2.25 2.25 0 1 1 8 7m.75 2.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0m3.5 3.5a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0",
                clipRule: "evenodd"
            })
        ]
    });
}
const CloudModelIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCloudModelIcon
    });
});
CloudModelIcon.displayName = 'CloudModelIcon';

function SvgCloudOffIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M13.97 14.53 2.47 3.03l-1 1 1.628 1.628a4.252 4.252 0 0 0 .723 8.32A.8.8 0 0 0 4 14h7.44l1.53 1.53zM4.077 7.005a.75.75 0 0 0 .29-.078L9.939 12.5H4.115l-.062-.007a2.75 2.75 0 0 1 .024-5.488",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M4.8 3.24a4.75 4.75 0 0 1 7.945 3.293 3.75 3.75 0 0 1 1.928 6.58l-1.067-1.067A2.25 2.25 0 0 0 12.25 8H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-5.388-2.448z"
            })
        ]
    });
}
const CloudOffIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCloudOffIcon
    });
});
CloudOffIcon.displayName = 'CloudOffIcon';

function SvgCloudUploadIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 2a4.75 4.75 0 0 0-4.606 3.586 4.251 4.251 0 0 0 .427 8.393A.8.8 0 0 0 4 14v-1.511a2.75 2.75 0 0 1 .077-5.484.75.75 0 0 0 .697-.657 3.25 3.25 0 0 1 6.476.402v.5c0 .414.336.75.75.75h.25a2.25 2.25 0 1 1-.188 4.492L12 12.49V14l.077-.004q.086.004.173.004a3.75 3.75 0 0 0 .495-7.468A4.75 4.75 0 0 0 8 2"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m8.75 8.81 2.22 2.22 1.06-1.06L8 5.94 3.97 9.97l1.06 1.06 2.22-2.22V14h1.5z"
            })
        ]
    });
}
const CloudUploadIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCloudUploadIcon
    });
});
CloudUploadIcon.displayName = 'CloudUploadIcon';

function SvgCodeIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M4.03 12.06 5.091 11l-2.97-2.97 2.97-2.97L4.031 4 0 8.03zM12.091 4l4.03 4.03-4.03 4.03-1.06-1.06L14 8.03l-2.97-2.97z"
        })
    });
}
const CodeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCodeIcon
    });
});
CodeIcon.displayName = 'CodeIcon';

function SvgColorFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M7.5 1v1.59l4.88 4.88a.75.75 0 0 1 0 1.06l-4.242 4.243a2.75 2.75 0 0 1-3.89 0l-2.421-2.422a2.75 2.75 0 0 1 0-3.889L6 2.29V1zM6 8V4.41L2.888 7.524a1.25 1.25 0 0 0 0 1.768l2.421 2.421a1.25 1.25 0 0 0 1.768 0L10.789 8 7.5 4.71V8zm7.27 1.51a.76.76 0 0 0-1.092.001 8.5 8.5 0 0 0-1.216 1.636c-.236.428-.46.953-.51 1.501-.054.576.083 1.197.587 1.701a2.385 2.385 0 0 0 3.372 0c.505-.504.644-1.126.59-1.703-.05-.55-.274-1.075-.511-1.503a8.5 8.5 0 0 0-1.22-1.633m-.995 2.363c.138-.25.3-.487.451-.689.152.201.313.437.452.687.19.342.306.657.33.913.02.228-.03.377-.158.505a.885.885 0 0 1-1.25 0c-.125-.125-.176-.272-.155-.501.024-.256.14-.572.33-.915",
            clipRule: "evenodd"
        })
    });
}
const ColorFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgColorFillIcon
    });
});
ColorFillIcon.displayName = 'ColorFillIcon';

function SvgColumnIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M6.5 9V6h3v3zm3 1.5v3h-3v-3zm1.5-.75v-9a.75.75 0 0 0-.75-.75h-4.5A.75.75 0 0 0 5 .75v13.5c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75zM6.5 4.5v-3h3v3z",
            clipRule: "evenodd"
        })
    });
}
const ColumnIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgColumnIcon
    });
});
ColumnIcon.displayName = 'ColumnIcon';

function SvgColumnSplitIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h4.75v11zm6.25 0h4.75v-11H8.75z",
            clipRule: "evenodd"
        })
    });
}
const ColumnSplitIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgColumnSplitIcon
    });
});
ColumnSplitIcon.displayName = 'ColumnSplitIcon';

function SvgColumnsIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zM2.5 13.5v-11H5v11zm4 0h3v-11h-3zm4.5-11v11h2.5v-11z",
            clipRule: "evenodd"
        })
    });
}
const ColumnsIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgColumnsIcon
    });
});
ColumnsIcon.displayName = 'ColumnsIcon';

function SvgCommandIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M6.75 3.875A2.875 2.875 0 1 0 3.875 6.75H5.25v2.5H3.875a2.875 2.875 0 1 0 2.875 2.875V10.75h2.5v1.375a2.875 2.875 0 1 0 2.875-2.875H10.75v-2.5h1.375A2.875 2.875 0 1 0 9.25 3.875V5.25h-2.5zm0 5.375h2.5v-2.5h-2.5zm-1.5 1.5H3.875a1.375 1.375 0 1 0 1.375 1.375zm0-6.875V5.25H3.875A1.375 1.375 0 1 1 5.25 3.875m5.5 6.875v1.375a1.375 1.375 0 1 0 1.375-1.375zm1.375-5.5H10.75V3.875a1.375 1.375 0 1 1 1.375 1.375",
            clipRule: "evenodd"
        })
    });
}
const CommandIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCommandIcon
    });
});
CommandIcon.displayName = 'CommandIcon';

function SvgCommandPaletteIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            stroke: "currentColor",
            strokeWidth: 0.25,
            d: "M4.316 6.737H5.38v2.526H4.316c-1.346 0-2.441 1.057-2.441 2.415a2.441 2.441 0 1 0 4.882 0V10.62h2.486v1.057a2.441 2.441 0 1 0 4.882 0c0-1.358-1.094-2.415-2.44-2.415h-1.058V6.737h1.057c1.347 0 2.441-1.057 2.441-2.415a2.441 2.441 0 1 0-4.882 0V5.38H6.757V4.322a2.441 2.441 0 1 0-4.882 0c0 1.358 1.095 2.415 2.44 2.415ZM5.38 4.322v1.073H4.316A1.08 1.08 0 0 1 3.25 4.322c0-.585.485-1.064 1.065-1.064s1.065.479 1.065 1.064Zm7.368 0a1.08 1.08 0 0 1-1.065 1.073h-1.057V4.322c0-.587.479-1.064 1.057-1.064.58 0 1.066.479 1.066 1.064Zm-3.506 2.4v2.557H6.757V6.72zM3.251 11.67a1.08 1.08 0 0 1 1.065-1.073H5.38v1.073c0 .585-.486 1.064-1.065 1.064-.58 0-1.065-.479-1.065-1.064Zm7.376 0v-1.073h1.057a1.08 1.08 0 0 1 1.066 1.073c0 .585-.486 1.064-1.066 1.064a1.065 1.065 0 0 1-1.057-1.064Z"
        })
    });
}
const CommandPaletteIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCommandPaletteIcon
    });
});
CommandPaletteIcon.displayName = 'CommandPaletteIcon';

function SvgConnectIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M7.78 3.97 5.03 1.22a.75.75 0 0 0-1.06 0L1.22 3.97a.75.75 0 0 0 0 1.06l2.75 2.75a.75.75 0 0 0 1.06 0l2.75-2.75a.75.75 0 0 0 0-1.06m-1.59.53L4.5 6.19 2.81 4.5 4.5 2.81zM15 11.75a3.25 3.25 0 1 0-6.5 0 3.25 3.25 0 0 0 6.5 0M11.75 10a1.75 1.75 0 1 1 0 3.5 1.75 1.75 0 0 1 0-3.5",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M14.25 1H9v1.5h4.5V7H15V1.75a.75.75 0 0 0-.75-.75M1 9v5.25c0 .414.336.75.75.75H7v-1.5H2.5V9z"
            })
        ]
    });
}
const ConnectIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgConnectIcon
    });
});
ConnectIcon.displayName = 'ConnectIcon';

function SvgCopyIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.75 1a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75H5v3.25c0 .414.336.75.75.75h8.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75H11V1.75a.75.75 0 0 0-.75-.75zM9.5 5V2.5h-7v7H5V5.75A.75.75 0 0 1 5.75 5zm-3 8.5v-7h7v7z",
            clipRule: "evenodd"
        })
    });
}
const CopyIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCopyIcon
    });
});
CopyIcon.displayName = 'CopyIcon';

function SvgCreditCardIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M13 9H9v1.5h4z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 2A1.75 1.75 0 0 0 0 3.75v8.5C0 13.216.784 14 1.75 14h12.5A1.75 1.75 0 0 0 16 12.25v-8.5A1.75 1.75 0 0 0 14.25 2zM1.5 3.75a.25.25 0 0 1 .25-.25h12.5a.25.25 0 0 1 .25.25V5.5h-13zM1.5 7h13v5.25a.25.25 0 0 1-.25.25H1.75a.25.25 0 0 1-.25-.25z",
                clipRule: "evenodd"
            })
        ]
    });
}
const CreditCardIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCreditCardIcon
    });
});
CreditCardIcon.displayName = 'CreditCardIcon';

function SvgCursorClickIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M5.22 5.22a.75.75 0 0 1 .806-.167l9.5 3.761a.75.75 0 0 1-.077 1.421l-4.09 1.124-1.124 4.09a.75.75 0 0 1-1.42.077l-3.762-9.5a.75.75 0 0 1 .167-.806m4.164 7.668.643-2.337.032-.093a.75.75 0 0 1 .492-.43l2.337-.644-5.803-2.299z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M3.516 7.837.744 8.985.17 7.6 2.94 6.45zM3.519 4.156l-.574 1.386-2.771-1.15.573-1.385zM5.545 2.941l-1.386.575L3.012.744 4.397.17zM8.99.74 7.84 3.512l-1.385-.574L7.603.166z"
            })
        ]
    });
}
const CursorClickIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCursorClickIcon
    });
});
CursorClickIcon.displayName = 'CursorClickIcon';

function SvgCursorIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#CursorIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    fillRule: "evenodd",
                    d: "M1.22 1.22a.75.75 0 0 1 .802-.169l13.5 5.25a.75.75 0 0 1-.043 1.413L9.597 9.597l-1.883 5.882a.75.75 0 0 1-1.413.043l-5.25-13.5a.75.75 0 0 1 .169-.802m1.847 1.847 3.864 9.937 1.355-4.233a.75.75 0 0 1 .485-.485l4.233-1.355z",
                    clipRule: "evenodd"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M16 0H0v16h16z"
                    })
                })
            })
        ]
    });
}
const CursorIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCursorIcon
    });
});
CursorIcon.displayName = 'CursorIcon';

function SvgCursorTypeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 3.75h1c.69 0 1.25.56 1.25 1.25v6c0 .69-.56 1.25-1.25 1.25H8v1.5h1c.788 0 1.499-.331 2-.863a2.74 2.74 0 0 0 2 .863h1v-1.5h-1c-.69 0-1.25-.56-1.25-1.25V5c0-.69.56-1.25 1.25-1.25h1v-1.5h-1c-.788 0-1.499.331-2 .863a2.74 2.74 0 0 0-2-.863H8z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M5.936 8.003 3 5.058 4.062 4l3.993 4.004-3.993 4.005L3 10.948z"
            })
        ]
    });
}
const CursorTypeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCursorTypeIcon
    });
});
CursorTypeIcon.displayName = 'CursorTypeIcon';

function SvgCustomAppIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.751 4a.75.75 0 0 0-.583 1.222L2.81 7.25H.751a.75.75 0 0 0-.583 1.222l4.25 5.25a.75.75 0 0 0 .583.278h9.25a.75.75 0 0 0 .597-1.204L13.29 10.75h1.961a.75.75 0 0 0 .583-1.222l-2.237-2.764c-.364.345-.786.63-1.25.84L13.68 9.25H6.36L3.324 5.5H6.47A4.5 4.5 0 0 1 6.03 4zm3.667 6.472a.75.75 0 0 0 .583.278h5.405l1.332 1.75h-7.38L2.325 8.75H4v-.03z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M10.501 0a.875.875 0 0 0-.862.725l-.178 1.023a.875.875 0 0 1-.712.712l-1.023.178a.875.875 0 0 0 0 1.724l1.023.178a.875.875 0 0 1 .712.712l.178 1.023a.875.875 0 0 0 1.725 0l.177-1.023a.875.875 0 0 1 .712-.712l1.023-.178a.876.876 0 0 0 0-1.724l-1.023-.178a.875.875 0 0 1-.712-.712L11.364.725A.875.875 0 0 0 10.5 0"
            })
        ]
    });
}
const CustomAppIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgCustomAppIcon
    });
});
CustomAppIcon.displayName = 'CustomAppIcon';

function SvgDagHorizontalIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M9 2.75A.75.75 0 0 1 9.75 2h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5a.75.75 0 0 1-.748-.692L7.311 8l1.691 1.692A.75.75 0 0 1 9.75 9h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5a.75.75 0 0 1-.75-.75v-1.44L6.998 9.809a.75.75 0 0 1-.748.692H.75A.75.75 0 0 1 0 9.75v-3.5a.75.75 0 0 1 .75-.75h5.5a.75.75 0 0 1 .748.692L9 4.189zm1.5 2.75h4v-2h-4zM5.5 7h-4v2h4zm5 3.5v2h4v-2z",
            clipRule: "evenodd"
        })
    });
}
const DagHorizontalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDagHorizontalIcon
    });
});
DagHorizontalIcon.displayName = 'DagHorizontalIcon';

function SvgDagIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 1.75A.75.75 0 0 1 8.75 1h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5A.75.75 0 0 1 8 5.25v-1H5.5c-.69 0-1.25.56-1.25 1.25h2a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-2c0 .69.56 1.25 1.25 1.25H8v-1a.75.75 0 0 1 .75-.75h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5a.75.75 0 0 1-.75-.75v-1H5.5a2.75 2.75 0 0 1-2.75-2.75h-2A.75.75 0 0 1 0 9.75v-3.5a.75.75 0 0 1 .75-.75h2A2.75 2.75 0 0 1 5.5 2.75H8zm1.5.75v2h4v-2zM1.5 9V7h4v2zm8 4.5v-2h4v2z",
            clipRule: "evenodd"
        })
    });
}
const DagIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDagIcon
    });
});
DagIcon.displayName = 'DagIcon';

function SvgDagVerticalIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M4.5 2.75A.75.75 0 0 1 5.25 2h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-.564l2.571 2h2.493a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5a.75.75 0 0 1-.75-.75v-3.5A.75.75 0 0 1 9.75 9h.564L8 7.2 5.686 9h.564a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75H.75a.75.75 0 0 1-.75-.75v-3.5A.75.75 0 0 1 .75 9h2.493l2.571-2H5.25a.75.75 0 0 1-.75-.75zM6 3.5v2h4v-2zm-4.5 7v2h4v-2zm9 0v2h4v-2z",
            clipRule: "evenodd"
        })
    });
}
const DagVerticalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDagVerticalIcon
    });
});
DagVerticalIcon.displayName = 'DagVerticalIcon';

function SvgDangerFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m15.78 11.533-4.242 4.243a.75.75 0 0 1-.53.22H4.996a.75.75 0 0 1-.53-.22L.224 11.533a.75.75 0 0 1-.22-.53v-6.01a.75.75 0 0 1 .22-.53L4.467.22a.75.75 0 0 1 .53-.22h6.01a.75.75 0 0 1 .53.22l4.243 4.242c.141.141.22.332.22.53v6.011a.75.75 0 0 1-.22.53m-8.528-.785a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0m1.5-5.75v4h-1.5v-4z",
            clipRule: "evenodd"
        })
    });
}
const DangerFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDangerFillIcon
    });
});
DangerFillIcon.displayName = 'DangerFillIcon';

function SvgDashIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M15 8.75H1v-1.5h14z",
            clipRule: "evenodd"
        })
    });
}
const DashIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDashIcon
    });
});
DashIcon.displayName = 'DashIcon';

function SvgDashboardIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5 8.75v3h4.75v-3zm0-1.5h4.75V2.5H2.5zm6.25-6.5v3h4.75v-3zm0 11V7h4.75v6.5z",
            clipRule: "evenodd"
        })
    });
}
const DashboardIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDashboardIcon
    });
});
DashboardIcon.displayName = 'DashboardIcon';

function SvgDataIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8.646.368a.75.75 0 0 0-1.292 0l-3.25 5.5A.75.75 0 0 0 4.75 7h6.5a.75.75 0 0 0 .646-1.132zM8 2.224 9.936 5.5H6.064zM8.5 9.25a.75.75 0 0 1 .75-.75h5a.75.75 0 0 1 .75.75v5a.75.75 0 0 1-.75.75h-5a.75.75 0 0 1-.75-.75zM10 10v3.5h3.5V10zM1 11.75a3.25 3.25 0 1 1 6.5 0 3.25 3.25 0 0 1-6.5 0M4.25 10a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5",
            clipRule: "evenodd"
        })
    });
}
const DataIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDataIcon
    });
});
DataIcon.displayName = 'DataIcon';

function SvgDatabaseClockIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12 8a4 4 0 1 1 0 8 4 4 0 0 1 0-8m-.75 4c0 .199.08.39.22.53l1.5 1.5 1.06-1.06-1.28-1.28V9.5h-1.5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8 1c1.79 0 3.442.26 4.674.703.613.22 1.161.501 1.571.85.407.346.755.832.755 1.447v3.39a5.5 5.5 0 0 0-1.5-.682v-.77a7 7 0 0 1-.826.359C11.442 6.74 9.789 7 8 7s-3.442-.26-4.674-.703a7 7 0 0 1-.826-.36V8c0 .007.002.113.228.305.222.19.589.394 1.107.58.843.304 1.982.52 3.28.589a5.5 5.5 0 0 0-.513 1.47c-1.244-.098-2.373-.322-3.276-.647a7 7 0 0 1-.826-.36V12c0 .007.002.113.228.305.222.19.589.394 1.107.58.75.27 1.735.471 2.857.561.151.554.387 1.072.692 1.542-1.55-.052-2.969-.299-4.058-.691-.612-.22-1.161-.501-1.571-.85C1.348 13.101 1 12.615 1 12V4c0-.615.348-1.101.755-1.447.41-.349.959-.63 1.571-.85C4.558 1.26 6.211 1 8 1m0 1.5c-1.662 0-3.135.244-4.165.614-.518.187-.885.392-1.107.581-.226.192-.228.298-.228.305s.002.113.228.305c.222.19.589.394 1.107.58C4.865 5.258 6.338 5.5 8 5.5s3.135-.244 4.165-.614c.518-.187.885-.392 1.108-.581.225-.192.227-.298.227-.305s-.002-.113-.227-.305c-.223-.19-.59-.394-1.108-.58C11.135 2.742 9.662 2.5 8 2.5",
                clipRule: "evenodd"
            })
        ]
    });
}
const DatabaseClockIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDatabaseClockIcon
    });
});
DatabaseClockIcon.displayName = 'DatabaseClockIcon';

function SvgDatabaseIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2.727 3.695c-.225.192-.227.298-.227.305s.002.113.227.305c.223.19.59.394 1.108.58C4.865 5.256 6.337 5.5 8 5.5s3.135-.244 4.165-.615c.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305s-.002-.113-.227-.305c-.223-.19-.59-.394-1.108-.58C11.135 2.744 9.663 2.5 8 2.5s-3.135.244-4.165.615c-.519.186-.885.39-1.108.58M13.5 5.94a7 7 0 0 1-.826.358C11.442 6.74 9.789 7 8 7s-3.442-.26-4.673-.703a7 7 0 0 1-.827-.358V8c0 .007.002.113.227.305.223.19.59.394 1.108.58C4.865 9.256 6.337 9.5 8 9.5s3.135-.244 4.165-.615c.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305zM15 8V4c0-.615-.348-1.1-.755-1.447-.41-.349-.959-.63-1.571-.85C11.442 1.26 9.789 1 8 1s-3.442.26-4.673.703c-.613.22-1.162.501-1.572.85C1.348 2.9 1 3.385 1 4v8c0 .615.348 1.1.755 1.447.41.349.959.63 1.572.85C4.558 14.74 6.21 15 8 15s3.441-.26 4.674-.703c.612-.22 1.161-.501 1.571-.85.407-.346.755-.832.755-1.447zm-1.5 1.939a7 7 0 0 1-.826.358C11.442 10.74 9.789 11 8 11s-3.442-.26-4.673-.703a7 7 0 0 1-.827-.358V12c0 .007.002.113.227.305.223.19.59.394 1.108.58 1.03.371 2.502.615 4.165.615s3.135-.244 4.165-.615c.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305z",
            clipRule: "evenodd"
        })
    });
}
const DatabaseIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDatabaseIcon
    });
});
DatabaseIcon.displayName = 'DatabaseIcon';

function SvgDatabaseImportIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M10.403 2.102c1.044.114 1.992.32 2.77.601.614.22 1.162.501 1.572.85.407.346.755.832.755 1.447v8c0 .615-.348 1.101-.755 1.447-.41.349-.958.63-1.571.85-1.232.443-2.885.703-4.674.703s-3.442-.26-4.674-.703c-.612-.22-1.161-.501-1.571-.85-.407-.346-.755-.832-.755-1.447V5c0-.615.348-1.101.755-1.447.41-.349.959-.63 1.571-.85.78-.28 1.727-.487 2.77-.601v1.51c-.875.106-1.647.281-2.261.502-.518.187-.885.392-1.107.581C3.002 4.887 3 4.993 3 5s.002.113.228.305c.91.774 1.743 1.653 2.601 2.487-.741-.12-1.42-.285-2.003-.495A7 7 0 0 1 3 6.938V9c0 .007.002.113.228.305.222.19.589.394 1.107.58 1.03.372 2.503.615 4.165.615s3.135-.243 4.165-.614c.518-.187.885-.392 1.108-.581.225-.192.227-.298.227-.305V6.938a7 7 0 0 1-.826.359c-.555.2-1.196.358-1.896.476.82-.827 1.609-1.715 2.495-2.468.225-.192.227-.298.227-.305s-.002-.113-.227-.305c-.223-.19-.59-.394-1.108-.58-.614-.222-1.386-.397-2.262-.503zM14 10.939a7 7 0 0 1-.826.358C11.942 11.74 10.289 12 8.5 12s-3.442-.26-4.674-.703A7 7 0 0 1 3 10.938V13c0 .007.002.113.228.305.222.19.589.394 1.107.58 1.03.372 2.503.615 4.165.615s3.135-.243 4.165-.614c.518-.187.885-.392 1.108-.581.225-.192.227-.298.227-.305zM9.25 6.392l1.19-1.19 1.061 1.06-3.001 3-3.001-3 1.06-1.06 1.191 1.19V.507h1.5z"
        })
    });
}
const DatabaseImportIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDatabaseImportIcon
    });
});
DatabaseImportIcon.displayName = 'DatabaseImportIcon';

function SvgDecimalIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M3 10a3 3 0 1 0 6 0V6a3 3 0 0 0-6 0zm3 1.5A1.5 1.5 0 0 1 4.5 10V6a1.5 1.5 0 1 1 3 0v4A1.5 1.5 0 0 1 6 11.5M10 10a3 3 0 1 0 6 0V6a3 3 0 1 0-6 0zm3 1.5a1.5 1.5 0 0 1-1.5-1.5V6a1.5 1.5 0 0 1 3 0v4a1.5 1.5 0 0 1-1.5 1.5",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M1 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2"
            })
        ]
    });
}
const DecimalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDecimalIcon
    });
});
DecimalIcon.displayName = 'DecimalIcon';

function SvgDeprecatedIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8 2a6 6 0 1 1 0 12A6 6 0 0 1 8 2m-3.92 9.103q.36.454.815.816l7.026-7.023a5 5 0 0 0-.817-.817z"
        })
    });
}
const DeprecatedIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDeprecatedIcon
    });
});
DeprecatedIcon.displayName = 'DeprecatedIcon';

function SvgDeprecatedSmallIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8 3a5 5 0 1 1 0 10A5 5 0 0 1 8 3m-3.268 7.585q.301.379.68.68l5.856-5.85a4.2 4.2 0 0 0-.682-.683z"
        })
    });
}
const DeprecatedSmallIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDeprecatedSmallIcon
    });
});
DeprecatedSmallIcon.displayName = 'DeprecatedSmallIcon';

function SvgDollarIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M12.75 1a2.5 2.5 0 0 1 2.5 2.5v9l-.013.256a2.5 2.5 0 0 1-2.231 2.231L12.75 15h-9l-.256-.013A2.5 2.5 0 0 1 1.25 12.5v-9a2.5 2.5 0 0 1 2.244-2.487L3.75 1zm-9 1.5a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1zm5.182 2.047c.35.097.663.265.92.481.409.344.716.852.716 1.43H9.205c0-.083-.051-.236-.231-.388a1.13 1.13 0 0 0-.724-.252c-.297 0-.55.107-.724.252-.18.152-.23.305-.23.388 0 .184.068.406.213.572.128.148.346.288.741.288.757 0 1.36.272 1.766.728.392.44.552.996.552 1.496 0 .98-.735 1.678-1.636 1.915V12.5H7.568v-1.048c-.91-.252-1.636-.992-1.636-1.91h1.364c0 .183.286.64.954.64.69 0 .955-.412.955-.64a.9.9 0 0 0-.207-.589c-.12-.134-.336-.271-.748-.271-.774 0-1.374-.3-1.771-.758a2.27 2.27 0 0 1-.547-1.466c0-.578.307-1.086.715-1.43.258-.216.572-.384.921-.481V3.5h1.364z"
        })
    });
}
const DollarIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDollarIcon
    });
});
DollarIcon.displayName = 'DollarIcon';

function SvgDomainCirclesThree(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M11.75 13.375a2.625 2.625 0 1 0 0-5.25 2.625 2.625 0 0 0 0 5.25m0-1.25a1.375 1.375 0 1 0 0-2.75 1.375 1.375 0 0 0 0 2.75M4.25 13.375a2.625 2.625 0 1 0 0-5.25 2.625 2.625 0 0 0 0 5.25m0-1.25a1.375 1.375 0 1 0 0-2.75 1.375 1.375 0 0 0 0 2.75M8 7.375a2.625 2.625 0 1 0 0-5.25 2.625 2.625 0 0 0 0 5.25m0-1.25a1.375 1.375 0 1 0 0-2.75 1.375 1.375 0 0 0 0 2.75",
            clipRule: "evenodd"
        })
    });
}
const DomainCirclesThree = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDomainCirclesThree
    });
});
DomainCirclesThree.displayName = 'DomainCirclesThree';

function SvgDotsCircleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#DotsCircleIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M6 8a.75.75 0 1 1-1.5 0A.75.75 0 0 1 6 8M8 8.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5M10.75 8.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0",
                        clipRule: "evenodd"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const DotsCircleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDotsCircleIcon
    });
});
DotsCircleIcon.displayName = 'DotsCircleIcon';

function SvgDownloadIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M1 13.5h14V15H1zM12.53 6.53l-1.06-1.06-2.72 2.72V1h-1.5v7.19L4.53 5.47 3.47 6.53 8 11.06z"
        })
    });
}
const DownloadIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDownloadIcon
    });
});
DownloadIcon.displayName = 'DownloadIcon';

function SvgDragIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M5.25 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M10.75 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M5.25 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M10.75 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M5.25 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M10.75 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5"
        })
    });
}
const DragIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgDragIcon
    });
});
DragIcon.displayName = 'DragIcon';

function SvgErdIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M10 1.75a.75.75 0 0 1 .75-.75h3.5a.75.75 0 0 1 .75.75v3a.75.75 0 0 1-.75.75h-3.062l-.692.922.004.078v3l-.004.078.691.922h3.063a.75.75 0 0 1 .75.75v3a.75.75 0 0 1-.75.75h-3.5a.75.75 0 0 1-.75-.75v-2.833l-.875-1.167h-2.25L6 11.417v2.833a.75.75 0 0 1-.75.75h-3.5a.75.75 0 0 1-.75-.75v-3a.75.75 0 0 1 .75-.75h3.063l.691-.922A1 1 0 0 1 5.5 9.5v-3q0-.039.004-.078L4.813 5.5H1.75A.75.75 0 0 1 1 4.75v-3A.75.75 0 0 1 1.75 1h3.5a.75.75 0 0 1 .75.75v2.833l.875 1.167h2.25L10 4.583zm1.5.75V4h2V2.5zm0 11V12h2v1.5zM2.5 4V2.5h2V4zm0 8v1.5h2V12zM7 8.75v-1.5h2v1.5z",
            clipRule: "evenodd"
        })
    });
}
const ErdIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgErdIcon
    });
});
ErdIcon.displayName = 'ErdIcon';

function SvgExpandLessIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 17",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M12.06 1.06 11 0 8.03 2.97 5.06 0 4 1.06l4.03 4.031zM4 15l4.03-4.03L12.06 15 11 16.06l-2.97-2.969-2.97 2.97z"
        })
    });
}
const ExpandLessIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgExpandLessIcon
    });
});
ExpandLessIcon.displayName = 'ExpandLessIcon';

function SvgExpandMoreIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 17",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "m4 4.03 1.06 1.061 2.97-2.97L11 5.091l1.06-1.06L8.03 0zM12.06 12.091l-4.03 4.03L4 12.091l1.06-1.06L8.03 14 11 11.03z"
        })
    });
}
const ExpandMoreIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgExpandMoreIcon
    });
});
ExpandMoreIcon.displayName = 'ExpandMoreIcon';

function SvgFaceFrownIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#FaceFrownIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M6 5.25a.75.75 0 0 0 0 1.5h.007a.75.75 0 0 0 0-1.5zM9.25 6a.75.75 0 0 1 .75-.75h.007a.75.75 0 0 1 0 1.5H10A.75.75 0 0 1 9.25 6M10.07 11.12a.75.75 0 0 0 1.197-.903l-.001-.001v-.001l-.003-.003-.005-.006-.015-.02a3 3 0 0 0-.217-.246 4.7 4.7 0 0 0-.626-.546C9.858 9 9.04 8.584 8 8.584s-1.858.416-2.4.81a4.7 4.7 0 0 0-.795.733l-.048.06-.015.019-.005.006-.002.003-.031.044.03-.042a.75.75 0 1 0 1.22.875q.032-.039.103-.115c.096-.1.24-.235.426-.37.375-.273.89-.523 1.517-.523s1.142.25 1.517.523a3.2 3.2 0 0 1 .529.485l.021.025z"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M8 .583a7.417 7.417 0 1 0 0 14.834A7.417 7.417 0 0 0 8 .583M2.083 8a5.917 5.917 0 1 1 11.834 0A5.917 5.917 0 0 1 2.083 8",
                        clipRule: "evenodd"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const FaceFrownIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFaceFrownIcon
    });
});
FaceFrownIcon.displayName = 'FaceFrownIcon';

function SvgFaceNeutralIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                fillRule: "evenodd",
                clipPath: "url(#FaceNeutralIcon_svg__a)",
                clipRule: "evenodd",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M8 2.084a5.917 5.917 0 1 0 0 11.833A5.917 5.917 0 0 0 8 2.084M.583 8a7.417 7.417 0 1 1 14.834 0A7.417 7.417 0 0 1 .583 8"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M4.583 10a.75.75 0 0 1 .75-.75h5.334a.75.75 0 1 1 0 1.5H5.333a.75.75 0 0 1-.75-.75M5.25 6A.75.75 0 0 1 6 5.25h.007a.75.75 0 0 1 0 1.5H6A.75.75 0 0 1 5.25 6M9.25 6a.75.75 0 0 1 .75-.75h.007a.75.75 0 1 1 0 1.5H10A.75.75 0 0 1 9.25 6"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const FaceNeutralIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFaceNeutralIcon
    });
});
FaceNeutralIcon.displayName = 'FaceNeutralIcon';

function SvgFaceSmileIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#FaceSmileIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M8 2.084a5.917 5.917 0 1 0 0 11.833A5.917 5.917 0 0 0 8 2.084M.583 8a7.417 7.417 0 1 1 14.834 0A7.417 7.417 0 0 1 .583 8",
                        clipRule: "evenodd"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M4.883 8.734a.75.75 0 0 1 1.048.146l.002.003.021.026q.032.038.103.114c.096.1.24.235.426.37.375.274.89.524 1.517.524s1.142-.25 1.517-.523a3.2 3.2 0 0 0 .55-.511.75.75 0 0 1 1.2.9l.029-.042-.03.043-.001.002-.002.002-.005.007-.015.019-.048.059q-.06.073-.17.188c-.143.15-.354.348-.626.546-.54.393-1.359.81-2.399.81s-1.858-.417-2.4-.81a4.7 4.7 0 0 1-.795-.734l-.048-.059-.015-.02-.005-.006-.002-.002v-.002h-.002a.75.75 0 0 1 .15-1.05"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M5.25 6A.75.75 0 0 1 6 5.25h.007a.75.75 0 0 1 0 1.5H6A.75.75 0 0 1 5.25 6M9.25 6a.75.75 0 0 1 .75-.75h.007a.75.75 0 1 1 0 1.5H10A.75.75 0 0 1 9.25 6",
                        clipRule: "evenodd"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const FaceSmileIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFaceSmileIcon
    });
});
FaceSmileIcon.displayName = 'FaceSmileIcon';

function SvgFileCodeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 17",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16H2zm7.5 1.81 1.94 1.94H9.5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.47 9.97 4.44 13l3.03 3.03 1.06-1.06L6.56 13l1.97-1.97zM11.03 9.97l-1.06 1.06L11.94 13l-1.97 1.97 1.06 1.06L14.06 13z"
            })
        ]
    });
}
const FileCodeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFileCodeIcon
    });
});
FileCodeIcon.displayName = 'FileCodeIcon';

function SvgFileCubeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M11.762 8.04a.75.75 0 0 1 .553.03l3.25 1.5q.045.02.09.048l.004.003q.016.011.031.024.023.014.044.032.059.05.106.111a.75.75 0 0 1 .16.462v3.5a.75.75 0 0 1-.435.68l-3.242 1.496-.006.003-.002.002-.013.005a.8.8 0 0 1-.251.062h-.102a.8.8 0 0 1-.25-.062l-.014-.005-.01-.005-3.24-1.495A.75.75 0 0 1 8 13.75v-3.5q0-.053.007-.104l.006-.03a.8.8 0 0 1 .047-.159l.007-.019.025-.048.017-.029.01-.015.023-.035.024-.03a.76.76 0 0 1 .268-.212h.002l3.25-1.5zM9.5 13.27l1.75.807V12.23l-1.75-.807zm3.25-1.04v1.847l1.75-.807v-1.848zm-2.21-1.981 1.46.675 1.46-.674L12 9.575z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8.75 0c.199 0 .39.08.53.22l4.5 4.5.094.114A.75.75 0 0 1 14 5.25v1.944l-1.057-.487-.199-.08a2 2 0 0 0-.244-.07V6H8.75A.75.75 0 0 1 8 5.25V1.5H3.5v12h3v.25c0 .455.138.887.38 1.25H2.75a.75.75 0 0 1-.75-.75V.75l.004-.077A.75.75 0 0 1 2.75 0zm.75 4.5h1.94L9.5 2.56z",
                clipRule: "evenodd"
            })
        ]
    });
}
const FileCubeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFileCubeIcon
    });
});
FileCubeIcon.displayName = 'FileCubeIcon';

function SvgFileDocumentIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16H2zm7.5 1.81 1.94 1.94H9.5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M5 11.5V13h9v-1.5zM14 16H5v-1.5h9z"
            })
        ]
    });
}
const FileDocumentIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFileDocumentIcon
    });
});
FileDocumentIcon.displayName = 'FileDocumentIcon';

function SvgFileIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53v9a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75zm1.5.75v12h9V7H8.75A.75.75 0 0 1 8 6.25V2.5zm6 1.06 1.94 1.94H9.5z",
            clipRule: "evenodd"
        })
    });
}
const FileIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFileIcon
    });
});
FileIcon.displayName = 'FileIcon';

function SvgFileImageIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16H2zm7.5 1.81 1.94 1.94H9.5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M10.466 10a.75.75 0 0 0-.542.27l-3.75 4.5A.75.75 0 0 0 6.75 16h6.5a.75.75 0 0 0 .75-.75V13.5a.75.75 0 0 0-.22-.53l-2.75-2.75a.75.75 0 0 0-.564-.22m2.034 3.81v.69H8.351l2.2-2.639zM6.5 7.25a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5M5.75 9.5a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0",
                clipRule: "evenodd"
            })
        ]
    });
}
const FileImageIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFileImageIcon
    });
});
FileImageIcon.displayName = 'FileImageIcon';

function SvgFileLockIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                fillRule: "evenodd",
                clipPath: "url(#FileLockIcon_svg__a)",
                clipRule: "evenodd",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M2.75 0A.75.75 0 0 0 2 .75v13.5c0 .414.336.75.75.75H7.5v-1.5h-4v-12H8v3.75c0 .414.336.75.75.75h3.75v1H14V5.25a.75.75 0 0 0-.22-.53L9.28.22A.75.75 0 0 0 8.75 0zm8.69 4.5L9.5 2.56V4.5z"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M14 10v.688h.282a.75.75 0 0 1 .75.75v3.874a.75.75 0 0 1-.75.75H9.718a.75.75 0 0 1-.75-.75v-3.874a.75.75 0 0 1 .75-.75H10V10a2 2 0 0 1 4 0m-1.5 0v.688h-1V10a.5.5 0 0 1 1 0m1.032 2.188v2.374h-3.064v-2.374z"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const FileLockIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFileLockIcon
    });
});
FileLockIcon.displayName = 'FileLockIcon';

function SvgFileModelIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M2.75 1a.75.75 0 0 0-.75.75V16h1.5V2.5H8v3.75c0 .414.336.75.75.75h3.75v3H14V6.25a.75.75 0 0 0-.22-.53l-4.5-4.5A.75.75 0 0 0 8.75 1zm8.69 4.5L9.5 3.56V5.5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M11.75 11.5a2.25 2.25 0 1 1-2.03 1.28l-.5-.5a2.25 2.25 0 1 1 1.06-1.06l.5.5c.294-.141.623-.22.97-.22m.75 2.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0M8.25 9.5a.75.75 0 1 1 0 1.5.75.75 0 0 1 0-1.5",
                clipRule: "evenodd"
            })
        ]
    });
}
const FileModelIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFileModelIcon
    });
});
FileModelIcon.displayName = 'FileModelIcon';

function SvgFileNewIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M2 .75A.75.75 0 0 1 2.75 0h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V7.5h-1.5V6H8.75A.75.75 0 0 1 8 5.25V1.5H3.5v12h4V15H2.75a.75.75 0 0 1-.75-.75zm7.5 1.81 1.94 1.94H9.5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M11.25 15v-2.25H9v-1.5h2.25V9h1.5v2.25H15v1.5h-2.25V15z"
            })
        ]
    });
}
const FileNewIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFileNewIcon
    });
});
FileNewIcon.displayName = 'FileNewIcon';

function SvgFilePipelineIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M2 .75A.75.75 0 0 1 2.75 0h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V8h-1.5V6H8.75A.75.75 0 0 1 8 5.25V1.5H3.5v12h6V15H2.75a.75.75 0 0 1-.75-.75zm7.5 1.81 1.94 1.94H9.5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M9.25 8.5a.75.75 0 0 0-.75.75v2.5c0 .414.336.75.75.75h.785a3.5 3.5 0 0 0 3.465 3h1.25a.75.75 0 0 0 .75-.75v-2.5a.75.75 0 0 0-.75-.75h-.785a3.5 3.5 0 0 0-3.465-3zM10 11v-1h.5a2 2 0 0 1 2 2 1 1 0 0 0 1 1h.5v1h-.5a2 2 0 0 1-2-2 1 1 0 0 0-1-1z",
                clipRule: "evenodd"
            })
        ]
    });
}
const FilePipelineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFilePipelineIcon
    });
});
FilePipelineIcon.displayName = 'FilePipelineIcon';

function SvgFilterIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75V4a.75.75 0 0 1-.22.53L10 9.31v4.94a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1-.75-.75V9.31L1.22 4.53A.75.75 0 0 1 1 4zm1.5.75v1.19l4.78 4.78c.141.14.22.331.22.53v4.5h1V9a.75.75 0 0 1 .22-.53l4.78-4.78V2.5z",
            clipRule: "evenodd"
        })
    });
}
const FilterIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFilterIcon
    });
});
FilterIcon.displayName = 'FilterIcon';

function SvgFlagPointerIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M3 2.5h5.439a.5.5 0 0 1 .39.188l4 5a.5.5 0 0 1 0 .624l-4 5a.5.5 0 0 1-.39.188H3a.5.5 0 0 1-.5-.5V3a.5.5 0 0 1 .5-.5M1 3a2 2 0 0 1 2-2h5.439A2 2 0 0 1 10 1.75l4 5a2 2 0 0 1 0 2.5l-4 5a2 2 0 0 1-1.562.75H3a2 2 0 0 1-2-2zm6 7a2 2 0 1 0 0-4 2 2 0 0 0 0 4",
            clipRule: "evenodd"
        })
    });
}
const FlagPointerIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFlagPointerIcon
    });
});
FlagPointerIcon.displayName = 'FlagPointerIcon';

function SvgFloatIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M0 5.25A2.25 2.25 0 0 0 2.25 3h1.5v8.5H6V13H0v-1.5h2.25V6c-.627.471-1.406.75-2.25.75zM10 5.75A2.75 2.75 0 0 1 12.75 3h.39a2.86 2.86 0 0 1 1.57 5.252l-2.195 1.44a2.25 2.25 0 0 0-1.014 1.808H16V13h-6v-1.426a3.75 3.75 0 0 1 1.692-3.135l2.194-1.44A1.36 1.36 0 0 0 13.14 4.5h-.389c-.69 0-1.25.56-1.25 1.25V6H10zM8 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2"
        })
    });
}
const FloatIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFloatIcon
    });
});
FloatIcon.displayName = 'FloatIcon';

function SvgFlowIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 15 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "m4.94 8.018.732.732H3.75a2 2 0 1 0 0 4H9V12a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-4a1 1 0 0 1-1-1v-.75H3.75a3.5 3.5 0 1 1 0-7h1.957zM10.5 14.5h3v-2h-3zM5 0a1 1 0 0 1 1 1v.75h5.25a3.5 3.5 0 0 1 0 7H9.025l1.035 1.036L9 10.846 6.172 8.019 9 5.189l1.06 1.061-1 1h2.19a2 2 0 0 0 0-4H6V4a1 1 0 0 1-1 1H1a1 1 0 0 1-1-1V1a1 1 0 0 1 1-1zM1.5 3.5h3v-2h-3z"
        })
    });
}
const FlowIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFlowIcon
    });
});
FlowIcon.displayName = 'FlowIcon';

function SvgFolderBranchFillIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75H5.5c0-.98.403-1.866 1.05-2.5a3.5 3.5 0 1 1 5.945-2.661 3.5 3.5 0 0 1 1.505-.339c.744 0 1.433.232 2 .627V4.75a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M9.75 12.145a2 2 0 1 1-1.5 0v-1.29a2 2 0 1 1 2.538-.957c.3.585.812 1.017 1.416 1.221a2 2 0 1 1-.096 1.53 4 4 0 0 1-2.358-1.577zM8.5 14a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0m5.5-2.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1M8.5 9a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0",
                clipRule: "evenodd"
            })
        ]
    });
}
const FolderBranchFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderBranchFillIcon
    });
});
FolderBranchFillIcon.displayName = 'FolderBranchFillIcon';

function SvgFolderBranchIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 0 1 .75.75V8h-1.5V5.5h-7a.75.75 0 0 1-.53-.22L5.555 3.866a1.25 1.25 0 0 0-.883-.366H1.5v9H5V14H.75a.75.75 0 0 1-.75-.75zM9 8.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1M7 9a2 2 0 1 1 3.778.917c.376.58.888 1.031 1.414 1.227a2 2 0 1 1-.072 1.54c-.977-.207-1.795-.872-2.37-1.626v1.087a2 2 0 1 1-1.5 0v-1.29A2 2 0 0 1 7 9m7 2.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1m-5 2a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1",
            clipRule: "evenodd"
        })
    });
}
const FolderBranchIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderBranchIcon
    });
});
FolderBranchIcon.displayName = 'FolderBranchIcon';

function SvgFolderCloudFilledIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#FolderCloudFilledIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 0 1 .75.75v5.02a4.4 4.4 0 0 0-.921-.607 5.11 5.11 0 0 0-9.512-.753A4.75 4.75 0 0 0 2.917 14H.75a.75.75 0 0 1-.75-.75z"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M6.715 9.595a3.608 3.608 0 0 1 7.056.688C15.07 10.572 16 11.739 16 13.107 16 14.688 14.757 16 13.143 16H7.32a.8.8 0 0 1-.163-.018 3.25 3.25 0 0 1-.443-6.387m.703 4.905-.044-.004a1.75 1.75 0 0 1 .015-3.493.75.75 0 0 0 .698-.657 2.108 2.108 0 0 1 4.199.261v.357c0 .415.335.75.75.75h.107c.753 0 1.357.607 1.357 1.393s-.604 1.393-1.357 1.393q-.036 0-.07-.002a1 1 0 0 0-.1.002z",
                        clipRule: "evenodd"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const FolderCloudFilledIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderCloudFilledIcon
    });
});
FolderCloudFilledIcon.displayName = 'FolderCloudFilledIcon';

function SvgFolderCloudIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#FolderCloudIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75H3v-1.5H1.5v-9h3.172c.331 0 .649.132.883.366L6.97 5.28c.14.141.331.22.53.22h7V8H16V4.75a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2z"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M10.179 7a3.61 3.61 0 0 0-3.464 2.595 3.251 3.251 0 0 0 .443 6.387.8.8 0 0 0 .163.018h5.821C14.758 16 16 14.688 16 13.107c0-1.368-.931-2.535-2.229-2.824A3.61 3.61 0 0 0 10.18 7m-2.805 7.496q.023 0 .044.004h5.555a1 1 0 0 1 .1-.002l.07.002c.753 0 1.357-.607 1.357-1.393s-.604-1.393-1.357-1.393h-.107a.75.75 0 0 1-.75-.75v-.357a2.107 2.107 0 0 0-4.199-.26.75.75 0 0 1-.698.656 1.75 1.75 0 0 0-.015 3.493",
                        clipRule: "evenodd"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const FolderCloudIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderCloudIcon
    });
});
FolderCloudIcon.displayName = 'FolderCloudIcon';

function SvgFolderCubeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M4.672 2c.73 0 1.429.29 1.944.806L7.811 4h7.439l.077.004A.75.75 0 0 1 16 4.75v3.367l-3.057-1.41-.199-.08a2.25 2.25 0 0 0-1.254-.068l-.205.057-.077.029-.076.03-.075.032-3.25 1.5-.005.002a2 2 0 0 0-.145.073l-.12.072-.053.036a2.3 2.3 0 0 0-.476.437l-.002.002-.007.008c-.01.012-.04.049-.074.096l-.031.046-.04.058-.015.026-.023.037-.04.07-.026.049-.024.049-.023.047-.025.059-.053.137-.043.138-.045.2.005-.027-.009.041-.012.077q-.022.166-.021.31v3.5q0 .126.016.25H.75a.75.75 0 0 1-.75-.75V2.75l.004-.077A.75.75 0 0 1 .75 2z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M11.762 8.04a.75.75 0 0 1 .553.03l3.25 1.5q.045.02.09.048l.004.003q.016.011.031.024.023.014.044.032.059.05.106.111a.75.75 0 0 1 .16.462v3.5a.75.75 0 0 1-.435.68l-3.242 1.496-.006.003-.002.002-.013.005a.8.8 0 0 1-.251.062h-.102a.8.8 0 0 1-.25-.062l-.014-.005-.01-.005-3.24-1.495A.75.75 0 0 1 8 13.75v-3.5q0-.053.007-.104l.006-.03a.8.8 0 0 1 .047-.159l.007-.019.025-.048.017-.029.01-.015.023-.035.024-.03a.8.8 0 0 1 .162-.151l.018-.012a1 1 0 0 1 .088-.049h.002l3.25-1.5zM9.5 13.27l1.75.807V12.23l-1.75-.807zm3.25-1.04v1.847l1.75-.807v-1.848zm-2.21-1.981 1.46.675 1.46-.674L12 9.575z",
                clipRule: "evenodd"
            })
        ]
    });
}
const FolderCubeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderCubeIcon
    });
});
FolderCubeIcon.displayName = 'FolderCubeIcon';

function SvgFolderCubeOutlineIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75H7.5v-1.5h-6v-9h3.172c.331 0 .649.132.883.366L6.97 5.28c.14.141.331.22.53.22h7V8H16V4.75a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12.762 8.039a.75.75 0 0 1 .553.03l3.25 1.5q.045.02.09.049l.004.003.031.023q.023.015.044.033.059.05.106.11a.75.75 0 0 1 .16.463v3.5a.75.75 0 0 1-.436.68l-3.24 1.495-.007.003-.002.002-.013.005a.8.8 0 0 1-.251.063h-.102a.8.8 0 0 1-.25-.063l-.014-.005-.01-.005-3.24-1.495A.75.75 0 0 1 9 13.75v-3.5q0-.053.007-.104l.006-.03a.8.8 0 0 1 .047-.16l.007-.018.025-.049.017-.028.01-.016.023-.035.024-.03a.8.8 0 0 1 .162-.15l.018-.012a1 1 0 0 1 .088-.049h.002l3.25-1.5zm-2.262 5.23 1.75.808v-1.848l-1.75-.807zm3.25-1.04v1.848l1.75-.808v-1.847zm-2.21-1.98 1.46.675 1.46-.674L13 9.575z",
                clipRule: "evenodd"
            })
        ]
    });
}
const FolderCubeOutlineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderCubeOutlineIcon
    });
});
FolderCubeOutlineIcon.displayName = 'FolderCubeOutlineIcon';

function SvgFolderFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2z"
        })
    });
}
const FolderFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderFillIcon
    });
});
FolderFillIcon.displayName = 'FolderFillIcon';

function SvgFolderHomeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 20",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M.75 4a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h6.22v-1.5H1.5v-9h3.172c.331 0 .649.132.883.366L6.97 7.28c.14.141.331.22.53.22h7V10H16V6.75a.75.75 0 0 0-.75-.75H7.81L6.617 4.805A2.75 2.75 0 0 0 4.672 4z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12.457 9.906a.75.75 0 0 0-.914 0l-3.25 2.5A.75.75 0 0 0 8 13v4.25c0 .414.336.75.75.75h6.5a.75.75 0 0 0 .75-.75V13a.75.75 0 0 0-.293-.594zM9.5 16.5v-3.13l2.5-1.924 2.5 1.923V16.5h-1.75V14h-1.5v2.5z",
                clipRule: "evenodd"
            })
        ]
    });
}
const FolderHomeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderHomeIcon
    });
});
FolderHomeIcon.displayName = 'FolderHomeIcon';

function SvgFolderIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 0 1 .75.75v8.5a.75.75 0 0 1-.75.75H.75a.75.75 0 0 1-.75-.75zm1.5.75v9h13v-7h-7a.75.75 0 0 1-.53-.22L5.555 3.866a1.25 1.25 0 0 0-.883-.366z",
            clipRule: "evenodd"
        })
    });
}
const FolderIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderIcon
    });
});
FolderIcon.displayName = 'FolderIcon';

function SvgFolderNewIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75H7.5v-1.5h-6v-9h3.172c.331 0 .649.132.883.366L6.97 5.28c.14.141.331.22.53.22h7V8H16V4.75a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M11.25 11.25V9h1.5v2.25H15v1.5h-2.25V15h-1.5v-2.25H9v-1.5z"
            })
        ]
    });
}
const FolderNewIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderNewIcon
    });
});
FolderNewIcon.displayName = 'FolderNewIcon';

function SvgFolderNodeIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M5.217 1c.713 0 1.4.277 1.912.772l.58.561h5.541l.077.004a.75.75 0 0 1 .673.746V9.75a.75.75 0 0 1-.75.75h-4.5v1.13A2.25 2.25 0 0 1 10.12 13H14v1.5h-3.88a2.248 2.248 0 0 1-4.24 0H2V13h3.88a2.25 2.25 0 0 1 1.37-1.37V10.5h-4.5A.75.75 0 0 1 2 9.75v-8l.004-.077A.75.75 0 0 1 2.75 1zM8 13a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5M3.5 9h9V3.833H7.405a.75.75 0 0 1-.408-.12l-.113-.09-.798-.771a1.25 1.25 0 0 0-.87-.352H3.5z",
            clipRule: "evenodd"
        })
    });
}
const FolderNodeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderNodeIcon
    });
});
FolderNodeIcon.displayName = 'FolderNodeIcon';

function SvgFolderOpenBranchIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M9 7a2 2 0 0 1 1.786 2.895l.074.135c.307.519.788.901 1.346 1.09A1.998 1.998 0 0 1 16 12a2 2 0 0 1-3.892.646A3.98 3.98 0 0 1 9.75 11.07v1.077a2 2 0 1 1-1.5 0v-1.294A1.999 1.999 0 0 1 9 7m0 6.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1m5-2a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1m-5-3a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M15.25 5a.75.75 0 0 1 .658 1.11l-1.33 2.438a3.52 3.52 0 0 0-2.082.29A3.5 3.5 0 1 0 6.551 11.5q-.235.23-.423.5H.75a1 1 0 0 1-.095-.007l-.011-.002a.8.8 0 0 1-.167-.045q-.025-.01-.048-.021a.7.7 0 0 1-.203-.139L.22 11.78l-.008-.01a.7.7 0 0 1-.097-.123l-.01-.014a1 1 0 0 1-.044-.088l-.01-.024a.8.8 0 0 1-.05-.24L0 11.263V.75L.004.673A.75.75 0 0 1 .75 0h3.922c.73 0 1.429.29 1.944.806L7.811 2h5.439l.077.004A.75.75 0 0 1 14 2.75V5zM12.5 5V3.5h-5a.75.75 0 0 1-.53-.22L5.556 1.866a1.25 1.25 0 0 0-.884-.366H1.5v6.809L3.092 5.39A.75.75 0 0 1 3.75 5z",
                clipRule: "evenodd"
            })
        ]
    });
}
const FolderOpenBranchIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderOpenBranchIcon
    });
});
FolderOpenBranchIcon.displayName = 'FolderOpenBranchIcon';

function SvgFolderOpenCloudIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M10.39 7.313a3.487 3.487 0 0 1 3.47 3.153c1.247.288 2.14 1.412 2.14 2.73C16 14.728 14.795 16 13.23 16l-.066-.002-.036.002H7.65a.8.8 0 0 1-.173-.023 3.143 3.143 0 0 1-.43-6.171 3.49 3.49 0 0 1 3.342-2.493m0 1.5a1.99 1.99 0 0 0-1.974 1.742.75.75 0 0 1-.565.636l-.132.02a1.646 1.646 0 0 0-.015 3.285q.012.001.024.004h5.343a1 1 0 0 1 .093-.002l.066.002c.704 0 1.27-.567 1.27-1.304 0-.736-.566-1.303-1.27-1.303h-.102a.75.75 0 0 1-.75-.75V10.8c0-1.098-.89-1.988-1.988-1.989",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M15.249 5a.75.75 0 0 1 .66 1.11l-1.31 2.397C13.792 7.017 12.18 6 10.32 6 8.393 6 6.73 7.09 5.952 8.669c-1.377.595-2.388 1.84-2.617 3.331H.749l-.003-.001a1 1 0 0 1-.09-.006q-.008 0-.017-.003l-.055-.01a1 1 0 0 1-.12-.037l-.03-.015a.7.7 0 0 1-.212-.147l-.008-.008a1 1 0 0 1-.078-.094l-.014-.02q-.008-.014-.018-.026a1 1 0 0 1-.043-.088l-.01-.024a1 1 0 0 1-.04-.143l-.005-.034L0 11.27 0 11.264V.75L.004.673A.75.75 0 0 1 .75 0h3.922c.73 0 1.429.29 1.944.806L7.811 2h5.439l.077.004A.75.75 0 0 1 14 2.75V5zM12.5 5V3.5h-5a.75.75 0 0 1-.53-.22L5.556 1.866a1.25 1.25 0 0 0-.884-.366H1.5v6.807L3.09 5.39A.75.75 0 0 1 3.75 5z",
                clipRule: "evenodd"
            })
        ]
    });
}
const FolderOpenCloudIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderOpenCloudIcon
    });
});
FolderOpenCloudIcon.displayName = 'FolderOpenCloudIcon';

function SvgFolderOpenCubeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M10.762 8.04a.75.75 0 0 1 .553.03l3.24 1.494a1 1 0 0 1 .1.054l.004.003.016.012.06.044q.039.034.073.072a.75.75 0 0 1 .192.501v3.5a.75.75 0 0 1-.435.68l-3.242 1.496-.006.003-.002.002a.8.8 0 0 1-.156.05q-.009.001-.018.004l-.042.006q-.024.004-.048.007h-.102q-.024-.003-.049-.007l-.042-.006q-.009-.001-.018-.005a.8.8 0 0 1-.155-.05l-.01-.004-3.24-1.495A.75.75 0 0 1 7 13.75v-3.5a.8.8 0 0 1 .06-.293l.007-.019.025-.048.035-.057.037-.05.027-.034a.8.8 0 0 1 .133-.116l.022-.015a1 1 0 0 1 .098-.054l3.241-1.495zM8.5 13.27l1.75.807V12.23l-1.75-.807zm3.25-1.04v1.847l1.75-.807v-1.848zm-2.21-1.98 1.46.674 1.46-.674L11 9.575z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M4.672 0c.73 0 1.429.29 1.944.806L7.811 2h5.439l.077.004A.75.75 0 0 1 14 2.75V5h1.25a.75.75 0 0 1 .658 1.11l-1.04 1.908-2.762-1.275a2.64 2.64 0 0 0-2.212 0l-2.86 1.32A2.64 2.64 0 0 0 5.5 10.461V12H.75a1 1 0 0 1-.094-.007q-.008 0-.017-.003l-.055-.01a1 1 0 0 1-.12-.037l-.03-.015a.7.7 0 0 1-.208-.142L.22 11.78l-.008-.01a.7.7 0 0 1-.094-.119l-.013-.018a1 1 0 0 1-.044-.088l-.01-.024a1 1 0 0 1-.04-.143l-.005-.034L0 11.285 0 11.264V.75L.004.673A.75.75 0 0 1 .75 0zM1.5 8.309 3.092 5.39A.75.75 0 0 1 3.75 5h8.75V3.5h-5a.75.75 0 0 1-.53-.22L5.556 1.866a1.25 1.25 0 0 0-.884-.366H1.5z",
                clipRule: "evenodd"
            })
        ]
    });
}
const FolderOpenCubeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderOpenCubeIcon
    });
});
FolderOpenCubeIcon.displayName = 'FolderOpenCubeIcon';

function SvgFolderOpenIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h5.439a.75.75 0 0 1 .75.75V7h1.25a.75.75 0 0 1 .658 1.11l-3 5.5a.75.75 0 0 1-.658.39H.75a.747.747 0 0 1-.75-.75zm1.5 7.559L3.092 7.39A.75.75 0 0 1 3.75 7h8.75V5.5h-5a.75.75 0 0 1-.53-.22L5.555 3.866a1.25 1.25 0 0 0-.883-.366H1.5z",
            clipRule: "evenodd"
        })
    });
}
const FolderOpenIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderOpenIcon
    });
});
FolderOpenIcon.displayName = 'FolderOpenIcon';

function SvgFolderOpenPipelineIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M0 1.75A.75.75 0 0 1 .75 1h3.922c.729 0 1.428.29 1.944.805L7.811 3h5.439a.75.75 0 0 1 .75.75V6h1.25a.75.75 0 0 1 .658 1.11L14.56 9.58A4.99 4.99 0 0 0 10.5 7.5H9.25A2.25 2.25 0 0 0 7 9.75v2.5q.002.396.128.75H.75a.747.747 0 0 1-.75-.75zm1.5 7.559L3.092 6.39A.75.75 0 0 1 3.75 6h8.75V4.5h-5a.75.75 0 0 1-.53-.22L5.555 2.866a1.25 1.25 0 0 0-.883-.366H1.5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8.5 9.75A.75.75 0 0 1 9.25 9h1.25a3.5 3.5 0 0 1 3.465 3h.785a.75.75 0 0 1 .75.75v2.5a.75.75 0 0 1-.75.75H13.5a3.5 3.5 0 0 1-3.465-3H9.25a.75.75 0 0 1-.75-.75zm1.5.75v1h.5a1 1 0 0 1 1 1 2 2 0 0 0 2 2h.5v-1h-.5a1 1 0 0 1-1-1 2 2 0 0 0-2-2z",
                clipRule: "evenodd"
            })
        ]
    });
}
const FolderOpenPipelineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderOpenPipelineIcon
    });
});
FolderOpenPipelineIcon.displayName = 'FolderOpenPipelineIcon';

function SvgFolderOutlinePipelineIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 0 1 .75.75V10h-1.5V5.5h-7a.75.75 0 0 1-.53-.22L5.555 3.866a1.25 1.25 0 0 0-.883-.366H1.5v9h6V14H.75a.75.75 0 0 1-.75-.75z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M9.25 8.5a.75.75 0 0 0-.75.75v2.5c0 .414.336.75.75.75h.785a3.5 3.5 0 0 0 3.465 3h1.25a.75.75 0 0 0 .75-.75v-2.5a.75.75 0 0 0-.75-.75h-.785a3.5 3.5 0 0 0-3.465-3zM10 11v-1h.5a2 2 0 0 1 2 2 1 1 0 0 0 1 1h.5v1h-.5a2 2 0 0 1-2-2 1 1 0 0 0-1-1z",
                clipRule: "evenodd"
            })
        ]
    });
}
const FolderOutlinePipelineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderOutlinePipelineIcon
    });
});
FolderOutlinePipelineIcon.displayName = 'FolderOutlinePipelineIcon';

function SvgFolderSolidPipelineIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h8.166l-.01-.026A2.25 2.25 0 0 1 7 11.75v-2.5A2.25 2.25 0 0 1 9.25 7h1.25a5 5 0 0 1 4.595 3.026c.33.051.638.174.905.353V4.75a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M9.25 8.5a.75.75 0 0 0-.75.75v2.5c0 .414.336.75.75.75h.785a3.5 3.5 0 0 0 3.465 3h1.25a.75.75 0 0 0 .75-.75v-2.5a.75.75 0 0 0-.75-.75h-.785a3.5 3.5 0 0 0-3.465-3zM10 11v-1h.5a2 2 0 0 1 2 2 1 1 0 0 0 1 1h.5v1h-.5a2 2 0 0 1-2-2 1 1 0 0 0-1-1z",
                clipRule: "evenodd"
            })
        ]
    });
}
const FolderSolidPipelineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFolderSolidPipelineIcon
    });
});
FolderSolidPipelineIcon.displayName = 'FolderSolidPipelineIcon';

function SvgFontIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#FontIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    fillRule: "evenodd",
                    d: "M5.197 3.473a.75.75 0 0 0-1.393-.001L-.006 13H1.61l.6-1.5h4.562l.596 1.5h1.614zM6.176 10 4.498 5.776 2.809 10zm4.07-2.385c.593-.205 1.173-.365 1.754-.365a1.5 1.5 0 0 1 1.42 1.014A3.8 3.8 0 0 0 12 8c-.741 0-1.47.191-2.035.607A2.3 2.3 0 0 0 9 10.5c0 .81.381 1.464.965 1.893.565.416 1.294.607 2.035.607.524 0 1.042-.096 1.5-.298V13H15V8.75a3 3 0 0 0-3-3c-.84 0-1.614.23-2.245.448zM13.5 10.5a.8.8 0 0 0-.353-.685C12.897 9.631 12.5 9.5 12 9.5c-.5 0-.897.131-1.146.315a.8.8 0 0 0-.354.685c0 .295.123.515.354.685.25.184.645.315 1.146.315.502 0 .897-.131 1.147-.315.23-.17.353-.39.353-.685",
                    clipRule: "evenodd"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const FontIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFontIcon
    });
});
FontIcon.displayName = 'FontIcon';

function SvgForkHorizontalIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.5 4.75a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0M2.75 2a2.75 2.75 0 1 0 2.646 3.5H6.75v3.75A2.75 2.75 0 0 0 9.5 12h.104a2.751 2.751 0 1 0 0-1.5H9.5c-.69 0-1.25-.56-1.25-1.25V5.5h1.354a2.751 2.751 0 1 0 0-1.5H5.396A2.75 2.75 0 0 0 2.75 2m9.5 1.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m0 6.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5",
            clipRule: "evenodd"
        })
    });
}
const ForkHorizontalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgForkHorizontalIcon
    });
});
ForkHorizontalIcon.displayName = 'ForkHorizontalIcon';

function SvgForkIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2 2.75a2.75 2.75 0 1 1 3.5 2.646V6.75h3.75A2.75 2.75 0 0 1 12 9.5v.104a2.751 2.751 0 1 1-1.5 0V9.5c0-.69-.56-1.25-1.25-1.25H5.5v1.354a2.751 2.751 0 1 1-1.5 0V5.396A2.75 2.75 0 0 1 2 2.75M4.75 1.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5M3.5 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0m6.5 0a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0",
            clipRule: "evenodd"
        })
    });
}
const ForkIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgForkIcon
    });
});
ForkIcon.displayName = 'ForkIcon';

function SvgFullscreenExitIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M6 1v4.25a.75.75 0 0 1-.75.75H1V4.5h3.5V1zM10 15v-4.25a.75.75 0 0 1 .75-.75H15v1.5h-3.5V15zM10.75 6H15V4.5h-3.5V1H10v4.25c0 .414.336.75.75.75M1 10h4.25a.75.75 0 0 1 .75.75V15H4.5v-3.5H1z"
        })
    });
}
const FullscreenExitIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFullscreenExitIcon
    });
});
FullscreenExitIcon.displayName = 'FullscreenExitIcon';

function SvgFullscreenIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M6 1H1.75a.75.75 0 0 0-.75.75V6h1.5V2.5H6zM10 2.5V1h4.25a.75.75 0 0 1 .75.75V6h-1.5V2.5zM10 13.5h3.5V10H15v4.25a.75.75 0 0 1-.75.75H10zM2.5 10v3.5H6V15H1.75a.75.75 0 0 1-.75-.75V10z"
        })
    });
}
const FullscreenIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFullscreenIcon
    });
});
FullscreenIcon.displayName = 'FullscreenIcon';

function SvgFunctionIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#FunctionIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    fillRule: "evenodd",
                    d: "M9.93 2.988c-.774-.904-2.252-.492-2.448.682L7.094 6h2.005a2.75 2.75 0 0 1 2.585 1.81l.073.202 2.234-2.063 1.018 1.102-2.696 2.489.413 1.137c.18.494.65.823 1.175.823H15V13h-1.1a2.75 2.75 0 0 1-2.585-1.81l-.198-.547-2.61 2.408-1.017-1.102 3.07-2.834-.287-.792A1.25 1.25 0 0 0 9.099 7.5H6.844l-.846 5.076c-.405 2.43-3.464 3.283-5.067 1.412l1.139-.976c.774.904 2.252.492 2.448-.682l.805-4.83H3V6h2.573l.43-2.576C6.407.994 9.465.14 11.07 2.012z",
                    clipRule: "evenodd"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M16 0H0v16h16z"
                    })
                })
            })
        ]
    });
}
const FunctionIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgFunctionIcon
    });
});
FunctionIcon.displayName = 'FunctionIcon';

function SvgGavelIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 24 24",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M4 21v-2h12v2zm5.65-4.85L4 10.5l2.1-2.15L11.8 14zM16 9.8l-5.65-5.7L12.5 2l5.65 5.65zM20.6 20 7.55 6.95l1.4-1.4L22 18.6z"
        })
    });
}
const GavelIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgGavelIcon
    });
});
GavelIcon.displayName = 'GavelIcon';

function SvgGearFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M7.966 0q-.51 0-1.005.063a.75.75 0 0 0-.62.51l-.639 1.946q-.315.13-.61.294L3.172 2.1a.75.75 0 0 0-.784.165c-.481.468-.903.996-1.255 1.572a.75.75 0 0 0 .013.802l1.123 1.713a6 6 0 0 0-.15.66L.363 8.07a.75.75 0 0 0-.36.716c.067.682.22 1.34.447 1.962a.75.75 0 0 0 .635.489l2.042.19q.195.276.422.529l-.27 2.032a.75.75 0 0 0 .336.728 8 8 0 0 0 1.812.874.75.75 0 0 0 .778-.192l1.422-1.478a6 6 0 0 0 .677 0l1.422 1.478a.75.75 0 0 0 .778.192 8 8 0 0 0 1.812-.874.75.75 0 0 0 .335-.728l-.269-2.032a6 6 0 0 0 .422-.529l2.043-.19a.75.75 0 0 0 .634-.49c.228-.621.38-1.279.447-1.961a.75.75 0 0 0-.36-.716l-1.756-1.056a6 6 0 0 0-.15-.661l1.123-1.713a.75.75 0 0 0 .013-.802 8 8 0 0 0-1.255-1.572.75.75 0 0 0-.784-.165l-1.92.713q-.295-.163-.61-.294L9.589.573a.75.75 0 0 0-.619-.51A8 8 0 0 0 7.965 0m.018 10.25a2.25 2.25 0 1 0 0-4.5 2.25 2.25 0 0 0 0 4.5",
            clipRule: "evenodd"
        })
    });
}
const GearFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgGearFillIcon
    });
});
GearFillIcon.displayName = 'GearFillIcon';

function SvgGearIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                fillRule: "evenodd",
                clipPath: "url(#GearIcon_svg__a)",
                clipRule: "evenodd",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M7.984 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6m-1.5 3a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M7.966 0q-.51 0-1.005.063a.75.75 0 0 0-.62.51l-.639 1.946q-.315.13-.61.294L3.172 2.1a.75.75 0 0 0-.784.165c-.481.468-.903.996-1.255 1.572a.75.75 0 0 0 .013.802l1.123 1.713a6 6 0 0 0-.15.66L.363 8.07a.75.75 0 0 0-.36.716c.067.682.22 1.34.447 1.962a.75.75 0 0 0 .635.489l2.042.19q.195.276.422.529l-.27 2.032a.75.75 0 0 0 .336.728 8 8 0 0 0 1.812.874.75.75 0 0 0 .778-.192l1.422-1.478a6 6 0 0 0 .677 0l1.422 1.478a.75.75 0 0 0 .778.192 8 8 0 0 0 1.812-.874.75.75 0 0 0 .335-.728l-.269-2.032a6 6 0 0 0 .422-.529l2.043-.19a.75.75 0 0 0 .634-.49c.228-.621.38-1.279.447-1.961a.75.75 0 0 0-.36-.716l-1.756-1.056a6 6 0 0 0-.15-.661l1.123-1.713a.75.75 0 0 0 .013-.802 8 8 0 0 0-1.255-1.572.75.75 0 0 0-.784-.165l-1.92.713q-.295-.163-.61-.294L9.589.573a.75.75 0 0 0-.619-.51A8 8 0 0 0 7.965 0m-.95 3.328.597-1.819a7 7 0 0 1 .705 0l.597 1.819a.75.75 0 0 0 .472.476q.519.177.97.468a.75.75 0 0 0 .668.073l1.795-.668q.234.264.44.552l-1.05 1.6a.75.75 0 0 0-.078.667q.181.501.24 1.05a.75.75 0 0 0 .359.567l1.642.988q-.06.351-.156.687l-1.909.178a.75.75 0 0 0-.569.353q-.287.463-.672.843a.75.75 0 0 0-.219.633l.252 1.901a7 7 0 0 1-.635.306l-1.33-1.381a.75.75 0 0 0-.63-.225 4.5 4.5 0 0 1-1.08 0 .75.75 0 0 0-.63.225l-1.33 1.381a7 7 0 0 1-.634-.306l.252-1.9a.75.75 0 0 0-.219-.634 4.5 4.5 0 0 1-.672-.843.75.75 0 0 0-.569-.353l-1.909-.178a7 7 0 0 1-.156-.687L3.2 8.113a.75.75 0 0 0 .36-.567q.056-.549.239-1.05a.75.75 0 0 0-.078-.666L2.67 4.229q.206-.288.44-.552l1.795.668a.75.75 0 0 0 .667-.073c.3-.193.626-.351.97-.468a.75.75 0 0 0 .472-.476"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const GearIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgGearIcon
    });
});
GearIcon.displayName = 'GearIcon';

function SvgGenieDeepResearchIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M6.25 8c0 1.415.114 2.716.308 3.683.095.477.217.911.375 1.245q.04.086.092.176c.189.496.389.864.583 1.102.228.279.364.294.392.294s.164-.015.392-.294c.222-.273.456-.713.667-1.324q.026-.081.051-.164.826-.046 1.605-.156c-.074.284-.15.555-.238.81-.242.7-.547 1.322-.922 1.782C9.184 15.61 8.663 16 8 16c-.632 0-1.447-.594-2.146-2.125-.615-1.344-1.03-3.202-1.094-5.302q-.009-.282-.01-.57a15 15 0 0 1 1.502-.188z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M12.562 5.283q.427.11.81.24c.7.242 1.322.547 1.782.922.455.371.846.892.846 1.555 0 .632-.594 1.447-2.125 2.146-1.28.586-3.029.99-5.008 1.083q-.426.019-.864.02a15 15 0 0 1-.188-1.502q.093.002.185.003c1.415 0 2.716-.114 3.683-.308.477-.095.911-.217 1.245-.375q.087-.041.176-.093c.496-.189.864-.388 1.102-.582.279-.228.294-.364.294-.392s-.015-.164-.294-.392c-.273-.222-.713-.456-1.324-.667l-.164-.052q-.045-.826-.156-1.606M7.996 4.75c.079.428.144.936.188 1.502L8 6.25c-1.415 0-2.716.114-3.683.308-.477.095-.911.217-1.245.375a2 2 0 0 0-.177.092c-.496.189-.863.39-1.101.583-.279.228-.294.364-.294.392s.015.164.294.392c.273.222.713.456 1.324.667q.08.026.163.051.046.826.156 1.606a11 11 0 0 1-.808-.24c-.701-.241-1.323-.546-1.783-.921C.39 9.184 0 8.663 0 8c0-.632.594-1.447 2.125-2.146 1.343-.614 3.2-1.03 5.297-1.094q.285-.009.574-.01"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 0c.632 0 1.447.594 2.146 2.125.608 1.329 1.02 3.16 1.092 5.23q.01.318.011.641a15 15 0 0 1-1.502.188L9.75 8c0-1.415-.114-2.716-.308-3.683-.095-.477-.217-.911-.375-1.245a2 2 0 0 0-.093-.177c-.189-.496-.388-.863-.582-1.101C8.164 1.515 8.028 1.5 8 1.5s-.164.015-.392.294c-.222.273-.456.713-.667 1.324q-.027.08-.052.163-.826.046-1.606.156.11-.426.24-.808c.242-.701.547-1.323.922-1.783C6.816.39 7.337 0 8 0"
            })
        ]
    });
}
const GenieDeepResearchIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgGenieDeepResearchIcon
    });
});
GenieDeepResearchIcon.displayName = 'GenieDeepResearchIcon';

function SvgGiftIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M3 3.25A2.25 2.25 0 0 1 5.25 1C6.365 1 7.36 1.522 8 2.335A3.5 3.5 0 0 1 10.75 1a2.25 2.25 0 0 1 2.122 3h1.378a.75.75 0 0 1 .75.75v3a.75.75 0 0 1-.75.75H14v5.75a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75V8.5h-.25A.75.75 0 0 1 1 7.75v-3A.75.75 0 0 1 1.75 4h1.378A2.3 2.3 0 0 1 3 3.25M5.25 4h1.937A2 2 0 0 0 5.25 2.5a.75.75 0 0 0 0 1.5m2 1.5H2.5V7h4.75zm0 3H3.5v5h3.75zm1.5 5v-5h3.75v5zm0-6.5V5.5h4.75V7zm.063-3h1.937a.75.75 0 0 0 0-1.5A2 2 0 0 0 8.813 4",
            clipRule: "evenodd"
        })
    });
}
const GiftIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgGiftIcon
    });
});
GiftIcon.displayName = 'GiftIcon';

function SvgGitCommitIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 5.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5M4.07 7.25a4.001 4.001 0 0 1 7.86 0H16v1.5h-4.07a4.001 4.001 0 0 1-7.86 0H0v-1.5z",
            clipRule: "evenodd"
        })
    });
}
const GitCommitIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgGitCommitIcon
    });
});
GitCommitIcon.displayName = 'GitCommitIcon';

function SvgGlobeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#GlobeIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    fillRule: "evenodd",
                    d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m5.354-5.393q.132-.347.287-.666A6.51 6.51 0 0 0 1.543 7.25h2.971c.067-1.777.368-3.399.84-4.643m.661 4.643c.066-1.627.344-3.062.742-4.11.23-.607.485-1.046.73-1.32.247-.274.421-.32.513-.32s.266.046.512.32.501.713.731 1.32c.398 1.048.676 2.483.742 4.11zm3.97 1.5h-3.97c.066 1.627.344 3.062.742 4.11.23.607.485 1.046.73 1.32.247.274.421.32.513.32s.266-.046.512-.32.501-.713.731-1.32c.398-1.048.676-2.483.742-4.11m1.501-1.5c-.067-1.777-.368-3.399-.84-4.643a8 8 0 0 0-.287-.666 6.51 6.51 0 0 1 4.098 5.309zm2.971 1.5h-2.971c-.067 1.777-.368 3.399-.84 4.643a8 8 0 0 1-.287.666 6.51 6.51 0 0 0 4.098-5.309m-9.943 0H1.543a6.51 6.51 0 0 0 4.098 5.309 8 8 0 0 1-.287-.666c-.472-1.244-.773-2.866-.84-4.643",
                    clipRule: "evenodd"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 16h16V0H0z"
                    })
                })
            })
        ]
    });
}
const GlobeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgGlobeIcon
    });
});
GlobeIcon.displayName = 'GlobeIcon';

function SvgGridDashIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M1 1.75V4h1.5V2.5H4V1H1.75a.75.75 0 0 0-.75.75M15 14.25V12h-1.5v1.5H12V15h2.25a.75.75 0 0 0 .75-.75M12 1h2.25a.75.75 0 0 1 .75.75V4h-1.5V2.5H12zM1.75 15H4v-1.5H2.5V12H1v2.25a.75.75 0 0 0 .75.75M10 2.5H6V1h4zM6 15h4v-1.5H6zM13.5 10V6H15v4zM1 6v4h1.5V6z"
        })
    });
}
const GridDashIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgGridDashIcon
    });
});
GridDashIcon.displayName = 'GridDashIcon';

function SvgGridIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.75 1a.75.75 0 0 0-.75.75v4.5c0 .414.336.75.75.75h4.5A.75.75 0 0 0 7 6.25v-4.5A.75.75 0 0 0 6.25 1zm.75 4.5v-3h3v3zM1.75 9a.75.75 0 0 0-.75.75v4.5c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-4.5A.75.75 0 0 0 6.25 9zm.75 4.5v-3h3v3zM9 1.75A.75.75 0 0 1 9.75 1h4.5a.75.75 0 0 1 .75.75v4.49a.75.75 0 0 1-.75.75h-4.5A.75.75 0 0 1 9 6.24zm1.5.75v2.99h3V2.5zM9.75 9a.75.75 0 0 0-.75.75v4.5c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-4.5a.75.75 0 0 0-.75-.75zm.75 4.5v-3h3v3z",
            clipRule: "evenodd"
        })
    });
}
const GridIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgGridIcon
    });
});
GridIcon.displayName = 'GridIcon';

function SvgH1Icon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M1 3v10h1.5V8.75H6V13h1.5V3H6v4.25H2.5V3zM11.25 3A2.25 2.25 0 0 1 9 5.25v1.5c.844 0 1.623-.279 2.25-.75v5.5H9V13h6v-1.5h-2.25V3z"
        })
    });
}
const H1Icon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgH1Icon
    });
});
H1Icon.displayName = 'H1Icon';

function SvgH2Icon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M1 3v10h1.5V8.75H6V13h1.5V3H6v4.25H2.5V3zM11.75 3A2.75 2.75 0 0 0 9 5.75V6h1.5v-.25c0-.69.56-1.25 1.25-1.25h.39a1.36 1.36 0 0 1 .746 2.498L10.692 8.44A3.75 3.75 0 0 0 9 11.574V13h6v-1.5h-4.499a2.25 2.25 0 0 1 1.014-1.807l2.194-1.44A2.86 2.86 0 0 0 12.14 3z"
        })
    });
}
const H2Icon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgH2Icon
    });
});
H2Icon.displayName = 'H2Icon';

function SvgH3Icon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M1 3h1.5v4.25H6V3h1.5v10H6V8.75H2.5V13H1zM9 5.75A2.75 2.75 0 0 1 11.75 3h.375a2.875 2.875 0 0 1 1.937 5 2.875 2.875 0 0 1-1.937 5h-.375A2.75 2.75 0 0 1 9 10.25V10h1.5v.25c0 .69.56 1.25 1.25 1.25h.375a1.375 1.375 0 1 0 0-2.75H11v-1.5h1.125a1.375 1.375 0 1 0 0-2.75h-.375c-.69 0-1.25.56-1.25 1.25V6H9z"
        })
    });
}
const H3Icon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgH3Icon
    });
});
H3Icon.displayName = 'H3Icon';

function SvgH4Icon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M2.5 7.25H6V3h1.5v10H6V8.75H2.5V13H1V3h1.5zM13.249 3a.75.75 0 0 1 .75.75L14 9.5h1V11h-1v2h-1.5v-2H9.25a.75.75 0 0 1-.75-.75V9.5a.75.75 0 0 1 .097-.37l3.25-5.75.055-.083A.75.75 0 0 1 12.5 3zm-3.137 6.5H12.5l-.001-4.224z"
        })
    });
}
const H4Icon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgH4Icon
    });
});
H4Icon.displayName = 'H4Icon';

function SvgH5Icon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M2.5 7.25H6V3h1.5v10H6V8.75H2.5V13H1V3h1.5zM14.5 4.5h-4v2.097a3.4 3.4 0 0 1 2.013-.27C13.803 6.55 15 7.556 15 9.25V10a3 3 0 0 1-6 0h1.5a1.5 1.5 0 0 0 3 0v-.75c0-.834-.539-1.324-1.244-1.446-.679-.118-1.381.133-1.756.71V8.5H9V3.75A.75.75 0 0 1 9.75 3h4.75z"
        })
    });
}
const H5Icon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgH5Icon
    });
});
H5Icon.displayName = 'H5Icon';

function SvgH6Icon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M2.5 7.25H6V3h1.5v10H6V8.75H2.5V13H1V3h1.5zM12.125 3c1.167 0 2.17.695 2.62 1.69l-1.366.62a1.38 1.38 0 0 0-1.254-.81h-.375c-.69 0-1.25.56-1.25 1.25v1.154A3 3 0 0 1 15 9.5v.5a3 3 0 0 1-5.996.154L9 10V5.75A2.75 2.75 0 0 1 11.75 3zM12 8a1.5 1.5 0 0 0-1.5 1.5v.5a1.5 1.5 0 0 0 3 0v-.5A1.5 1.5 0 0 0 12 8"
        })
    });
}
const H6Icon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgH6Icon
    });
});
H6Icon.displayName = 'H6Icon';

function SvgHashIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M6.868 4.836h3.359l.58-3.495h1.521l-.58 3.495h2.475l-.252 1.5h-2.473l-.444 2.675h2.467l-.252 1.5h-2.464l-.689 4.148H8.897l-.293-.049.681-4.1h-3.36l-.688 4.15H3.704l.702-4.15H2.295l.259-1.5h2.101L5.1 6.337H3.017l.26-1.5h2.07l.58-3.495H7.45zM6.175 9.01h3.36l.443-2.675H6.619z"
        })
    });
}
const HashIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgHashIcon
    });
});
HashIcon.displayName = 'HashIcon';

function SvgHistoryIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#HistoryIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "m3.507 7.73.963-.962 1.06 1.06-2.732 2.732L-.03 7.732l1.06-1.06.979.978a7 7 0 1 1 2.041 5.3l1.061-1.06a5.5 5.5 0 1 0-1.604-4.158"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M8.25 8V4h1.5v3.69l1.78 1.78-1.06 1.06-2-2A.75.75 0 0 1 8.25 8"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const HistoryIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgHistoryIcon
    });
});
HistoryIcon.displayName = 'HistoryIcon';

function SvgHomeIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M7.625 1.1a.75.75 0 0 1 .75 0l6.25 3.61a.75.75 0 0 1 .375.65v8.89a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75V10H7v4.25a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75V5.355a.75.75 0 0 1 .375-.65zM2.5 5.79V13.5h3V9.25a.75.75 0 0 1 .75-.75h3.5a.75.75 0 0 1 .75.75v4.25h3V5.792L8 2.616z",
            clipRule: "evenodd"
        })
    });
}
const HomeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgHomeIcon
    });
});
HomeIcon.displayName = 'HomeIcon';

function SvgImageIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M6.25 3.998a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5m-.75 2.25a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.492a.75.75 0 0 1-.75.75H5.038l-.009.009-.008-.009H1.75a.75.75 0 0 1-.75-.75zm12.5 11.742H6.544l4.455-4.436 2.47 2.469.031-.03zm0-10.992v6.934l-1.97-1.968a.75.75 0 0 0-1.06-.001l-6.052 6.027H2.5V2.5z",
                clipRule: "evenodd"
            })
        ]
    });
}
const ImageIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgImageIcon
    });
});
ImageIcon.displayName = 'ImageIcon';

function SvgIndentDecreaseIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M16 2H0v1.5h16zM16 5.5H8V7h8zM16 9H8v1.5h8zM0 12.5V14h16v-1.5zM6.06 6.03 5 4.97 1.97 8 5 11.03l1.06-1.06L4.092 8z"
        })
    });
}
const IndentDecreaseIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgIndentDecreaseIcon
    });
});
IndentDecreaseIcon.displayName = 'IndentDecreaseIcon';

function SvgIndentIncreaseIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M16 2H0v1.5h16zM16 5.5H8V7h8zM16 9H8v1.5h8zM0 12.5V14h16v-1.5zM1.97 6.03l1.06-1.06L6.06 8l-3.03 3.03-1.06-1.06L3.94 8z"
        })
    });
}
const IndentIncreaseIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgIndentIncreaseIcon
    });
});
IndentIncreaseIcon.displayName = 'IndentIncreaseIcon';

function SvgInfinityIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m8 6.94 1.59-1.592a3.75 3.75 0 1 1 0 5.304L8 9.06l-1.591 1.59a3.75 3.75 0 1 1 0-5.303zm2.652-.531a2.25 2.25 0 1 1 0 3.182L9.06 8zM6.939 8 5.35 6.409a2.25 2.25 0 1 0 0 3.182l1.588-1.589z",
            clipRule: "evenodd"
        })
    });
}
const InfinityIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgInfinityIcon
    });
});
InfinityIcon.displayName = 'InfinityIcon';

function SvgInfoBookIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M11.75 4.5a.75.75 0 1 1 0 1.5.75.75 0 0 1 0-1.5M12.5 7.75a.75.75 0 0 0-.75-.75h-1.5v1.5H11V11h1.5z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M0 2.75A.75.75 0 0 1 .75 2h4.396A3.75 3.75 0 0 1 8 3.317 3.75 3.75 0 0 1 10.854 2h4.396a.75.75 0 0 1 .75.75v10.5a.75.75 0 0 1-.75.75h-4.396a2.25 2.25 0 0 0-2.012 1.244l-.171.341a.75.75 0 0 1-1.342 0l-.17-.341A2.25 2.25 0 0 0 5.145 14H.75a.75.75 0 0 1-.75-.75zm1.5.75v9h3.646c.765 0 1.494.233 2.104.646V4.927l-.092-.183A2.25 2.25 0 0 0 5.146 3.5zm7.25 1.427v8.219a3.75 3.75 0 0 1 2.104-.646H14.5v-9h-3.646a2.25 2.25 0 0 0-2.012 1.244z",
                clipRule: "evenodd"
            })
        ]
    });
}
const InfoBookIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgInfoBookIcon
    });
});
InfoBookIcon.displayName = 'InfoBookIcon';

function SvgInfoFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0m-8.75 3V7h1.5v4zM8 4.5A.75.75 0 1 1 8 6a.75.75 0 0 1 0-1.5",
            clipRule: "evenodd"
        })
    });
}
const InfoFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgInfoFillIcon
    });
});
InfoFillIcon.displayName = 'InfoFillIcon';

function SvgInfoIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0m.75 5.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0M7.25 11V7h1.5v4z",
            clipRule: "evenodd"
        })
    });
}
const InfoIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgInfoIcon
    });
});
InfoIcon.displayName = 'InfoIcon';

function SvgIngestionIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M15 2.5a.75.75 0 0 0-.75-.75h-3a.75.75 0 0 0-.75.75V6H12V3.25h1.5v9.5H12V10h-1.5v3.5c0 .414.336.75.75.75h3a.75.75 0 0 0 .75-.75z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M3.75 0c1.26 0 2.322.848 2.648 2.004A2.75 2.75 0 0 1 9 4.75v2.5h3v1.5H9v2.5a2.75 2.75 0 0 1-2.602 2.746 2.751 2.751 0 1 1-3.47-3.371 2.751 2.751 0 0 1 0-5.25A2.751 2.751 0 0 1 3.75 0M5 2.75a1.25 1.25 0 1 0-2.5 0 1.25 1.25 0 0 0 2.5 0m-.428 2.625a2.76 2.76 0 0 0 1.822-1.867A1.25 1.25 0 0 1 7.5 4.75v2.5H6.396a2.76 2.76 0 0 0-1.824-1.875M6.396 8.75H7.5v2.5a1.25 1.25 0 0 1-1.106 1.242 2.76 2.76 0 0 0-1.822-1.867A2.76 2.76 0 0 0 6.396 8.75M3.75 12a1.25 1.25 0 1 1 0 2.5 1.25 1.25 0 0 1 0-2.5m0-5.25a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5",
                clipRule: "evenodd"
            })
        ]
    });
}
const IngestionIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgIngestionIcon
    });
});
IngestionIcon.displayName = 'IngestionIcon';

function SvgItalicIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M9.648 4.5H12V3H6v1.5h2.102l-1.75 7H4V13h6v-1.5H7.898z",
            clipRule: "evenodd"
        })
    });
}
const ItalicIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgItalicIcon
    });
});
ItalicIcon.displayName = 'ItalicIcon';

function SvgJoinOperatorIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M10.25 2.25A5.75 5.75 0 1 1 8 13.292 5.75 5.75 0 1 1 8 2.707a5.7 5.7 0 0 1 2.25-.457m0 1.5q-.298 0-.586.04a5.73 5.73 0 0 1 1.167 6.897q-.28.062-.581.063a2.7 2.7 0 0 1-1.085-.223 4.22 4.22 0 0 0 .827-2.77l-.008-.099a4 4 0 0 0-.02-.198q0-.018-.004-.036a4.2 4.2 0 0 0-.265-1.002l-.017-.046a4 4 0 0 0-.347-.663l-.031-.05a4 4 0 0 0-.29-.388l-.055-.064-.101-.112-.066-.07a4 4 0 0 0-.137-.133l-.03-.029A4.27 4.27 0 0 0 6.334 3.79l-.031-.004a4 4 0 0 0-.208-.021l-.091-.007a4.25 4.25 0 1 0 .331 8.451A5.73 5.73 0 0 1 4.5 8c0-.971.242-1.885.667-2.687a2.76 2.76 0 0 1 1.667.159 4.23 4.23 0 0 0-.827 2.77l.008.099q.006.099.02.198 0 .018.004.036.036.263.102.513l.014.05q.024.087.052.173l.03.088c.17.492.43.943.76 1.333l.024.03q.069.079.14.155l.036.038a4.23 4.23 0 0 0 2.735 1.281l.04.003.06.005.218.006a4.25 4.25 0 0 0 0-8.5",
            clipRule: "evenodd"
        })
    });
}
const JoinOperatorIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgJoinOperatorIcon
    });
});
JoinOperatorIcon.displayName = 'JoinOperatorIcon';

function SvgKeyIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M0 8a4 4 0 0 1 7.93-.75h7.32A.75.75 0 0 1 16 8v3h-1.5V8.75H13V11h-1.5V8.75H7.93A4.001 4.001 0 0 1 0 8m4-2.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5",
            clipRule: "evenodd"
        })
    });
}
const KeyIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgKeyIcon
    });
});
KeyIcon.displayName = 'KeyIcon';

function SvgKeyboardIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75zm.75 10.5v-9h13v9zm2.75-8h-1.5V6h1.5zm1.5 0V6h1.5V4.5zm3 0V6h1.5V4.5zm3 0V6h1.5V4.5zm-1.5 2.75h-1.5v1.5h1.5zm1.5 1.5v-1.5h1.5v1.5zm-4.5 0v-1.5h-1.5v1.5zm-3 0v-1.5h-1.5v1.5zM11 10H5v1.5h6z",
            clipRule: "evenodd"
        })
    });
}
const KeyboardIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgKeyboardIcon
    });
});
KeyboardIcon.displayName = 'KeyboardIcon';

function SvgLayerGraphIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1 3.75a2.75 2.75 0 1 1 3.5 2.646v3.208c.916.259 1.637.98 1.896 1.896h3.208a2.751 2.751 0 1 1 0 1.5H6.396A2.751 2.751 0 1 1 3 9.604V6.396A2.75 2.75 0 0 1 1 3.75m11.25 9.75a1.25 1.25 0 1 1 0-2.5 1.25 1.25 0 0 1 0 2.5m-8.5-11a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m0 8.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M10 1.5h3.75a.75.75 0 0 1 .75.75V6H13V3h-3z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M7.75 4a.75.75 0 0 0-.75.75v3.5c0 .414.336.75.75.75h3.5a.75.75 0 0 0 .75-.75v-3.5a.75.75 0 0 0-.75-.75zm.75 3.5v-2h2v2z",
                clipRule: "evenodd"
            })
        ]
    });
}
const LayerGraphIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLayerGraphIcon
    });
});
LayerGraphIcon.displayName = 'LayerGraphIcon';

function SvgLayerIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M13.5 2.5H7V1h7.25a.75.75 0 0 1 .75.75V9h-1.5z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1 7.75A.75.75 0 0 1 1.75 7h6.5a.75.75 0 0 1 .75.75v6.5a.75.75 0 0 1-.75.75h-6.5a.75.75 0 0 1-.75-.75zm1.5.75v5h5v-5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M4 5.32h6.5V12H12V4.57a.75.75 0 0 0-.75-.75H4z"
            })
        ]
    });
}
const LayerIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLayerIcon
    });
});
LayerIcon.displayName = 'LayerIcon';

function SvgLeafIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M5 6a2.75 2.75 0 0 0 2.75 2.75h2.395a2 2 0 1 0 0-1.5H7.75C7.06 7.25 6.5 6.69 6.5 6V2H5zm6.5 2a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0",
            clipRule: "evenodd"
        })
    });
}
const LeafIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLeafIcon
    });
});
LeafIcon.displayName = 'LeafIcon';

function SvgLetterFormatIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M3.03 4a1 1 0 0 1 .976.78l.015.092L4.942 12H3.43l-.194-1.5h-1.47L1.57 12H.058l.92-7.128.016-.092A1 1 0 0 1 1.97 4zM1.958 9h1.084l-.451-3.5h-.182zM7.75 4A2.25 2.25 0 0 1 10 6.25v.25c0 .453-.136.874-.367 1.228.527.411.867 1.051.867 1.772v.25A2.25 2.25 0 0 1 8.25 12h-1.5a.75.75 0 0 1-.75-.75v-6.5A.75.75 0 0 1 6.75 4zm-.25 6.5h.75A.75.75 0 0 0 9 9.75V9.5a.75.75 0 0 0-.75-.75H7.5zm0-3.25h.25a.75.75 0 0 0 .75-.75v-.25a.75.75 0 0 0-.75-.75H7.5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M13.75 4A2.25 2.25 0 0 1 16 6.25v.25h-1.5v-.25a.75.75 0 0 0-1.5 0v3.5a.75.75 0 0 0 1.5 0V9.5H16v.25a2.25 2.25 0 0 1-4.5 0v-3.5A2.25 2.25 0 0 1 13.75 4"
            })
        ]
    });
}
const LetterFormatIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLetterFormatIcon
    });
});
LetterFormatIcon.displayName = 'LetterFormatIcon';

function SvgLettersIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M6.25 1h2.174a2.126 2.126 0 0 1 1.81 3.243 2.126 2.126 0 0 1-1.36 3.761H6.25a.75.75 0 0 1-.75-.75V1.75A.75.75 0 0 1 6.25 1M7 6.504V5.252h1.874a.626.626 0 1 1 0 1.252zm2.05-3.378c0 .345-.28.625-.625.626H7.001L7 2.5h1.424c.346 0 .626.28.626.626M3.307 6a.75.75 0 0 1 .697.473L6.596 13H4.982l-.238-.6H1.855l-.24.6H0l2.61-6.528A.75.75 0 0 1 3.307 6m-.003 2.776.844 2.124H2.455z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M12.5 15a2.5 2.5 0 0 0 2.5-2.5h-1.5a1 1 0 1 1-2 0v-1.947c0-.582.472-1.053 1.053-1.053.523 0 .947.424.947.947v.053H15v-.053A2.447 2.447 0 0 0 12.553 8 2.553 2.553 0 0 0 10 10.553V12.5a2.5 2.5 0 0 0 2.5 2.5"
            })
        ]
    });
}
const LettersIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLettersIcon
    });
});
LettersIcon.displayName = 'LettersIcon';

function SvgLettersNumbersIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M4.273 1.534a.75.75 0 0 0-1.429-.023L1 7h1.582l.137-.407h1.509L4.35 7h1.566zm-.496 3.559h-.554l.292-.87z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M13.055 1a2 2 0 0 0-2 2v2a2 2 0 1 0 4 0h-1.5a.5.5 0 0 1-1 0V3a.5.5 0 1 1 1 0h1.5a2 2 0 0 0-2-2M2.305 9a1 1 0 0 1-1 1h-.25v1.5h.25c.356 0 .694-.074 1-.208V13.5h-1.25V15h4v-1.5h-1.25V9zM5.555 11.012c0-1.111.9-2.012 2.012-2.012h.656a1.876 1.876 0 0 1 .665 3.63l-1.302.495a.82.82 0 0 0-.43.375h2.9V15h-4.5v-1.106c-.001-.965.596-1.83 1.498-2.171l1.302-.495a.376.376 0 0 0-.133-.728h-.656a.51.51 0 0 0-.512.512zM13.44 10.512a.38.38 0 0 1 .383.374.376.376 0 0 1-.368.381h-.903l.006 1.5h.9a.366.366 0 0 1-.002.733h-.883a.5.5 0 0 1-.5-.5h-1.5a2 2 0 0 0 2 2h.883a1.866 1.866 0 0 0 1.496-2.983c.238-.319.377-.716.37-1.145a1.89 1.89 0 0 0-1.905-1.86l-.89.01a1.973 1.973 0 0 0-1.954 1.975l1.5.006c0-.264.212-.479.477-.481z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M6.305 1.75a.75.75 0 0 1 .75-.75H8.43a1.875 1.875 0 0 1 1.611 2.835A1.875 1.875 0 0 1 8.68 7H7.055a.75.75 0 0 1-.75-.75zm2.5 1.125a.375.375 0 0 1-.375.375h-.625V2.5h.625c.207 0 .375.168.375.375m-1 2.625v-.75h.876a.375.375 0 1 1 0 .75z",
                clipRule: "evenodd"
            })
        ]
    });
}
const LettersNumbersIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLettersNumbersIcon
    });
});
LettersNumbersIcon.displayName = 'LettersNumbersIcon';

function SvgLibrariesIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "m8.301 1.522 5.25 13.5 1.398-.544-5.25-13.5zM1 15V1h1.5v14zM5 15V1h1.5v14z"
        })
    });
}
const LibrariesIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLibrariesIcon
    });
});
LibrariesIcon.displayName = 'LibrariesIcon';

function SvgLifesaverIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#LifesaverIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    d: "M8.5 0a8 8 0 1 1 0 16 8 8 0 0 1 0-16M4.446 13.08A6.47 6.47 0 0 0 8.5 14.5a6.47 6.47 0 0 0 4.028-1.401l-1.419-1.433a4.48 4.48 0 0 1-2.609.835l-.231-.006a4.5 4.5 0 0 1-2.405-.847zm7.674-7.754A4.48 4.48 0 0 1 13.001 8l-.006.231a4.47 4.47 0 0 1-.825 2.373l1.42 1.435A6.47 6.47 0 0 0 15 8a6.47 6.47 0 0 0-1.462-4.106zM3.441 3.918A6.47 6.47 0 0 0 2 8c0 1.516.52 2.91 1.39 4.015l1.422-1.437A4.5 4.5 0 0 1 4 8c0-.99.32-1.905.861-2.647zM8.5 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6m0-3.5a6.47 6.47 0 0 0-3.987 1.368l1.42 1.436A4.48 4.48 0 0 1 8.5 3.5c.942 0 1.817.29 2.54.785l1.423-1.436A6.47 6.47 0 0 0 8.5 1.5"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M.5 0h16.25v16H.5z"
                    })
                })
            })
        ]
    });
}
const LifesaverIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLifesaverIcon
    });
});
LifesaverIcon.displayName = 'LifesaverIcon';

function SvgLightbulbIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.25 0v2h1.5V0zM16 7.25h-2v1.5h2zM0 7.25h2v1.5H0zM13.127 1.813l-1.415 1.414 1.061 1.06 1.414-1.414zM2.874 1.813l1.414 1.414-1.06 1.06-1.415-1.414z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M3.25 8.221C3.25 5.61 5.382 3.5 8 3.5s4.75 2.109 4.75 4.721a4.7 4.7 0 0 1-.985 2.879c-.754.973-1.33 1.776-1.33 2.644v1.506a.75.75 0 0 1-.75.75h-3.37a.75.75 0 0 1-.75-.75v-1.506c0-.868-.576-1.67-1.33-2.644A4.7 4.7 0 0 1 3.25 8.22M8 5C6.2 5 4.75 6.447 4.75 8.221c0 .738.25 1.417.67 1.96l.044.056c.284.366.612.789.897 1.263h3.278c.285-.474.613-.897.897-1.263l.043-.056c.422-.543.671-1.222.671-1.96C11.25 6.447 9.8 5 8 5m-.934 8.744c0-.256-.03-.504-.081-.744h2.03q-.079.36-.08.744v.756h-1.87z",
                clipRule: "evenodd"
            })
        ]
    });
}
const LightbulbIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLightbulbIcon
    });
});
LightbulbIcon.displayName = 'LightbulbIcon';

function SvgLightningIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M9.49.04a.75.75 0 0 1 .51.71V6h3.25a.75.75 0 0 1 .596 1.206l-6.5 8.5A.75.75 0 0 1 6 15.25V10H2.75a.75.75 0 0 1-.596-1.206l6.5-8.5A.75.75 0 0 1 9.491.04M4.269 8.5H6.75a.75.75 0 0 1 .75.75v3.785L11.732 7.5H9.25a.75.75 0 0 1-.75-.75V2.965z",
            clipRule: "evenodd"
        })
    });
}
const LightningIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLightningIcon
    });
});
LightningIcon.displayName = 'LightningIcon';

function SvgLinkIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M4 4h3v1.5H4a2.5 2.5 0 0 0 0 5h3V12H4a4 4 0 0 1 0-8M12 10.5H9V12h3a4 4 0 0 0 0-8H9v1.5h3a2.5 2.5 0 0 1 0 5"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M4 8.75h8v-1.5H4z"
            })
        ]
    });
}
const LinkIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLinkIcon
    });
});
LinkIcon.displayName = 'LinkIcon';

function SvgLinkOffIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M14.035 11.444A4 4 0 0 0 12 4H9v1.5h3a2.5 2.5 0 0 1 .917 4.826zM14 13.53 2.47 2l-1 1 1.22 1.22A4.002 4.002 0 0 0 4 12h3v-1.5H4a2.5 2.5 0 0 1-.03-5l1.75 1.75H4v1.5h3.22L13 14.53z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m9.841 7.25 1.5 1.5H12v-1.5z"
            })
        ]
    });
}
const LinkOffIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLinkOffIcon
    });
});
LinkOffIcon.displayName = 'LinkOffIcon';

function SvgListBorderIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M12 8.75H7v-1.5h5zM7 5.5h5V4H7zM12 12H7v-1.5h5zM4.75 5.5a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5M5.5 8A.75.75 0 1 1 4 8a.75.75 0 0 1 1.5 0M4.75 12a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75v11h11v-11z",
                clipRule: "evenodd"
            })
        ]
    });
}
const ListBorderIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgListBorderIcon
    });
});
ListBorderIcon.displayName = 'ListBorderIcon';

function SvgListClearIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                fill: "currentColor",
                clipPath: "url(#ListClearIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    d: "M15.03 13.97 13.06 12l1.97-1.97-1.06-1.06L12 10.94l-1.97-1.97-1.06 1.06L10.94 12l-1.97 1.97 1.06 1.06L12 13.06l1.97 1.97zM5 11.5H1V10h4zM11 3.5H1V2h10zM7 7.5H1V6h6z"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 16h16V0H0z"
                    })
                })
            })
        ]
    });
}
const ListClearIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgListClearIcon
    });
});
ListClearIcon.displayName = 'ListClearIcon';

function SvgListNumberIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M4.76 8a2.24 2.24 0 0 1 .883 4.299l-1.431.612c-.273.117-.484.33-.604.589H7V15H2v-1.009c0-1.07.638-2.036 1.621-2.458l1.43-.613A.74.74 0 0 0 4.76 9.5h-.371a.89.89 0 0 0-.889.889H2C2 9.069 3.07 8 4.389 8zM14 12.75H9v-1.5h5zM5.25 5.5H7V7H2V5.5h1.75V2.595A3 3 0 0 1 2.25 3H2V1.5h.25A1.5 1.5 0 0 0 3.75 0h1.5zM14 4.75H9v-1.5h5z"
        })
    });
}
const ListNumberIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgListNumberIcon
    });
});
ListNumberIcon.displayName = 'ListNumberIcon';

function SvgLockFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M12 6V4a4 4 0 0 0-8 0v2H2.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75zM5.5 6h5V4a2.5 2.5 0 0 0-5 0zm1.75 7V9h1.5v4z",
            clipRule: "evenodd"
        })
    });
}
const LockFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLockFillIcon
    });
});
LockFillIcon.displayName = 'LockFillIcon';

function SvgLockShareIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M13.962 6.513a3.24 3.24 0 0 0-2.057.987H3.5v6.95H8v1.5H2.75A.75.75 0 0 1 2 15.2V6.75A.75.75 0 0 1 2.75 6H4V4a4 4 0 1 1 8 0v2h1.25a.75.75 0 0 1 .712.513M10.5 4v2h-5V4a2.5 2.5 0 0 1 5 0",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M11.5 12.036v-.072l1.671-.836a1.75 1.75 0 1 0-.67-1.342l-1.672.836a1.75 1.75 0 1 0 0 2.756l1.671.836v.036a1.75 1.75 0 1 0 .671-1.378z"
            })
        ]
    });
}
const LockShareIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLockShareIcon
    });
});
LockShareIcon.displayName = 'LockShareIcon';

function SvgLockUnlockedIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M10 11.75v-1.5H6v1.5z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M13.25 6H5.5V4a2.5 2.5 0 0 1 5 0v.5H12V4a4 4 0 0 0-8 0v2H2.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75M3.5 7.5h9v7h-9z",
                clipRule: "evenodd"
            })
        ]
    });
}
const LockUnlockedIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLockUnlockedIcon
    });
});
LockUnlockedIcon.displayName = 'LockUnlockedIcon';

function SvgLoopIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 17",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M3.75 2A2.75 2.75 0 0 0 1 4.75v6.5A2.75 2.75 0 0 0 3.75 14H5.5v-1.5H3.75c-.69 0-1.25-.56-1.25-1.25v-6.5c0-.69.56-1.25 1.25-1.25h8.5c.69 0 1.25.56 1.25 1.25v6.5c0 .69-.56 1.25-1.25 1.25H9.81l.97-.97-1.06-1.06-2.78 2.78 2.78 2.78 1.06-1.06-.97-.97h2.44A2.75 2.75 0 0 0 15 11.25v-6.5A2.75 2.75 0 0 0 12.25 2z"
        })
    });
}
const LoopIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgLoopIcon
    });
});
LoopIcon.displayName = 'LoopIcon';

function SvgMailIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75zm.75 2.347V12.5h13V4.347L9.081 8.604a1.75 1.75 0 0 1-2.162 0zM13.15 3.5H2.85l4.996 3.925a.25.25 0 0 0 .308 0z",
            clipRule: "evenodd"
        })
    });
}
const MailIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgMailIcon
    });
});
MailIcon.displayName = 'MailIcon';

function SvgMapIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m2.058 13.934 3.827-1.723 3.675 2.646.015.011a.75.75 0 0 0 .735.065l4.248-1.912a.75.75 0 0 0 .442-.684V2.75a.75.75 0 0 0-1.058-.684L10.115 3.79 6.44 1.143l-.015-.011a.75.75 0 0 0-.735-.065L1.442 2.979A.75.75 0 0 0 1 3.663v9.587a.75.75 0 0 0 1.058.684M2.5 4.148 5.25 2.91v7.942L2.5 12.09zm8.25 1v7.942l2.75-1.238V3.91zm-1.5-.134-2.5-1.8v7.772l2.5 1.8z",
            clipRule: "evenodd"
        })
    });
}
const MapIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgMapIcon
    });
});
MapIcon.displayName = 'MapIcon';

function SvgMarkdownIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "m13.75 10.125 1.207-1.268 1.086 1.035L13 13.088 9.957 9.892l1.086-1.035 1.207 1.268V6h1.5zM7.743 3.297A.752.752 0 0 1 9.05 3.8V13h-1.5V5.746L5.056 8.503a.75.75 0 0 1-1.118-.008L1.55 5.785V13H.05V3.8a.75.75 0 0 1 1.312-.496l3.145 3.569z"
        })
    });
}
const MarkdownIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgMarkdownIcon
    });
});
MarkdownIcon.displayName = 'MarkdownIcon';

function SvgMcpIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                fillRule: "evenodd",
                clipPath: "url(#McpIcon_svg__a)",
                clipRule: "evenodd",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M10.459 1.562a1.725 1.725 0 0 0-2.407 0L1.635 7.855a.575.575 0 0 1-.925-.18.55.55 0 0 1 .123-.606L7.25.775a2.875 2.875 0 0 1 4.01 0 2.74 2.74 0 0 1 .803 2.36 2.87 2.87 0 0 1 2.406.787l.034.033a2.743 2.743 0 0 1 0 3.934L8.699 13.58a.18.18 0 0 0 0 .262l1.192 1.17a.55.55 0 0 1 0 .786.576.576 0 0 1-.802 0L7.897 14.63a1.28 1.28 0 0 1 0-1.836L13.7 7.101a1.645 1.645 0 0 0 0-2.36l-.034-.032a1.725 1.725 0 0 0-2.404-.002L6.48 9.397H6.48l-.065.065a.575.575 0 0 1-.926-.18.55.55 0 0 1 .123-.607l4.849-4.755a1.645 1.645 0 0 0-.002-2.358"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M9.657 3.135a.55.55 0 0 0 0-.786.575.575 0 0 0-.803 0L4.108 7.003a2.743 2.743 0 0 0 0 3.934 2.876 2.876 0 0 0 4.01 0l4.747-4.655a.55.55 0 0 0 0-.787.575.575 0 0 0-.802 0L7.317 10.15a1.725 1.725 0 0 1-2.407 0 1.647 1.647 0 0 1 0-2.36z"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const McpIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgMcpIcon
    });
});
McpIcon.displayName = 'McpIcon';

function SvgMeasureIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M10.22.72a.75.75 0 0 1 1.06 0l4 4a.75.75 0 0 1 0 1.06l-9.5 9.5a.75.75 0 0 1-1.06 0l-4-4a.75.75 0 0 1 0-1.06zm.53 1.59-8.44 8.44 2.94 2.94 1.314-1.315-1.47-1.47 1.061-1.06 1.47 1.47L8.939 10 7.47 8.53 8.53 7.47 10 8.94l1.314-1.315-1.47-1.47 1.061-1.06 1.47 1.47 1.314-1.315z",
            clipRule: "evenodd"
        })
    });
}
const MeasureIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgMeasureIcon
    });
});
MeasureIcon.displayName = 'MeasureIcon';

function SvgMenuIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M15 4H1V2.5h14zm0 4.75H1v-1.5h14zm0 4.75H1V12h14z",
            clipRule: "evenodd"
        })
    });
}
const MenuIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgMenuIcon
    });
});
MenuIcon.displayName = 'MenuIcon';

function SvgMinusCircleFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16m3.5-7.25h-7v-1.5h7z",
            clipRule: "evenodd"
        })
    });
}
const MinusCircleFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgMinusCircleFillIcon
    });
});
MinusCircleFillIcon.displayName = 'MinusCircleFillIcon';

function SvgMinusCircleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M4.5 8.75v-1.5h7v1.5z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13",
                clipRule: "evenodd"
            })
        ]
    });
}
const MinusCircleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgMinusCircleIcon
    });
});
MinusCircleIcon.displayName = 'MinusCircleIcon';

function SvgMinusSquareIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M11.5 8.75h-7v-1.5h7z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z",
                clipRule: "evenodd"
            })
        ]
    });
}
const MinusSquareIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgMinusSquareIcon
    });
});
MinusSquareIcon.displayName = 'MinusSquareIcon';

function SvgModelsIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#ModelsIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    fillRule: "evenodd",
                    d: "M0 4.75a2.75 2.75 0 0 1 5.145-1.353l4.372-.95a2.75 2.75 0 1 1 3.835 2.823l.282 2.257a2.75 2.75 0 1 1-2.517 4.46l-2.62 1.145.003.118a2.75 2.75 0 1 1-4.415-2.19L3.013 7.489A2.75 2.75 0 0 1 0 4.75M2.75 3.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m2.715 1.688q.027-.164.033-.333l4.266-.928a2.75 2.75 0 0 0 2.102 1.546l.282 2.257c-.377.165-.71.412-.976.719zM4.828 6.55a2.8 2.8 0 0 1-.413.388l1.072 3.573q.13-.012.263-.012c.945 0 1.778.476 2.273 1.202l2.5-1.093a2.8 2.8 0 0 1 .012-.797zM12 10.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0M5.75 12a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5M11 2.75a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0",
                    clipRule: "evenodd"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const ModelsIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgModelsIcon
    });
});
ModelsIcon.displayName = 'ModelsIcon';

function SvgMoonIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M5.25 5c0-.682.119-1.336.337-1.943a5.5 5.5 0 1 0 7.354 7.355A5.75 5.75 0 0 1 5.25 5m1.5 0a4.25 4.25 0 0 0 6.962 3.271.75.75 0 0 1 1.222.678A7 7 0 1 1 7.05 1.065l.114-.006a.75.75 0 0 1 .564 1.228A4.23 4.23 0 0 0 6.75 5"
        })
    });
}
const MoonIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgMoonIcon
    });
});
MoonIcon.displayName = 'MoonIcon';

function SvgNeonProjectIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8.25.51a2 2 0 0 1 1.962 2.38l3.146 3.147Q13.544 6 13.74 6a2 2 0 1 1-.38 3.962l-3.147 3.146q.037.185.038.38a2 2 0 1 1-3.963-.38L3.141 9.962q-.185.037-.38.038a2 2 0 1 1 .38-3.963l3.146-3.146A2 2 0 0 1 8.25.51m1.118 3.658a1.99 1.99 0 0 1-2.237 0L4.419 6.88c.216.32.343.705.343 1.12s-.127.8-.343 1.12l2.71 2.71c.32-.216.706-.342 1.121-.342.414 0 .799.127 1.118.342l2.713-2.712A2 2 0 0 1 11.74 8c0-.415.126-.8.342-1.12zM8.25 5.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5m0 1.5a1 1 0 1 0 0 2 1 1 0 0 0 0-2"
        })
    });
}
const NeonProjectIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgNeonProjectIcon
    });
});
NeonProjectIcon.displayName = 'NeonProjectIcon';

function SvgNewChatIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8.318 2.5H3.75c-.69 0-1.25.56-1.25 1.25v8.5c0 .69.56 1.25 1.25 1.25h8.5c.69 0 1.25-.56 1.25-1.25V7.682l1.5-1.5v6.068A2.75 2.75 0 0 1 12.25 15h-8.5A2.75 2.75 0 0 1 1 12.25v-8.5A2.75 2.75 0 0 1 3.75 1h6.068z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12.263.677a1.75 1.75 0 0 1 2.474 0l.586.586a1.75 1.75 0 0 1 0 2.475L9.28 9.78a.75.75 0 0 1-.53.22h-2A.75.75 0 0 1 6 9.25v-2c0-.2.08-.39.22-.531zM7.5 7.561v.94h.94l4-4-.94-.94zm6.177-5.823a.25.25 0 0 0-.354 0l-.763.762.94.94.763-.763a.25.25 0 0 0 0-.353z",
                clipRule: "evenodd"
            })
        ]
    });
}
const NewChatIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgNewChatIcon
    });
});
NewChatIcon.displayName = 'NewChatIcon';

function SvgNoIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.5 8a6.5 6.5 0 0 1 10.535-5.096l-9.131 9.131A6.47 6.47 0 0 1 1.5 8m2.465 5.096a6.5 6.5 0 0 0 9.131-9.131z",
            clipRule: "evenodd"
        })
    });
}
const NoIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgNoIcon
    });
});
NoIcon.displayName = 'NoIcon';

function SvgNotebookIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M3 1.75A.75.75 0 0 1 3.75 1h10.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H3.75a.75.75 0 0 1-.75-.75V12.5H1V11h2V8.75H1v-1.5h2V5H1V3.5h2zm1.5.75v11H6v-11zm3 0v11h6v-11z",
            clipRule: "evenodd"
        })
    });
}
const NotebookIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgNotebookIcon
    });
});
NotebookIcon.displayName = 'NotebookIcon';

function SvgNotebookPipelineIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M3 1.75A.75.75 0 0 1 3.75 1h10.5a.75.75 0 0 1 .75.75V8h-1.5V2.5h-6v11h2V15H3.75a.75.75 0 0 1-.75-.75V12.5H1V11h2V8.75H1v-1.5h2V5H1V3.5h2zm1.5.75v11H6v-11z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M9.25 8.5a.75.75 0 0 0-.75.75v2.5c0 .414.336.75.75.75h.785a3.5 3.5 0 0 0 3.465 3h1.25a.75.75 0 0 0 .75-.75v-2.5a.75.75 0 0 0-.75-.75h-.785a3.5 3.5 0 0 0-3.465-3zM10 11v-1h.5a2 2 0 0 1 2 2 1 1 0 0 0 1 1h.5v1h-.5a2 2 0 0 1-2-2 1 1 0 0 0-1-1z",
                clipRule: "evenodd"
            })
        ]
    });
}
const NotebookPipelineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgNotebookPipelineIcon
    });
});
NotebookPipelineIcon.displayName = 'NotebookPipelineIcon';

function SvgNotificationIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 1a5 5 0 0 0-5 5v1.99c0 .674-.2 1.332-.573 1.892l-1.301 1.952A.75.75 0 0 0 1.75 13h3.5v.25a2.75 2.75 0 1 0 5.5 0V13h3.5a.75.75 0 0 0 .624-1.166l-1.301-1.952A3.4 3.4 0 0 1 13 7.99V6a5 5 0 0 0-5-5m1.25 12h-2.5v.25a1.25 1.25 0 1 0 2.5 0zM4.5 6a3.5 3.5 0 1 1 7 0v1.99c0 .97.287 1.918.825 2.724l.524.786H3.15l.524-.786A4.9 4.9 0 0 0 4.5 7.99z",
            clipRule: "evenodd"
        })
    });
}
const NotificationIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgNotificationIcon
    });
});
NotificationIcon.displayName = 'NotificationIcon';

function SvgNotificationOffIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m14.47 13.53-12-12-1 1L3.28 4.342A5 5 0 0 0 3 6v1.99c0 .674-.2 1.332-.573 1.892l-1.301 1.952A.75.75 0 0 0 1.75 13h3.5v.25a2.75 2.75 0 1 0 5.5 0V13h1.19l1.53 1.53zM13.038 8.5A3.4 3.4 0 0 1 13 7.99V6a5 5 0 0 0-7.965-4.026l1.078 1.078A3.5 3.5 0 0 1 11.5 6v1.99q0 .238.023.472l.038.038zM4.5 6q0-.21.024-.415L10.44 11.5H3.151l.524-.786A4.9 4.9 0 0 0 4.5 7.99zm2.25 7.25V13h2.5v.25a1.25 1.25 0 1 1-2.5 0",
            clipRule: "evenodd"
        })
    });
}
const NotificationOffIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgNotificationOffIcon
    });
});
NotificationOffIcon.displayName = 'NotificationOffIcon';

function SvgNumberFormatIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M2.738 10.5H4V12H0v-1.5h1.238V6.29a2.5 2.5 0 0 1-1 .21H0V5h.238a1 1 0 0 0 1-1h1.5zM7.75 4A2.25 2.25 0 0 1 10 6.25v.292c0 1.024-.579 1.96-1.495 2.419l-.814.407A1.25 1.25 0 0 0 7 10.486v.014h3V12H5.5v-1.514a2.75 2.75 0 0 1 1.52-2.46l.814-.407c.408-.204.666-.62.666-1.077V6.25a.75.75 0 0 0-1.5 0v.25H5.5v-.25A2.25 2.25 0 0 1 7.75 4M13.615 4A2.39 2.39 0 0 1 16 6.386c0 .627-.246 1.194-.644 1.617.399.425.644.994.644 1.622a2.375 2.375 0 1 1-4.75 0V9.5h1.5v.125a.875.875 0 1 0 .875-.875h-.622L13 8l-.004-.75h.633a.87.87 0 0 0 .871-.871.88.88 0 0 0-.879-.879.87.87 0 0 0-.871.87v.133l-1.5-.006v-.133A2.363 2.363 0 0 1 13.615 4"
        })
    });
}
const NumberFormatIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgNumberFormatIcon
    });
});
NumberFormatIcon.displayName = 'NumberFormatIcon';

function SvgNumbersIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M7.889 1A2.39 2.39 0 0 0 5.5 3.389H7c0-.491.398-.889.889-.889h.371a.74.74 0 0 1 .292 1.42l-1.43.613A2.68 2.68 0 0 0 5.5 6.992V8h5V6.5H7.108c.12-.26.331-.472.604-.588l1.43-.613A2.24 2.24 0 0 0 8.26 1zM2.75 6a1.5 1.5 0 0 1-1.5 1.5H1V9h.25c.546 0 1.059-.146 1.5-.401V11.5H1V13h5v-1.5H4.25V6zM10 12.85A2.15 2.15 0 0 0 12.15 15h.725a2.125 2.125 0 0 0 1.617-3.504 2.138 2.138 0 0 0-1.656-3.521l-.713.008A2.15 2.15 0 0 0 10 10.133v.284h1.5v-.284a.65.65 0 0 1 .642-.65l.712-.009a.638.638 0 1 1 .008 1.276H12v1.5h.875a.625.625 0 1 1 0 1.25h-.725a.65.65 0 0 1-.65-.65v-.267H10z"
        })
    });
}
const NumbersIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgNumbersIcon
    });
});
NumbersIcon.displayName = 'NumbersIcon';

function SvgOfficeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M4 8.75h8v-1.5H4zM7 5.75H4v-1.5h3zM4 11.75h8v-1.5H4z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V5a.75.75 0 0 0-.75-.75H10v-2.5A.75.75 0 0 0 9.25 1zm.75 1.5h6V5c0 .414.336.75.75.75h4.25v7.75h-11z",
                clipRule: "evenodd"
            })
        ]
    });
}
const OfficeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgOfficeIcon
    });
});
OfficeIcon.displayName = 'OfficeIcon';

function SvgOverflowIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8 1a1.75 1.75 0 1 0 0 3.5A1.75 1.75 0 0 0 8 1M8 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M8 11.5A1.75 1.75 0 1 0 8 15a1.75 1.75 0 0 0 0-3.5"
        })
    });
}
const OverflowIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgOverflowIcon
    });
});
OverflowIcon.displayName = 'OverflowIcon';

function SvgPageBottomIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 3.06 2.06 2l5.97 5.97L14 2l1.06 1.06-7.03 7.031zm14.03 10.47v1.5h-14v-1.5z",
            clipRule: "evenodd"
        })
    });
}
const PageBottomIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPageBottomIcon
    });
});
PageBottomIcon.displayName = 'PageBottomIcon';

function SvgPageFirstIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m12.97 1 1.06 1.06-5.97 5.97L14.03 14l-1.06 1.06-7.03-7.03zM2.5 15.03H1v-14h1.5z",
            clipRule: "evenodd"
        })
    });
}
const PageFirstIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPageFirstIcon
    });
});
PageFirstIcon.displayName = 'PageFirstIcon';

function SvgPageLastIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M3.06 1 2 2.06l5.97 5.97L2 14l1.06 1.06 7.031-7.03zm10.47 14.03h1.5v-14h-1.5z",
            clipRule: "evenodd"
        })
    });
}
const PageLastIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPageLastIcon
    });
});
PageLastIcon.displayName = 'PageLastIcon';

function SvgPageTopIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m1 12.97 1.06 1.06 5.97-5.97L14 14.03l1.06-1.06-7.03-7.03zM15.03 2.5V1h-14v1.5z",
            clipRule: "evenodd"
        })
    });
}
const PageTopIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPageTopIcon
    });
});
PageTopIcon.displayName = 'PageTopIcon';

function SvgPanelDockedIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H14.3a.7.7 0 0 0 .7-.7V1.75a.75.75 0 0 0-.75-.75zM8 13.5V8.7a.7.7 0 0 1 .7-.7h4.8V2.5h-11v11z",
            clipRule: "evenodd"
        })
    });
}
const PanelDockedIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPanelDockedIcon
    });
});
PanelDockedIcon.displayName = 'PanelDockedIcon';

function SvgPanelFloatingIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11zM5.6 5a.6.6 0 0 0-.6.6v4.8a.6.6 0 0 0 .6.6h4.8a.6.6 0 0 0 .6-.6V5.6a.6.6 0 0 0-.6-.6z",
            clipRule: "evenodd"
        })
    });
}
const PanelFloatingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPanelFloatingIcon
    });
});
PanelFloatingIcon.displayName = 'PanelFloatingIcon';

function SvgPaperclipIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M11.536 2.343a2.25 2.25 0 0 0-3.182 0l-4.95 4.95a3.75 3.75 0 1 0 5.303 5.303l4.066-4.066 1.06 1.06-4.065 4.067a5.25 5.25 0 1 1-7.425-7.425l4.95-4.95a3.75 3.75 0 1 1 5.303 5.304l-4.95 4.95a2.25 2.25 0 1 1-3.182-3.182l5.48-5.48 1.061 1.06-5.48 5.48a.75.75 0 1 0 1.06 1.06l4.95-4.949a2.25 2.25 0 0 0 0-3.182",
            clipRule: "evenodd"
        })
    });
}
const PaperclipIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPaperclipIcon
    });
});
PaperclipIcon.displayName = 'PaperclipIcon';

function SvgPauseIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M10 12V4h1.5v8zm-5.5 0V4H6v8z",
            clipRule: "evenodd"
        })
    });
}
const PauseIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPauseIcon
    });
});
PauseIcon.displayName = 'PauseIcon';

function SvgPencilFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M11.013 1.513a1.75 1.75 0 0 1 2.474 0l1.086 1.085a1.75 1.75 0 0 1 0 2.475l-1.512 1.513L9.5 3.026zM8.439 4.086l-7.22 7.22a.75.75 0 0 0-.219.53v2.5c0 .414.336.75.75.75h2.5a.75.75 0 0 0 .53-.22L12 7.646z"
        })
    });
}
const PencilFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPencilFillIcon
    });
});
PencilFillIcon.displayName = 'PencilFillIcon';

function SvgPencilIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M13.487 1.513a1.75 1.75 0 0 0-2.474 0L1.22 11.306a.75.75 0 0 0-.22.53v2.5c0 .414.336.75.75.75h2.5a.75.75 0 0 0 .53-.22l9.793-9.793a1.75 1.75 0 0 0 0-2.475zm-1.414 1.06a.25.25 0 0 1 .354 0l1.086 1.086a.25.25 0 0 1 0 .354L12 5.525l-1.44-1.44zM9.5 5.146l-7 7v1.44h1.44l7-7z",
            clipRule: "evenodd"
        })
    });
}
const PencilIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPencilIcon
    });
});
PencilIcon.displayName = 'PencilIcon';

function SvgPencilSparkleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12.073 2.573a.25.25 0 0 1 .354 0l1.086 1.086a.25.25 0 0 1 0 .354L12 5.525l-1.44-1.44zM9.5 5.146l-7 7v1.44h1.44l7-7zm3.987-3.633a1.75 1.75 0 0 0-2.474 0L1.22 11.306a.75.75 0 0 0-.22.53v2.5c0 .414.336.75.75.75h2.5a.75.75 0 0 0 .53-.22l9.793-9.793a1.75 1.75 0 0 0 0-2.475z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M12.31 11.09 12.5 10l.19 1.09a1.5 1.5 0 0 0 1.22 1.22l1.09.19-1.09.19a1.5 1.5 0 0 0-1.22 1.22L12.5 15l-.19-1.09a1.5 1.5 0 0 0-1.22-1.22L10 12.5l1.09-.19a1.5 1.5 0 0 0 1.22-1.22"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12.5 9.25a.75.75 0 0 1 .739.621l.19 1.09a.75.75 0 0 0 .61.61l1.09.19a.75.75 0 0 1 0 1.478l-1.09.19a.75.75 0 0 0-.61.61l-.19 1.09a.75.75 0 0 1-1.478 0l-.19-1.09a.75.75 0 0 0-.61-.61l-1.09-.19a.75.75 0 0 1 0-1.478l1.09-.19a.75.75 0 0 0 .61-.61l.345.06-.344-.06.19-1.09a.75.75 0 0 1 .738-.621m0 3.094q-.075.081-.156.156.081.075.156.156.075-.081.156-.156a2 2 0 0 1-.156-.156",
                clipRule: "evenodd"
            })
        ]
    });
}
const PencilSparkleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPencilSparkleIcon
    });
});
PencilSparkleIcon.displayName = 'PencilSparkleIcon';

function SvgPieChartIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M9.25 0a.75.75 0 0 0-.75.75v6c0 .414.336.75.75.75h6a.75.75 0 0 0 .75-.75A6.75 6.75 0 0 0 9.25 0M10 1.553A5.25 5.25 0 0 1 14.447 6H10z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M6.75 2.5a6.75 6.75 0 1 0 6.75 6.75.75.75 0 0 0-.75-.75H7.5V3.25a.75.75 0 0 0-.75-.75M1.5 9.25A5.25 5.25 0 0 1 6 4.053V9.25c0 .414.336.75.75.75h5.197A5.251 5.251 0 0 1 1.5 9.25",
                clipRule: "evenodd"
            })
        ]
    });
}
const PieChartIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPieChartIcon
    });
});
PieChartIcon.displayName = 'PieChartIcon';

function SvgPinCancelIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M5.75 0A.75.75 0 0 0 5 .75v1.19l9 9V9a.75.75 0 0 0-.22-.53l-2.12-2.122a2.25 2.25 0 0 1-.66-1.59V.75a.75.75 0 0 0-.75-.75zM10.94 12l2.53 2.53 1.06-1.06-11.5-11.5-1.06 1.06 2.772 2.773q-.157.301-.4.545L2.22 8.47A.75.75 0 0 0 2 9v2.25c0 .414.336.75.75.75h4.5v4h1.5v-4z"
        })
    });
}
const PinCancelIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPinCancelIcon
    });
});
PinCancelIcon.displayName = 'PinCancelIcon';

function SvgPinFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M5 .75A.75.75 0 0 1 5.75 0h4.5a.75.75 0 0 1 .75.75v4.007c0 .597.237 1.17.659 1.591L13.78 8.47c.141.14.22.331.22.53v2.25a.75.75 0 0 1-.75.75h-4.5v4h-1.5v-4h-4.5a.75.75 0 0 1-.75-.75V9a.75.75 0 0 1 .22-.53L4.34 6.348A2.25 2.25 0 0 0 5 4.758z"
        })
    });
}
const PinFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPinFillIcon
    });
});
PinFillIcon.displayName = 'PinFillIcon';

function SvgPinIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M5.75 0A.75.75 0 0 0 5 .75v4.007a2.25 2.25 0 0 1-.659 1.591L2.22 8.47A.75.75 0 0 0 2 9v2.25c0 .414.336.75.75.75h4.5v4h1.5v-4h4.5a.75.75 0 0 0 .75-.75V9a.75.75 0 0 0-.22-.53L11.66 6.348A2.25 2.25 0 0 1 11 4.758V.75a.75.75 0 0 0-.75-.75zm.75 4.757V1.5h3v3.257a3.75 3.75 0 0 0 1.098 2.652L12.5 9.311V10.5h-9V9.31L5.402 7.41A3.75 3.75 0 0 0 6.5 4.757",
            clipRule: "evenodd"
        })
    });
}
const PinIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPinIcon
    });
});
PinIcon.displayName = 'PinIcon';

function SvgPipelineCodeIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 17",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "m10.53 11.06-1.97 1.97L10.53 15l-1.06 1.06-3.03-3.03L9.47 10zM16.06 13.03l-3.03 3.03L11.97 15l1.97-1.97-1.97-1.97L13.03 10zM5 1a5.75 5.75 0 0 1 5.75 5.75V9h-1.5V6.75A4.25 4.25 0 0 0 5.5 2.53v2.793A1.75 1.75 0 0 1 6.75 7v2.25q.001.47.098.91l-1.196 1.204A5.7 5.7 0 0 1 5.25 9.25V7A.25.25 0 0 0 5 6.75H1.75A.75.75 0 0 1 1 6V1.75A.75.75 0 0 1 1.75 1zM2.5 5.25H4V2.5H2.5z"
        })
    });
}
const PipelineCodeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPipelineCodeIcon
    });
});
PipelineCodeIcon.displayName = 'PipelineCodeIcon';

function SvgPipelineCubeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M11.686 8.07c.199-.093.43-.093.629 0l3.24 1.494a1 1 0 0 1 .1.054l.02.015q.074.051.133.116.014.016.027.034l.037.05.018.028a.7.7 0 0 1 .082.189.8.8 0 0 1 .028.2v3.5a.75.75 0 0 1-.435.68l-3.242 1.496-.006.003-.002.002a.8.8 0 0 1-.156.05q-.009.001-.018.004l-.042.006q-.024.004-.048.007h-.102q-.024-.003-.049-.007l-.042-.006q-.009-.001-.018-.005a.8.8 0 0 1-.155-.05l-.01-.004-3.24-1.495A.75.75 0 0 1 8 13.75v-3.5a.8.8 0 0 1 .06-.293l.007-.019.025-.048.035-.057.037-.05.027-.034a.8.8 0 0 1 .133-.116l.022-.015a1 1 0 0 1 .098-.054zM9.5 13.27l1.75.807V12.23l-1.75-.807zm3.25-1.04v1.847l1.75-.807v-1.848zm-2.21-1.98 1.46.674 1.46-.674L12 9.575z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M5 1a5.75 5.75 0 0 1 5.75 5.75v.175l-1.5.692V6.75A4.25 4.25 0 0 0 5.5 2.53v2.793A1.75 1.75 0 0 1 6.75 7v6.122a5.73 5.73 0 0 1-1.5-3.872V7A.25.25 0 0 0 5 6.75H1.75A.75.75 0 0 1 1 6V1.75A.75.75 0 0 1 1.75 1zM2.5 5.25H4V2.5H2.5z",
                clipRule: "evenodd"
            })
        ]
    });
}
const PipelineCubeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPipelineCubeIcon
    });
});
PipelineCubeIcon.displayName = 'PipelineCubeIcon';

function SvgPipelineIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M10.75 6.75A5.75 5.75 0 0 0 5 1H1.75a.75.75 0 0 0-.75.75V6c0 .414.336.75.75.75H5a.25.25 0 0 1 .25.25v2.25A5.75 5.75 0 0 0 11 15h3.25a.75.75 0 0 0 .75-.75V10a.75.75 0 0 0-.75-.75H11a.25.25 0 0 1-.25-.25zM5.5 2.53a4.25 4.25 0 0 1 3.75 4.22V9a1.75 1.75 0 0 0 1.25 1.678v2.793A4.25 4.25 0 0 1 6.75 9.25V7A1.75 1.75 0 0 0 5.5 5.322zM4 2.5v2.75H2.5V2.5zm9.5 8.25H12v2.75h1.5z",
            clipRule: "evenodd"
        })
    });
}
const PipelineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPipelineIcon
    });
});
PipelineIcon.displayName = 'PipelineIcon';

function SvgPlayCircleFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m7.125-2.815A.75.75 0 0 0 6 5.835v4.33a.75.75 0 0 0 1.125.65l3.75-2.166a.75.75 0 0 0 0-1.299z",
            clipRule: "evenodd"
        })
    });
}
const PlayCircleFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPlayCircleFillIcon
    });
});
PlayCircleFillIcon.displayName = 'PlayCircleFillIcon';

function SvgPlayCircleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M11.25 8a.75.75 0 0 1-.375.65l-3.75 2.165A.75.75 0 0 1 6 10.165v-4.33a.75.75 0 0 1 1.125-.65l3.75 2.165a.75.75 0 0 1 .375.65"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0",
                clipRule: "evenodd"
            })
        ]
    });
}
const PlayCircleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPlayCircleIcon
    });
});
PlayCircleIcon.displayName = 'PlayCircleIcon';

function SvgPlayDoubleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M2.371 3.853a.75.75 0 0 1 .745-.007l6.25 3.5a.75.75 0 0 1 0 1.308l-6.25 3.5A.75.75 0 0 1 2 11.5v-7l.007-.099a.75.75 0 0 1 .364-.548"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M14.636 7.357a.75.75 0 0 1 0 1.287l-5.833 3.5-.772-1.287L12.792 8 7.864 5.044l.772-1.287z"
            })
        ]
    });
}
const PlayDoubleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPlayDoubleIcon
    });
});
PlayDoubleIcon.displayName = 'PlayDoubleIcon';

function SvgPlayIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M12.125 8.864a.75.75 0 0 0 0-1.3l-6-3.464A.75.75 0 0 0 5 4.75v6.928a.75.75 0 0 0 1.125.65z"
        })
    });
}
const PlayIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPlayIcon
    });
});
PlayIcon.displayName = 'PlayIcon';

function SvgPlayMultipleIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M6.63 3.98a.5.5 0 0 1 .683-.182l5.4 3.117.175.118a1.5 1.5 0 0 1-.176 2.48L7.313 12.63l-.092.042a.5.5 0 0 1-.49-.849l.082-.06 5.4-3.117a.5.5 0 0 0 .058-.826l-.059-.039-5.399-3.118a.5.5 0 0 1-.183-.683m-3.517.12a.75.75 0 0 1 .75 0l6 3.464a.75.75 0 0 1 0 1.3l-6 3.464a.75.75 0 0 1-1.125-.65V4.75a.75.75 0 0 1 .375-.65"
        })
    });
}
const PlayMultipleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPlayMultipleIcon
    });
});
PlayMultipleIcon.displayName = 'PlayMultipleIcon';

function SvgPlugIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m14.168 2.953.893-.892L14 1l-.893.893a4 4 0 0 0-5.077.48l-.884.884a.75.75 0 0 0 0 1.061l4.597 4.596a.75.75 0 0 0 1.06 0l.884-.884a4 4 0 0 0 .48-5.077M12.627 6.97l-.354.353-3.536-3.535.354-.354a2.5 2.5 0 1 1 3.536 3.536M7.323 10.152 5.91 8.737l1.414-1.414-1.06-1.06-1.415 1.414-.53-.53a.75.75 0 0 0-1.06 0l-.885.883a4 4 0 0 0-.48 5.077L1 14l1.06 1.06.893-.892a4 4 0 0 0 5.077-.48l.884-.885a.75.75 0 0 0 0-1.06l-.53-.53 1.414-1.415-1.06-1.06zm-3.889 2.475a2.5 2.5 0 0 0 3.536 0l.353-.354-3.535-3.536-.354.354a2.5 2.5 0 0 0 0 3.536",
            clipRule: "evenodd"
        })
    });
}
const PlugIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPlugIcon
    });
});
PlugIcon.displayName = 'PlugIcon';

function SvgPlusCircleFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16m-.75-4.5V8.75H4.5v-1.5h2.75V4.5h1.5v2.75h2.75v1.5H8.75v2.75z",
            clipRule: "evenodd"
        })
    });
}
const PlusCircleFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPlusCircleFillIcon
    });
});
PlusCircleFillIcon.displayName = 'PlusCircleFillIcon';

function SvgPlusCircleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.25 11.5V8.75H4.5v-1.5h2.75V4.5h1.5v2.75h2.75v1.5H8.75v2.75z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0",
                clipRule: "evenodd"
            })
        ]
    });
}
const PlusCircleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPlusCircleIcon
    });
});
PlusCircleIcon.displayName = 'PlusCircleIcon';

function SvgPlusIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M7.25 7.25V1h1.5v6.25H15v1.5H8.75V15h-1.5V8.75H1v-1.5z",
            clipRule: "evenodd"
        })
    });
}
const PlusIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPlusIcon
    });
});
PlusIcon.displayName = 'PlusIcon';

function SvgPlusMinusSquareIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.25 4.25V6H5.5v1.5h1.75v1.75h1.5V7.5h1.75V6H8.75V4.25zM10.5 10.5h-5V12h5z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z",
                clipRule: "evenodd"
            })
        ]
    });
}
const PlusMinusSquareIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPlusMinusSquareIcon
    });
});
PlusMinusSquareIcon.displayName = 'PlusMinusSquareIcon';

function SvgPlusSquareIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.25 7.25V4.5h1.5v2.75h2.75v1.5H8.75v2.75h-1.5V8.75H4.5v-1.5z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75v11h11v-11z",
                clipRule: "evenodd"
            })
        ]
    });
}
const PlusSquareIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPlusSquareIcon
    });
});
PlusSquareIcon.displayName = 'PlusSquareIcon';

function SvgPullRequestIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M4.25 1A2.998 2.998 0 0 1 5 6.901v2.187a2.998 2.998 0 0 1-.75 5.902 2.998 2.998 0 0 1-.75-5.902V6.9A2.998 2.998 0 0 1 4.25 1m7.395.991-1.156 1.155a3.25 3.25 0 0 1 2.874 3.228v2.74a3 3 0 1 1-1.5-.053V6.374c0-.86-.62-1.574-1.438-1.722l1.22 1.22-1.061 1.06-3.001-3L10.585.93zm-7.395 8.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3m8.259 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3M4.25 2.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3"
        })
    });
}
const PullRequestIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPullRequestIcon
    });
});
PullRequestIcon.displayName = 'PullRequestIcon';

function SvgPuzzleIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M12.015 1.338v4.985c1.758-.263 3.485-.095 3.485 2.88s-1.75 3.14-3.49 2.915v3.608H.5V1.297c1.262-.24 2.892-.434 4.22-.57a.3.3 0 0 1 .327.268l.077.772a.5.5 0 0 1-.452.547c-.189.018-.355.145-.359.334-.015.653.394 1.82 1.877 1.82 1.59 0 2.046-.903 1.901-1.859a.45.45 0 0 0-.388-.363l-.078-.011a.5.5 0 0 1-.425-.522l.044-.82a.295.295 0 0 1 .315-.28c1.445.104 2.917.407 4.456.725m-2.73 1.088c.591 1.949-1.063 3.695-3.095 3.695-2.003 0-3.522-1.746-3.117-3.646-.238.033-.752.108-.997.147V14.15h8.451v-3.914h1.483l.048.236c.492.447 1.996.634 1.996-1.27s-1.548-1.701-2.039-1.334l-.014.22-1.487.077v-5.59c-.313-.05-.924-.11-1.23-.149"
        })
    });
}
const PuzzleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgPuzzleIcon
    });
});
PuzzleIcon.displayName = 'PuzzleIcon';

function SvgQueryEditorIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M12 12H8v-1.5h4zM5.53 11.53 7.56 9.5 5.53 7.47 4.47 8.53l.97.97-.97.97z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 3V2.5h11V4zm0 1.5v8h11v-8z",
                clipRule: "evenodd"
            })
        ]
    });
}
const QueryEditorIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgQueryEditorIcon
    });
});
QueryEditorIcon.displayName = 'QueryEditorIcon';

function SvgQueryIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#QueryIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16h-.75a.75.75 0 0 1-.75-.75zm7.5 1.81 1.94 1.94H9.5z",
                        clipRule: "evenodd"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M5.53 9.97 8.56 13l-3.03 3.03-1.06-1.06L6.44 13l-1.97-1.97zM14 14.5H9V16h5z"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const QueryIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgQueryIcon
    });
});
QueryIcon.displayName = 'QueryIcon';

function SvgQuestionMarkFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16m2.207-10.189a2.25 2.25 0 0 1-1.457 2.56V9h-1.5V7.75A.75.75 0 0 1 8 7a.75.75 0 1 0-.75-.75h-1.5a2.25 2.25 0 0 1 4.457-.439M7.25 10.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0",
            clipRule: "evenodd"
        })
    });
}
const QuestionMarkFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgQuestionMarkFillIcon
    });
});
QuestionMarkFillIcon.displayName = 'QuestionMarkFillIcon';

function SvgQuestionMarkIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.25 10.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0M10.079 7.111A2.25 2.25 0 1 0 5.75 6.25h1.5A.75.75 0 1 1 8 7a.75.75 0 0 0-.75.75V9h1.5v-.629a2.25 2.25 0 0 0 1.329-1.26"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13",
                clipRule: "evenodd"
            })
        ]
    });
}
const QuestionMarkIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgQuestionMarkIcon
    });
});
QuestionMarkIcon.displayName = 'QuestionMarkIcon';

function SvgRadioIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m5 0a3 3 0 1 1 6 0 3 3 0 0 1-6 0m3-1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3",
            clipRule: "evenodd"
        })
    });
}
const RadioIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgRadioIcon
    });
});
RadioIcon.displayName = 'RadioIcon';

function SvgReaderModeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M13 4.5h-3V6h3zM13 7.25h-3v1.5h3zM13 10h-3v1.5h3z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75zm.75 10.5v-9h5.75v9zm7.25 0h5.75v-9H8.75z",
                clipRule: "evenodd"
            })
        ]
    });
}
const ReaderModeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgReaderModeIcon
    });
});
ReaderModeIcon.displayName = 'ReaderModeIcon';

function SvgRedoIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#RedoIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    fillRule: "evenodd",
                    d: "m13.19 5-2.72-2.72 1.06-1.06 4.53 4.53-4.53 4.53-1.06-1.06 2.72-2.72H4.5a3 3 0 1 0 0 6H9V14H4.5a4.5 4.5 0 0 1 0-9z",
                    clipRule: "evenodd"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 16h16V0H0z"
                    })
                })
            })
        ]
    });
}
const RedoIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgRedoIcon
    });
});
RedoIcon.displayName = 'RedoIcon';

function SvgRefreshIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 8a7 7 0 0 1 11.85-5.047l.65.594V2H15v4h-4V4.5h1.32l-.496-.453-.007-.007a5.5 5.5 0 1 0 .083 7.839l1.063 1.058A7 7 0 0 1 1 8",
            clipRule: "evenodd"
        })
    });
}
const RefreshIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgRefreshIcon
    });
});
RefreshIcon.displayName = 'RefreshIcon';

function SvgRefreshPlayIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 1c1.878 0 3.583.74 4.84 1.943l.66.596V2H15v4h-4V4.5h1.326l-.491-.443-.009-.008-.01-.009a5.5 5.5 0 1 0 .083 7.839l1.064 1.057A7 7 0 1 1 8 1"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M6.375 5.186a.75.75 0 0 1 .75 0l3.75 2.165.083.055a.75.75 0 0 1-.083 1.243l-3.75 2.166A.75.75 0 0 1 6 10.165v-4.33a.75.75 0 0 1 .375-.65"
            })
        ]
    });
}
const RefreshPlayIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgRefreshPlayIcon
    });
});
RefreshPlayIcon.displayName = 'RefreshPlayIcon';

function SvgRefreshXIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 1c1.878 0 3.583.74 4.84 1.943l.66.596V2H15v4h-4V4.5h1.326l-.491-.443-.009-.008-.01-.009a5.5 5.5 0 1 0 .083 7.839l1.064 1.057A7 7 0 1 1 8 1"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M5 6.05 6.95 8 5 9.95 6.05 11 8 9.05 9.95 11 11 9.95 9.05 8 11 6.05 9.95 5 8 6.95 6.05 5z"
            })
        ]
    });
}
const RefreshXIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgRefreshXIcon
    });
});
RefreshXIcon.displayName = 'RefreshXIcon';

function SvgReplyIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("mask", {
                id: "ReplyIcon_svg__a",
                width: 16,
                height: 16,
                x: 0,
                y: 0,
                maskUnits: "userSpaceOnUse",
                style: {
                    maskType: 'alpha'
                },
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    d: "M0 0h16v16H0z"
                })
            }),
            /*#__PURE__*/ jsx("g", {
                mask: "url(#ReplyIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    d: "M3.333 3.333V6q0 .834.584 1.417Q4.5 8 5.333 8h6.117l-2.4-2.4.95-.933 4 4-4 4-.95-.934 2.4-2.4H5.333a3.21 3.21 0 0 1-2.358-.975A3.21 3.21 0 0 1 2 6V3.333z"
                })
            })
        ]
    });
}
const ReplyIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgReplyIcon
    });
});
ReplyIcon.displayName = 'ReplyIcon';

function SvgResizeIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M15 6.75H1v-1.5h14zm0 4.75H1V10h14z",
            clipRule: "evenodd"
        })
    });
}
const ResizeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgResizeIcon
    });
});
ResizeIcon.displayName = 'ResizeIcon';

function SvgRichTextIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 18 18",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M16 16H2v-1.5h14zM16 12.75H2v-1.5h14zM9 3.5H6.25v6.25h-1.5V3.5H2V2h7zM16 9.75h-5.5v-1.5H16zM16 6.75h-5.5v-1.5H16zM16 3.5h-5.5V2H16z"
        })
    });
}
const RichTextIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgRichTextIcon
    });
});
RichTextIcon.displayName = 'RichTextIcon';

function SvgRobotIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 0a.75.75 0 0 1 .75.75V3h5.5a.75.75 0 0 1 .75.75V6h.25a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-.75.75H15v2.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75V12H.75a.75.75 0 0 1-.75-.75v-4.5A.75.75 0 0 1 .75 6H1V3.75A.75.75 0 0 1 1.75 3h5.5V.75A.75.75 0 0 1 8 0M2.5 4.5v9h11v-9zM5 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2m7-1a1 1 0 1 1-2 0 1 1 0 0 1 2 0m-6.25 2.25a.75.75 0 0 0 0 1.5h4.5a.75.75 0 0 0 0-1.5z",
            clipRule: "evenodd"
        })
    });
}
const RobotIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgRobotIcon
    });
});
RobotIcon.displayName = 'RobotIcon';

function SvgRocketIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M13.25 2a.75.75 0 0 1 .75.75v.892a8.75 8.75 0 0 1-3.07 6.656h.015v.626a4.75 4.75 0 0 1-2.017 3.884l-1.496 1.053a.75.75 0 0 1-1.163-.446l-.72-3.148-1.814-1.815L.589 9.75a.75.75 0 0 1-.451-1.162L1.193 7.08a4.75 4.75 0 0 1 3.891-2.025h.618v.015A8.75 8.75 0 0 1 12.358 2zM7.105 12.341l.377 1.65.583-.41a3.25 3.25 0 0 0 1.353-2.245q-.405.22-.837.397zM4.267 7.419l-.61 1.48L2.01 8.53l.413-.589a3.25 3.25 0 0 1 2.242-1.358q-.22.404-.397.836M12.5 3.5h-.142a7.2 7.2 0 0 0-2.754.543l2.353 2.353a7.2 7.2 0 0 0 .543-2.754zM5.654 7.99a7.24 7.24 0 0 1 2.576-3.2l2.98 2.98a7.24 7.24 0 0 1-3.2 2.576l-1.601.66L4.995 9.59z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m2.22 10.72-.122.121A3.75 3.75 0 0 0 1 13.493v.757c0 .414.336.75.75.75h.757a3.75 3.75 0 0 0 2.652-1.098l.121-.122-1.06-1.06-.122.121a2.25 2.25 0 0 1-1.59.659H2.5v-.007c0-.597.237-1.17.659-1.591l.121-.122z"
            })
        ]
    });
}
const RocketIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgRocketIcon
    });
});
RocketIcon.displayName = 'RocketIcon';

function SvgRowsIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M14.25 1a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75V1.75A.75.75 0 0 1 1.75 1zM2.5 2.5h11V5h-11zm0 4v3h11v-3zm11 4.5h-11v2.5h11z",
            clipRule: "evenodd"
        })
    });
}
const RowsIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgRowsIcon
    });
});
RowsIcon.displayName = 'RowsIcon';

function SvgRunIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M15 13.225 12.232 16l-1.056-1.059.965-.967h-7.78A3.365 3.365 0 0 1 1 10.604a3.365 3.365 0 0 1 3.36-3.368h7.22a1.87 1.87 0 0 0 1.866-1.871 1.87 1.87 0 0 0-1.867-1.872H6.37A2.74 2.74 0 0 1 3.738 5.49 2.74 2.74 0 0 1 1 2.745 2.74 2.74 0 0 1 3.738 0C4.991 0 6.045.845 6.37 1.996h5.21a3.365 3.365 0 0 1 3.36 3.369 3.365 3.365 0 0 1-3.36 3.368H4.36a1.87 1.87 0 0 0-1.866 1.872 1.87 1.87 0 0 0 1.866 1.871h7.781l-.965-.968 1.056-1.058zM2.494 2.745c0 .689.557 1.247 1.244 1.247.688 0 1.245-.558 1.245-1.247s-.557-1.248-1.245-1.248c-.687 0-1.244.559-1.244 1.248"
        })
    });
}
const RunIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgRunIcon
    });
});
RunIcon.displayName = 'RunIcon';

function SvgRunningIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#RunningIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    fillRule: "evenodd",
                    d: "M8 1.5A6.5 6.5 0 0 0 1.5 8H0a8 8 0 0 1 8-8zm0 13A6.5 6.5 0 0 0 14.5 8H16a8 8 0 0 1-8 8z",
                    clipRule: "evenodd"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const RunningIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgRunningIcon
    });
});
RunningIcon.displayName = 'RunningIcon';

function SvgSaveClockIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8m-.75-6.5v2.81l1.72 1.72 1.06-1.06-1.28-1.28V9.5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h5.941a5.2 5.2 0 0 1-.724-1.5H2.5v-11H5v3.75c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75V2.81l2.5 2.5v1.657a5.2 5.2 0 0 1 1.5.724V5a.75.75 0 0 0-.22-.53l-3.25-3.25A.75.75 0 0 0 11 1zM6.5 2.5h3v3h-3z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.527 9.25H5v1.5h1.9a5.2 5.2 0 0 1 .627-1.5"
            })
        ]
    });
}
const SaveClockIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSaveClockIcon
    });
});
SaveClockIcon.displayName = 'SaveClockIcon';

function SvgSaveIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M10 9.25H6v1.5h4z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1 1.75A.75.75 0 0 1 1.75 1H11a.75.75 0 0 1 .53.22l3.25 3.25c.141.14.22.331.22.53v9.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75H5v3.75c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75V2.81l2.5 2.5v8.19h-11zm4 0h3v3h-3z",
                clipRule: "evenodd"
            })
        ]
    });
}
const SaveIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSaveIcon
    });
});
SaveIcon.displayName = 'SaveIcon';

function SvgSchemaIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2.75 0A.75.75 0 0 0 2 .75v3a.75.75 0 0 0 .75.75h1v7a2.75 2.75 0 0 0 2.75 2.75H7v1c0 .414.336.75.75.75h5.5a.75.75 0 0 0 .75-.75v-3a.75.75 0 0 0-.75-.75h-5.5a.75.75 0 0 0-.75.75v.5h-.5c-.69 0-1.25-.56-1.25-1.25V8.45c.375.192.8.3 1.25.3H7v.75c0 .414.336.75.75.75h5.5A.75.75 0 0 0 14 9.5v-3a.75.75 0 0 0-.75-.75h-5.5A.75.75 0 0 0 7 6.5v.75h-.5c-.69 0-1.25-.56-1.25-1.25V4.5h8a.75.75 0 0 0 .75-.75v-3a.75.75 0 0 0-.75-.75zm.75 3V1.5h9V3zm5 10v1.5h4V13zm0-4.25v-1.5h4v1.5z",
            clipRule: "evenodd"
        })
    });
}
const SchemaIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSchemaIcon
    });
});
SchemaIcon.displayName = 'SchemaIcon';

function SvgSchoolIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M16 7a.75.75 0 0 0-.37-.647l-7.25-4.25a.75.75 0 0 0-.76 0L.37 6.353a.75.75 0 0 0 0 1.294L3 9.188V12a.75.75 0 0 0 .4.663l4.25 2.25a.75.75 0 0 0 .7 0l4.25-2.25A.75.75 0 0 0 13 12V9.188l1.5-.879V12H16zm-7.62 4.897 3.12-1.83v1.481L8 13.401l-3.5-1.853v-1.48l3.12 1.829a.75.75 0 0 0 .76 0M8 3.619 2.233 7 8 10.38 13.767 7z",
            clipRule: "evenodd"
        })
    });
}
const SchoolIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSchoolIcon
    });
});
SchoolIcon.displayName = 'SchoolIcon';

function SvgSearchDataIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.651 3.128a.75.75 0 0 0-1.302 0l-1 1.75A.75.75 0 0 0 6 6h2a.75.75 0 0 0 .651-1.122zM4.75 6.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M7.5 7.25a.75.75 0 0 1 .75-.75h2a.75.75 0 0 1 .75.75v2a.75.75 0 0 1-.75.75h-2a.75.75 0 0 1-.75-.75z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M0 7a7 7 0 1 1 12.45 4.392l2.55 2.55-1.06 1.061-2.55-2.55A7 7 0 0 1 0 7m7-5.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11",
                clipRule: "evenodd"
            })
        ]
    });
}
const SearchDataIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSearchDataIcon
    });
});
SearchDataIcon.displayName = 'SearchDataIcon';

function SvgSendIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M16 8a.75.75 0 0 1-.435.68l-13.5 6.25a.75.75 0 0 1-1.02-.934L3.202 8 1.044 2.004a.75.75 0 0 1 1.021-.935l13.5 6.25A.75.75 0 0 1 16 8m-11.473.75-1.463 4.065L13.464 8l-10.4-4.815L4.527 7.25H8v1.5z",
            clipRule: "evenodd"
        })
    });
}
const SendIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSendIcon
    });
});
SendIcon.displayName = 'SendIcon';

function SvgShareIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M3.97 5.03 8 1l4.03 4.03-1.06 1.061-2.22-2.22v7.19h-1.5V3.87l-2.22 2.22z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M2.5 13.56v-6.5H1v7.25c0 .415.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V7.06h-1.5v6.5z"
            })
        ]
    });
}
const ShareIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgShareIcon
    });
});
ShareIcon.displayName = 'ShareIcon';

function SvgShieldCheckIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2 1.75A.75.75 0 0 1 2.75 1h10.5a.75.75 0 0 1 .75.75v7.465a5.75 5.75 0 0 1-2.723 4.889l-2.882 1.784a.75.75 0 0 1-.79 0l-2.882-1.784A5.75 5.75 0 0 1 2 9.214zm1.5.75v6.715a4.25 4.25 0 0 0 2.013 3.613L8 14.368l2.487-1.54A4.25 4.25 0 0 0 12.5 9.215V2.5zm6.22 2.97 1.06 1.06-3.53 3.53-2.03-2.03 1.06-1.06.97.97z",
            clipRule: "evenodd"
        })
    });
}
const ShieldCheckIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgShieldCheckIcon
    });
});
ShieldCheckIcon.displayName = 'ShieldCheckIcon';

function SvgShieldIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2 1.75A.75.75 0 0 1 2.75 1h10.5a.75.75 0 0 1 .75.75v7.465a5.75 5.75 0 0 1-2.723 4.889l-2.882 1.784a.75.75 0 0 1-.79 0l-2.882-1.784A5.75 5.75 0 0 1 2 9.214zm1.5.75V7h3.75V2.5zm5.25 0V7h3.75V2.5zm3.75 6H8.75v5.404l1.737-1.076A4.25 4.25 0 0 0 12.5 9.215zm-5.25 5.404V8.5H3.5v.715a4.25 4.25 0 0 0 2.013 3.613z",
            clipRule: "evenodd"
        })
    });
}
const ShieldIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgShieldIcon
    });
});
ShieldIcon.displayName = 'ShieldIcon';

function SvgShieldOffIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M13.378 11.817A5.75 5.75 0 0 0 14 9.215V1.75a.75.75 0 0 0-.75-.75H2.75a.8.8 0 0 0-.17.02L4.06 2.5h8.44v6.715c0 .507-.09 1.002-.26 1.464z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "m1.97 2.53-1 1L2 4.56v4.655a5.75 5.75 0 0 0 2.723 4.889l2.882 1.784a.75.75 0 0 0 .79 0l2.882-1.784.162-.104 1.53 1.53 1-1zM3.5 9.215V6.06l6.852 6.851L8 14.368l-2.487-1.54A4.25 4.25 0 0 1 3.5 9.215",
                clipRule: "evenodd"
            })
        ]
    });
}
const ShieldOffIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgShieldOffIcon
    });
});
ShieldOffIcon.displayName = 'ShieldOffIcon';

function SvgShortcutIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M14.25 14H9v-1.5h4.5v-10h-10V6H2V1.75A.75.75 0 0 1 2.75 1h11.5a.75.75 0 0 1 .75.75v11.5a.75.75 0 0 1-.75.75"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M2 8h5v5H5.5v-2.872a2.251 2.251 0 0 0 .75 4.372V16A3.75 3.75 0 0 1 3.7 9.5H2z"
            })
        ]
    });
}
const ShortcutIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgShortcutIcon
    });
});
ShortcutIcon.displayName = 'ShortcutIcon';

function SvgSidebarAutoIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1zM4 2.5H2.5v11H4z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m9.06 8 1.97 1.97-1.06 1.06L6.94 8l3.03-3.03 1.06 1.06zM11.97 6.03 13.94 8l-1.97 1.97 1.06 1.06L16.06 8l-3.03-3.03z"
            })
        ]
    });
}
const SidebarAutoIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSidebarAutoIcon
    });
});
SidebarAutoIcon.displayName = 'SidebarAutoIcon';

function SvgSidebarCollapseIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1zM4 2.5H2.5v11H4z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m9.81 8.75 1.22 1.22-1.06 1.06L6.94 8l3.03-3.03 1.06 1.06-1.22 1.22H14v1.5z"
            })
        ]
    });
}
const SidebarCollapseIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSidebarCollapseIcon
    });
});
SidebarCollapseIcon.displayName = 'SidebarCollapseIcon';

function SvgSidebarExpandIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1zM4 2.5H2.5v11H4z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M11.19 8.75 9.97 9.97l1.06 1.06L14.06 8l-3.03-3.03-1.06 1.06 1.22 1.22H7v1.5z"
            })
        ]
    });
}
const SidebarExpandIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSidebarExpandIcon
    });
});
SidebarExpandIcon.displayName = 'SidebarExpandIcon';

function SvgSidebarIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11H4v11zm3 0h8v-11h-8z",
            clipRule: "evenodd"
        })
    });
}
const SidebarIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSidebarIcon
    });
});
SidebarIcon.displayName = 'SidebarIcon';

function SvgSidebarSyncIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1 3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3.5a.5.5 0 0 1-.5.5H14a.5.5 0 0 1-.5-.5v-3H5v9h2.5a.5.5 0 0 1 .5.5v.5a.5.5 0 0 1-.5.5H2a1 1 0 0 1-1-1zm3 1.5a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0 0 1h1a.5.5 0 0 0 .5-.5m0 2a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0 0 1h1a.5.5 0 0 0 .5-.5",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M12 8c.794 0 1.525.282 2.082.752a.627.627 0 0 1-.809.958A1.97 1.97 0 0 0 12 9.253c-.415 0-.794.124-1.102.33h.263a.627.627 0 0 1 0 1.253H9.626A.627.627 0 0 1 9 10.209V8.626a.626.626 0 0 1 1.241-.11A3.24 3.24 0 0 1 12 8m3 5.374a.627.627 0 0 1-1.242.108A3.23 3.23 0 0 1 12 14a3.22 3.22 0 0 1-2.082-.753.627.627 0 0 1 .809-.957c.333.281.778.457 1.273.457.415 0 .794-.124 1.102-.33h-.263a.626.626 0 0 1 0-1.253h1.535c.345 0 .626.281.626.627z"
            })
        ]
    });
}
const SidebarSyncIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSidebarSyncIcon
    });
});
SidebarSyncIcon.displayName = 'SidebarSyncIcon';

function SvgSlashSquareIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m8.654 4-2.912 8h1.596l2.912-8z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z",
                clipRule: "evenodd"
            })
        ]
    });
}
const SlashSquareIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSlashSquareIcon
    });
});
SlashSquareIcon.displayName = 'SlashSquareIcon';

function SvgSlidersIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M4 15v-2.354a2.751 2.751 0 0 1 0-5.292V1h1.5v6.354a2.751 2.751 0 0 1 0 5.292V15zm.75-3.75a1.25 1.25 0 1 1 0-2.5 1.25 1.25 0 0 1 0 2.5M10.5 1v2.354a2.751 2.751 0 0 0 0 5.292V15H12V8.646a2.751 2.751 0 0 0 0-5.292V1zm.75 3.75a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5",
            clipRule: "evenodd"
        })
    });
}
const SlidersIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSlidersIcon
    });
});
SlidersIcon.displayName = 'SlidersIcon';

function SvgSortAscendingIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "m11.5.94 4.03 4.03-1.06 1.06-2.22-2.22V10h-1.5V3.81L8.53 6.03 7.47 4.97zM1 4.5h4V6H1zM1 12.5h10V14H1zM8 8.5H1V10h7z"
        })
    });
}
const SortAscendingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortAscendingIcon
    });
});
SortAscendingIcon.displayName = 'SortAscendingIcon';

function SvgSortCustomHorizontalIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8.75 9.137q.35.11.647.311l.872-.695.935 1.173-.874.696q.129.331.16.698l1.09.248-.334 1.463-1.09-.248a2.5 2.5 0 0 1-.447.559l.485 1.007-1.35.651-.486-1.007A3 3 0 0 1 8 14.02a2.5 2.5 0 0 1-.358-.027L7.157 15l-1.351-.65.484-1.006a2.5 2.5 0 0 1-.446-.56l-1.09.248-.333-1.462 1.088-.249q.031-.367.16-.698l-.873-.697.935-1.172.873.695q.296-.2.645-.311L7.25 8.02h1.501zM8 10.52a1 1 0 1 0 .002 2.002A1 1 0 0 0 8 10.521",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m15.06 4-3.53 3.53-1.06-1.06 1.72-1.72H3.81l1.72 1.72-1.06 1.06L.94 4 4.47.47l1.06 1.06-1.72 1.72h8.38l-1.72-1.72L11.53.47z"
            })
        ]
    });
}
const SortCustomHorizontalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortCustomHorizontalIcon
    });
});
SortCustomHorizontalIcon.displayName = 'SortCustomHorizontalIcon';

function SvgSortCustomVerticalIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.53 4.47 6.47 5.53 4.75 3.81v8.38l1.72-1.72 1.06 1.06L4 15.06.47 11.53l1.06-1.06 1.72 1.72V3.81L1.53 5.53.47 4.47 4 .94z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12.17 5.626c.233.073.45.18.648.312l.872-.696.935 1.173-.874.696q.129.331.16.7L15 8.059l-.333 1.462-1.09-.248a2.5 2.5 0 0 1-.447.559l.485 1.007-1.35.65-.486-1.007a2.5 2.5 0 0 1-.358.029q-.183-.002-.358-.029l-.485 1.007-1.351-.65.485-1.006a2.5 2.5 0 0 1-.447-.56l-1.09.248-.333-1.462L8.93 7.81q.03-.369.159-.7l-.873-.696.935-1.173.872.696a2.5 2.5 0 0 1 .646-.312V4.511l1.502-.001zm-.75 1.385a1 1 0 1 0 .002 1.999 1 1 0 0 0-.001-2",
                clipRule: "evenodd"
            })
        ]
    });
}
const SortCustomVerticalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortCustomVerticalIcon
    });
});
SortCustomVerticalIcon.displayName = 'SortCustomVerticalIcon';

function SvgSortDescendingIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 3.5h10V2H1zm0 8h4V10H1zm7-4H1V6h7zm3.5 7.56 4.03-4.03-1.06-1.06-2.22 2.22V6h-1.5v6.19L8.53 9.97l-1.06 1.06z",
            clipRule: "evenodd"
        })
    });
}
const SortDescendingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortDescendingIcon
    });
});
SortDescendingIcon.displayName = 'SortDescendingIcon';

function SvgSortHorizontalAscendingIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.47 5.03.94 4.5l.53-.53 3.5-3.5 1.06 1.06-2.22 2.22H10v1.5H3.81l2.22 2.22-1.06 1.06zM4.5 15v-4H6v4zm8 0V5H14v10zm-4-7v7H10V8z",
            clipRule: "evenodd"
        })
    });
}
const SortHorizontalAscendingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortHorizontalAscendingIcon
    });
});
SortHorizontalAscendingIcon.displayName = 'SortHorizontalAscendingIcon';

function SvgSortHorizontalDescendingIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M3.5 15V5H2v10zm8 0v-4H10v4zm-4-7v7H6V8zm7.03-2.97.53-.53-.53-.53-3.5-3.5-1.06 1.06 2.22 2.22H6v1.5h6.19L9.97 7.47l1.06 1.06z",
            clipRule: "evenodd"
        })
    });
}
const SortHorizontalDescendingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortHorizontalDescendingIcon
    });
});
SortHorizontalDescendingIcon.displayName = 'SortHorizontalDescendingIcon';

function SvgSortLetterHorizontalAscendingIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#SortLetterHorizontalAscendingIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "m14.06 4-4.03 4.03-1.06-1.06 2.22-2.22H5v-1.5h6.19L8.97 1.03l1.06-1.06z"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M4.307 9a.75.75 0 0 1 .697.473L7.596 16H5.982l-.238-.6H2.855l-.24.6H1l2.61-6.528A.75.75 0 0 1 4.307 9m-.852 4.9h1.693l-.844-2.124z",
                        clipRule: "evenodd"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M11.777 10.5H8.5V9h4.75a.75.75 0 0 1 .607 1.191l-3.134 4.31H14V16H9.25a.75.75 0 0 1-.607-1.192z"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const SortLetterHorizontalAscendingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortLetterHorizontalAscendingIcon
    });
});
SortLetterHorizontalAscendingIcon.displayName = 'SortLetterHorizontalAscendingIcon';

function SvgSortLetterHorizontalDescendingIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#SortLetterHorizontalDescendingIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M.94 4 4.97-.03l1.06 1.06-2.22 2.22H10v1.5H3.81l2.22 2.22-1.06 1.06z"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M4.307 9a.75.75 0 0 1 .697.473L7.596 16H5.982l-.238-.6H2.855l-.24.6H1l2.61-6.528A.75.75 0 0 1 4.307 9m-.852 4.9h1.693l-.844-2.124z",
                        clipRule: "evenodd"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M11.777 10.5H8.5V9h4.75a.75.75 0 0 1 .607 1.191L10.723 14.5H14V16H9.25a.75.75 0 0 1-.607-1.191z"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const SortLetterHorizontalDescendingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortLetterHorizontalDescendingIcon
    });
});
SortLetterHorizontalDescendingIcon.displayName = 'SortLetterHorizontalDescendingIcon';

function SvgSortLetterUnsortedIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m11.5.94-.53.53-3.5 3.5 1.06 1.06 2.22-2.22v8.379999999999999L8.53 9.97l-1.06 1.06 3.5 3.5.53.53.53-.53 3.5-3.5-1.06-1.06-2.22 2.22V3.81l2.22 2.22 1.06-1.06-3.5-3.5zM4 1c.274 0 .52.173.623.437L7 7.533H5.549L5.185 6.6h-2.37l-.364.933H1l2.377-6.096A.67.67 0 0 1 4 1m-.639 4.2H4.64L4 3.561zM4.598 9.867H1.311v-1.4h4.706a.67.67 0 0 1 .608.4.72.72 0 0 1-.087.743L3.402 13.6H6.69V15H1.983a.67.67 0 0 1-.608-.4.72.72 0 0 1 .087-.743z",
            clipRule: "evenodd"
        })
    });
}
const SortLetterUnsortedIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortLetterUnsortedIcon
    });
});
SortLetterUnsortedIcon.displayName = 'SortLetterUnsortedIcon';

function SvgSortLetterVerticalAscendingIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#SortLetterVerticalAscendingIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M4.307 0a.75.75 0 0 1 .697.473L7.596 7H5.982l-.238-.6H2.855l-.24.6H1L3.61.472A.75.75 0 0 1 4.307 0m-.852 4.9h1.693l-.844-2.124z",
                        clipRule: "evenodd"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M4.777 9.5H1.5V8h4.75a.75.75 0 0 1 .607 1.191L3.723 13.5H7V15H2.25a.75.75 0 0 1-.607-1.191zM12 .94l4.03 4.03-1.06 1.06-2.22-2.22V10h-1.5V3.81L9.03 6.03 7.97 4.97z"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const SortLetterVerticalAscendingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortLetterVerticalAscendingIcon
    });
});
SortLetterVerticalAscendingIcon.displayName = 'SortLetterVerticalAscendingIcon';

function SvgSortLetterVerticalDescendingIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#SortLetterVerticalDescendingIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "M4.307 0a.75.75 0 0 1 .697.473L7.596 7H5.982l-.238-.6H2.855l-.24.6H1L3.61.472A.75.75 0 0 1 4.307 0m-.852 4.9h1.693l-.844-2.124z",
                        clipRule: "evenodd"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M4.777 9.5H1.5V8h4.75a.75.75 0 0 1 .607 1.191L3.723 13.5H7V15H2.25a.75.75 0 0 1-.607-1.191zM12 15.06l-4.03-4.03 1.06-1.06 2.22 2.22V6h1.5v6.19l2.22-2.22 1.06 1.06z"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const SortLetterVerticalDescendingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortLetterVerticalDescendingIcon
    });
});
SortLetterVerticalDescendingIcon.displayName = 'SortLetterVerticalDescendingIcon';

function SvgSortUnsortedIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M11.5.94 7.47 4.97l1.06 1.06 2.22-2.22v8.38L8.53 9.97l-1.06 1.06 4.03 4.03 4.03-4.03-1.06-1.06-2.22 2.22V3.81l2.22 2.22 1.06-1.06zM6 3.5H1V5h5zM6 11.5H1V13h5zM1 7.5h5V9H1z"
        })
    });
}
const SortUnsortedIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortUnsortedIcon
    });
});
SortUnsortedIcon.displayName = 'SortUnsortedIcon';

function SvgSortVerticalAscendingIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "m10.97 1.47.53-.53.53.53 3.5 3.5-1.06 1.06-2.22-2.22V10h-1.5V3.81L8.53 6.03 7.47 4.97zM1 4.5h4V6H1zm0 8h10V14H1zm7-4H1V10h7z",
            clipRule: "evenodd"
        })
    });
}
const SortVerticalAscendingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortVerticalAscendingIcon
    });
});
SortVerticalAscendingIcon.displayName = 'SortVerticalAscendingIcon';

function SvgSortVerticalDescendingIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 3.5h10V2H1zm0 8h4V10H1zm7-4H1V6h7zm2.97 7.03.53.53.53-.53 3.5-3.5-1.06-1.06-2.22 2.22V6h-1.5v6.19L8.53 9.97l-1.06 1.06z",
            clipRule: "evenodd"
        })
    });
}
const SortVerticalDescendingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSortVerticalDescendingIcon
    });
});
SortVerticalDescendingIcon.displayName = 'SortVerticalDescendingIcon';

function SvgSparkleDoubleFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M4.739 9.622a.75.75 0 0 0-1.478 0l-.152.876a.75.75 0 0 1-.61.61l-.878.153a.75.75 0 0 0 0 1.478l.877.152a.75.75 0 0 1 .61.61l.153.878a.75.75 0 0 0 1.478 0l.152-.877a.75.75 0 0 1 .61-.61l.878-.153a.75.75 0 0 0 0-1.478l-.877-.152a.75.75 0 0 1-.61-.61zM10.737.611a.75.75 0 0 0-1.474 0l-.264 1.398A3.75 3.75 0 0 1 6.01 5l-1.398.264a.75.75 0 0 0 0 1.474l1.398.264A3.75 3.75 0 0 1 9 9.99l.264 1.398a.75.75 0 0 0 1.474 0l.264-1.398A3.75 3.75 0 0 1 13.99 7l1.398-.264a.75.75 0 0 0 0-1.474l-1.398-.264A3.75 3.75 0 0 1 11 2.01z",
            clipRule: "evenodd"
        })
    });
}
const SparkleDoubleFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSparkleDoubleFillIcon
    });
});
SparkleDoubleFillIcon.displayName = 'SparkleDoubleFillIcon';

function SvgSparkleDoubleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M3.848 10.627 4 9.75l.152.877a1.5 1.5 0 0 0 1.221 1.22L6.25 12l-.877.152a1.5 1.5 0 0 0-1.22 1.221L4 14.25l-.152-.877a1.5 1.5 0 0 0-1.221-1.22L1.75 12l.877-.152a1.5 1.5 0 0 0 1.22-1.221"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M4 9a.75.75 0 0 1 .739.621l.152.877a.75.75 0 0 0 .61.61l.878.153a.75.75 0 0 1 0 1.478l-.877.152a.75.75 0 0 0-.61.61l-.153.878a.75.75 0 0 1-1.478 0l-.152-.877a.75.75 0 0 0-.61-.61l-.878-.153a.75.75 0 0 1 0-1.478l.877-.152a.75.75 0 0 0 .61-.61l.153-.878A.75.75 0 0 1 4 9m0 2.92-.08.08q.042.039.08.08.038-.042.08-.08zM10 0c.36 0 .67.257.737.611l.264 1.398A3.75 3.75 0 0 0 13.99 5l1.398.264a.75.75 0 0 1 0 1.474l-1.398.264A3.75 3.75 0 0 0 11 9.99l-.264 1.398a.75.75 0 0 1-1.474 0l-.264-1.398A3.75 3.75 0 0 0 6.01 7l-1.398-.264a.75.75 0 0 1 0-1.474l1.398-.264A3.75 3.75 0 0 0 9 2.01L9.263.611A.75.75 0 0 1 10 0m0 3.682A5.26 5.26 0 0 1 7.682 6 5.26 5.26 0 0 1 10 8.318 5.26 5.26 0 0 1 12.318 6 5.26 5.26 0 0 1 10 3.682",
                clipRule: "evenodd"
            })
        ]
    });
}
const SparkleDoubleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSparkleDoubleIcon
    });
});
SparkleDoubleIcon.displayName = 'SparkleDoubleIcon';

function SvgSparkleFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M13.667 6.575c1.376.452 1.376 2.398 0 2.85l-2.472.813a1.5 1.5 0 0 0-.957.957l-.813 2.472c-.452 1.376-2.398 1.376-2.85 0l-.813-2.472a1.5 1.5 0 0 0-.956-.957l-2.473-.813c-1.376-.452-1.376-2.398 0-2.85l2.473-.813a1.5 1.5 0 0 0 .956-.956l.813-2.473c.452-1.376 2.398-1.376 2.85 0l.813 2.473a1.5 1.5 0 0 0 .957.956z",
            clipRule: "evenodd"
        })
    });
}
const SparkleFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSparkleFillIcon
    });
});
SparkleFillIcon.displayName = 'SparkleFillIcon';

function SvgSparkleIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M10.726 8.813 13.199 8l-2.473-.813a3 3 0 0 1-1.913-1.913L8 2.801l-.813 2.473a3 3 0 0 1-1.913 1.913L2.801 8l2.473.813a3 3 0 0 1 1.913 1.913L8 13.199l.813-2.473a3 3 0 0 1 1.913-1.913m2.941.612c1.376-.452 1.376-2.398 0-2.85l-2.472-.813a1.5 1.5 0 0 1-.957-.956l-.813-2.473c-.452-1.376-2.398-1.376-2.85 0l-.813 2.473a1.5 1.5 0 0 1-.956.956l-2.473.813c-1.376.452-1.376 2.398 0 2.85l2.473.813a1.5 1.5 0 0 1 .956.957l.813 2.472c.452 1.376 2.398 1.376 2.85 0l.813-2.472a1.5 1.5 0 0 1 .957-.957z",
            clipRule: "evenodd"
        })
    });
}
const SparkleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSparkleIcon
    });
});
SparkleIcon.displayName = 'SparkleIcon';

function SvgSparkleRectangleIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M0 2.75A.75.75 0 0 1 .75 2H8v1.5H1.5v9h13V10H16v3.25a.75.75 0 0 1-.75.75H.75a.75.75 0 0 1-.75-.75zm12.987-.14a.75.75 0 0 0-1.474 0l-.137.728a1.93 1.93 0 0 1-1.538 1.538l-.727.137a.75.75 0 0 0 0 1.474l.727.137c.78.147 1.39.758 1.538 1.538l.137.727a.75.75 0 0 0 1.474 0l.137-.727c.147-.78.758-1.39 1.538-1.538l.727-.137a.75.75 0 0 0 0-1.474l-.727-.137a1.93 1.93 0 0 1-1.538-1.538z",
            clipRule: "evenodd"
        })
    });
}
const SparkleRectangleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSparkleRectangleIcon
    });
});
SparkleRectangleIcon.displayName = 'SparkleRectangleIcon';

function SvgSpeechBubbleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M6 7a.75.75 0 1 1-1.5 0A.75.75 0 0 1 6 7M8 7.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5M10.75 7.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M6 1a6 6 0 1 0 0 12v2.25a.75.75 0 0 0 1.28.53L10.061 13A6 6 0 0 0 10 1zM1.5 7A4.5 4.5 0 0 1 6 2.5h4a4.5 4.5 0 1 1 0 9h-.25a.75.75 0 0 0-.53.22L7.5 13.44v-1.19a.75.75 0 0 0-.75-.75H6A4.5 4.5 0 0 1 1.5 7",
                clipRule: "evenodd"
            })
        ]
    });
}
const SpeechBubbleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSpeechBubbleIcon
    });
});
SpeechBubbleIcon.displayName = 'SpeechBubbleIcon';

function SvgSpeechBubblePlusIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M7.25 9.5V7.75H5.5v-1.5h1.75V4.5h1.5v1.75h1.75v1.5H8.75V9.5z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M6 1a6 6 0 0 0-6 6v.25a5.75 5.75 0 0 0 5 5.701v2.299a.75.75 0 0 0 1.28.53L9.06 13H10a6 6 0 0 0 0-12zM1.5 7A4.5 4.5 0 0 1 6 2.5h4a4.5 4.5 0 1 1 0 9H8.75a.75.75 0 0 0-.53.22L6.5 13.44v-1.19a.75.75 0 0 0-.75-.75A4.25 4.25 0 0 1 1.5 7.25z",
                clipRule: "evenodd"
            })
        ]
    });
}
const SpeechBubblePlusIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSpeechBubblePlusIcon
    });
});
SpeechBubblePlusIcon.displayName = 'SpeechBubblePlusIcon';

function SvgSpeechBubbleQuestionMarkFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M0 7a6 6 0 0 1 6-6h4a6 6 0 0 1 0 12h-.94l-2.78 2.78A.75.75 0 0 1 5 15.25v-2.299A5.75 5.75 0 0 1 0 7.25zm10.079-.389A2.25 2.25 0 1 0 5.75 5.75h1.5A.75.75 0 1 1 8 6.5h-.75V8H8a2.25 2.25 0 0 0 2.079-1.389M8 10.5A.75.75 0 1 1 8 9a.75.75 0 0 1 0 1.5",
            clipRule: "evenodd"
        })
    });
}
const SpeechBubbleQuestionMarkFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSpeechBubbleQuestionMarkFillIcon
    });
});
SpeechBubbleQuestionMarkFillIcon.displayName = 'SpeechBubbleQuestionMarkFillIcon';

function SvgSpeechBubbleQuestionMarkIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M6 1a6 6 0 0 0-6 6v.25a5.75 5.75 0 0 0 5 5.701v2.299a.75.75 0 0 0 1.28.53L9.06 13H10a6 6 0 0 0 0-12zM1.5 7A4.5 4.5 0 0 1 6 2.5h4a4.5 4.5 0 1 1 0 9H8.75a.75.75 0 0 0-.53.22L6.5 13.44v-1.19a.75.75 0 0 0-.75-.75A4.25 4.25 0 0 1 1.5 7.25zm8.707-1.689A2.25 2.25 0 0 1 8 8h-.75V6.5H8a.75.75 0 1 0-.75-.75h-1.5a2.25 2.25 0 0 1 4.457-.439M7.25 9.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0",
            clipRule: "evenodd"
        })
    });
}
const SpeechBubbleQuestionMarkIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSpeechBubbleQuestionMarkIcon
    });
});
SpeechBubbleQuestionMarkIcon.displayName = 'SpeechBubbleQuestionMarkIcon';

function SvgSpeechBubbleStarIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 3.5a.5.5 0 0 1 .476.345l.56 1.728h1.817a.5.5 0 0 1 .294.904l-1.47 1.068.562 1.728a.5.5 0 0 1-.77.559L8 8.764 6.53 9.832a.5.5 0 0 1-.769-.56l.561-1.727-1.47-1.068a.5.5 0 0 1 .295-.904h1.816l.561-1.728A.5.5 0 0 1 8 3.5"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M6 1a6 6 0 1 0 0 12v2.25a.75.75 0 0 0 1.28.53L10.061 13A6 6 0 0 0 10 1zM1.5 7A4.5 4.5 0 0 1 6 2.5h4a4.5 4.5 0 1 1 0 9h-.25a.75.75 0 0 0-.53.22L7.5 13.44v-1.19a.75.75 0 0 0-.75-.75H6A4.5 4.5 0 0 1 1.5 7",
                clipRule: "evenodd"
            })
        ]
    });
}
const SpeechBubbleStarIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSpeechBubbleStarIcon
    });
});
SpeechBubbleStarIcon.displayName = 'SpeechBubbleStarIcon';

function SvgSpeedometerIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M5.749 7.82a.75.75 0 0 1 .896.186v.003l.006.006.022.025.08.095c.07.081.172.198.293.34.242.286.57.675.91 1.086.337.409.688.844.979 1.221.272.354.541.723.664.969a1.75 1.75 0 0 1-3.135 1.558c-.126-.253-.252-.7-.362-1.133a44 44 0 0 1-.355-1.534 94 94 0 0 1-.288-1.395l-.088-.443-.023-.124-.006-.033-.002-.008v-.002l-.001-.001-.013-.124a.75.75 0 0 1 .423-.692"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 1a8 8 0 0 1 6.927 12H13.12a6.47 6.47 0 0 0 1.334-3.254l-1.953.004-.004-1.5 1.957-.004a6.47 6.47 0 0 0-1.363-3.284l-1.378 1.385-1.064-1.058 1.38-1.387a6.47 6.47 0 0 0-3.285-1.359V4.5h-1.5V2.544a6.47 6.47 0 0 0-3.278 1.36l1.38 1.385-1.064 1.058-1.378-1.382a6.47 6.47 0 0 0-1.361 3.281l1.958.004-.004 1.5-1.954-.004A6.47 6.47 0 0 0 2.879 13H1.073A8 8 0 0 1 8 1"
            })
        ]
    });
}
const SpeedometerIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSpeedometerIcon
    });
});
SpeedometerIcon.displayName = 'SpeedometerIcon';

function SvgStarFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M7.995 0a.75.75 0 0 1 .714.518l1.459 4.492h4.723a.75.75 0 0 1 .44 1.356l-3.82 2.776 1.459 4.492a.75.75 0 0 1-1.154.838l-3.82-2.776-3.821 2.776a.75.75 0 0 1-1.154-.838L4.48 9.142.66 6.366A.75.75 0 0 1 1.1 5.01h4.723L7.282.518A.75.75 0 0 1 7.995 0"
        })
    });
}
const StarFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgStarFillIcon
    });
});
StarFillIcon.displayName = 'StarFillIcon';

function SvgStarIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M7.995 0a.75.75 0 0 1 .714.518l1.459 4.492h4.723a.75.75 0 0 1 .44 1.356l-3.82 2.776 1.459 4.492a.75.75 0 0 1-1.154.838l-3.82-2.776-3.821 2.776a.75.75 0 0 1-1.154-.838L4.48 9.142.66 6.366A.75.75 0 0 1 1.1 5.01h4.723L7.282.518A.75.75 0 0 1 7.995 0m0 3.177-.914 2.814a.75.75 0 0 1-.713.519h-2.96l2.394 1.739a.75.75 0 0 1 .273.839l-.915 2.814 2.394-1.74a.75.75 0 0 1 .882 0l2.394 1.74-.914-2.814a.75.75 0 0 1 .272-.839l2.394-1.74H9.623a.75.75 0 0 1-.713-.518z",
            clipRule: "evenodd"
        })
    });
}
const StarIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgStarIcon
    });
});
StarIcon.displayName = 'StarIcon';

function SvgStopCircleFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16M6.125 5.5a.625.625 0 0 0-.625.625v3.75c0 .345.28.625.625.625h3.75c.345 0 .625-.28.625-.625v-3.75a.625.625 0 0 0-.625-.625z",
            clipRule: "evenodd"
        })
    });
}
const StopCircleFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgStopCircleFillIcon
    });
});
StopCircleFillIcon.displayName = 'StopCircleFillIcon';

function SvgStopCircleIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M5.5 6a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5H6a.5.5 0 0 1-.5-.5z",
            clipRule: "evenodd"
        })
    });
}
const StopCircleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgStopCircleIcon
    });
});
StopCircleIcon.displayName = 'StopCircleIcon';

function SvgStopIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M4.5 4a.5.5 0 0 0-.5.5v7a.5.5 0 0 0 .5.5h7a.5.5 0 0 0 .5-.5v-7a.5.5 0 0 0-.5-.5z"
        })
    });
}
const StopIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgStopIcon
    });
});
StopIcon.displayName = 'StopIcon';

function SvgStoredProcedureIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.514 11.853c.019-.244.075-.478.16-.698L.8 10.458l.936-1.172.872.695q.297-.201.646-.312V8.554h1.5v1.115q.35.111.647.312l.872-.696.935 1.173-.874.697q.129.331.16.698l1.09.248-.334 1.463-1.09-.248a2.5 2.5 0 0 1-.447.559l.486 1.007-1.351.65-.486-1.006a2.5 2.5 0 0 1-.358.027q-.181 0-.358-.027l-.484 1.006-1.352-.65.485-1.006a2.5 2.5 0 0 1-.447-.56l-1.089.248-.334-1.462zm1.49.2a1 1 0 1 0 2 .001 1 1 0 0 0-2 0",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("circle", {
                cx: 12.25,
                cy: 3.75,
                r: 2,
                stroke: "currentColor",
                strokeWidth: 1.5
            }),
            /*#__PURE__*/ jsx("path", {
                stroke: "currentColor",
                strokeWidth: 1.5,
                d: "M10.25 3.75H6a2 2 0 0 0-2 2V7.5M13.5 12.25h-5M11.5 9.75l2.5 2.5-2.5 2.5"
            })
        ]
    });
}
const StoredProcedureIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgStoredProcedureIcon
    });
});
StoredProcedureIcon.displayName = 'StoredProcedureIcon';

function SvgStorefrontIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M3.52 2.3a.75.75 0 0 1 .6-.3h7.76a.75.75 0 0 1 .6.3l2.37 3.158a.75.75 0 0 1 .15.45v.842q0 .059-.009.115A2.31 2.31 0 0 1 14 8.567v5.683a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75V8.567A2.31 2.31 0 0 1 1 6.75v-.841a.75.75 0 0 1 .15-.45zm7.605 6.068c.368.337.847.557 1.375.6V13.5h-9V8.968a2.3 2.3 0 0 0 1.375-.6c.411.377.96.607 1.563.607.602 0 1.15-.23 1.562-.607.411.377.96.607 1.563.607.602 0 1.15-.23 1.562-.607m2.375-2.21v.532l-.001.019a.813.813 0 0 1-1.623 0l-.008-.076a1 1 0 0 0 .012-.133V4zm-3.113.445a1 1 0 0 0-.013.106.813.813 0 0 1-1.624-.019V3.5h1.63v3q0 .053.007.103M7.25 3.5v3.19l-.001.019a.813.813 0 0 1-1.623 0l-.006-.064V3.5zM4.12 4 2.5 6.16v.531l.001.019a.813.813 0 0 0 1.619.045z",
            clipRule: "evenodd"
        })
    });
}
const StorefrontIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgStorefrontIcon
    });
});
StorefrontIcon.displayName = 'StorefrontIcon';

function SvgStreamIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.52 7.48a6.5 6.5 0 0 1 12.722-1.298l-.09-.091a3.75 3.75 0 0 0-5.304 0L6.091 8.848a2.25 2.25 0 0 1-3.182 0L1.53 7.47zm.238 2.338A6.5 6.5 0 0 0 14.48 8.52l-.01.01-1.379-1.378a2.25 2.25 0 0 0-3.182 0L7.152 9.909a3.75 3.75 0 0 1-5.304 0z",
            clipRule: "evenodd"
        })
    });
}
const StreamIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgStreamIcon
    });
});
StreamIcon.displayName = 'StreamIcon';

function SvgStrikeThroughIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M7.784 4C6.6 4 5.75 4.736 5.75 5.72c0 .384.07.625.152.78.08.15.191.262.35.356.365.216.894.3 1.634.4l.07.01c.381.052.827.113 1.263.234H15V9H1V7.5h3.764a2.4 2.4 0 0 1-.188-.298c-.222-.421-.326-.916-.326-1.482 0-2.056 1.789-3.22 3.534-3.22 1.746 0 3.535 1.164 3.535 3.22h-1.5c0-.984-.85-1.72-2.035-1.72M4.257 10.5c.123 1.92 1.845 3 3.527 3s3.405-1.08 3.528-3H9.804c-.116.871-.925 1.5-2.02 1.5s-1.903-.629-2.02-1.5z"
        })
    });
}
const StrikeThroughIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgStrikeThroughIcon
    });
});
StrikeThroughIcon.displayName = 'StrikeThroughIcon';

function SvgSunIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8.75 16h-1.5v-3h1.5zM4.995 12.065l-2.121 2.123-1.06-1.061 2.12-2.122zM14.188 13.127l-1.061 1.06-2.121-2.122 1.06-1.06z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8 4.25a3.75 3.75 0 1 1 0 7.5 3.75 3.75 0 0 1 0-7.5m0 1.5a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M3 8.75H0v-1.5h3zM16 8.75h-3v-1.5h3zM4.995 3.935l-1.06 1.06-2.122-2.122 1.061-1.06zM14.188 2.873l-2.122 2.122-1.06-1.06 2.121-2.122zM8.75 3h-1.5V0h1.5z"
            })
        ]
    });
}
const SunIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSunIcon
    });
});
SunIcon.displayName = 'SunIcon';

function SvgSyncIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8 2.5a5.48 5.48 0 0 1 3.817 1.54l.009.009.5.451H11V6h4V2h-1.5v1.539l-.651-.588A7 7 0 0 0 1 8h1.5A5.5 5.5 0 0 1 8 2.5M1 10h4v1.5H3.674l.5.451.01.01A5.5 5.5 0 0 0 13.5 8h1.499a7 7 0 0 1-11.849 5.048L2.5 12.46V14H1z"
        })
    });
}
const SyncIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSyncIcon
    });
});
SyncIcon.displayName = 'SyncIcon';

function SvgSyncToFileIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M14.25 6.22a.75.75 0 0 1 .75.75v.53c0 3.175-2.574 5.53-5.75 5.53H4.06l.69.688A.751.751 0 0 1 3.69 14.78l-1.97-1.97a.75.75 0 0 1 0-1.06l1.97-1.97a.751.751 0 0 1 1.061 1.062l-.69.688h5.19c2.347 0 4.25-1.683 4.25-4.03v-.53a.75.75 0 0 1 .75-.75M11 1.22a.75.75 0 0 1 1.062 0l1.968 1.97c.293.262.293.737 0 1.06l-1.968 1.97A.751.751 0 0 1 11 5.158l.69-.688H6.5c-2.347 0-4.25 1.682-4.25 4.03v.53a.75.75 0 0 1-1.5 0V8.5c0-3.176 2.574-5.53 5.75-5.53h5.19L11 2.28a.75.75 0 0 1 0-1.061"
        })
    });
}
const SyncToFileIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgSyncToFileIcon
    });
});
SyncToFileIcon.displayName = 'SyncToFileIcon';

function SvgTableClockIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12 8a4 4 0 1 1 0 8 4 4 0 0 1 0-8m-.75 4.31 1.72 1.72 1.06-1.06-1.28-1.28V9.5h-1.5z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M14.327 1.004A.75.75 0 0 1 15 1.75V7H6.5v8H1.75a.75.75 0 0 1-.75-.75V1.75l.004-.077A.75.75 0 0 1 1.75 1h12.5zM2.5 13.5H5V7H2.5zm0-8h11v-3h-11z",
                clipRule: "evenodd"
            })
        ]
    });
}
const TableClockIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTableClockIcon
    });
});
TableClockIcon.displayName = 'TableClockIcon';

function SvgTableCombineIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M11.327 1.004A.75.75 0 0 1 12 1.75V5.5H5.5V12H1.75l-.077-.004a.75.75 0 0 1-.67-.669L1 11.25v-9.5a.75.75 0 0 1 .673-.746L1.75 1h9.5zM2.5 5.5v5H4v-5zm0-1.5h8V2.5h-8z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M14.327 4.004A.75.75 0 0 1 15 4.75v9.5l-.004.077a.75.75 0 0 1-.669.67L14.25 15h-9.5l-.077-.004a.75.75 0 0 1-.67-.669L4 14.25v-9.5a.75.75 0 0 1 .673-.746L4.75 4h9.5zM5.5 13.5H7v-5H5.5zm6.5 0h1.5v-5H12zm-3.5 0h2v-5h-2zM5.5 7h8V5.5h-8z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M11.327 1.004A.75.75 0 0 1 12 1.75V4h2.25l.077.004A.75.75 0 0 1 15 4.75v9.5l-.004.077a.75.75 0 0 1-.669.67L14.25 15h-9.5l-.077-.004a.75.75 0 0 1-.67-.669L4 14.25V12H1.75l-.077-.004a.75.75 0 0 1-.67-.669L1 11.25v-9.5a.75.75 0 0 1 .673-.746L1.75 1h9.5zM5.5 13.5H7v-5H5.5zm3 0h2v-5h-2zm3.5 0h1.5v-5H12zm-9.5-3H4v-5H2.5zm3-3.5h8V5.5h-8zm-3-3h8V2.5h-8z",
                clipRule: "evenodd"
            })
        ]
    });
}
const TableCombineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTableCombineIcon
    });
});
TableCombineIcon.displayName = 'TableCombineIcon';

function SvgTableGlassesIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H4v-1.5H2.5V7H5v2h1.5V7h3v2H11V7h2.5v2H15V1.75a.75.75 0 0 0-.75-.75zM13.5 5.5v-3h-11v3z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M11.75 10a.75.75 0 0 0-.707.5H9.957a.75.75 0 0 0-.708-.5H5.75a.75.75 0 0 0-.75.75v1.75a2.5 2.5 0 0 0 5 0V12h1v.5a2.5 2.5 0 0 0 5 0v-1.75a.75.75 0 0 0-.75-.75zm.75 2.5v-1h2v1a1 1 0 1 1-2 0m-6-1v1a1 1 0 1 0 2 0v-1z",
                clipRule: "evenodd"
            })
        ]
    });
}
const TableGlassesIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTableGlassesIcon
    });
});
TableGlassesIcon.displayName = 'TableGlassesIcon';

function SvgTableGlobeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H7v-1.5h-.5V7H15V1.75a.75.75 0 0 0-.75-.75zM5 7v6.5H2.5V7zm8.5-1.5v-3h-11v3z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M11.625 7.25a4.375 4.375 0 1 0 0 8.75 4.375 4.375 0 0 0 0-8.75M9.952 9.287a10.5 10.5 0 0 0-.185 1.588H8.85a2.88 2.88 0 0 1 1.103-1.588m1.547-.02c-.116.41-.196.963-.23 1.608h.712c-.034-.646-.114-1.198-.23-1.608a2.5 2.5 0 0 0-.126-.353q-.06.13-.126.353m0 4.716c-.116-.41-.196-.963-.23-1.608h.712c-.034.646-.114 1.198-.23 1.608-.043.15-.086.265-.126.353a2.5 2.5 0 0 1-.126-.353m1.799-4.696c.098.475.158 1.016.185 1.588h.918a2.88 2.88 0 0 0-1.103-1.588m.185 3.088h.918a2.88 2.88 0 0 1-1.103 1.588c.098-.475.158-1.016.185-1.588m-4.634 0h.918c.027.572.087 1.113.185 1.588a2.88 2.88 0 0 1-1.103-1.588",
                clipRule: "evenodd"
            })
        ]
    });
}
const TableGlobeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTableGlobeIcon
    });
});
TableGlobeIcon.displayName = 'TableGlobeIcon';

function SvgTableIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75v3h11v-3zm0 11V7H5v6.5zm4 0h3V7h-3zM11 7v6.5h2.5V7z",
            clipRule: "evenodd"
        })
    });
}
const TableIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTableIcon
    });
});
TableIcon.displayName = 'TableIcon';

function SvgTableLightningIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H8v-1.5H6.5V7h7v2H15V1.75a.75.75 0 0 0-.75-.75zM5 7H2.5v6.5H5zm8.5-1.5v-3h-11v3z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m8.43 11.512 3-3.5 1.14.976-1.94 2.262H14a.75.75 0 0 1 .57 1.238l-3 3.5-1.14-.976 1.94-2.262H9a.75.75 0 0 1-.57-1.238"
            })
        ]
    });
}
const TableLightningIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTableLightningIcon
    });
});
TableLightningIcon.displayName = 'TableLightningIcon';

function SvgTableMeasureIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H4v-1.5H2.5V7H5v2h1.5V7h3v2H11V7h2.5v2H15V1.75a.75.75 0 0 0-.75-.75zM13.5 5.5v-3h-11v3z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M5 11v3.25c0 .414.336.75.75.75h9.5a.75.75 0 0 0 .75-.75V11h-1.5v2.5h-.875V12h-1.5v1.5h-.875V11h-1.5v2.5h-.875V12h-1.5v1.5H6.5V11z"
            })
        ]
    });
}
const TableMeasureIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTableMeasureIcon
    });
});
TableMeasureIcon.displayName = 'TableMeasureIcon';

function SvgTableModelIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H6.5V7H15V1.75a.75.75 0 0 0-.75-.75zM5 7v6.5H2.5V7zm8.5-1.5v-3h-11v3z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M7.25 8.5a1.25 1.25 0 1 1 2.488.177l1.48 1.481a2 2 0 0 1 1.563 0l.731-.731a1.25 1.25 0 1 1 1.06 1.06l-.73.732a2 2 0 0 1 0 1.562l.731.731a1.25 1.25 0 1 1-1.06 1.06l-.732-.73a2 2 0 0 1-2.636-1.092H9.5a1.25 1.25 0 1 1 0-1.5h.645l.013-.031-1.481-1.481A1.25 1.25 0 0 1 7.25 8.5M11.5 12a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0",
                clipRule: "evenodd"
            })
        ]
    });
}
const TableModelIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTableModelIcon
    });
});
TableModelIcon.displayName = 'TableModelIcon';

function SvgTableStreamIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H6.5V7h3v1H11V7h2.5v1H15V1.75a.75.75 0 0 0-.75-.75zM5 7v6.5H2.5V7zm8.5-1.5v-3h-11v3z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M9.024 10.92a1.187 1.187 0 0 1 1.876.03 2.687 2.687 0 0 0 4.247.066l.439-.548-1.172-.937-.438.548a1.187 1.187 0 0 1-1.876-.03 2.687 2.687 0 0 0-4.247-.066l-.439.548 1.172.937z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M9.024 13.92a1.187 1.187 0 0 1 1.876.03 2.687 2.687 0 0 0 4.247.066l.439-.548-1.172-.937-.438.548a1.187 1.187 0 0 1-1.876-.03 2.687 2.687 0 0 0-4.247-.066l-.439.548 1.172.937z"
            })
        ]
    });
}
const TableStreamIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTableStreamIcon
    });
});
TableStreamIcon.displayName = 'TableStreamIcon';

function SvgTableVectorIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H8v-1.5H6.5V7h7v2H15V1.75a.75.75 0 0 0-.75-.75zM5 7H2.5v6.5H5zm8.5-1.5v-3h-11v3z",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("circle", {
                cx: 12,
                cy: 12,
                r: 1,
                fill: "currentColor"
            }),
            /*#__PURE__*/ jsx("circle", {
                cx: 12.5,
                cy: 9.5,
                r: 0.5,
                fill: "currentColor"
            }),
            /*#__PURE__*/ jsx("circle", {
                cx: 9.5,
                cy: 12.5,
                r: 0.5,
                fill: "currentColor"
            }),
            /*#__PURE__*/ jsx("circle", {
                cx: 13.75,
                cy: 13.75,
                r: 0.75,
                fill: "currentColor"
            }),
            /*#__PURE__*/ jsx("circle", {
                cx: 9.75,
                cy: 9.75,
                r: 0.75,
                fill: "currentColor"
            }),
            /*#__PURE__*/ jsx("path", {
                stroke: "currentColor",
                strokeWidth: 0.3,
                d: "M13.5 13.5 12 12m0 0 .5-2.5M12 12l-2.5.5M10 10l2 2"
            })
        ]
    });
}
const TableVectorIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTableVectorIcon
    });
});
TableVectorIcon.displayName = 'TableVectorIcon';

function SvgTableViewIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                fillRule: "evenodd",
                clipPath: "url(#TableViewIcon_svg__a)",
                clipRule: "evenodd",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H4v-1.5H2.5V7H5v2h1.5V7h3v2H11V7h2.5v2H15V1.75a.75.75 0 0 0-.75-.75zM13.5 5.5v-3h-11v3z"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M11.75 10a.75.75 0 0 0-.707.5H9.957a.75.75 0 0 0-.707-.5h-3.5a.75.75 0 0 0-.75.75v1.75a2.5 2.5 0 0 0 5 0V12h1v.5a2.5 2.5 0 0 0 5 0v-1.75a.75.75 0 0 0-.75-.75zm.75 2.5v-1h2v1a1 1 0 1 1-2 0m-6-1v1a1 1 0 1 0 2 0v-1z"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const TableViewIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTableViewIcon
    });
});
TableViewIcon.displayName = 'TableViewIcon';

function SvgTagIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M5 6a1 1 0 1 0 0-2 1 1 0 0 0 0 2"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.768 1.018a.75.75 0 0 0-.75.75v6.1c0 .199.079.39.22.53l6.884 6.885a.75.75 0 0 0 1.06 0l6.101-6.1a.75.75 0 0 0 0-1.061L8.4 1.237a.75.75 0 0 0-.53-.22zm6.884 12.674L2.518 7.557v-5.04h5.04l6.134 6.135z",
                clipRule: "evenodd"
            })
        ]
    });
}
const TagIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTagIcon
    });
});
TagIcon.displayName = 'TagIcon';

function SvgTargetIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8 10.667a2.667 2.667 0 1 0 0-5.334 2.667 2.667 0 0 0 0 5.334m0-1.334a1.333 1.333 0 1 0 0-2.666 1.333 1.333 0 0 0 0 2.666",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8 3.667c.184 0 .334.149.334.333v2a.333.333 0 0 1-.667 0V4c0-.184.15-.333.333-.333M12.334 8c0 .184-.15.333-.334.333h-2a.333.333 0 1 1 0-.666h2c.184 0 .334.149.334.333M8 9.667c.184 0 .334.149.334.333v2a.333.333 0 0 1-.667 0v-2c0-.184.15-.333.333-.333M6.334 8c0 .184-.15.333-.334.333H4a.333.333 0 0 1 0-.666h2c.184 0 .333.149.333.333",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M11.334 2.667a.667.667 0 1 1 0-1.334H12A2.667 2.667 0 0 1 14.667 4v.667a.667.667 0 1 1-1.333 0V4c0-.736-.598-1.333-1.334-1.333zM5.334 2a.667.667 0 0 1-.667.667H4c-.736 0-1.333.597-1.333 1.333v.667a.667.667 0 0 1-1.333 0V4A2.667 2.667 0 0 1 4 1.333h.667c.368 0 .667.299.667.667M1.334 12v-.667a.667.667 0 1 1 1.333 0V12c0 .736.597 1.333 1.333 1.333h.667a.667.667 0 0 1 0 1.334H4A2.667 2.667 0 0 1 1.334 12M11.334 13.333a.667.667 0 0 0 0 1.334H12A2.667 2.667 0 0 0 14.667 12v-.667a.667.667 0 1 0-1.333 0V12c0 .736-.598 1.333-1.334 1.333z"
            })
        ]
    });
}
const TargetIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTargetIcon
    });
});
TargetIcon.displayName = 'TargetIcon';

function SvgTerminalIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M5.03 4.97 8.06 8l-3.03 3.03-1.06-1.06L5.94 8 3.97 6.03zM12 9.5H8V11h4z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75v11h11v-11z",
                clipRule: "evenodd"
            })
        ]
    });
}
const TerminalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTerminalIcon
    });
});
TerminalIcon.displayName = 'TerminalIcon';

function SvgTextBoxIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11zM5 6h2.25v5.5h1.5V6H11V4.5H5z",
            clipRule: "evenodd"
        })
    });
}
const TextBoxIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTextBoxIcon
    });
});
TextBoxIcon.displayName = 'TextBoxIcon';

function SvgTextIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M7.118 13V4.62H4.083V3.135h7.84v1.483H8.883V13z"
        })
    });
}
const TextIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTextIcon
    });
});
TextIcon.displayName = 'TextIcon';

function SvgTextJustifyIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M15 15H1v-1.5h14zM15 11.75H1v-1.5h14zM15 8.75H1v-1.5h14zM15 5.75H1v-1.5h14zM15 2.5H1V1h14z"
        })
    });
}
const TextJustifyIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTextJustifyIcon
    });
});
TextJustifyIcon.displayName = 'TextJustifyIcon';

function SvgTextUnderlineIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8.5 1H10v3.961a3.4 3.4 0 0 1 1.75-.461c.857 0 1.674.287 2.283.863.616.582.967 1.411.967 2.387s-.351 1.805-.967 2.387c-.61.576-1.426.863-2.283.863a3.4 3.4 0 0 1-1.75-.461V11H8.5zM10 7.75c0 .602.208 1.023.498 1.297.295.28.728.453 1.252.453s.957-.174 1.252-.453c.29-.274.498-.695.498-1.297s-.208-1.023-.498-1.297C12.708 6.173 12.275 6 11.75 6s-.957.174-1.252.453c-.29.274-.498.695-.498 1.297M4 5.25c-.582 0-1.16.16-1.755.365l-.49-1.417C2.385 3.979 3.159 3.75 4 3.75a3 3 0 0 1 3 3V11H5.5v-.298A3.7 3.7 0 0 1 4 11c-.741 0-1.47-.191-2.035-.607A2.3 2.3 0 0 1 1 8.5c0-.81.381-1.464.965-1.893C2.529 6.19 3.259 6 4 6c.494 0 .982.085 1.42.264A1.5 1.5 0 0 0 4 5.25m1.147 2.565c.23.17.353.39.353.685a.8.8 0 0 1-.353.685C4.897 9.369 4.5 9.5 4 9.5s-.897-.131-1.147-.315A.8.8 0 0 1 2.5 8.5c0-.295.123-.515.353-.685C3.103 7.631 3.5 7.5 4 7.5s.897.131 1.147.315",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M1 12.5h14V14H1z"
            })
        ]
    });
}
const TextUnderlineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTextUnderlineIcon
    });
});
TextUnderlineIcon.displayName = 'TextUnderlineIcon';

function SvgThreeDotsIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 18 15",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M9 7.375A3.437 3.437 0 1 1 9 .5a3.437 3.437 0 0 1 0 6.875M13.688 8a3.438 3.438 0 1 0 0 6.875 3.438 3.438 0 0 0 0-6.875M4.313 8a3.437 3.437 0 1 0 0 6.875 3.437 3.437 0 0 0 0-6.875"
        })
    });
}
const ThreeDotsIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgThreeDotsIcon
    });
});
ThreeDotsIcon.displayName = 'ThreeDotsIcon';

function SvgThumbsDownIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#ThumbsDownIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    fillRule: "evenodd",
                    d: "M13.655 2.274a.8.8 0 0 0-.528-.19h-1.044v5.833h1.044a.79.79 0 0 0 .79-.643V2.725a.8.8 0 0 0-.262-.451m-3.072 6.233V2.083H3.805a.58.58 0 0 0-.583.496v.001l-.92 6a.585.585 0 0 0 .583.67h3.782a.75.75 0 0 1 .75.75v2.667a1.25 1.25 0 0 0 .8 1.166zm1.238.91L9.352 14.97a.75.75 0 0 1-.685.446 2.75 2.75 0 0 1-2.75-2.75V10.75h-3.02A2.082 2.082 0 0 1 .82 8.354l.92-6A2.085 2.085 0 0 1 3.816.584h9.29a2.29 2.29 0 0 1 2.303 1.982 1 1 0 0 1 .007.1v4.667a1 1 0 0 1-.007.1 2.29 2.29 0 0 1-2.303 1.984z",
                    clipRule: "evenodd"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const ThumbsDownIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgThumbsDownIcon
    });
});
ThumbsDownIcon.displayName = 'ThumbsDownIcon';

function SvgThumbsUpIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#ThumbsUpIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    fillRule: "evenodd",
                    d: "M6.648 1.029a.75.75 0 0 1 .685-.446 2.75 2.75 0 0 1 2.75 2.75V5.25h3.02a2.083 2.083 0 0 1 2.079 2.396l-.92 6a2.085 2.085 0 0 1-2.08 1.77H2.668a2.083 2.083 0 0 1-2.084-2.082V8.667a2.083 2.083 0 0 1 2.084-2.083h1.512zM3.917 8.084h-1.25a.583.583 0 0 0-.584.583v4.667a.583.583 0 0 0 .584.583h1.25zm1.5 5.833h6.778a.58.58 0 0 0 .583-.496l.92-6a.584.584 0 0 0-.583-.67H9.333a.75.75 0 0 1-.75-.75V3.332a1.25 1.25 0 0 0-.8-1.166L5.417 7.493z",
                    clipRule: "evenodd"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const ThumbsUpIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgThumbsUpIcon
    });
});
ThumbsUpIcon.displayName = 'ThumbsUpIcon';

function SvgTokenIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 12 12",
        ...props,
        children: [
            /*#__PURE__*/ jsx("mask", {
                id: "TokenIcon_svg__a",
                fill: "#fff",
                children: /*#__PURE__*/ jsx("path", {
                    d: "M5.596 10.799a5 5 0 1 0 .082-9.621l.258.94a4.025 4.025 0 1 1-.066 7.745z"
                })
            }),
            /*#__PURE__*/ jsx("path", {
                stroke: "currentColor",
                strokeWidth: 2,
                d: "M5.596 10.799a5 5 0 1 0 .082-9.621l.258.94a4.025 4.025 0 1 1-.066 7.745z",
                mask: "url(#TokenIcon_svg__a)"
            }),
            /*#__PURE__*/ jsx("circle", {
                cx: 5,
                cy: 6,
                r: 4.5,
                stroke: "currentColor"
            }),
            /*#__PURE__*/ jsx("circle", {
                cx: 5,
                cy: 6,
                r: 1.5,
                stroke: "currentColor"
            })
        ]
    });
}
const TokenIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTokenIcon
    });
});
TokenIcon.displayName = 'TokenIcon';

function SvgTrashIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M6 0a.75.75 0 0 0-.712.513L4.46 3H1v1.5h1.077l1.177 10.831A.75.75 0 0 0 4 16h8a.75.75 0 0 0 .746-.669L13.923 4.5H15V3h-3.46L10.713.513A.75.75 0 0 0 10 0zm3.96 3-.5-1.5H6.54L6.04 3zM3.585 4.5l1.087 10h6.654l1.087-10z",
            clipRule: "evenodd"
        })
    });
}
const TrashIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTrashIcon
    });
});
TrashIcon.displayName = 'TrashIcon';

function SvgTreeIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2.004 9.602a2.751 2.751 0 1 0 3.371 3.47 2.751 2.751 0 0 0 5.25 0 2.751 2.751 0 1 0 3.371-3.47A2.75 2.75 0 0 0 11.25 7h-2.5v-.604a2.751 2.751 0 1 0-1.5 0V7h-2.5a2.75 2.75 0 0 0-2.746 2.602M2.75 11a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m4.5-2.5h-2.5a1.25 1.25 0 0 0-1.242 1.106 2.76 2.76 0 0 1 1.867 1.822A2.76 2.76 0 0 1 7.25 9.604zm1.5 0v1.104c.892.252 1.6.942 1.875 1.824a2.76 2.76 0 0 1 1.867-1.822A1.25 1.25 0 0 0 11.25 8.5zM12 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0m-5.25 0a1.25 1.25 0 1 0 2.5 0 1.25 1.25 0 0 0-2.5 0M8 5a1.25 1.25 0 1 1 0-2.5A1.25 1.25 0 0 1 8 5",
            clipRule: "evenodd"
        })
    });
}
const TreeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTreeIcon
    });
});
TreeIcon.displayName = 'TreeIcon';

function SvgTrendingIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M6.332.14a.99.99 0 0 1 .98-.018c4.417 2.052 6.918 6.313 6.764 9.876-.066 1.497-.602 2.883-1.64 3.895-1.04 1.012-2.531 1.604-4.42 1.607a5.745 5.745 0 0 1-6.097-5.504v-.008a4.85 4.85 0 0 1 2.495-4.366.554.554 0 0 1 .776.261c.27.613.648 1.17 1.115 1.646.547-.714.8-1.637.792-2.652-.009-1.208-.388-2.502-1.024-3.605A.84.84 0 0 1 6.333.14"
        })
    });
}
const TrendingIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTrendingIcon
    });
});
TrendingIcon.displayName = 'TrendingIcon';

function SvgTriangleIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8 3a.75.75 0 0 1 .65.375l4.33 7.5A.75.75 0 0 1 12.33 12H3.67a.75.75 0 0 1-.65-1.125l4.33-7.5.056-.083A.75.75 0 0 1 8 3"
        })
    });
}
const TriangleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgTriangleIcon
    });
});
TriangleIcon.displayName = 'TriangleIcon';

function SvgUnderlineIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M4.544 6.466 4.6 2.988l1.5.024-.056 3.478A1.978 1.978 0 1 0 10 6.522V3h1.5v3.522a3.478 3.478 0 1 1-6.956-.056M12 13H4v-1.5h8z",
            clipRule: "evenodd"
        })
    });
}
const UnderlineIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUnderlineIcon
    });
});
UnderlineIcon.displayName = 'UnderlineIcon';

function SvgUndoIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#UndoIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    d: "M2.81 6.5h8.69a3 3 0 0 1 0 6H7V14h4.5a4.5 4.5 0 0 0 0-9H2.81l2.72-2.72-1.06-1.06-4.53 4.53 4.53 4.53 1.06-1.06z"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M16 16H0V0h16z"
                    })
                })
            })
        ]
    });
}
const UndoIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUndoIcon
    });
});
UndoIcon.displayName = 'UndoIcon';

function SvgUploadIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M1 13.56h14v1.5H1zM12.53 5.53l-1.06 1.061-2.72-2.72v7.19h-1.5V3.87l-2.72 2.72-1.06-1.06L8 1z"
        })
    });
}
const UploadIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUploadIcon
    });
});
UploadIcon.displayName = 'UploadIcon';

function SvgUsbIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M8 0a.75.75 0 0 1 .65.375l1.299 2.25a.75.75 0 0 1-.65 1.125H8.75V9.5h2.75V8h-.25a.75.75 0 0 1-.75-.75v-2a.75.75 0 0 1 .75-.75h2a.75.75 0 0 1 .75.75v2a.75.75 0 0 1-.75.75H13v2.25a.75.75 0 0 1-.75.75h-3.5v1.668a1.75 1.75 0 1 1-1.5 0V11h-3.5a.75.75 0 0 1-.75-.75V7.832a1.75 1.75 0 1 1 1.5 0V9.5h2.75V3.75h-.549a.75.75 0 0 1-.65-1.125l1.3-2.25A.75.75 0 0 1 8 0"
        })
    });
}
const UsbIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUsbIcon
    });
});
UsbIcon.displayName = 'UsbIcon';

function SvgUserBadgeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8 5.25a2.75 2.75 0 1 0 0 5.5 2.75 2.75 0 0 0 0-5.5M6.75 8a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "m4.401 2.5.386-.867A2.75 2.75 0 0 1 7.3 0h1.4a2.75 2.75 0 0 1 2.513 1.633l.386.867h1.651a.75.75 0 0 1 .75.75v12a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75v-12a.75.75 0 0 1 .75-.75zm1.756-.258A1.25 1.25 0 0 1 7.3 1.5h1.4c.494 0 .942.29 1.143.742l.114.258H6.043zM8 12a8.7 8.7 0 0 0-4.5 1.244V4h9v9.244A8.7 8.7 0 0 0 8 12m0 1.5c1.342 0 2.599.364 3.677 1H4.323A7.2 7.2 0 0 1 8 13.5",
                clipRule: "evenodd"
            })
        ]
    });
}
const UserBadgeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUserBadgeIcon
    });
});
UserBadgeIcon.displayName = 'UserBadgeIcon';

function SvgUserCircleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M5.25 6.75a2.75 2.75 0 1 1 5.5 0 2.75 2.75 0 0 1-5.5 0M8 5.5A1.25 1.25 0 1 0 8 8a1.25 1.25 0 0 0 0-2.5",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 0 0-4.773 10.912A8.73 8.73 0 0 1 8 11c1.76 0 3.4.52 4.773 1.412A6.5 6.5 0 0 0 8 1.5m3.568 11.934A7.23 7.23 0 0 0 8 12.5a7.23 7.23 0 0 0-3.568.934A6.47 6.47 0 0 0 8 14.5a6.47 6.47 0 0 0 3.568-1.066",
                clipRule: "evenodd"
            })
        ]
    });
}
const UserCircleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUserCircleIcon
    });
});
UserCircleIcon.displayName = 'UserCircleIcon';

function SvgUserGroupFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M2.25 3.75a2.75 2.75 0 1 1 5.5 0 2.75 2.75 0 0 1-5.5 0M9.502 14H.75a.75.75 0 0 1-.75-.75V11a.75.75 0 0 1 .164-.469C1.298 9.115 3.077 8 5.125 8c1.76 0 3.32.822 4.443 1.952A5.55 5.55 0 0 1 11.75 9.5c1.642 0 3.094.745 4.041 1.73a.75.75 0 0 1 .209.52v1.5a.75.75 0 0 1-.75.75zM11.75 3.5a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5"
        })
    });
}
const UserGroupFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUserGroupFillIcon
    });
});
UserGroupFillIcon.displayName = 'UserGroupFillIcon';

function SvgUserGroupIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M2.25 3.75a2.75 2.75 0 1 1 5.5 0 2.75 2.75 0 0 1-5.5 0M5 2.5A1.25 1.25 0 1 0 5 5a1.25 1.25 0 0 0 0-2.5M9.502 14H.75a.75.75 0 0 1-.75-.75V11a.75.75 0 0 1 .164-.469C1.298 9.115 3.077 8 5.125 8c1.76 0 3.32.822 4.443 1.952A5.55 5.55 0 0 1 11.75 9.5c1.642 0 3.094.745 4.041 1.73a.75.75 0 0 1 .209.52v1.5a.75.75 0 0 1-.75.75zM1.5 12.5v-1.228C2.414 10.228 3.72 9.5 5.125 9.5c1.406 0 2.71.728 3.625 1.772V12.5zm8.75 0h4.25v-.432A4.17 4.17 0 0 0 11.75 11c-.53 0-1.037.108-1.5.293zM11.75 3.5a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5M11 5.75a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0",
            clipRule: "evenodd"
        })
    });
}
const UserGroupIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUserGroupIcon
    });
});
UserGroupIcon.displayName = 'UserGroupIcon';

function SvgUserIcon$1(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8 1a3.25 3.25 0 1 0 0 6.5A3.25 3.25 0 0 0 8 1M6.25 4.25a1.75 1.75 0 1 1 3.5 0 1.75 1.75 0 0 1-3.5 0M8 9a8.74 8.74 0 0 0-6.836 3.287.75.75 0 0 0-.164.469v1.494c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75v-1.494a.75.75 0 0 0-.164-.469A8.74 8.74 0 0 0 8 9m-5.5 4.5v-.474A7.23 7.23 0 0 1 8 10.5c2.2 0 4.17.978 5.5 2.526v.474z",
            clipRule: "evenodd"
        })
    });
}
const UserIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUserIcon$1
    });
});
UserIcon.displayName = 'UserIcon';

function SvgUserKeyIconIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M4.75 4.25a3.25 3.25 0 1 1 6.5 0 3.25 3.25 0 0 1-6.5 0M8 2.5A1.75 1.75 0 1 0 8 6a1.75 1.75 0 0 0 0-3.5M12.75 12.372a2.251 2.251 0 1 0-1.5 0v2.878c0 .414.336.75.75.75h2v-2.75h-1.25zM12 9.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M1.75 15h8v-1.5H2.5v-.474a7.23 7.23 0 0 1 5.759-2.521 3.8 3.8 0 0 1 .2-1.493 8.735 8.735 0 0 0-7.295 3.275.75.75 0 0 0-.164.469v1.494c0 .414.336.75.75.75"
            })
        ]
    });
}
const UserKeyIconIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUserKeyIconIcon
    });
});
UserKeyIconIcon.displayName = 'UserKeyIconIcon';

function SvgUserSparkleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8 1c.664 0 1.282.2 1.797.542l-.014.072-.062.357-.357.062c-.402.07-.765.245-1.06.493a1.75 1.75 0 1 0 0 3.447c.295.25.658.424 1.06.494l.357.062.062.357.014.072A3.25 3.25 0 1 1 8 1"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M9.59 4.983A.75.75 0 0 1 9.62 3.51l.877-.152a.75.75 0 0 0 .61-.61l.153-.878a.75.75 0 0 1 1.478 0l.152.877a.75.75 0 0 0 .61.61l.878.153a.75.75 0 0 1 0 1.478l-.877.152a.75.75 0 0 0-.61.61l-.153.878a.75.75 0 0 1-1.478 0l-.152-.877a.75.75 0 0 0-.61-.61l-.878-.153z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1.164 12.287A8.74 8.74 0 0 1 8 9a8.74 8.74 0 0 1 6.836 3.287.75.75 0 0 1 .164.469v1.494a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75v-1.494a.75.75 0 0 1 .164-.469m1.336.74v.473h11v-.474A7.23 7.23 0 0 0 8 10.5c-2.2 0-4.17.978-5.5 2.526",
                clipRule: "evenodd"
            })
        ]
    });
}
const UserSparkleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUserSparkleIcon
    });
});
UserSparkleIcon.displayName = 'UserSparkleIcon';

function SvgUserTeamIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "M4.283 10.062c2.38-2.098 4.797-2.1 7.615-.005.245.181.382.469.382.77v2.519l-.004.08a.81.81 0 0 1-.725.716l-.083.004H4.782l-.084-.004a.81.81 0 0 1-.724-.715l-.004-.081v-2.573c0-.27.11-.532.313-.711m-.986.527c-.281 0-.446.046-.588.115-.167.081-.342.212-.64.461a.25.25 0 0 0-.089.192v1.204c0 .027.023.05.051.05h1.266v1.5H2.03c-.869 0-1.574-.695-1.574-1.55v-1.204c0-.511.227-1 .626-1.334.291-.244.595-.49.95-.664.38-.185.783-.27 1.264-.27zm9.656-1.5c.481 0 .883.085 1.264.27.355.173.66.42.95.664.399.333.626.823.626 1.334v1.204c0 .855-.705 1.55-1.574 1.55h-1.266v-1.5h1.266a.05.05 0 0 0 .05-.05v-1.204a.25.25 0 0 0-.088-.192c-.297-.249-.472-.38-.639-.46-.142-.07-.307-.116-.589-.116zm-5.026.898c-.728.001-1.52.275-2.434 1.027v1.632h5.264V11.09c-1.15-.81-2.073-1.104-2.83-1.104m-4.758-6c1.262 0 2.285 1.008 2.285 2.25 0 1.243-1.023 2.25-2.285 2.25S.884 7.48.884 6.237c0-1.242 1.023-2.25 2.285-2.25m9.966 0c1.262 0 2.285 1.008 2.285 2.25 0 1.243-1.023 2.25-2.285 2.25S10.85 7.48 10.85 6.237c0-1.242 1.023-2.25 2.285-2.25M8 1.943a2.684 2.684 0 0 1 2.684 2.683l-.015.274A2.684 2.684 0 0 1 8 7.31l-.274-.015A2.683 2.683 0 0 1 8 1.943M3.169 5.487a.756.756 0 0 0-.762.75c0 .415.341.75.762.75.42 0 .762-.336.762-.75a.756.756 0 0 0-.762-.75m9.966 0a.756.756 0 0 0-.762.75c0 .414.341.75.762.75.42 0 .761-.335.761-.75a.756.756 0 0 0-.761-.75M8 3.443a1.183 1.183 0 1 0 .001 2.367A1.183 1.183 0 0 0 8 3.443"
        })
    });
}
const UserTeamIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgUserTeamIcon
    });
});
UserTeamIcon.displayName = 'UserTeamIcon';

function SvgVisibleFillIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("g", {
                clipPath: "url(#VisibleFillIcon_svg__a)",
                children: /*#__PURE__*/ jsx("path", {
                    fill: "currentColor",
                    fillRule: "evenodd",
                    d: "M8 2A8.39 8.39 0 0 0 .028 7.777a.75.75 0 0 0 0 .466 8.389 8.389 0 0 0 15.944 0 .75.75 0 0 0 0-.466A8.39 8.39 0 0 0 8 2M6.5 8a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0M8 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6",
                    clipRule: "evenodd"
                })
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const VisibleFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgVisibleFillIcon
    });
});
VisibleFillIcon.displayName = 'VisibleFillIcon';

function SvgVisibleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                fillRule: "evenodd",
                clipPath: "url(#VisibleIcon_svg__a)",
                clipRule: "evenodd",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        d: "M8 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6M6.5 8a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M8 2A8.39 8.39 0 0 0 .028 7.777a.75.75 0 0 0 0 .466 8.389 8.389 0 0 0 15.944 0 .75.75 0 0 0 0-.466A8.39 8.39 0 0 0 8 2m0 10.52a6.89 6.89 0 0 1-6.465-4.51 6.888 6.888 0 0 1 12.93 0A6.89 6.89 0 0 1 8 12.52"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const VisibleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgVisibleIcon
    });
});
VisibleIcon.displayName = 'VisibleIcon';

function SvgVisibleOffIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsxs("g", {
                fill: "currentColor",
                clipPath: "url(#VisibleOffIcon_svg__a)",
                children: [
                    /*#__PURE__*/ jsx("path", {
                        fillRule: "evenodd",
                        d: "m11.634 13.195 1.335 1.335 1.061-1.06-11.5-11.5-1.06 1.06 1.027 1.028a8.4 8.4 0 0 0-2.469 3.72.75.75 0 0 0 0 .465 8.39 8.39 0 0 0 11.606 4.951m-1.14-1.14-1.301-1.301a3 3 0 0 1-3.946-3.946L3.56 5.121A6.9 6.9 0 0 0 1.535 8.01a6.89 6.89 0 0 0 8.96 4.045",
                        clipRule: "evenodd"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M15.972 8.243a8.4 8.4 0 0 1-1.946 3.223l-1.06-1.06a6.9 6.9 0 0 0 1.499-2.396 6.89 6.89 0 0 0-8.187-4.293L5.082 2.522a8.389 8.389 0 0 1 10.89 5.256.75.75 0 0 1 0 .465"
                    }),
                    /*#__PURE__*/ jsx("path", {
                        d: "M11 8q0 .21-.028.411L7.589 5.028q.201-.027.41-.028a3 3 0 0 1 3 3"
                    })
                ]
            }),
            /*#__PURE__*/ jsx("defs", {
                children: /*#__PURE__*/ jsx("clipPath", {
                    children: /*#__PURE__*/ jsx("path", {
                        fill: "#fff",
                        d: "M0 0h16v16H0z"
                    })
                })
            })
        ]
    });
}
const VisibleOffIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgVisibleOffIcon
    });
});
VisibleOffIcon.displayName = 'VisibleOffIcon';

function SvgWarningFillIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M8.649 1.374a.75.75 0 0 0-1.298 0l-7.25 12.5A.75.75 0 0 0 .75 15h14.5a.75.75 0 0 0 .649-1.126zM7.25 10V6.5h1.5V10zm1.5 1.75a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0",
            clipRule: "evenodd"
        })
    });
}
const WarningFillIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgWarningFillIcon
    });
});
WarningFillIcon.displayName = 'WarningFillIcon';

function SvgWorkflowCodeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 17 17",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m10.53 11.06-1.97 1.97L10.53 15l-1.06 1.06-3.03-3.03L9.47 10zM16.06 13.03l-3.03 3.03L11.97 15l1.97-1.97-1.97-1.97L13.03 10z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M2.75 0c1.258 0 2.317.846 2.644 2h5.231a3.375 3.375 0 1 1 0 6.75h-.365L7 6.42 4.79 8l2.14 1.528-1.106 1.053L3.27 8.755a1.873 1.873 0 0 0 .106 3.745H5V14H3.375a3.375 3.375 0 0 1-.118-6.748L6.564 4.89l.102-.062a.75.75 0 0 1 .77.062l3.295 2.354a1.873 1.873 0 0 0-.106-3.744H5.394A2.749 2.749 0 0 1 0 2.75 2.75 2.75 0 0 1 2.75 0m0 1.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5"
            })
        ]
    });
}
const WorkflowCodeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgWorkflowCodeIcon
    });
});
WorkflowCodeIcon.displayName = 'WorkflowCodeIcon';

function SvgWorkflowCubeIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M11.686 8.07a.75.75 0 0 1 .629 0l3.25 1.5q.045.02.09.048a.8.8 0 0 1 .153.132l.029.036.033.044a.7.7 0 0 1 .116.286l.006.03a1 1 0 0 1 .008.104v3.5a.75.75 0 0 1-.435.68l-3.242 1.496-.006.003-.002.002-.013.005a.8.8 0 0 1-.253.062h-.099a.8.8 0 0 1-.252-.062l-.013-.005-.01-.005-3.24-1.495A.75.75 0 0 1 8 13.75v-3.5q0-.053.007-.104l.006-.03a.8.8 0 0 1 .047-.159l.007-.019.025-.048.015-.027.022-.033q.016-.023.033-.044l.03-.036a.8.8 0 0 1 .242-.18h.002zM9.5 13.27l1.75.807V12.23l-1.75-.807zm3.25-1.04v1.847l1.75-.807v-1.848zm-2.21-1.98 1.46.674 1.46-.674L12 9.575z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M6.564 4.89a.75.75 0 0 1 .873 0l2.933 2.094-1.57.724L7 6.422 4.79 8l1.94 1.385c-.149.332-.23.698-.23 1.076v.602L3.268 8.756a1.873 1.873 0 0 0 .107 3.745H6.5v1.04q0 .235.042.46H3.375a3.375 3.375 0 0 1-.12-6.748z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M2.75 0c1.258 0 2.317.846 2.644 2h5.231a3.375 3.375 0 0 1 2.974 4.97l-.493-.227a2.6 2.6 0 0 0-.986-.24A1.875 1.875 0 0 0 10.625 3.5H5.394A2.749 2.749 0 0 1 0 2.75 2.75 2.75 0 0 1 2.75 0m0 1.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5"
            })
        ]
    });
}
const WorkflowCubeIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgWorkflowCubeIcon
    });
});
WorkflowCubeIcon.displayName = 'WorkflowCubeIcon';

function SvgWorkflowsIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M3.75 4a1.25 1.25 0 1 0 0-2.5 1.25 1.25 0 0 0 0 2.5m2.646-.5a2.751 2.751 0 1 1 0-1.5h5.229a3.375 3.375 0 0 1 .118 6.748L8.436 11.11a.75.75 0 0 1-.872 0l-3.3-2.357a1.875 1.875 0 0 0 .111 3.747h5.229a2.751 2.751 0 1 1 0 1.5H4.375a3.375 3.375 0 0 1-.118-6.748L7.564 4.89a.75.75 0 0 1 .872 0l3.3 2.357a1.875 1.875 0 0 0-.111-3.747zm7.104 9.75a1.25 1.25 0 1 1-2.5 0 1.25 1.25 0 0 1 2.5 0M8 6.422 5.79 8 8 9.578 10.21 8z",
            clipRule: "evenodd"
        })
    });
}
const WorkflowsIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgWorkflowsIcon
    });
});
WorkflowsIcon.displayName = 'WorkflowsIcon';

function SvgWorkspacesIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M2.5 1a.75.75 0 0 0-.75.75v3c0 .414.336.75.75.75H6V4H3.25V2.5h9.5V4H10v1.5h3.5a.75.75 0 0 0 .75-.75v-3A.75.75 0 0 0 13.5 1z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M0 12.25c0-1.26.848-2.322 2.004-2.648A2.75 2.75 0 0 1 4.75 7h2.5V4h1.5v3h2.5a2.75 2.75 0 0 1 2.746 2.602 2.751 2.751 0 1 1-3.371 3.47 2.751 2.751 0 0 1-5.25 0A2.751 2.751 0 0 1 0 12.25M2.75 11a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m2.625.428a2.76 2.76 0 0 0-1.867-1.822A1.25 1.25 0 0 1 4.75 8.5h2.5v1.104c-.892.252-1.6.942-1.875 1.824M8.75 9.604V8.5h2.5c.642 0 1.17.483 1.242 1.106a2.76 2.76 0 0 0-1.867 1.822A2.76 2.76 0 0 0 8.75 9.604M12 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0m-5.25 0a1.25 1.25 0 1 0 2.5 0 1.25 1.25 0 0 0-2.5 0",
                clipRule: "evenodd"
            })
        ]
    });
}
const WorkspacesIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgWorkspacesIcon
    });
});
WorkspacesIcon.displayName = 'WorkspacesIcon';

function SvgWrenchIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M14.367 3.29a.75.75 0 0 1 .547.443 5.001 5.001 0 0 1-6.072 6.736l-3.187 3.186a2.341 2.341 0 0 1-3.31-3.31L5.53 7.158a5.001 5.001 0 0 1 6.736-6.072.75.75 0 0 1 .237 1.22L10.5 4.312V5.5h1.19l2.003-2.004a.75.75 0 0 1 .674-.206m-.56 2.214L12.53 6.78A.75.75 0 0 1 12 7H9.75A.75.75 0 0 1 9 6.25V4a.75.75 0 0 1 .22-.53l1.275-1.276a3.501 3.501 0 0 0-3.407 4.865.75.75 0 0 1-.16.823l-3.523 3.523a.84.84 0 1 0 1.19 1.19L8.118 9.07a.75.75 0 0 1 .823-.16 3.5 3.5 0 0 0 4.865-3.407",
            clipRule: "evenodd"
        })
    });
}
const WrenchIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgWrenchIcon
    });
});
WrenchIcon.displayName = 'WrenchIcon';

function SvgWrenchSparkleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M12.31 11.09 12.5 10l.19 1.09a1.5 1.5 0 0 0 1.22 1.22l1.09.19-1.09.19a1.5 1.5 0 0 0-1.22 1.22L12.5 15l-.19-1.09a1.5 1.5 0 0 0-1.22-1.22L10 12.5l1.09-.19a1.5 1.5 0 0 0 1.22-1.22"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M12.5 9.25a.75.75 0 0 1 .739.621l.19 1.09a.75.75 0 0 0 .61.61l1.09.19a.75.75 0 0 1 0 1.478l-1.09.19a.75.75 0 0 0-.61.61l-.19 1.09a.75.75 0 0 1-1.478 0l-.19-1.09a.75.75 0 0 0-.61-.61l-1.09-.19a.75.75 0 0 1 0-1.478l1.09-.19a.75.75 0 0 0 .61-.61l.345.06-.344-.06.19-1.09a.75.75 0 0 1 .738-.621m0 3.094q-.075.081-.156.156.081.075.156.156.075-.081.156-.156a2 2 0 0 1-.156-.156",
                clipRule: "evenodd"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "m3.125 13.604 3.98-3.979q.416.167.905.26.49.094.99.094 2.063 0 3.531-1.458Q14.001 7.063 14 5a5.6 5.6 0 0 0-.198-1.498 4.7 4.7 0 0 0-.594-1.314l-3 3L8.75 3.729l3-3a5.3 5.3 0 0 0-1.302-.541A5.6 5.6 0 0 0 9 0Q6.916 0 5.458 1.48 4 2.957 4 5.024q0 .475.094.881t.26.823L.292 10.771a1 1 0 0 0-.292.722q0 .423.292.715l1.416 1.396a.97.97 0 0 0 .71.292.96.96 0 0 0 .707-.292m-.708-1.416-.73-.709 4.48-4.437a2.5 2.5 0 0 1-.542-1.094Q5.5 5.354 5.5 5.02q0-1.393.99-2.426.99-1.032 2.385-1.095L7.167 3.188a.75.75 0 0 0 .006 1.087L9.68 6.766a.74.74 0 0 0 .531.234q.296 0 .517-.23L12.5 5q0 1.438-1.031 2.458T9 8.48q-.354 0-.98-.146a2.9 2.9 0 0 1-1.166-.583z"
            })
        ]
    });
}
const WrenchSparkleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgWrenchSparkleIcon
    });
});
WrenchSparkleIcon.displayName = 'WrenchSparkleIcon';

function SvgXCircleIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M6.94 8 4.97 6.03l1.06-1.06L8 6.94l1.97-1.97 1.06 1.06L9.06 8l1.97 1.97-1.06 1.06L8 9.06l-1.97 1.97-1.06-1.06z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13",
                clipRule: "evenodd"
            })
        ]
    });
}
const XCircleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgXCircleIcon
    });
});
XCircleIcon.displayName = 'XCircleIcon';

function SvgZaHorizontalIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M11.654 4.5a.75.75 0 0 1 .695.468L15 11.5h-1.619l-.406-1h-2.643l-.406 1H8.307l2.652-6.532a.75.75 0 0 1 .695-.468M10.94 9h1.425l-.712-1.756zM4.667 6H1V4.5h5.25a.75.75 0 0 1 .58 1.225L3.333 10H7v1.5H1.75a.75.75 0 0 1-.58-1.225z",
            clipRule: "evenodd"
        })
    });
}
const ZaHorizontalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgZaHorizontalIcon
    });
});
ZaHorizontalIcon.displayName = 'ZaHorizontalIcon';

function SvgZaVerticalIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            fillRule: "evenodd",
            d: "M7.996 8a.75.75 0 0 0-.695.468L4.65 15h1.619l.406-1h2.643l.406 1h1.619L8.69 8.468A.75.75 0 0 0 7.996 8m.713 4.5H7.284l.712-1.756zM8.664 1.5H4.996V0h5.25a.75.75 0 0 1 .58 1.225L7.33 5.5h3.667V7h-5.25a.75.75 0 0 1-.58-1.225z",
            clipRule: "evenodd"
        })
    });
}
const ZaVerticalIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgZaVerticalIcon
    });
});
ZaVerticalIcon.displayName = 'ZaVerticalIcon';

function SvgZoomInIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 17",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M8.75 7.25H11v1.5H8.75V11h-1.5V8.75H5v-1.5h2.25V5h1.5z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M8 1a7 7 0 1 0 4.39 12.453l2.55 2.55 1.06-1.06-2.55-2.55A7 7 0 0 0 8 1M2.5 8a5.5 5.5 0 1 1 11 0 5.5 5.5 0 0 1-11 0",
                clipRule: "evenodd"
            })
        ]
    });
}
const ZoomInIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgZoomInIcon
    });
});
ZoomInIcon.displayName = 'ZoomInIcon';

function SvgZoomMarqueeSelection(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M1 1.75V4h1.5V2.5H4V1H1.75a.75.75 0 0 0-.75.75M14.25 1H12v1.5h1.5V4H15V1.75a.75.75 0 0 0-.75-.75M4 15H1.75a.75.75 0 0 1-.75-.75V12h1.5v1.5H4zM6 2.5h4V1H6zM1 10V6h1.5v4z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M4.053 9.27a5.217 5.217 0 1 1 9.397 3.122l1.69 1.69-1.062 1.06-1.688-1.69A5.217 5.217 0 0 1 4.053 9.27M9.27 5.555a3.717 3.717 0 1 0 0 7.434 3.717 3.717 0 0 0 0-7.434",
                clipRule: "evenodd"
            })
        ]
    });
}
const ZoomMarqueeSelection = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgZoomMarqueeSelection
    });
});
ZoomMarqueeSelection.displayName = 'ZoomMarqueeSelection';

function SvgZoomOutIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 17",
        ...props,
        children: [
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                d: "M11 7.25H5v1.5h6z"
            }),
            /*#__PURE__*/ jsx("path", {
                fill: "currentColor",
                fillRule: "evenodd",
                d: "M1 8a7 7 0 1 1 12.45 4.392l2.55 2.55-1.06 1.061-2.55-2.55A7 7 0 0 1 1 8m7-5.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11",
                clipRule: "evenodd"
            })
        ]
    });
}
const ZoomOutIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgZoomOutIcon
    });
});
ZoomOutIcon.displayName = 'ZoomOutIcon';

function SvgZoomToFitIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "1em",
        height: "1em",
        fill: "none",
        viewBox: "0 0 16 16",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            fill: "currentColor",
            d: "m2.5 3.56 2.97 2.97 1.06-1.06L3.56 2.5H6V1H1v5h1.5zM10.53 6.53l2.97-2.97V6H15V1h-5v1.5h2.44L9.47 5.47zM9.47 10.53l2.97 2.97H10V15h5v-5h-1.5v2.44l-2.97-2.97zM5.47 9.47 2.5 12.44V10H1v5h5v-1.5H3.56l2.97-2.97z"
        })
    });
}
const ZoomToFitIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: SvgZoomToFitIcon
    });
});
ZoomToFitIcon.displayName = 'ZoomToFitIcon';

const getButtonStyles = (theme)=>{
    return /*#__PURE__*/ css({
        color: theme.colors.textPlaceholder,
        fontSize: theme.typography.fontSizeSm,
        marginLeft: theme.spacing.xs,
        ':hover': {
            color: theme.colors.actionTertiaryTextHover
        }
    });
};
const ClearSelectionButton = ({ ...restProps })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(XCircleFillIcon, {
        "aria-hidden": "false",
        css: getButtonStyles(theme),
        role: "button",
        "aria-label": "Clear selection",
        ...restProps
    });
};

const dialogComboboxContextDefaults = {
    componentId: 'codegen_design-system_src_design-system_dialogcombobox_providers_dialogcomboboxcontext.tsx_27',
    id: '',
    label: '',
    value: [],
    isInsideDialogCombobox: false,
    multiSelect: false,
    setValue: ()=>{},
    setIsControlled: ()=>{},
    stayOpenOnSelection: false,
    isOpen: false,
    setIsOpen: ()=>{},
    contentWidth: undefined,
    setContentWidth: ()=>{},
    textOverflowMode: 'multiline',
    setTextOverflowMode: ()=>{},
    scrollToSelectedElement: true,
    rememberLastScrollPosition: false,
    disableMouseOver: false,
    setDisableMouseOver: ()=>{},
    onView: ()=>{}
};
const DialogComboboxContext = /*#__PURE__*/ createContext(dialogComboboxContextDefaults);
const DialogComboboxContextProvider = ({ children, value })=>{
    return /*#__PURE__*/ jsx(DialogComboboxContext.Provider, {
        value: value,
        children: children
    });
};

const useDialogComboboxContext = ()=>{
    return useContext(DialogComboboxContext);
};

const EmptyResults = ({ emptyText, id })=>{
    const { theme } = useDesignSystemTheme();
    const { emptyText: emptyTextFromContext } = useDialogComboboxContext();
    return /*#__PURE__*/ jsx("div", {
        "aria-live": "assertive",
        id: id,
        css: {
            color: theme.colors.textSecondary,
            textAlign: 'center',
            padding: '6px 12px',
            width: '100%',
            boxSizing: 'border-box'
        },
        children: emptyTextFromContext ?? emptyText ?? 'No results found'
    });
};

const HintRow$1 = ({ disabled, children })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        css: {
            color: theme.colors.textSecondary,
            fontSize: theme.typography.fontSizeSm,
            ...disabled && {
                color: theme.colors.actionDisabledText
            }
        },
        children: children
    });
};

const LoadingSpinner = (props)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(Spinner, {
        css: {
            display: 'flex',
            alignSelf: 'center',
            justifyContent: 'center',
            alignItems: 'center',
            height: theme.general.heightSm,
            width: theme.general.heightSm,
            '> span': {
                fontSize: 20
            }
        },
        ...props
    });
};

const SectionHeader = ({ children, ...props })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        ...props,
        css: {
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'flex-start',
            padding: `${theme.spacing.xs}px ${theme.spacing.lg / 2}px`,
            alignSelf: 'stretch',
            fontWeight: 400,
            color: theme.colors.textSecondary
        },
        children: children
    });
};

const Separator$2 = (props)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        ...props,
        css: {
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            margin: `${theme.spacing.xs}px ${theme.spacing.lg / 2}px`,
            border: `1px solid ${theme.colors.borderDecorative}`,
            borderBottom: 0,
            alignSelf: 'stretch'
        }
    });
};

const getCommonTabsListStyles = (theme)=>{
    return {
        display: 'flex',
        borderBottom: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
        marginBottom: theme.spacing.md,
        height: theme.general.heightSm,
        boxSizing: 'border-box'
    };
};
const getCommonTabsTriggerStyles = (theme)=>{
    return {
        display: 'flex',
        fontWeight: theme.typography.typographyBoldFontWeight,
        fontSize: theme.typography.fontSizeMd,
        backgroundColor: 'transparent',
        marginRight: theme.spacing.md
    };
};

const setupDesignSystemEventProviderForTesting = (eventCallback)=>{
    const eventCallbackWrapper = (arg)=>{
        if (arg?.componentViewId) {
            const { componentViewId, ...rest } = arg; // strip componentViewId out of jest call args
            eventCallback(rest);
            return;
        }
        eventCallback(arg);
    };
    return {
        DesignSystemEventProviderForTest: ({ children })=>/*#__PURE__*/ jsx(DesignSystemEventProvider, {
                callback: eventCallbackWrapper,
                children: children
            })
    };
};

function getAccordionEmotionStyles({ clsPrefix, theme, alignContentToEdge, isLeftAligned }) {
    const classItem = `.${clsPrefix}-item`;
    const classItemActive = `${classItem}-active`;
    const classHeader = `.${clsPrefix}-header`;
    const classContent = `.${clsPrefix}-content`;
    const classContentBox = `.${clsPrefix}-content-box`;
    const classArrow = `.${clsPrefix}-arrow`;
    const styles = {
        border: '0 none',
        background: 'none',
        [classItem]: {
            border: '0 none',
            [`&:hover`]: {
                [classHeader]: {
                    color: theme.colors.actionPrimaryBackgroundHover
                },
                [classArrow]: {
                    color: theme.colors.actionPrimaryBackgroundHover
                }
            },
            [`&:active`]: {
                [classHeader]: {
                    color: theme.colors.actionPrimaryBackgroundPress
                },
                [classArrow]: {
                    color: theme.colors.actionPrimaryBackgroundPress
                }
            }
        },
        [classHeader]: {
            color: theme.colors.textPrimary,
            fontWeight: 600,
            '&:focus-visible': {
                outlineColor: `${theme.colors.actionDefaultBorderFocus} !important`,
                outlineStyle: 'auto !important'
            }
        },
        [`& > ${classItem} > ${classHeader} > ${classArrow}`]: {
            fontSize: theme.general.iconFontSize,
            right: alignContentToEdge || isLeftAligned ? 0 : 12,
            ...isLeftAligned && {
                verticalAlign: 'middle',
                marginTop: -2
            }
        },
        [classArrow]: {
            color: theme.colors.textSecondary
        },
        [`& > ${classItemActive} > ${classHeader} > ${classArrow}`]: {
            transform: isLeftAligned ? 'rotate(180deg)' : 'translateY(-50%) rotate(180deg)'
        },
        [classContent]: {
            border: '0 none',
            backgroundColor: theme.colors.backgroundPrimary
        },
        [classContentBox]: {
            padding: alignContentToEdge ? '8px 0px 16px' : '8px 16px 16px'
        },
        [`& > ${classItem} > ${classHeader}`]: {
            padding: '6px 44px 6px 0',
            lineHeight: theme.typography.lineHeightBase
        },
        ...getAnimationCss(theme.options.enableAnimation)
    };
    return /*#__PURE__*/ css(styles);
}
const AccordionPanel = ({ dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, children, ...props })=>{
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Collapse.Panel, {
            ...props,
            ...dangerouslySetAntdProps,
            css: dangerouslyAppendEmotionCSS,
            children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                children: children
            })
        })
    });
};
const Accordion = /* #__PURE__ */ (()=>{
    const Accordion = ({ dangerouslySetAntdProps, dangerouslyAppendEmotionCSS, displayMode = 'multiple', analyticsEvents, componentId, valueHasNoPii, onChange, alignContentToEdge = false, chevronAlignment = 'right', ...props })=>{
        const emitOnView = safex('databricks.fe.observability.defaultComponentView.accordion', false);
        const { theme, getPrefixedClassName } = useDesignSystemTheme();
        // While this component is called `Accordion` for correctness, in AntD it is called `Collapse`.
        const clsPrefix = getPrefixedClassName('collapse');
        const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView
            ] : [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
            ]), [
            analyticsEvents,
            emitOnView
        ]);
        const eventContext = useDesignSystemEventComponentCallbacks({
            componentType: DesignSystemEventProviderComponentTypes.Accordion,
            componentId,
            analyticsEvents: memoizedAnalyticsEvents,
            valueHasNoPii
        });
        const { elementRef: accordionRef } = useNotifyOnFirstView({
            onView: eventContext.onView
        });
        const onChangeWrapper = useCallback((newValue)=>{
            if (Array.isArray(newValue)) {
                eventContext.onValueChange(JSON.stringify(newValue));
            } else {
                eventContext.onValueChange(newValue);
            }
            onChange?.(newValue);
        }, [
            eventContext,
            onChange
        ]);
        return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsx(Collapse, {
                // eslint-disable-next-line @databricks/no-unstable-nested-components -- go/no-nested-components
                expandIcon: ()=>/*#__PURE__*/ jsx(ChevronDownIcon, {
                        ...eventContext.dataComponentProps,
                        ref: accordionRef
                    }),
                expandIconPosition: chevronAlignment,
                accordion: displayMode === 'single',
                ...props,
                ...dangerouslySetAntdProps,
                css: [
                    getAccordionEmotionStyles({
                        clsPrefix,
                        theme,
                        alignContentToEdge,
                        isLeftAligned: chevronAlignment === 'left'
                    }),
                    dangerouslyAppendEmotionCSS,
                    addDebugOutlineStylesIfEnabled(theme)
                ],
                onChange: onChangeWrapper
            })
        });
    };
    Accordion.Panel = AccordionPanel;
    return Accordion;
})();

// TODO: Replace with custom icons
// TODO: Reuse in Alert
const filledIconsMap = {
    error: DangerFillIcon,
    warning: WarningFillIcon,
    success: CheckCircleFillIcon,
    info: InfoFillIcon
};
const SeverityIcon = /*#__PURE__*/ forwardRef(function(props, ref) {
    const FilledIcon = filledIconsMap[props.severity];
    return /*#__PURE__*/ jsx(FilledIcon, {
        ref: ref,
        ...props
    });
});

const Alert = ({ componentId, analyticsEvents = [
    DesignSystemEventProviderAnalyticsEventTypes.OnView
], dangerouslySetAntdProps, closable = true, closeIconLabel = 'Close alert', onClose, actions, showMoreContent, showMoreText = 'Show details', showMoreModalTitle = 'Details', forceVerticalActionsPlacement = false, size = 'large', ...props })=>{
    const { theme, getPrefixedClassName } = useDesignSystemTheme();
    const useNewBorderRadii = safex('databricks.fe.designsystem.useNewBorderRadii', false);
    const useNewLargeAlertSizing = safex('databricks.fe.designsystem.useNewLargeAlertSizing', true);
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents, [
        analyticsEvents
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Alert,
        componentId,
        componentSubType: DesignSystemEventProviderComponentSubTypeMap[props.type],
        analyticsEvents: memoizedAnalyticsEvents
    });
    const closeButtonEventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentId: componentId ? `${componentId}.close` : 'codegen_design_system_src_design_system_alert_alert.tsx_50',
        analyticsEvents: [
            DesignSystemEventProviderAnalyticsEventTypes.OnClick
        ]
    });
    const { elementRef } = useNotifyOnFirstView({
        onView: eventContext.onView
    });
    const clsPrefix = getPrefixedClassName('alert');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const mergedProps = {
        ...props,
        type: props.type || 'error',
        showIcon: true,
        closable
    };
    const closeIconRef = useRef(null);
    useEffect(()=>{
        if (closeIconRef.current) {
            closeIconRef.current.removeAttribute('aria-label');
            closeIconRef.current.closest('button')?.setAttribute('aria-label', closeIconLabel);
        }
    }, [
        mergedProps.closable,
        closeIconLabel,
        closeIconRef
    ]);
    const onCloseWrapper = (e)=>{
        closeButtonEventContext.onClick(e);
        onClose?.(e);
    };
    const memoizedActions = useMemo(()=>{
        if (!actions?.length) return null;
        return actions.map((action, index)=>/*#__PURE__*/ jsx(Button, {
                size: "small",
                ...action
            }, index));
    }, [
        actions
    ]);
    // Determine action placement based on number of actions
    const actionsPlacement = actions?.length === 1 && !forceVerticalActionsPlacement ? 'horizontal' : 'vertical';
    const description = /*#__PURE__*/ jsx("div", {
        css: {
            ...useNewLargeAlertSizing ? {
                display: 'flex',
                flexDirection: 'row',
                gap: theme.spacing.sm,
                justifyContent: 'space-between'
            } : {}
        },
        children: /*#__PURE__*/ jsxs("div", {
            css: {
                ...actionsPlacement === 'horizontal' && {
                    display: 'flex',
                    flexDirection: 'row',
                    justifyContent: 'space-between',
                    gap: theme.spacing.sm,
                    alignItems: 'flex-start'
                }
            },
            children: [
                /*#__PURE__*/ jsxs("div", {
                    css: {
                        overflowWrap: 'anywhere',
                        wordBreak: 'break-word'
                    },
                    children: [
                        props.description,
                        ' ',
                        showMoreContent && /*#__PURE__*/ jsx(Typography.Link, {
                            href: "#",
                            componentId: componentId ? `${componentId}.show_more` : 'alert.show_more',
                            onClick: (e)=>{
                                e.preventDefault();
                                setIsModalOpen(true);
                            },
                            children: showMoreText
                        })
                    ]
                }),
                !useNewLargeAlertSizing && memoizedActions && actionsPlacement === 'horizontal' && /*#__PURE__*/ jsx("div", {
                    css: {
                        display: 'flex',
                        gap: theme.spacing.sm,
                        ...useNewLargeAlertSizing ? {
                            alignSelf: 'center'
                        } : {}
                    },
                    children: memoizedActions
                })
            ]
        })
    });
    // Create a separate section for vertical actions if needed
    const verticalActions = memoizedActions && actionsPlacement === 'vertical' && /*#__PURE__*/ jsx("div", {
        css: {
            display: 'flex',
            gap: theme.spacing.sm,
            marginTop: useNewLargeAlertSizing ? theme.spacing.xs : theme.spacing.sm,
            marginBottom: theme.spacing.xs
        },
        children: memoizedActions
    });
    // Create the final description that includes both the description and vertical actions if present
    const finalDescription = /*#__PURE__*/ jsxs(Fragment, {
        children: [
            description,
            verticalActions
        ]
    });
    return /*#__PURE__*/ jsxs(DesignSystemAntDConfigProvider, {
        children: [
            /*#__PURE__*/ jsx(Alert$1, {
                ...addDebugOutlineIfEnabled(),
                ...mergedProps,
                onClose: onCloseWrapper,
                className: classnames(mergedProps.className),
                css: getAlertEmotionStyles(clsPrefix, theme, mergedProps, size, Boolean(actions?.length), forceVerticalActionsPlacement, useNewBorderRadii, useNewLargeAlertSizing, useNewBorderColors),
                icon: /*#__PURE__*/ jsx(SeverityIcon, {
                    severity: mergedProps.type,
                    ref: elementRef
                }),
                // Antd calls this prop `closeText` but we can use it to set any React element to replace the close icon.
                closeText: mergedProps.closable && (useNewLargeAlertSizing ? /*#__PURE__*/ jsx("div", {
                    css: {
                        marginTop: actionsPlacement === 'horizontal' ? theme.spacing.xs : 0
                    },
                    children: /*#__PURE__*/ jsx(CloseSmallIcon, {
                        ref: closeIconRef,
                        "aria-label": closeIconLabel,
                        css: {
                            alignSelf: 'center'
                        }
                    })
                }) : /*#__PURE__*/ jsx(CloseIcon, {
                    ref: closeIconRef,
                    "aria-label": closeIconLabel,
                    css: {
                        fontSize: theme.general.iconSize
                    }
                })),
                // Always set a description for consistent styling (e.g. icon size)
                description: finalDescription,
                action: mergedProps.action ? mergedProps.action : useNewLargeAlertSizing && actionsPlacement === 'horizontal' ? memoizedActions : undefined,
                ...dangerouslySetAntdProps,
                ...eventContext.dataComponentProps
            }),
            showMoreContent && /*#__PURE__*/ jsx(Modal, {
                title: showMoreModalTitle,
                visible: isModalOpen,
                onCancel: ()=>setIsModalOpen(false),
                componentId: componentId ? `${componentId}.show_more_modal` : 'alert.show_more_modal',
                footer: null,
                size: "wide",
                children: showMoreContent
            })
        ]
    });
};
const getAlertEmotionStyles = (clsPrefix, theme, props, size, hasActions, isVertical, useNewBorderRadii, useNewLargeAlertSizing, useNewBorderColors)=>{
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
        ...useNewLargeAlertSizing && {
            padding: `${LARGE_SIZE_PADDING - BORDER_SIZE}px ${LARGE_SIZE_PADDING}px`,
            boxSizing: 'border-box',
            ...isSmall && {
                padding: `${theme.spacing.xs + BORDER_SIZE}px ${SMALL_SIZE_PADDING}px`
            },
            [classAction]: {
                alignSelf: 'center'
            }
        },
        ...useNewBorderRadii && {
            borderRadius: theme.borders.borderRadiusSm
        },
        ...useNewBorderColors && {
            borderColor: theme.colors.border
        },
        [`${classMessage}, &${classWithDescription} ${classMessage}`]: {
            fontSize: theme.typography.fontSizeBase,
            fontWeight: theme.typography.typographyBoldFontWeight,
            lineHeight: theme.typography.lineHeightBase,
            ...useNewLargeAlertSizing && {
                marginBottom: 0
            }
        },
        [`${classDescription}`]: {
            lineHeight: theme.typography.lineHeightBase
        },
        // Icons
        [classCloseButton]: {
            fontSize: ALERT_ICON_FONT_SIZE,
            marginRight: 12
        },
        [classCloseIcon]: {
            '&:focus-visible': {
                outlineStyle: 'auto',
                outlineColor: theme.colors.actionDefaultBorderFocus
            }
        },
        [`${classCloseIcon}, ${classCloseButton}`]: {
            lineHeight: theme.typography.lineHeightBase,
            height: ALERT_ICON_HEIGHT,
            width: ALERT_ICON_HEIGHT,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
        },
        [classWithIcon]: {
            fontSize: ALERT_ICON_FONT_SIZE,
            marginTop: 2,
            ...useNewLargeAlertSizing && {
                alignSelf: 'flex-start',
                marginTop: 0,
                display: 'inline-flex',
                height: theme.typography.lineHeightBase,
                alignItems: 'center'
            }
        },
        [`${classCloseIcon}, ${classCloseButton}, ${classCloseText} > span`]: {
            lineHeight: theme.typography.lineHeightBase,
            height: ALERT_ICON_HEIGHT,
            width: ALERT_ICON_HEIGHT,
            fontSize: ALERT_ICON_FONT_SIZE,
            marginTop: 2,
            ...useNewLargeAlertSizing && props.description && hasActions && !isVertical && {
                marginTop: 0
            },
            '& > span, & > span > span': {
                lineHeight: theme.typography.lineHeightBase,
                display: 'inline-flex',
                alignItems: 'center'
            }
        },
        // No description
        ...!props.description && {
            display: 'flex',
            alignItems: 'center',
            [classWithIcon]: {
                fontSize: ALERT_ICON_FONT_SIZE,
                marginTop: 0,
                ...useNewLargeAlertSizing && {
                    display: 'inline-flex',
                    height: theme.typography.lineHeightBase,
                    alignItems: 'center',
                    ...isVertical && {
                        alignSelf: 'flex-start'
                    }
                }
            },
            [classMessage]: {
                margin: 0
            },
            [classDescription]: {
                display: 'none'
            },
            [classCloseIcon]: {
                alignSelf: 'baseline'
            }
        },
        // No description with icons
        ...!props.description && hasActions && {
            ...!isVertical && {
                [classContent]: {
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between'
                }
            },
            [classDescription]: {
                display: 'flex'
            }
        },
        // Warning
        ...props.type === 'warning' && {
            color: theme.colors.textValidationWarning,
            borderColor: theme.colors.yellow300
        },
        // Error
        ...props.type === 'error' && {
            color: theme.colors.textValidationDanger,
            borderColor: theme.colors.red300
        },
        // Banner
        ...props.banner && {
            borderStyle: 'solid',
            borderWidth: `${theme.general.borderWidth}px 0`
        },
        // After closed
        '&[data-show="false"]': {
            borderWidth: 0,
            padding: 0,
            width: 0,
            height: 0
        },
        ...getAnimationCss(theme.options.enableAnimation)
    };
    return /*#__PURE__*/ css(styles);
};

const getGlobalStyles = (theme)=>{
    return /*#__PURE__*/ css({
        'body, .mfe-root': {
            backgroundColor: theme.colors.backgroundPrimary,
            color: theme.colors.textPrimary,
            '--dubois-global-background-color': theme.colors.backgroundPrimary,
            '--dubois-global-color': theme.colors.textPrimary
        }
    });
};
const ApplyGlobalStyles = ()=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(Global, {
        styles: [
            getGlobalStyles(theme)
        ]
    });
};

/**
 * @deprecated Use `TypeaheadCombobox` instead.
 */ const AutoComplete = /* #__PURE__ */ (()=>{
    const AutoComplete = ({ dangerouslySetAntdProps, ...props })=>{
        const { theme } = useDesignSystemTheme();
        const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
        return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsx(AutoComplete$1, {
                ...addDebugOutlineIfEnabled(),
                dropdownStyle: {
                    boxShadow: theme.general.shadowLow,
                    ...getDarkModePortalStyles(theme, useNewBorderColors)
                },
                ...props,
                ...dangerouslySetAntdProps,
                css: css(getAnimationCss(theme.options.enableAnimation))
            })
        });
    };
    AutoComplete.Option = AutoComplete$1.Option;
    return AutoComplete;
})();

// eslint-disable-next-line no-restricted-imports -- grandfathering, see go/ui-bestpractices
function SvgDatabricksIcon(props) {
    return /*#__PURE__*/ jsxs("svg", {
        viewBox: "0 0 24 24",
        width: "100%",
        height: "100%",
        xmlns: "http://www.w3.org/2000/svg",
        ...props,
        children: [
            /*#__PURE__*/ jsx("rect", {
                width: "24",
                height: "24",
                fill: "#FF3621"
            }),
            /*#__PURE__*/ jsx("path", {
                d: "m18.8 10.515-6.8228 3.945-7.3059-4.215-0.35138 0.195v3.06l7.6573 4.41 6.8228-3.93v1.62l-6.8228 3.945-7.3059-4.215-0.35138 0.195v0.525l7.6573 4.41 7.6427-4.41v-3.06l-0.3514-0.195-7.2913 4.2-6.8374-3.93v-1.62l6.8374 3.93 7.6427-4.41v-3.015l-0.3807-0.225-7.262 4.185-6.486-3.72 6.486-3.735 5.3294 3.075 0.4685-0.27v-0.375l-5.7979-3.345-7.6573 4.41v0.48l7.6573 4.41 6.8228-3.945v1.62z",
                fill: "#fff"
            })
        ]
    });
}

// eslint-disable-next-line no-restricted-imports -- grandfathering, see go/ui-bestpractices
function SvgUserIcon(props) {
    return /*#__PURE__*/ jsx("svg", {
        xmlns: "http://www.w3.org/2000/svg",
        width: "100%",
        height: "auto",
        fill: "none",
        viewBox: "0 0 400 400",
        ...props,
        children: /*#__PURE__*/ jsx("path", {
            d: "M200 69.333c-44.873 0-81.25 36.377-81.25 81.25s36.377 81.25 81.25 81.25 81.25-36.377 81.25-81.25-36.377-81.25-81.25-81.25Zm0 200c-69.162 0-130.835 32.119-170.89 82.181A18.748 18.748 0 0 0 25 363.228v37.355c0 10.356 8.395 18.75 18.75 18.75h312.5c10.355 0 18.75-8.394 18.75-18.75v-37.355c0-4.258-1.449-8.39-4.11-11.714-40.055-50.062-101.728-82.181-170.89-82.181Z",
            fill: "#5F7281"
        })
    });
}

/**
 * `LegacyTooltip` is deprecated in favor of the new `Tooltip` component
 * @deprecated
 */ const LegacyTooltip = ({ children, title, placement = 'top', dataTestId, dangerouslySetAntdProps, silenceScreenReader = false, useAsLabel = false, ...props })=>{
    const { theme } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const tooltipRef = useRef(null);
    const duboisId = useUniqueId('dubois-tooltip-component-');
    const id = dangerouslySetAntdProps?.id ? dangerouslySetAntdProps?.id : duboisId;
    if (!title) {
        return /*#__PURE__*/ jsx(React__default.Fragment, {
            children: children
        });
    }
    const titleProps = silenceScreenReader ? {} : {
        'aria-live': 'polite',
        'aria-relevant': 'additions'
    };
    if (dataTestId) {
        titleProps['data-testid'] = dataTestId;
    }
    const liveTitle = title && /*#__PURE__*/ React__default.isValidElement(title) ? /*#__PURE__*/ React__default.cloneElement(title, titleProps) : /*#__PURE__*/ jsx("span", {
        ...titleProps,
        children: title
    });
    const ariaProps = {
        'aria-hidden': false
    };
    const addAriaProps = (e)=>{
        if (!tooltipRef.current || e.currentTarget.hasAttribute('aria-describedby') || e.currentTarget.hasAttribute('aria-labelledby')) {
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
    const removeAriaProps = (e)=>{
        if (!tooltipRef || !e.currentTarget.hasAttribute('aria-describedby') && !e.currentTarget.hasAttribute('aria-labelledby')) {
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
        onMouseEnter: (e)=>{
            addAriaProps(e);
        },
        onMouseLeave: (e)=>{
            removeAriaProps(e);
        },
        onFocus: (e)=>{
            addAriaProps(e);
        },
        onBlur: (e)=>{
            removeAriaProps(e);
        }
    };
    const childWithProps = /*#__PURE__*/ React__default.isValidElement(children) ? /*#__PURE__*/ React__default.cloneElement(children, {
        ...ariaProps,
        ...interactionProps,
        ...children.props
    }) : isNil(children) ? children : /*#__PURE__*/ jsx("span", {
        ...ariaProps,
        ...interactionProps,
        children: children
    });
    const { overlayInnerStyle, overlayStyle, ...delegatedDangerouslySetAntdProps } = dangerouslySetAntdProps || {};
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Tooltip, {
            id: id,
            ref: tooltipRef,
            title: liveTitle,
            placement: placement,
            // Always trigger on hover and focus
            trigger: [
                'hover',
                'focus'
            ],
            overlayInnerStyle: {
                backgroundColor: '#2F3941',
                lineHeight: '22px',
                padding: '4px 8px',
                boxShadow: theme.general.shadowLow,
                ...overlayInnerStyle,
                ...getDarkModePortalStyles(theme, useNewBorderColors)
            },
            overlayStyle: {
                zIndex: theme.options.zIndexBase + 70,
                ...overlayStyle
            },
            css: {
                ...getAnimationCss(theme.options.enableAnimation)
            },
            ...delegatedDangerouslySetAntdProps,
            ...props,
            children: childWithProps
        })
    });
};

/**
 * `LegacyInfoTooltip` is deprecated in favor of the new `InfoTooltip` component
 * @deprecated
 */ const LegacyInfoTooltip = ({ title, tooltipProps, iconTitle, isKeyboardFocusable = true, ...iconProps })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(LegacyTooltip, {
        useAsLabel: true,
        title: title,
        ...tooltipProps,
        children: /*#__PURE__*/ jsx("span", {
            ...addDebugOutlineIfEnabled(),
            style: {
                display: 'inline-flex'
            },
            children: /*#__PURE__*/ jsx(InfoCircleOutlined, {
                tabIndex: isKeyboardFocusable ? 0 : -1,
                "aria-hidden": "false",
                "aria-label": iconTitle,
                alt: iconTitle,
                css: {
                    fontSize: theme.typography.fontSizeSm,
                    color: theme.colors.textSecondary
                },
                ...iconProps
            })
        })
    });
};

const InfoPopover = ({ children, popoverProps, iconTitle, iconProps, isKeyboardFocusable = true, ariaLabel = 'More details' })=>{
    const { theme } = useDesignSystemTheme();
    const { isInsideModal } = useModalContext();
    const [open, setOpen] = useState(false);
    const handleKeyDown = (event)=>{
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            setOpen(!open);
        }
    };
    const { onKeyDown, ...restPopoverProps } = popoverProps || {};
    return /*#__PURE__*/ jsxs(Root$8, {
        componentId: "codegen_design-system_src_design-system_popover_infopopover.tsx_36",
        open: open,
        onOpenChange: setOpen,
        children: [
            /*#__PURE__*/ jsx(Trigger$4, {
                asChild: true,
                children: /*#__PURE__*/ jsx("span", {
                    style: {
                        display: 'inline-flex',
                        cursor: 'pointer'
                    },
                    // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
                    tabIndex: isKeyboardFocusable ? 0 : -1,
                    onKeyDown: handleKeyDown,
                    "aria-label": iconTitle ? undefined : ariaLabel,
                    role: "button",
                    onClick: (e)=>{
                        e.preventDefault();
                        e.stopPropagation();
                        setOpen(!open);
                    },
                    children: /*#__PURE__*/ jsx(InfoSmallIcon, {
                        "aria-hidden": iconTitle ? false : true,
                        title: iconTitle,
                        "aria-label": iconTitle,
                        css: {
                            color: theme.colors.textSecondary
                        },
                        ...iconProps
                    })
                })
            }),
            /*#__PURE__*/ jsxs(Content$5, {
                align: "start",
                onKeyDown: (e)=>{
                    if (e.key === 'Escape') {
                        // If inside an AntD Modal, stop propagation of Escape key so that the modal doesn't close.
                        // This is specifically for that case, so we only do it if inside a modal to limit the blast radius.
                        if (isInsideModal) {
                            e.stopPropagation();
                            // If stopping propagation, we also need to manually close the popover since the radix
                            // library expects the event to bubble up to the parent components.
                            setOpen(false);
                        }
                    }
                    onKeyDown?.(e);
                },
                ...restPopoverProps,
                children: [
                    children,
                    /*#__PURE__*/ jsx(Arrow$2, {})
                ]
            })
        ]
    });
};

const OverflowPopover = ({ items, renderLabel, tooltipText, ariaLabel = 'More items', ...props })=>{
    const { theme } = useDesignSystemTheme();
    const [showTooltip, setShowTooltip] = useState(true);
    const label = `+${items.length}`;
    let trigger = /*#__PURE__*/ jsx("span", {
        css: {
            lineHeight: 0
        },
        ...addDebugOutlineIfEnabled(),
        children: /*#__PURE__*/ jsx(Trigger$4, {
            asChild: true,
            children: /*#__PURE__*/ jsx(Button, {
                componentId: "something",
                type: "link",
                children: renderLabel ? renderLabel(label) : label
            })
        })
    });
    if (showTooltip) {
        trigger = /*#__PURE__*/ jsx(LegacyTooltip, {
            title: tooltipText || 'See more items',
            children: trigger
        });
    }
    return /*#__PURE__*/ jsxs(Root$8, {
        componentId: "codegen_design-system_src_design-system_overflow_overflowpopover.tsx_37",
        onOpenChange: (open)=>setShowTooltip(!open),
        children: [
            trigger,
            /*#__PURE__*/ jsx(Content$5, {
                align: "start",
                "aria-label": ariaLabel,
                ...props,
                ...addDebugOutlineIfEnabled(),
                children: /*#__PURE__*/ jsx("div", {
                    css: {
                        display: 'flex',
                        flexDirection: 'column',
                        gap: theme.spacing.xs
                    },
                    children: items.map((item, index)=>/*#__PURE__*/ jsx("div", {
                            children: item
                        }, `overflow-${index}`))
                })
            })
        ]
    });
};

const SIZE = new Map([
    [
        'xl',
        {
            avatarSize: 48,
            fontSize: 18,
            groupShift: 12,
            iconSize: 24
        }
    ],
    [
        'lg',
        {
            avatarSize: 40,
            fontSize: 16,
            groupShift: 8,
            iconSize: 20
        }
    ],
    [
        'md',
        {
            avatarSize: 32,
            fontSize: 14,
            groupShift: 4,
            iconSize: 16
        }
    ],
    [
        'sm',
        {
            avatarSize: 24,
            fontSize: 12,
            groupShift: 4,
            iconSize: 14
        }
    ],
    [
        'xs',
        {
            avatarSize: 20,
            fontSize: 12,
            groupShift: 2,
            iconSize: 12
        }
    ],
    [
        'xxs',
        {
            avatarSize: 16,
            fontSize: 11,
            groupShift: 2,
            iconSize: 12
        }
    ]
]);
const DEFAULT_SIZE = 'sm';
function getAvatarEmotionStyles({ backgroundColor, size = DEFAULT_SIZE, theme }) {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
    const { avatarSize, fontSize, iconSize } = SIZE.get(size);
    return {
        abbr: {
            color: theme.colors.tagText,
            textDecoration: 'none',
            textTransform: 'uppercase'
        },
        default: {
            height: avatarSize,
            width: avatarSize,
            fontSize,
            [`.${theme.general.iconfontCssPrefix}`]: {
                fontSize: iconSize
            }
        },
        icon: {
            alignItems: 'center',
            color: backgroundColor ? theme.colors.tagText : theme.colors.textSecondary,
            backgroundColor: backgroundColor ? theme.colors[backgroundColor] : theme.colors.tagDefault,
            display: 'flex',
            justifyContent: 'center'
        },
        img: {
            objectFit: 'cover',
            objectPosition: 'center'
        },
        system: {
            borderRadius: theme.borders.borderRadiusSm,
            overflow: 'hidden'
        },
        user: {
            borderRadius: '100%',
            overflow: 'hidden'
        },
        userIcon: {
            alignItems: 'flex-end'
        }
    };
}
/** Generate random number from a string between 0 - (maxRange - 1) */ function getRandomNumberFromString({ value, maxRange }) {
    let hash = 0;
    let char = 0;
    if (value.length === 0) return hash;
    for(let i = 0; i < value.length; i++){
        char = value.charCodeAt(i);
        hash = (hash << 5) - hash + char;
        hash = hash & hash;
    }
    const idx = Math.abs(hash % maxRange);
    return idx;
}
function getAvatarBackgroundColor(label, theme) {
    const randomNumber = getRandomNumberFromString({
        value: label,
        maxRange: 5
    });
    switch(randomNumber){
        case 0:
            return theme.colors.indigo;
        case 1:
            return theme.colors.teal;
        case 2:
            return theme.colors.pink;
        case 3:
            return theme.colors.brown;
        case 4:
        default:
            return theme.colors.purple;
    }
}
function Avatar(props) {
    const { theme } = useDesignSystemTheme();
    const styles = getAvatarEmotionStyles({
        size: props.size,
        theme,
        backgroundColor: 'backgroundColor' in props ? props.backgroundColor : undefined
    });
    switch(props.type){
        case 'entity':
            if ('src' in props && props.src) {
                return /*#__PURE__*/ jsx("img", {
                    css: [
                        styles.default,
                        styles.img,
                        styles.system
                    ],
                    src: props.src,
                    alt: props.label
                });
            }
            if ('icon' in props && props.icon) {
                return /*#__PURE__*/ jsx("div", {
                    css: [
                        styles.default,
                        styles.system,
                        styles.icon
                    ],
                    role: "img",
                    "aria-label": props.label,
                    children: props.icon
                });
            }
            // display first initial of name when no image / icon is provided
            return /*#__PURE__*/ jsx("div", {
                css: [
                    styles.default,
                    styles.system,
                    styles.icon,
                    {
                        backgroundColor: getAvatarBackgroundColor(props.label, theme)
                    }
                ],
                children: /*#__PURE__*/ jsx("abbr", {
                    css: styles.abbr,
                    title: props.label,
                    children: props.label.substring(0, 1)
                })
            });
        case 'user':
            if ('label' in props && props.label.trim()) {
                if (props.src) {
                    return /*#__PURE__*/ jsx("img", {
                        css: [
                            styles.default,
                            styles.img,
                            styles.user
                        ],
                        src: props.src,
                        alt: props.label
                    });
                } else if (props.icon) {
                    return /*#__PURE__*/ jsx("div", {
                        css: [
                            styles.default,
                            styles.user,
                            styles.icon
                        ],
                        role: "img",
                        "aria-label": props.label,
                        children: props.icon
                    });
                }
                // display first initial of name when no image / icon is provided
                return /*#__PURE__*/ jsx("div", {
                    css: [
                        styles.default,
                        styles.user,
                        styles.icon,
                        {
                            backgroundColor: getAvatarBackgroundColor(props.label, theme)
                        }
                    ],
                    children: /*#__PURE__*/ jsx("abbr", {
                        css: styles.abbr,
                        title: props.label,
                        children: props.label.substring(0, 1)
                    })
                });
            }
            // default to user icon when no user info is provided
            return /*#__PURE__*/ jsx("div", {
                css: [
                    styles.default,
                    styles.user,
                    styles.icon,
                    styles.userIcon
                ],
                role: "img",
                "aria-label": "user",
                children: /*#__PURE__*/ jsx(SvgUserIcon, {})
            });
    }
}
function DBAssistantAvatar({ size }) {
    return /*#__PURE__*/ jsx(Avatar, {
        size: size,
        type: "entity",
        label: "Assistant",
        icon: /*#__PURE__*/ jsx(SvgDatabricksIcon, {})
    });
}
function AssistantAvatar({ backgroundColor, size }) {
    return /*#__PURE__*/ jsx(Avatar, {
        backgroundColor: backgroundColor,
        size: size,
        type: "entity",
        label: "Assistant",
        icon: /*#__PURE__*/ jsx(SparkleDoubleIcon, {})
    });
}
const MAX_AVATAR_GROUP_USERS = 3;
function getAvatarGroupEmotionStyles(theme) {
    return {
        container: {
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.xs
        },
        avatarsContainer: {
            display: 'flex'
        },
        avatar: {
            display: 'flex',
            borderRadius: '100%',
            border: `1px solid ${theme.colors.backgroundPrimary}`,
            position: 'relative'
        }
    };
}
function AvatarGroup({ size = DEFAULT_SIZE, users }) {
    const { theme } = useDesignSystemTheme();
    const styles = getAvatarGroupEmotionStyles(theme);
    const displayedUsers = useMemo(()=>users.slice(0, MAX_AVATAR_GROUP_USERS), [
        users
    ]);
    const extraUsers = useMemo(()=>users.slice(MAX_AVATAR_GROUP_USERS), [
        users
    ]);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
    const { avatarSize, groupShift: avatarShift, fontSize } = SIZE.get(size);
    return /*#__PURE__*/ jsxs("div", {
        css: styles.container,
        children: [
            /*#__PURE__*/ jsx("div", {
                css: {
                    ...styles.avatarsContainer,
                    width: (avatarSize + 2 - avatarShift) * displayedUsers.length + avatarShift
                },
                children: displayedUsers.map((user, idx)=>/*#__PURE__*/ jsx("div", {
                        css: {
                            ...styles.avatar,
                            left: -avatarShift * idx
                        },
                        children: /*#__PURE__*/ jsx(Avatar, {
                            size: size,
                            type: "user",
                            ...user
                        })
                    }, `${user.label}-idx`))
            }),
            extraUsers.length > 0 && /*#__PURE__*/ jsx(OverflowPopover, {
                items: extraUsers.map((user)=>user.label),
                tooltipText: "Show more users",
                renderLabel: (label)=>/*#__PURE__*/ jsx("span", {
                        css: {
                            fontSize: `${fontSize}px !important`
                        },
                        children: label
                    })
            })
        ]
    });
}

const Breadcrumb = /* #__PURE__ */ (()=>{
    const Breadcrumb = ({ dangerouslySetAntdProps, includeTrailingCaret = true, ...props })=>{
        const { theme, classNamePrefix } = useDesignSystemTheme();
        const separatorClass = `.${classNamePrefix}-breadcrumb-separator`;
        const styles = css({
            // `antd` forces the last anchor to be black, so that it doesn't look like an anchor
            // (even though it is one). This undoes that; if the user wants to make the last
            // text-colored, they can do that by not using an anchor.
            'span:last-child a': {
                color: theme.colors.primary,
                // TODO: Need to pull a global color for anchor hover/focus. Discuss with Ginny.
                ':hover, :focus': {
                    color: '#2272B4'
                }
            },
            // TODO: Consider making this global within dubois components
            a: {
                '&:focus-visible': {
                    outlineColor: `${theme.colors.actionDefaultBorderFocus} !important`,
                    outlineStyle: 'auto !important'
                }
            },
            [separatorClass]: {
                fontSize: theme.general.iconFontSize,
                '& .anticon': {
                    fontSize: 13
                }
            },
            '& > span': {
                display: 'inline-flex',
                alignItems: 'center'
            }
        });
        return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsxs(Breadcrumb$1, {
                ...addDebugOutlineIfEnabled(),
                separator: /*#__PURE__*/ jsx(ChevronRightIcon, {}),
                ...props,
                ...dangerouslySetAntdProps,
                css: css(getAnimationCss(theme.options.enableAnimation), styles),
                children: [
                    props.children,
                    includeTrailingCaret && props.children && /*#__PURE__*/ jsx(Breadcrumb.Item, {
                        children: " "
                    })
                ]
            })
        });
    };
    Breadcrumb.Item = Breadcrumb$1.Item;
    Breadcrumb.Separator = Breadcrumb$1.Separator;
    return Breadcrumb;
})();

// This is a very simple PRNG that is seeded (so that the output is deterministic).
// We need this in order to produce a random ragged edge for the table skeleton.
function pseudoRandomNumberGeneratorFromSeed(seed) {
    // This is a simple way to get a consistent number from a string;
    // `charCodeAt` returns a number between 0 and 65535, and we then just add them all up.
    const seedValue = seed.split('').map((char)=>char.charCodeAt(0)).reduce((prev, curr)=>prev + curr, 0);
    // This is a simple sine wave function that will always return a number between 0 and 1.
    // This produces a value akin to `Math.random()`, but has deterministic output.
    // Of course, sine curves are not a perfectly random distribution between 0 and 1, but
    // it's close enough for our purposes.
    return Math.sin(seedValue) / 2 + 0.5;
}
// This is a simple Fisher-Yates shuffler using the above PRNG.
function shuffleArray(arr, seed) {
    for(let i = arr.length - 1; i > 0; i--){
        const j = Math.floor(pseudoRandomNumberGeneratorFromSeed(seed + String(i)) * (i + 1));
        [arr[i], arr[j]] = [
            arr[j],
            arr[i]
        ];
    }
    return arr;
}
// Finally, we shuffle a list off offsets to apply to the widths of the cells.
// This ensures that the cells are not all the same width, but that they are
// random to produce a more realistic looking skeleton.
function getOffsets(seed) {
    return shuffleArray([
        48,
        24,
        0
    ], seed);
}
const skeletonLoading = /*#__PURE__*/ keyframes({
    '0%': {
        backgroundPosition: '100% 50%'
    },
    '100%': {
        backgroundPosition: '0 50%'
    }
});
const genSkeletonAnimatedColor = (theme, frameRate = 60)=>{
    // TODO: Pull this from the themes; it's not currently available.
    const color = theme.isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(31, 38, 45, 0.1)';
    // Light mode value copied from Ant's Skeleton animation
    const colorGradientEnd = theme.isDarkMode ? 'rgba(99, 99, 99, 0.24)' : 'rgba(129, 129, 129, 0.24)';
    return /*#__PURE__*/ css({
        animationDuration: '1.4s',
        background: `linear-gradient(90deg, ${color} 25%, ${colorGradientEnd} 37%, ${color} 63%)`,
        backgroundSize: '400% 100%',
        animationName: skeletonLoading,
        animationTimingFunction: `steps(${frameRate}, end)`,
        // Based on data from perf dashboard, p95 loading time goes up to about 20s, so about 14 iterations is needed.
        animationIterationCount: 14
    });
};

const GenericContainerStyles = /*#__PURE__*/ css({
    cursor: 'progress',
    borderRadius: 'var(--border-radius)'
});
const GenericSkeleton = ({ label, frameRate = 60, style, loading = true, loadingDescription = 'GenericSkeleton', ...restProps })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsxs("div", {
        ...addDebugOutlineIfEnabled(),
        css: [
            GenericContainerStyles,
            genSkeletonAnimatedColor(theme, frameRate)
        ],
        style: {
            ...style,
            ['--border-radius']: `${theme.general.borderRadiusBase}px`
        },
        ...restProps,
        children: [
            loading && /*#__PURE__*/ jsx(LoadingState, {
                description: loadingDescription
            }),
            /*#__PURE__*/ jsx("span", {
                css: visuallyHidden,
                children: label
            })
        ]
    });
};

const paragraphContainerStyles = /*#__PURE__*/ css({
    cursor: 'progress',
    width: '100%',
    height: 20,
    display: 'flex',
    justifyContent: 'flex-start',
    alignItems: 'center'
});
const paragraphFillStyles = /*#__PURE__*/ css({
    borderRadius: 'var(--border-radius)',
    height: 8
});
const ParagraphSkeleton = ({ label, seed = '', frameRate = 60, style, loading = true, loadingDescription = 'ParagraphSkeleton', ...restProps })=>{
    const { theme } = useDesignSystemTheme();
    const offsetWidth = getOffsets(seed)[0];
    return /*#__PURE__*/ jsxs("div", {
        ...addDebugOutlineIfEnabled(),
        css: paragraphContainerStyles,
        style: {
            ...style,
            ['--border-radius']: `${theme.general.borderRadiusBase}px`
        },
        ...restProps,
        children: [
            loading && /*#__PURE__*/ jsx(LoadingState, {
                description: loadingDescription
            }),
            /*#__PURE__*/ jsx("span", {
                css: visuallyHidden,
                children: label
            }),
            /*#__PURE__*/ jsx("div", {
                "aria-hidden": true,
                css: [
                    paragraphFillStyles,
                    genSkeletonAnimatedColor(theme, frameRate),
                    {
                        width: `calc(100% - ${offsetWidth}px)`
                    }
                ]
            })
        ]
    });
};

const titleContainerStyles = /*#__PURE__*/ css({
    cursor: 'progress',
    width: '100%',
    height: 28,
    display: 'flex',
    justifyContent: 'flex-start',
    alignItems: 'center'
});
const titleFillStyles = /*#__PURE__*/ css({
    borderRadius: 'var(--border-radius)',
    height: 12,
    width: '100%'
});
const TitleSkeleton = ({ label, seed = '', frameRate = 60, style, loading = true, loadingDescription = 'TitleSkeleton', ...restProps })=>{
    const { theme } = useDesignSystemTheme();
    const offsetWidth = getOffsets(seed)[0];
    return /*#__PURE__*/ jsxs("div", {
        ...addDebugOutlineIfEnabled(),
        css: titleContainerStyles,
        style: {
            ...style,
            ['--border-radius']: `${theme.general.borderRadiusBase}px`
        },
        ...restProps,
        children: [
            loading && /*#__PURE__*/ jsx(LoadingState, {
                description: loadingDescription
            }),
            /*#__PURE__*/ jsx("span", {
                css: visuallyHidden,
                children: label
            }),
            /*#__PURE__*/ jsx("div", {
                "aria-hidden": true,
                css: [
                    titleFillStyles,
                    genSkeletonAnimatedColor(theme, frameRate),
                    {
                        width: `calc(100% - ${offsetWidth}px)`
                    }
                ]
            })
        ]
    });
};

// Class names that can be used to reference children within
// Should not be used outside of design system
// TODO: PE-239 Maybe we could add "dangerous" into the names or make them completely random.
function randomString() {
    return times(20, ()=>random(35).toString(36)).join('');
}
const tableClassNames = {
    cell: `js--ds-table-cell-${randomString()}`,
    header: `js--ds-table-header-${randomString()}`,
    row: `js--ds-table-row-${randomString()}`
};
// We do not want to use `css=` for elements that can appear on the screen more than ~100 times.
// Instead, we define them here and nest the styling in the styles for the table component below.
// For details see: https://emotion.sh/docs/performance
const repeatingElementsStyles = {
    cell: /*#__PURE__*/ css({
        display: 'inline-grid',
        position: 'relative',
        flex: 1,
        boxSizing: 'border-box',
        paddingLeft: 'var(--table-spacing-sm)',
        paddingRight: 'var(--table-spacing-sm)',
        wordBreak: 'break-word',
        overflow: 'hidden',
        '& .anticon': {
            verticalAlign: 'text-bottom'
        }
    }),
    header: /*#__PURE__*/ css({
        fontWeight: 'bold',
        alignItems: 'flex-end',
        display: 'flex',
        overflow: 'hidden',
        '&[aria-sort]': {
            cursor: 'pointer',
            userSelect: 'none'
        },
        '.table-header-text': {
            color: 'var(--table-header-text-color)'
        },
        '.table-header-icon-container': {
            color: 'var(--table-header-sort-icon-color)',
            display: 'none'
        },
        '&[aria-sort]:hover': {
            '.table-header-icon-container, .table-header-text': {
                color: 'var(--table-header-focus-color)'
            }
        },
        '&[aria-sort]:active': {
            '.table-header-icon-container, .table-header-text': {
                color: 'var(--table-header-active-color)'
            }
        },
        '&:hover, &[aria-sort="ascending"], &[aria-sort="descending"]': {
            '.table-header-icon-container': {
                display: 'inline'
            }
        }
    }),
    row: /*#__PURE__*/ css({
        display: 'flex',
        '&.table-isHeader': {
            '> *': {
                backgroundColor: 'var(--table-header-background-color)'
            },
            '.table-isScrollable &': {
                position: 'sticky',
                top: 0,
                zIndex: 1
            }
        },
        // Note: Next-sibling selector is necessary for Ant Checkboxes; if we move away
        // from those in the future we would need to adjust these styles.
        '.table-row-select-cell input[type="checkbox"] ~ *': {
            opacity: 'var(--row-checkbox-opacity, 0)'
        },
        '&:not(.table-row-isGrid)&:hover': {
            '&:not(.table-isHeader)': {
                backgroundColor: 'var(--table-row-hover)'
            },
            '.table-row-select-cell input[type="checkbox"] ~ *': {
                opacity: 1
            }
        },
        '.table-row-select-cell input[type="checkbox"]:focus ~ *': {
            opacity: 1
        },
        '> *': {
            paddingTop: 'var(--table-row-vertical-padding)',
            paddingBottom: 'var(--table-row-vertical-padding)',
            borderBottom: '1px solid',
            borderColor: 'var(--table-separator-color)'
        },
        '&.table-row-isGrid > *': {
            borderRight: '1px solid',
            borderColor: 'var(--table-separator-color)'
        },
        // Add left border to first cell in grid
        '&.table-row-isGrid > :first-of-type': {
            borderLeft: '1px solid',
            borderColor: 'var(--table-separator-color)'
        },
        // Add top border for first row in cell
        '&.table-row-isGrid.table-isHeader:first-of-type > *': {
            borderTop: '1px solid',
            borderColor: 'var(--table-separator-color)'
        }
    })
};
const hideIconButtonActionCellClassName = `hide-icon-button-${randomString()}`;
const skipHideIconButtonActionClassName = `skip-hide-icon-button-${randomString()}`;
const hideIconButtonRowStyles = /*#__PURE__*/ css({
    // Always show if skipHideIconButtonClassName is present (overrides all below)
    [`.${hideIconButtonActionCellClassName} button.${skipHideIconButtonActionClassName}`]: {
        opacity: '1 !important',
        transition: 'opacity 0.1s ease !important'
    },
    // Hide by default
    [`.${hideIconButtonActionCellClassName} button:has(> span.anticon[role="img"]:only-child),
    .${hideIconButtonActionCellClassName} button:has(> i.fa:only-child),
    .${hideIconButtonActionCellClassName} a:has(> span.anticon[role="img"]:only-child),
    .${hideIconButtonActionCellClassName} a:has(> i.fa:only-child)`]: {
        opacity: 0,
        transition: 'opacity 0.1s ease !important'
    },
    [`.${hideIconButtonActionCellClassName} button:focus-visible:has(> span.anticon[role="img"]:only-child),
    .${hideIconButtonActionCellClassName} button:focus-visible:has(> i.fa:only-child),
    .${hideIconButtonActionCellClassName} a:focus-visible:has(> span.anticon[role="img"]:only-child),
    .${hideIconButtonActionCellClassName} a:focus-visible:has(> i.fa:only-child)`]: {
        outlineStyle: 'solid !important'
    },
    // Keep visible when actively clicked or when dropdown is open
    [`&:hover .${hideIconButtonActionCellClassName} button:has(> span.anticon[role="img"]:only-child),
    .${hideIconButtonActionCellClassName} button:focus-visible:has(> span.anticon[role="img"]:only-child),
    &:hover .${hideIconButtonActionCellClassName} button:has(> i.fa:only-child),
    .${hideIconButtonActionCellClassName} button:focus-visible:has(> i.fa:only-child),
    &:hover .${hideIconButtonActionCellClassName} a:has(> span.anticon[role="img"]:only-child),
    .${hideIconButtonActionCellClassName} a:focus-visible:has(> span.anticon[role="img"]:only-child),
    &:hover .${hideIconButtonActionCellClassName} a:has(> i.fa:only-child),
    .${hideIconButtonActionCellClassName} a:focus-visible:has(> i.fa:only-child)
    .${hideIconButtonActionCellClassName} div[aria-expanded="true"] > button:has(span.anticon[role="img"]:only-child),
    .${hideIconButtonActionCellClassName} div[aria-expanded="true"] > button:has(i.fa:only-child),
    .${hideIconButtonActionCellClassName} button[aria-expanded="true"]:has(span.anticon[role="img"]:only-child),
    .${hideIconButtonActionCellClassName} button[aria-expanded="true"]:has(i.fa:only-child),`]: {
        opacity: 1
    }
});
// For performance, these styles are defined outside of the component so they are not redefined on every render.
// We're also using CSS Variables rather than any dynamic styles so that the style object remains static.
const tableStyles = {
    tableWrapper: /*#__PURE__*/ css({
        '&.table-isScrollable': {
            overflow: 'auto'
        },
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        // Inline repeating elements styles for performance reasons
        [`.${tableClassNames.cell}`]: repeatingElementsStyles.cell,
        [`.${tableClassNames.header}`]: repeatingElementsStyles.header,
        [`.${tableClassNames.row}`]: repeatingElementsStyles.row
    }),
    table: /*#__PURE__*/ css({
        '.table-isScrollable &': {
            overflow: 'auto',
            // Remove this flex:1 style when removing databricks.fe.designsystem.disableFlexOnScrollableTable
            flex: 1
        }
    }),
    headerButtonTarget: /*#__PURE__*/ css({
        alignItems: 'flex-end',
        display: 'flex',
        overflow: 'hidden',
        width: '100%',
        justifyContent: 'inherit',
        '&:focus': {
            '.table-header-text': {
                color: 'var(--table-header-focus-color)'
            },
            '.table-header-icon-container': {
                color: 'var(--table-header-focus-color)',
                display: 'inline'
            }
        },
        '&:active': {
            '.table-header-icon-container, .table-header-text': {
                color: 'var(--table-header-active-color)'
            }
        }
    }),
    sortHeaderIconOnRight: /*#__PURE__*/ css({
        marginLeft: 'var(--table-spacing-xs)'
    }),
    sortHeaderIconOnLeft: /*#__PURE__*/ css({
        marginRight: 'var(--table-spacing-xs)'
    }),
    checkboxCell: /*#__PURE__*/ css({
        display: 'flex',
        alignItems: 'center',
        flex: 0,
        paddingLeft: 'var(--table-spacing-sm)',
        paddingTop: 0,
        paddingBottom: 0,
        minWidth: 'var(--table-spacing-md)',
        maxWidth: 'var(--table-spacing-md)',
        boxSizing: 'content-box !important'
    }),
    resizeHandleContainer: /*#__PURE__*/ css({
        position: 'absolute',
        right: -3,
        top: 'var(--table-spacing-sm)',
        bottom: 'var(--table-spacing-sm)',
        width: 'var(--table-spacing-sm)',
        display: 'flex',
        justifyContent: 'center',
        cursor: 'col-resize',
        userSelect: 'none',
        touchAction: 'none',
        zIndex: 1
    }),
    resizeHandle: /*#__PURE__*/ css({
        width: 1,
        background: 'var(--table-resize-handle-color)'
    }),
    paginationContainer: /*#__PURE__*/ css({
        display: 'flex',
        justifyContent: 'flex-end',
        paddingTop: 'var(--table-spacing-sm)',
        paddingBottom: 'var(--table-spacing-sm)'
    })
};

const TableContext = /*#__PURE__*/ createContext({
    size: 'default',
    grid: false
});
const Table = /*#__PURE__*/ forwardRef(function Table({ children, size = 'default', someRowsSelected, style, pagination, empty, className, scrollable = false, grid = false, noMinHeight = false, onScroll, ...rest }, ref) {
    const { theme } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const tableContentRef = useRef(null);
    useImperativeHandle(ref, ()=>tableContentRef.current);
    const minHeightCss = noMinHeight ? {} : {
        minHeight: !empty && pagination ? 150 : 100
    };
    const disableFlexOnScrollableTable = safex('databricks.fe.designsystem.disableFlexOnScrollableTable', false);
    return /*#__PURE__*/ jsx(DesignSystemEventSuppressInteractionProviderContext.Provider, {
        value: DesignSystemEventSuppressInteractionTrueContextValue,
        children: /*#__PURE__*/ jsx(TableContext.Provider, {
            value: useMemo(()=>{
                return {
                    size,
                    someRowsSelected,
                    grid
                };
            }, [
                size,
                someRowsSelected,
                grid
            ]),
            children: /*#__PURE__*/ jsxs("div", {
                ...addDebugOutlineIfEnabled(),
                ...rest,
                // This is a performance optimization; we want to statically create the styles for the table,
                // but for the dynamic theme values, we need to use CSS variables.
                // See: https://emotion.sh/docs/best-practices#advanced-css-variables-with-style
                style: {
                    ...style,
                    ['--table-header-active-color']: theme.colors.actionDefaultTextPress,
                    ['colorScheme']: theme.isDarkMode ? 'dark' : undefined,
                    ['--table-header-background-color']: theme.colors.backgroundPrimary,
                    ['--table-header-focus-color']: theme.colors.actionDefaultTextHover,
                    ['--table-header-sort-icon-color']: theme.colors.textSecondary,
                    ['--table-header-text-color']: theme.colors.actionDefaultTextDefault,
                    ['--table-row-hover']: theme.colors.tableRowHover,
                    ['--table-separator-color']: useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative,
                    ['--table-resize-handle-color']: theme.colors.borderDecorative,
                    ['--table-spacing-md']: `${theme.spacing.md}px`,
                    ['--table-spacing-sm']: `${theme.spacing.sm}px`,
                    ['--table-spacing-xs']: `${theme.spacing.xs}px`
                },
                css: [
                    tableStyles.tableWrapper,
                    minHeightCss
                ],
                className: classnames({
                    'table-isScrollable': scrollable,
                    'table-isGrid': grid
                }, className),
                children: [
                    /*#__PURE__*/ jsxs("div", {
                        role: "table",
                        ref: tableContentRef,
                        css: [
                            tableStyles.table,
                            disableFlexOnScrollableTable && scrollable && {
                                flex: 'initial !important'
                            }
                        ],
                        // Needed to make panel body content focusable when scrollable for keyboard-only users to be able to focus & scroll
                        // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
                        tabIndex: scrollable ? 0 : -1,
                        onScroll: onScroll,
                        children: [
                            children,
                            empty && /*#__PURE__*/ jsx("div", {
                                css: {
                                    padding: theme.spacing.lg
                                },
                                children: empty
                            })
                        ]
                    }),
                    !empty && pagination && /*#__PURE__*/ jsx("div", {
                        css: tableStyles.paginationContainer,
                        children: pagination
                    })
                ]
            })
        })
    });
});

const TableCell = /*#__PURE__*/ forwardRef(function({ children, className, ellipsis = false, multiline = false, align = 'left', style, wrapContent = true, ...rest }, ref) {
    const { size, grid } = useContext(TableContext);
    const { classNamePrefix } = useDesignSystemTheme();
    const linkWithEllipsisCss = /*#__PURE__*/ css({
        [`& > .${classNamePrefix}-typography > a.${classNamePrefix}-typography-ellipsis`]: {
            display: 'block'
        }
    });
    let typographySize = 'md';
    if (size === 'small') {
        typographySize = 'sm';
    }
    const content = wrapContent === true ? /*#__PURE__*/ jsx(Typography.Text, {
        ellipsis: !multiline,
        size: typographySize,
        title: !multiline && typeof children === 'string' && children || undefined,
        // Needed for the button focus outline to be visible for the expand/collapse buttons
        css: {
            '&:has(> button)': {
                overflow: 'visible'
            }
        },
        children: children
    }) : children;
    return /*#__PURE__*/ jsx("div", {
        ...rest,
        role: "cell",
        style: {
            textAlign: align,
            ...style
        },
        ref: ref,
        // PE-259 Use more performance className for grid but keep css= for compatibility.
        css: [
            !grid ? repeatingElementsStyles.cell : undefined,
            linkWithEllipsisCss
        ],
        className: classnames(grid && tableClassNames.cell, className),
        children: content
    });
});

const TableRowContext = /*#__PURE__*/ createContext({
    isHeader: false
});
const TableRow = /*#__PURE__*/ forwardRef(function TableRow({ children, className, style, isHeader = false, skipIconHiding = false, verticalAlignment, ...rest }, ref) {
    const { size, grid } = useContext(TableContext);
    const { theme } = useDesignSystemTheme();
    // Vertical only be larger if the row is a header AND size is large.
    const shouldUseLargeVerticalPadding = isHeader && size === 'default';
    let rowPadding;
    if (shouldUseLargeVerticalPadding) {
        rowPadding = theme.spacing.sm;
    } else if (size === 'default') {
        rowPadding = 6;
    } else {
        rowPadding = theme.spacing.xs;
    }
    return /*#__PURE__*/ jsx(TableRowContext.Provider, {
        value: useMemo(()=>{
            return {
                isHeader
            };
        }, [
            isHeader
        ]),
        children: /*#__PURE__*/ jsx("div", {
            ...rest,
            ref: ref,
            role: "row",
            style: {
                ...style,
                ['--table-row-vertical-padding']: `${rowPadding}px`
            },
            // PE-259 Use more performance className for grid but keep css= for consistency.
            css: [
                !isHeader && !skipIconHiding && hideIconButtonRowStyles,
                !grid && repeatingElementsStyles.row
            ],
            className: classnames(className, grid && tableClassNames.row, {
                'table-isHeader': isHeader,
                'table-row-isGrid': grid
            }),
            children: children
        })
    });
});

const TableRowActionStyles = {
    container: /*#__PURE__*/ css({
        width: 32,
        paddingTop: 'var(--vertical-padding)',
        paddingBottom: 'var(--vertical-padding)',
        display: 'flex',
        alignItems: 'start',
        justifyContent: 'center'
    })
};
const TableRowAction = /*#__PURE__*/ forwardRef(function TableRowAction({ children, style, className, ...rest }, ref) {
    const { size } = useContext(TableContext);
    const { isHeader } = useContext(TableRowContext);
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        ...rest,
        ref: ref,
        role: isHeader ? 'columnheader' : 'cell',
        style: {
            ...style,
            ['--vertical-padding']: size === 'default' ? `${theme.spacing.xs}px` : 0
        },
        css: TableRowActionStyles.container,
        className: classnames(className, !isHeader && hideIconButtonActionCellClassName),
        children: children
    });
});
/** @deprecated Use `TableRowAction` instead */ const TableRowMenuContainer = TableRowAction;

const TableSkeletonStyles = {
    container: /*#__PURE__*/ css({
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start'
    }),
    cell: /*#__PURE__*/ css({
        width: '100%',
        height: 8,
        borderRadius: 4,
        background: 'var(--table-skeleton-color)',
        marginTop: 'var(--table-skeleton-row-vertical-margin)',
        marginBottom: 'var(--table-skeleton-row-vertical-margin)'
    })
};
const TableSkeleton = ({ lines = 1, seed = '', frameRate = 60, style, label, ...rest })=>{
    const { theme } = useDesignSystemTheme();
    const { size } = useContext(TableContext);
    const widths = getOffsets(seed);
    return /*#__PURE__*/ jsxs("div", {
        ...rest,
        ...addDebugOutlineIfEnabled(),
        "aria-busy": true,
        css: TableSkeletonStyles.container,
        role: "status",
        style: {
            ...style,
            // TODO: Pull this from the themes; it's not currently available.
            ['--table-skeleton-color']: theme.isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(31, 38, 45, 0.1)',
            ['--table-skeleton-row-vertical-margin']: size === 'small' ? '4px' : '6px'
        },
        children: [
            [
                ...Array(lines)
            ].map((_, idx)=>/*#__PURE__*/ jsx("div", {
                    css: [
                        TableSkeletonStyles.cell,
                        genSkeletonAnimatedColor(theme, frameRate),
                        {
                            width: `calc(100% - ${widths[idx % widths.length]}px)`
                        }
                    ]
                }, idx)),
            /*#__PURE__*/ jsx("span", {
                css: visuallyHidden,
                children: label
            })
        ]
    });
};
const TableSkeletonRows = ({ table, actionColumnIds = [], numRows = 3, loading = true, loadingDescription = 'Table skeleton rows', label })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsxs(Fragment, {
        children: [
            loading && /*#__PURE__*/ jsx(LoadingState, {
                description: loadingDescription
            }),
            /*#__PURE__*/ jsx("span", {
                css: visuallyHidden,
                children: label
            }),
            [
                ...Array(numRows).keys()
            ].map((i)=>/*#__PURE__*/ jsx(TableRow, {
                    children: table.getFlatHeaders().map((header)=>{
                        const meta = header.column.columnDef.meta;
                        return actionColumnIds.includes(header.id) ? /*#__PURE__*/ jsx(TableRowAction, {
                            children: /*#__PURE__*/ jsx(TableSkeleton, {
                                style: {
                                    width: theme.general.iconSize
                                }
                            })
                        }, `cell-${header.id}-${i}`) : /*#__PURE__*/ jsx(TableCell, {
                            style: meta?.styles ?? (meta?.width !== undefined ? {
                                maxWidth: meta.width
                            } : {}),
                            children: /*#__PURE__*/ jsx(TableSkeleton, {
                                seed: `skeleton-${header.id}-${i}`,
                                lines: meta?.numSkeletonLines ?? undefined
                            })
                        }, `cell-${header.id}-${i}`);
                    })
                }, i))
        ]
    });
};

// Loading state requires a width since it'll have no content
const LOADING_STATE_DEFAULT_WIDTH = 300;
function getStyles$1(args) {
    const { theme, width, hasTopBar, hasBottomBar, isInteractive, useNewBorderRadii } = args;
    const hoverOrFocusStyle = isInteractive ? {
        border: `1px solid ${theme.colors.actionDefaultBorderHover}`,
        boxShadow: theme.shadows.md
    } : {};
    return /*#__PURE__*/ css({
        color: theme.colors.textPrimary,
        backgroundColor: theme.colors.backgroundPrimary,
        position: 'relative',
        display: 'flex',
        justifyContent: 'flex-start',
        flexDirection: 'column',
        paddingRight: hasTopBar || hasBottomBar ? 0 : theme.spacing.md,
        paddingLeft: hasTopBar || hasBottomBar ? 0 : theme.spacing.md,
        paddingTop: hasTopBar ? 0 : theme.spacing.md,
        paddingBottom: hasBottomBar ? 0 : theme.spacing.md,
        width: width ?? 'fit-content',
        borderRadius: useNewBorderRadii ? theme.borders.borderRadiusMd : theme.legacyBorders.borderRadiusMd,
        borderColor: theme.colors.border,
        borderWidth: '1px',
        borderStyle: 'solid',
        '&:hover': hoverOrFocusStyle,
        '&:focus': hoverOrFocusStyle,
        cursor: !isInteractive ? 'default' : 'pointer',
        boxShadow: theme.shadows.sm,
        transition: `box-shadow 0.2s ease-in-out`,
        textDecoration: 'none !important',
        ...getAnimationCss(theme.options.enableAnimation)
    });
}
function getBottomBarStyles(theme, useNewBorderRadii) {
    return /*#__PURE__*/ css({
        marginTop: theme.spacing.sm,
        borderBottomRightRadius: useNewBorderRadii ? theme.borders.borderRadiusSm : theme.legacyBorders.borderRadiusMd,
        borderBottomLeftRadius: useNewBorderRadii ? theme.borders.borderRadiusSm : theme.legacyBorders.borderRadiusMd,
        overflow: 'hidden'
    });
}
function getTopBarStyles(theme, useNewBorderRadii) {
    return /*#__PURE__*/ css({
        marginBottom: theme.spacing.sm,
        borderTopRightRadius: useNewBorderRadii ? theme.borders.borderRadiusSm : theme.legacyBorders.borderRadiusMd,
        borderTopLeftRadius: useNewBorderRadii ? theme.borders.borderRadiusSm : theme.legacyBorders.borderRadiusMd,
        overflow: 'hidden'
    });
}
const Card = ({ children, customLoadingContent, dangerouslyAppendEmotionCSS, loading, loadingDescription, width, bottomBarContent, topBarContent, disableHover, onClick, href, navigateFn, anchorProps, componentId, analyticsEvents, shouldStartInteraction, ...dataAndAttributes })=>{
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.card', false);
    const { theme } = useDesignSystemTheme();
    const useNewBorderRadii = safex('databricks.fe.designsystem.useNewBorderRadii', false);
    const hasTopBar = !isUndefined(topBarContent);
    const hasBottomBar = !isUndefined(bottomBarContent);
    const hasHref = Boolean(href);
    const hasOnClick = Boolean(onClick);
    const hasNavigateFn = Boolean(navigateFn);
    const isInteractive = !disableHover && !loading && (hasHref || hasOnClick || hasNavigateFn);
    const cardStyle = /*#__PURE__*/ css(getStyles$1({
        theme,
        width,
        hasBottomBar,
        hasTopBar,
        isInteractive,
        useNewBorderRadii
    }));
    const ref = React__default.useRef(null);
    const bottomBar = bottomBarContent ? /*#__PURE__*/ jsx("div", {
        css: /*#__PURE__*/ css(getBottomBarStyles(theme, useNewBorderRadii)),
        children: bottomBarContent
    }) : null;
    const topBar = topBarContent ? /*#__PURE__*/ jsx("div", {
        css: /*#__PURE__*/ css(getTopBarStyles(theme, useNewBorderRadii)),
        children: topBarContent
    }) : null;
    const contentPadding = hasTopBar || hasBottomBar ? theme.spacing.lg : 0;
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnClick,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnClick
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Card,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        shouldStartInteraction
    });
    const { elementRef: cardRef } = useNotifyOnFirstView({
        onView: eventContext.onView
    });
    const mergedRef = useMergeRefs([
        ref,
        cardRef
    ]);
    const navigate = useCallback(async ()=>{
        if (navigateFn) {
            await navigateFn();
        }
    }, [
        navigateFn
    ]);
    const handleClick = useCallback(async (e)=>{
        eventContext.onClick(e);
        await navigate();
        onClick?.(e);
        ref.current?.blur();
    }, [
        navigate,
        eventContext,
        onClick
    ]);
    const handleSelection = useCallback(async (e)=>{
        eventContext.onClick(e);
        e.preventDefault();
        await navigate();
        onClick?.(e);
    }, [
        navigate,
        eventContext,
        onClick
    ]);
    const content = /*#__PURE__*/ jsx("div", {
        ref: mergedRef,
        // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
        tabIndex: !href ? 0 : undefined,
        ...addDebugOutlineIfEnabled(),
        css: href ? [] : [
            cardStyle,
            dangerouslyAppendEmotionCSS
        ],
        onClick: loading || href ? undefined : handleClick,
        ...dataAndAttributes,
        onKeyDown: async (e)=>{
            const isEnter = e.key === 'Enter';
            const isSpace = e.key === ' ';
            const isCardFocused = e.target === e.currentTarget; // ensures the element itself is focused
            if ((isEnter || isSpace) && isCardFocused) {
                await handleSelection(e);
            }
            dataAndAttributes.onKeyDown?.(e);
        },
        ...eventContext.dataComponentProps,
        children: loading ? /*#__PURE__*/ jsx(DefaultCardLoadingContent, {
            width: width,
            customLoadingContent: customLoadingContent,
            loadingDescription: loadingDescription
        }) : /*#__PURE__*/ jsxs(Fragment, {
            children: [
                topBar,
                /*#__PURE__*/ jsx("div", {
                    css: {
                        padding: `0px ${contentPadding}px`,
                        flexGrow: 1
                    },
                    children: children
                }),
                bottomBar
            ]
        })
    });
    return href ? /*#__PURE__*/ jsx("a", {
        css: [
            cardStyle,
            dangerouslyAppendEmotionCSS
        ],
        href: href,
        ...anchorProps,
        children: content
    }) : content;
};
function DefaultCardLoadingContent({ customLoadingContent, width, loadingDescription }) {
    if (customLoadingContent) {
        return /*#__PURE__*/ jsx(Fragment, {
            children: customLoadingContent
        });
    }
    return /*#__PURE__*/ jsxs("div", {
        css: {
            width: width ?? LOADING_STATE_DEFAULT_WIDTH
        },
        children: [
            /*#__PURE__*/ jsx(TitleSkeleton, {
                label: "Loading...",
                style: {
                    width: '50%'
                },
                loadingDescription: loadingDescription
            }),
            [
                ...Array(3).keys()
            ].map((i)=>/*#__PURE__*/ jsx(ParagraphSkeleton, {
                    label: "Loading..."
                }, i))
        ]
    });
}

function getCheckboxEmotionStyles(clsPrefix, theme, isHorizontal = false, useNewFormUISpacing, useNewBorderRadii) {
    const classInput = `.${clsPrefix}-input`;
    const classInner = `.${clsPrefix}-inner`;
    const classIndeterminate = `.${clsPrefix}-indeterminate`;
    const classChecked = `.${clsPrefix}-checked`;
    const classDisabled = `.${clsPrefix}-disabled`;
    const classDisabledWrapper = `.${clsPrefix}-wrapper-disabled`;
    const classContainer = `.${clsPrefix}-group`;
    const classWrapper = `.${clsPrefix}-wrapper`;
    const defaultSelector = `${classInput} + ${classInner}`;
    const hoverSelector = `${classInput}:hover + ${classInner}`;
    const pressSelector = `${classInput}:active + ${classInner}`;
    const cleanClsPrefix = `.${clsPrefix.replace('-checkbox', '')}`;
    const styles = {
        [`.${clsPrefix}`]: {
            top: 'unset',
            lineHeight: theme.typography.lineHeightBase,
            alignSelf: 'flex-start',
            display: 'flex',
            alignItems: 'center',
            height: theme.typography.lineHeightBase
        },
        [`&${classWrapper}, ${classWrapper}`]: {
            alignItems: 'center',
            lineHeight: theme.typography.lineHeightBase
        },
        // Top level styles are for the unchecked state
        [classInner]: {
            borderColor: theme.colors.actionDefaultBorderDefault,
            ...useNewBorderRadii && {
                borderRadius: theme.borders.borderRadiusSm
            }
        },
        // Style wrapper span added by Antd
        [`&> span:not(.${clsPrefix})`]: {
            display: 'inline-flex',
            alignItems: 'center'
        },
        // Layout styling
        [`&${classContainer}`]: {
            display: 'flex',
            flexDirection: 'column',
            rowGap: theme.spacing.sm,
            columnGap: 0,
            ...useNewFormUISpacing && {
                [`& + ${cleanClsPrefix}-form-message`]: {
                    marginTop: theme.spacing.sm
                }
            }
        },
        ...useNewFormUISpacing && {
            [`${cleanClsPrefix}-hint + &${classContainer}`]: {
                marginTop: theme.spacing.sm
            }
        },
        ...isHorizontal && {
            [`&${classContainer}`]: {
                display: 'flex',
                flexDirection: 'row',
                columnGap: theme.spacing.sm,
                rowGap: 0,
                [`& > ${classContainer}-item`]: {
                    marginRight: 0
                }
            }
        },
        // Keyboard focus
        [`${classInput}:focus-visible + ${classInner}`]: {
            outlineWidth: '2px',
            outlineColor: theme.colors.actionDefaultBorderFocus,
            outlineOffset: '4px',
            outlineStyle: 'solid'
        },
        // Hover
        [hoverSelector]: {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionPrimaryBackgroundHover
        },
        // Mouse pressed
        [pressSelector]: {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
            borderColor: theme.colors.actionPrimaryBackgroundPress
        },
        // Checked state
        [classChecked]: {
            [classInner]: {
                boxShadow: theme.shadows.xs
            },
            '&::after': {
                border: 'none'
            },
            [defaultSelector]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                borderColor: 'transparent'
            },
            // Checked hover
            [hoverSelector]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundHover,
                borderColor: 'transparent'
            },
            // Checked and mouse pressed
            [pressSelector]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundPress,
                borderColor: 'transparent'
            }
        },
        // Indeterminate
        [classIndeterminate]: {
            [classInner]: {
                boxShadow: theme.shadows.xs,
                backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                borderColor: theme.colors.actionPrimaryBackgroundDefault,
                // The after pseudo-element is used for the check image itself
                '&:after': {
                    backgroundColor: theme.colors.white,
                    height: '3px',
                    width: '8px',
                    borderRadius: '4px'
                }
            },
            // Indeterminate hover
            [hoverSelector]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundHover,
                borderColor: 'transparent'
            },
            // Indeterminate and mouse pressed
            [pressSelector]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundPress
            }
        },
        // Disabled state
        [`&${classDisabledWrapper}`]: {
            [classDisabled]: {
                // Disabled Checked
                [`&${classChecked}`]: {
                    [classInner]: {
                        backgroundColor: theme.colors.actionDisabledBackground,
                        borderColor: theme.colors.actionDisabledBorder,
                        '&:after': {
                            borderColor: theme.colors.actionDisabledText
                        }
                    },
                    // Disabled checked hover
                    [hoverSelector]: {
                        backgroundColor: theme.colors.actionDisabledBackground,
                        borderColor: theme.colors.actionDisabledBorder
                    }
                },
                // Disabled indeterminate
                [`&${classIndeterminate}`]: {
                    [classInner]: {
                        backgroundColor: theme.colors.actionDisabledBackground,
                        borderColor: theme.colors.actionDisabledBorder,
                        '&:after': {
                            borderColor: theme.colors.actionDisabledText,
                            backgroundColor: theme.colors.actionDisabledText
                        }
                    },
                    // Disabled indeterminate hover
                    [hoverSelector]: {
                        backgroundColor: theme.colors.actionDisabledBackground,
                        borderColor: theme.colors.actionDisabledBorder
                    }
                },
                // Disabled unchecked
                [classInner]: {
                    backgroundColor: theme.colors.actionDisabledBackground,
                    borderColor: theme.colors.actionDisabledBorder,
                    // The after pseudo-element is used for the check image itself
                    '&:after': {
                        borderColor: 'transparent'
                    }
                },
                // Disabled hover
                [hoverSelector]: {
                    backgroundColor: theme.colors.actionDisabledBackground,
                    borderColor: theme.colors.actionDisabledBorder
                },
                '& + span': {
                    color: theme.colors.actionDisabledText
                }
            }
        },
        // Animation
        ...getAnimationCss(theme.options.enableAnimation)
    };
    return styles;
}
const getWrapperStyle = ({ clsPrefix, theme, wrapperStyle = {}, useNewFormUISpacing })=>{
    const extraSelector = useNewFormUISpacing ? `, && + .${clsPrefix}-hint + .${clsPrefix}-form-message` : '';
    const styles = {
        height: theme.typography.lineHeightBase,
        lineHeight: theme.typography.lineHeightBase,
        [`&& + .${clsPrefix}-hint, && + .${clsPrefix}-form-message${extraSelector}`]: {
            paddingLeft: theme.spacing.lg,
            marginTop: 0
        },
        ...wrapperStyle
    };
    return /*#__PURE__*/ css(styles);
};
const DuboisCheckbox = /*#__PURE__*/ forwardRef(function Checkbox({ isChecked, onChange, children, isDisabled = false, style, wrapperStyle, dangerouslySetAntdProps, className, componentId, analyticsEvents, ...restProps }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.checkbox', false);
    const { theme, classNamePrefix, getPrefixedClassName } = useDesignSystemTheme();
    const useNewFormUISpacing = safex('databricks.fe.designsystem.useNewFormUISpacing', false);
    const useNewBorderRadii = safex('databricks.fe.designsystem.useNewBorderRadii', false);
    const clsPrefix = getPrefixedClassName('checkbox');
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Checkbox,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true
    });
    const { elementRef: checkboxRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: isChecked ?? restProps.defaultChecked
    });
    const onChangeHandler = (event)=>{
        eventContext.onValueChange(event.target.checked);
        onChange?.(event.target.checked, event);
    };
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx("div", {
            ...addDebugOutlineIfEnabled(),
            className: classnames(className, `${clsPrefix}-container`),
            css: getWrapperStyle({
                clsPrefix: classNamePrefix,
                theme,
                wrapperStyle,
                useNewFormUISpacing
            }),
            ref: checkboxRef,
            children: /*#__PURE__*/ jsx(Checkbox$1, {
                checked: isChecked === null ? undefined : isChecked,
                ref: ref,
                onChange: onChangeHandler,
                disabled: isDisabled,
                indeterminate: isChecked === null,
                // Individual checkboxes don't depend on isHorizontal flag, orientation and spacing is handled by end users
                css: /*#__PURE__*/ css(importantify(getCheckboxEmotionStyles(clsPrefix, theme, false, useNewFormUISpacing, useNewBorderRadii))),
                style: style,
                "aria-checked": isChecked === null ? 'mixed' : isChecked,
                ...restProps,
                ...dangerouslySetAntdProps,
                ...eventContext.dataComponentProps,
                children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                    children: children
                })
            })
        })
    });
});
const CheckboxGroup = /*#__PURE__*/ forwardRef(function CheckboxGroup({ children, layout = 'vertical', ...props }, ref) {
    const { theme, getPrefixedClassName } = useDesignSystemTheme();
    const clsPrefix = getPrefixedClassName('checkbox');
    const useNewFormUISpacing = safex('databricks.fe.designsystem.useNewFormUISpacing', false);
    const useNewBorderRadii = safex('databricks.fe.designsystem.useNewBorderRadii', false);
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Checkbox$1.Group, {
            ...addDebugOutlineIfEnabled(),
            ref: ref,
            ...props,
            css: getCheckboxEmotionStyles(clsPrefix, theme, layout === 'horizontal', useNewFormUISpacing, useNewBorderRadii),
            children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                children: children
            })
        })
    });
});
const CheckboxNamespace = /* #__PURE__ */ Object.assign(DuboisCheckbox, {
    Group: CheckboxGroup
});
const Checkbox = CheckboxNamespace;
// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
const __INTERNAL_DO_NOT_USE__Group = CheckboxGroup;

const getAllMenuItemsInContainer = (container)=>{
    return container.querySelectorAll('[role^="menuitem"]');
};
const focusNextItem = (e)=>{
    const container = e.currentTarget.closest('[role="menu"]');
    if (!container) {
        return;
    }
    const menuItems = getAllMenuItemsInContainer(container);
    const activeElement = document.activeElement;
    const shouldNavigateUp = e.key === 'ArrowUp' || e.key === 'Tab' && e.shiftKey;
    const activeIndex = Array.from(menuItems).findIndex((item)=>item === activeElement);
    let nextIndex = shouldNavigateUp ? activeIndex - 1 : activeIndex + 1;
    if (nextIndex < 0 || nextIndex >= menuItems.length) {
        nextIndex = shouldNavigateUp ? menuItems.length - 1 : 0;
    }
    const nextItem = menuItems[nextIndex];
    if (nextItem) {
        const isDisabled = nextItem.hasAttribute('data-disabled');
        if (isDisabled) {
            const tooltip = nextItem.querySelector('[data-disabled-tooltip]');
            tooltip?.setAttribute('tabindex', '0');
            if (tooltip) {
                e.preventDefault();
                tooltip.focus();
            }
        } else {
            nextItem.focus();
            nextItem.setAttribute('data-highlighted', 'true');
        }
    }
};
const blurTooltipAndFocusNextItem = (e)=>{
    const tooltip = document.activeElement;
    const parentItem = tooltip.closest('[role^="menuitem"]');
    const container = tooltip.closest('[role="menu"]');
    if (!container) {
        return;
    }
    const menuItems = getAllMenuItemsInContainer(container);
    const activeIndex = Array.from(menuItems).findIndex((item)=>item === parentItem);
    const shouldNavigateUp = e.key === 'ArrowUp' || e.key === 'Tab' && e.shiftKey;
    let nextIndex = shouldNavigateUp ? activeIndex - 1 : activeIndex + 1;
    if (nextIndex < 0 || nextIndex >= menuItems.length) {
        nextIndex = shouldNavigateUp ? menuItems.length - 1 : 0;
    }
    const nextItem = menuItems[nextIndex];
    if (nextItem) {
        tooltip.removeAttribute('tabindex');
        tooltip.blur();
        const isDisabled = nextItem.hasAttribute('data-disabled');
        if (isDisabled) {
            const tooltip = nextItem.querySelector('[data-disabled-tooltip]');
            tooltip?.setAttribute('tabindex', '0');
            if (tooltip) {
                e.preventDefault();
                tooltip.focus();
            }
        } else {
            nextItem.focus();
        }
    }
};
const handleKeyboardNavigation = (e)=>{
    const isItemFocused = document.activeElement?.getAttribute('role') === 'menuitem' || document.activeElement?.getAttribute('role') === 'menuitemcheckbox' || document.activeElement?.getAttribute('role') === 'menuitemradio';
    const isTooltipFocused = document.activeElement?.hasAttribute('data-disabled-tooltip');
    if (isItemFocused || !isTooltipFocused) {
        focusNextItem(e);
    } else {
        blurTooltipAndFocusNextItem(e);
    }
};

const infoIconStyles = (theme)=>({
        display: 'inline-flex',
        paddingLeft: theme.spacing.xs,
        color: theme.colors.textSecondary,
        pointerEvents: 'all'
    });
const getNewChildren = (children, props, disabledReason, ref)=>{
    const childCount = Children.count(children);
    const tooltip = /*#__PURE__*/ jsx(LegacyTooltip, {
        title: disabledReason,
        placement: "right",
        dangerouslySetAntdProps: {
            getPopupContainer: ()=>ref.current || document.body
        },
        children: /*#__PURE__*/ jsx("span", {
            "data-disabled-tooltip": true,
            css: (theme)=>infoIconStyles(theme),
            onClick: (e)=>{
                if (props.disabled) {
                    e.stopPropagation();
                }
            },
            children: /*#__PURE__*/ jsx(InfoSmallIcon, {
                role: "presentation",
                alt: "Disabled state reason",
                "aria-hidden": "false"
            })
        })
    });
    if (childCount === 1) {
        return getChild(children, Boolean(props['disabled']), disabledReason, tooltip, 0, childCount);
    }
    return Children.map(children, (child, idx)=>{
        return getChild(child, Boolean(props['disabled']), disabledReason, tooltip, idx, childCount);
    });
};
const getChild = (child, isDisabled, disabledReason, tooltip, index, siblingCount)=>{
    const HintColumnType = /*#__PURE__*/ jsx(HintColumn, {}).type;
    const isHintColumnType = Boolean(child && typeof child !== 'string' && typeof child !== 'number' && typeof child !== 'boolean' && 'type' in child && child?.type === HintColumnType);
    if (isDisabled && disabledReason && child && isHintColumnType) {
        return /*#__PURE__*/ jsxs(Fragment, {
            children: [
                tooltip,
                child
            ]
        });
    } else if (index === siblingCount - 1 && isDisabled && disabledReason) {
        return /*#__PURE__*/ jsxs(Fragment, {
            children: [
                child,
                tooltip
            ]
        });
    }
    return child;
};

const DropdownContext = /*#__PURE__*/ createContext({
    isOpen: false,
    setIsOpen: (isOpen)=>{}
});
const useDropdownContext = ()=>React__default.useContext(DropdownContext);
const Root$7 = ({ children, itemHtmlType, ...props })=>{
    const [isOpen, setIsOpen] = React__default.useState(Boolean(props.defaultOpen || props.open));
    const useExternalState = useRef(props.open !== undefined || props.onOpenChange !== undefined).current;
    useEffect(()=>{
        if (useExternalState) {
            setIsOpen(Boolean(props.open));
        }
    }, [
        useExternalState,
        props.open
    ]);
    const handleOpenChange = (isOpen)=>{
        if (!useExternalState) {
            setIsOpen(isOpen);
        }
        // In case the consumer doesn't manage open state but wants to listen to the callback
        if (props.onOpenChange) {
            props.onOpenChange(isOpen);
        }
    };
    return /*#__PURE__*/ jsx(DropdownMenu$1.Root, {
        ...props,
        ...!useExternalState && {
            open: isOpen,
            onOpenChange: handleOpenChange
        },
        children: /*#__PURE__*/ jsx(DropdownContext.Provider, {
            value: {
                isOpen: useExternalState ? props.open : isOpen,
                setIsOpen: useExternalState ? props.onOpenChange : handleOpenChange,
                itemHtmlType
            },
            children: children
        })
    });
};
const Content$4 = /*#__PURE__*/ forwardRef(function Content({ children, minWidth = 220, matchTriggerWidth, forceCloseOnEscape, onEscapeKeyDown, onKeyDown, ...props }, ref) {
    const { getPopupContainer } = useDesignSystemContext();
    const { theme } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const { setIsOpen } = useDropdownContext();
    const { isInsideModal } = useModalContext();
    return /*#__PURE__*/ jsx(DropdownMenu$1.Portal, {
        container: getPopupContainer && getPopupContainer(),
        children: /*#__PURE__*/ jsx(DropdownMenu$1.Content, {
            ...addDebugOutlineIfEnabled(),
            ref: ref,
            loop: true,
            css: [
                contentStyles(theme, useNewBorderColors),
                {
                    minWidth
                },
                matchTriggerWidth ? {
                    width: 'var(--radix-dropdown-menu-trigger-width)'
                } : {}
            ],
            sideOffset: 4,
            align: "start",
            onKeyDown: (e)=>{
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
            },
            ...props,
            onWheel: (e)=>{
                e.stopPropagation();
                props?.onWheel?.(e);
            },
            onTouchMove: (e)=>{
                e.stopPropagation();
                props?.onTouchMove?.(e);
            },
            children: children
        })
    });
});
const SubContent$1 = /*#__PURE__*/ forwardRef(function Content({ children, minWidth = 220, onKeyDown, ...props }, ref) {
    const { getPopupContainer } = useDesignSystemContext();
    const { theme } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const [contentFitsInViewport, setContentFitsInViewport] = React__default.useState(true);
    const [dataSide, setDataSide] = React__default.useState(null);
    const { isOpen } = useSubContext();
    const elemRef = useRef(null);
    useImperativeHandle(ref, ()=>elemRef.current);
    const checkAvailableWidth = useCallback(()=>{
        if (elemRef.current) {
            const elemStyle = getComputedStyle(elemRef.current);
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
    useEffect(()=>{
        window.addEventListener('resize', checkAvailableWidth);
        checkAvailableWidth();
        return ()=>{
            window.removeEventListener('resize', checkAvailableWidth);
        };
    }, [
        checkAvailableWidth
    ]);
    useEffect(()=>{
        if (isOpen) {
            setTimeout(()=>{
                checkAvailableWidth();
            }, 25);
        }
    }, [
        isOpen,
        checkAvailableWidth
    ]);
    let transformCalc = `calc(var(--radix-dropdown-menu-content-available-width) + var(--radix-dropdown-menu-trigger-width) * -1)`;
    if (dataSide === 'left') {
        transformCalc = `calc(var(--radix-dropdown-menu-trigger-width) - var(--radix-dropdown-menu-content-available-width))`;
    }
    const responsiveCss = `
    transform-origin: var(--radix-dropdown-menu-content-transform-origin) !important;
    transform: translateX(${transformCalc}) !important;
`;
    return /*#__PURE__*/ jsx(DropdownMenu$1.Portal, {
        container: getPopupContainer && getPopupContainer(),
        children: /*#__PURE__*/ jsx(DropdownMenu$1.SubContent, {
            ...addDebugOutlineIfEnabled(),
            ref: elemRef,
            loop: true,
            css: [
                contentStyles(theme, useNewBorderColors),
                {
                    minWidth
                },
                contentFitsInViewport ? '' : responsiveCss
            ],
            sideOffset: -2,
            alignOffset: -5,
            onKeyDown: (e)=>{
                if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                    e.stopPropagation();
                    handleKeyboardNavigation(e);
                }
                onKeyDown?.(e);
            },
            ...props,
            children: children
        })
    });
});
const Trigger$3 = /*#__PURE__*/ forwardRef(function Trigger({ children, ...props }, ref) {
    return /*#__PURE__*/ jsx(DropdownMenu$1.Trigger, {
        ...addDebugOutlineIfEnabled(),
        ref: ref,
        ...props,
        children: children
    });
});
const Item$2 = /*#__PURE__*/ forwardRef(function Item({ children, disabledReason, danger, onClick, componentId, analyticsEvents, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.dropdownMenu', false);
    const formContext = useFormContext();
    const { itemHtmlType } = useDropdownContext();
    const itemRef = useRef(null);
    useImperativeHandle(ref, ()=>itemRef.current);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnClick,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnClick
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.DropdownMenuItem,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        // If the item is a submit item and is part of a form, it is not the subject of the interaction, the form submission is
        isInteractionSubject: !(itemHtmlType === 'submit' && formContext.componentId)
    });
    const { elementRef: dropdownMenuItemRef } = useNotifyOnFirstView({
        onView: !props.asChild ? eventContext.onView : ()=>{}
    });
    const mergedRefs = useMergeRefs([
        itemRef,
        dropdownMenuItemRef
    ]);
    return /*#__PURE__*/ jsx(DropdownMenu$1.Item, {
        css: (theme)=>[
                dropdownItemStyles,
                danger && dangerItemStyles(theme)
            ],
        ref: mergedRefs,
        onClick: (e)=>{
            if (props.disabled) {
                e.preventDefault();
            } else {
                if (!props.asChild) {
                    eventContext.onClick(e);
                }
                if (itemHtmlType === 'submit' && formContext.formRef?.current) {
                    e.preventDefault();
                    formContext.formRef.current.requestSubmit();
                }
                onClick?.(e);
            }
        },
        onKeyDown: (e)=>{
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        },
        ...props,
        ...eventContext.dataComponentProps,
        children: getNewChildren(children, props, disabledReason, itemRef)
    });
});
const Label$2 = /*#__PURE__*/ forwardRef(function Label({ children, ...props }, ref) {
    return /*#__PURE__*/ jsx(DropdownMenu$1.Label, {
        ref: ref,
        css: [
            dropdownItemStyles,
            (theme)=>({
                    color: theme.colors.textSecondary,
                    '&:hover': {
                        cursor: 'default'
                    }
                })
        ],
        ...props,
        children: children
    });
});
const Separator$1 = /*#__PURE__*/ forwardRef(function Separator({ children, ...props }, ref) {
    return /*#__PURE__*/ jsx(DropdownMenu$1.Separator, {
        ref: ref,
        css: dropdownSeparatorStyles,
        ...props,
        children: children
    });
});
const SubTrigger$1 = /*#__PURE__*/ forwardRef(function TriggerItem({ children, disabledReason, ...props }, ref) {
    const subTriggerRef = useRef(null);
    useImperativeHandle(ref, ()=>subTriggerRef.current);
    return /*#__PURE__*/ jsxs(DropdownMenu$1.SubTrigger, {
        ref: subTriggerRef,
        css: [
            dropdownItemStyles,
            (theme)=>({
                    '&[data-state="open"]': {
                        backgroundColor: theme.colors.actionTertiaryBackgroundHover
                    }
                })
        ],
        onKeyDown: (e)=>{
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        },
        ...props,
        children: [
            getNewChildren(children, props, disabledReason, subTriggerRef),
            /*#__PURE__*/ jsx(HintColumn, {
                css: (theme)=>({
                        margin: CONSTANTS$1.subMenuIconMargin(theme),
                        display: 'flex',
                        alignSelf: 'stretch',
                        alignItems: 'center'
                    }),
                children: /*#__PURE__*/ jsx(ChevronRightIcon, {
                    css: (theme)=>({
                            fontSize: CONSTANTS$1.subMenuIconSize(theme)
                        })
                })
            })
        ]
    });
});
/**
 * Deprecated. Use `SubTrigger` instead.
 * @deprecated
 */ const TriggerItem = SubTrigger$1;
const CheckboxItem$1 = /*#__PURE__*/ forwardRef(function CheckboxItem({ children, disabledReason, componentId, analyticsEvents, onCheckedChange, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.dropdownMenu', false);
    const checkboxItemRef = useRef(null);
    useImperativeHandle(ref, ()=>checkboxItemRef.current);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.DropdownMenuCheckboxItem,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true
    });
    const { elementRef: checkboxItemOnViewRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.checked ?? props.defaultChecked
    });
    const mergedRefs = useMergeRefs([
        checkboxItemRef,
        checkboxItemOnViewRef
    ]);
    const onCheckedChangeWrapper = useCallback((checked)=>{
        eventContext.onValueChange(checked);
        onCheckedChange?.(checked);
    }, [
        eventContext,
        onCheckedChange
    ]);
    return /*#__PURE__*/ jsx(DropdownMenu$1.CheckboxItem, {
        ref: mergedRefs,
        css: (theme)=>[
                dropdownItemStyles,
                checkboxItemStyles(theme)
            ],
        onCheckedChange: onCheckedChangeWrapper,
        onKeyDown: (e)=>{
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        },
        ...props,
        ...eventContext.dataComponentProps,
        children: getNewChildren(children, props, disabledReason, checkboxItemRef)
    });
});
const RadioGroup$1 = /*#__PURE__*/ forwardRef(function RadioGroup({ children, componentId, analyticsEvents, onValueChange, valueHasNoPii, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.dropdownMenu', false);
    const radioGroupItemRef = useRef(null);
    useImperativeHandle(ref, ()=>radioGroupItemRef.current);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.DropdownMenuRadioGroup,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    const { elementRef: radioGroupItemOnViewRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.value ?? props.defaultValue
    });
    const mergedRef = useMergeRefs([
        radioGroupItemRef,
        radioGroupItemOnViewRef
    ]);
    const onValueChangeWrapper = useCallback((value)=>{
        eventContext.onValueChange(value);
        onValueChange?.(value);
    }, [
        eventContext,
        onValueChange
    ]);
    return /*#__PURE__*/ jsx(DropdownMenu$1.RadioGroup, {
        ref: mergedRef,
        onValueChange: onValueChangeWrapper,
        ...props,
        ...eventContext.dataComponentProps,
        children: children
    });
});
const ItemIndicator$1 = /*#__PURE__*/ forwardRef(function ItemIndicator({ children, ...props }, ref) {
    return /*#__PURE__*/ jsx(DropdownMenu$1.ItemIndicator, {
        ref: ref,
        css: (theme)=>({
                marginLeft: -(CONSTANTS$1.checkboxIconWidth(theme) + CONSTANTS$1.checkboxPaddingRight(theme)),
                position: 'absolute',
                fontSize: theme.general.iconFontSize
            }),
        ...props,
        children: children ?? /*#__PURE__*/ jsx(CheckIcon, {
            css: (theme)=>({
                    color: theme.colors.textSecondary
                })
        })
    });
});
const Arrow$1 = /*#__PURE__*/ forwardRef(function Arrow({ children, ...props }, ref) {
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(DropdownMenu$1.Arrow, {
        css: {
            fill: theme.colors.backgroundPrimary,
            stroke: theme.colors.borderDecorative,
            strokeDashoffset: -CONSTANTS$1.arrowBottomLength(),
            strokeDasharray: CONSTANTS$1.arrowBottomLength() + 2 * CONSTANTS$1.arrowSide(),
            strokeWidth: CONSTANTS$1.arrowStrokeWidth(),
            // TODO: This is a temporary fix for the alignment of the Arrow;
            // Radix has changed the implementation for v1.0.0 (uses floating-ui)
            // which has new behaviors for alignment that we don't want. Generally
            // we need to fix the arrow to always be aligned to the left of the menu (with
            // offset equal to border radius)
            position: 'relative',
            top: -1
        },
        ref: ref,
        width: 12,
        height: 6,
        ...props,
        children: children
    });
});
const RadioItem$1 = /*#__PURE__*/ forwardRef(function RadioItem({ children, disabledReason, ...props }, ref) {
    const radioItemRef = useRef(null);
    useImperativeHandle(ref, ()=>radioItemRef.current);
    return /*#__PURE__*/ jsx(DropdownMenu$1.RadioItem, {
        ref: radioItemRef,
        css: (theme)=>[
                dropdownItemStyles,
                checkboxItemStyles(theme)
            ],
        ...props,
        children: getNewChildren(children, props, disabledReason, radioItemRef)
    });
});
const SubContext = /*#__PURE__*/ createContext({
    isOpen: false
});
const useSubContext = ()=>React__default.useContext(SubContext);
const Sub$1 = ({ children, onOpenChange, ...props })=>{
    const [isOpen, setIsOpen] = React__default.useState(props.defaultOpen ?? false);
    const handleOpenChange = (isOpen)=>{
        onOpenChange?.(isOpen);
        setIsOpen(isOpen);
    };
    return /*#__PURE__*/ jsx(DropdownMenu$1.Sub, {
        onOpenChange: handleOpenChange,
        ...props,
        children: /*#__PURE__*/ jsx(SubContext.Provider, {
            value: {
                isOpen
            },
            children: children
        })
    });
};
// UNWRAPPED RADIX-UI-COMPONENTS
const Group$2 = DropdownMenu$1.Group;
// EXTRA COMPONENTS
const HintColumn = /*#__PURE__*/ forwardRef(function HintColumn({ children, ...props }, ref) {
    return /*#__PURE__*/ jsx("div", {
        ref: ref,
        css: [
            metaTextStyles,
            {
                marginLeft: 'auto'
            }
        ],
        ...props,
        children: children
    });
});
const HintRow = /*#__PURE__*/ forwardRef(function HintRow({ children, ...props }, ref) {
    return /*#__PURE__*/ jsx("div", {
        ref: ref,
        css: [
            metaTextStyles,
            {
                minWidth: '100%'
            }
        ],
        ...props,
        children: children
    });
});
const IconWrapper = /*#__PURE__*/ forwardRef(function IconWrapper({ children, ...props }, ref) {
    return /*#__PURE__*/ jsx("div", {
        ref: ref,
        css: (theme)=>({
                fontSize: 16,
                color: theme.colors.textSecondary,
                paddingRight: theme.spacing.sm
            }),
        ...props,
        children: children
    });
});
// CONSTANTS
const CONSTANTS$1 = {
    itemPaddingVertical (theme) {
        // The number from the mocks is the midpoint between constants
        return 0.5 * theme.spacing.xs + 0.5 * theme.spacing.sm;
    },
    itemPaddingHorizontal (theme) {
        return theme.spacing.sm;
    },
    checkboxIconWidth (theme) {
        return theme.general.iconFontSize;
    },
    checkboxPaddingLeft (theme) {
        return theme.spacing.sm + theme.spacing.xs;
    },
    checkboxPaddingRight (theme) {
        return theme.spacing.sm;
    },
    subMenuIconMargin (theme) {
        // Negative margin so the icons can be larger without increasing the overall item height
        const iconMarginVertical = this.itemPaddingVertical(theme) / 2;
        const iconMarginRight = -this.itemPaddingVertical(theme) + theme.spacing.sm * 1.5;
        return `${-iconMarginVertical}px ${-iconMarginRight}px ${-iconMarginVertical}px auto`;
    },
    subMenuIconSize (theme) {
        return theme.spacing.lg;
    },
    arrowBottomLength () {
        // The built in arrow is a polygon: 0,0 30,0 15,10
        return 30;
    },
    arrowHeight () {
        return 10;
    },
    arrowSide () {
        return 2 * (this.arrowHeight() ** 2 * 2) ** 0.5;
    },
    arrowStrokeWidth () {
        // This is eyeballed b/c relative to the svg viewbox coordinate system
        return 2;
    }
};
const dropdownContentStyles = (theme, useNewBorderColors)=>({
        backgroundColor: theme.colors.backgroundPrimary,
        color: theme.colors.textPrimary,
        lineHeight: theme.typography.lineHeightBase,
        border: `1px solid ${useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusSm,
        padding: `${theme.spacing.xs}px 0`,
        boxShadow: theme.shadows.lg,
        userSelect: 'none',
        // Allow for scrolling within the dropdown when viewport is too small
        overflowY: 'auto',
        maxHeight: 'var(--radix-dropdown-menu-content-available-height)',
        ...getDarkModePortalStyles(theme, useNewBorderColors),
        // Ant Design uses 1000s for their zIndex space; this ensures Radix works with that, but
        // we'll likely need to be sure that all Radix components are using the same zIndex going forward.
        //
        // Additionally, there is an issue where macOS overlay scrollbars in Chrome and Safari (sometimes!)
        // overlap other elements with higher zIndex, because the scrollbars themselves have zIndex 9999,
        // so we have to use a higher value than that: https://github.com/databricks/universe/pull/232825
        zIndex: getZIndex(),
        a: importantify({
            color: theme.colors.textPrimary,
            '&:hover, &:focus': {
                color: theme.colors.textPrimary,
                textDecoration: 'none'
            }
        })
    });
const contentStyles = (theme, useNewBorderColors)=>({
        ...dropdownContentStyles(theme, useNewBorderColors)
    });
const dropdownItemStyles = (theme)=>({
        padding: `${CONSTANTS$1.itemPaddingVertical(theme)}px ${CONSTANTS$1.itemPaddingHorizontal(theme)}px`,
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        outline: 'unset',
        '&:hover': {
            cursor: 'pointer'
        },
        '&:focus': {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
            '&:not(:hover)': {
                outline: `2px auto ${theme.colors.actionDefaultBorderFocus}`,
                outlineOffset: '-1px'
            }
        },
        '&[data-disabled]': {
            pointerEvents: 'none',
            color: `${theme.colors.actionDisabledText} !important`
        }
    });
const dangerItemStyles = (theme)=>({
        color: theme.colors.textValidationDanger,
        '&:hover, &:focus': {
            backgroundColor: theme.colors.actionDangerDefaultBackgroundHover
        }
    });
const checkboxItemStyles = (theme)=>({
        position: 'relative',
        paddingLeft: CONSTANTS$1.checkboxIconWidth(theme) + CONSTANTS$1.checkboxPaddingLeft(theme) + CONSTANTS$1.checkboxPaddingRight(theme)
    });
const metaTextStyles = (theme)=>({
        color: theme.colors.textSecondary,
        fontSize: theme.typography.fontSizeSm,
        '[data-disabled] &': {
            color: theme.colors.actionDisabledText
        }
    });
const dropdownSeparatorStyles = (theme)=>({
        height: 1,
        margin: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
        backgroundColor: theme.colors.borderDecorative
    });
function getZIndex() {
    return 10000;
}

var DropdownMenu = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Arrow: Arrow$1,
  CheckboxItem: CheckboxItem$1,
  Content: Content$4,
  Group: Group$2,
  HintColumn: HintColumn,
  HintRow: HintRow,
  IconWrapper: IconWrapper,
  Item: Item$2,
  ItemIndicator: ItemIndicator$1,
  Label: Label$2,
  RadioGroup: RadioGroup$1,
  RadioItem: RadioItem$1,
  Root: Root$7,
  Separator: Separator$1,
  Sub: Sub$1,
  SubContent: SubContent$1,
  SubTrigger: SubTrigger$1,
  Trigger: Trigger$3,
  TriggerItem: TriggerItem,
  dropdownContentStyles: dropdownContentStyles,
  dropdownItemStyles: dropdownItemStyles,
  dropdownSeparatorStyles: dropdownSeparatorStyles,
  getZIndex: getZIndex
});

const Trigger$2 = ContextMenuTrigger;
const ItemIndicator = ContextMenuItemIndicator;
const Group$1 = ContextMenuGroup;
const Arrow = ContextMenuArrow;
const Sub = ContextMenuSub;
const ContextMenuProps = /*#__PURE__*/ createContext({
    isOpen: false,
    setIsOpen: (isOpen)=>{}
});
const useContextMenuProps = ()=>React__default.useContext(ContextMenuProps);
const Root$6 = ({ children, onOpenChange, ...props })=>{
    const [isOpen, setIsOpen] = React__default.useState(false);
    const handleChange = (isOpen)=>{
        setIsOpen(isOpen);
        onOpenChange?.(isOpen);
    };
    return /*#__PURE__*/ jsx(ContextMenu$2, {
        onOpenChange: handleChange,
        ...props,
        children: /*#__PURE__*/ jsx(ContextMenuProps.Provider, {
            value: {
                isOpen,
                setIsOpen
            },
            children: children
        })
    });
};
const SubTrigger = ({ children, disabledReason, withChevron, ...props })=>{
    const { theme } = useDesignSystemTheme();
    const ref = useRef(null);
    return /*#__PURE__*/ jsxs(ContextMenuSubTrigger, {
        ...props,
        css: dropdownItemStyles(theme),
        ref: ref,
        onKeyDown: (e)=>{
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        },
        children: [
            getNewChildren(children, props, disabledReason, ref),
            withChevron && /*#__PURE__*/ jsx(ContextMenu.Hint, {
                children: /*#__PURE__*/ jsx(ChevronRightIcon, {})
            })
        ]
    });
};
const Content$3 = ({ children, minWidth, forceCloseOnEscape, onEscapeKeyDown, onKeyDown, ...childrenProps })=>{
    const { getPopupContainer } = useDesignSystemContext();
    const { theme } = useDesignSystemTheme();
    const { isInsideModal } = useModalContext();
    const { isOpen, setIsOpen } = useContextMenuProps();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    return /*#__PURE__*/ jsx(ContextMenuPortal, {
        container: getPopupContainer && getPopupContainer(),
        children: isOpen && /*#__PURE__*/ jsx(ContextMenuContent, {
            ...addDebugOutlineIfEnabled(),
            onWheel: (e)=>{
                e.stopPropagation();
            },
            onTouchMove: (e)=>{
                e.stopPropagation();
            },
            onKeyDown: (e)=>{
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
            },
            ...childrenProps,
            css: [
                dropdownContentStyles(theme, useNewBorderColors),
                {
                    minWidth
                }
            ],
            children: children
        })
    });
};
const SubContent = ({ children, minWidth, ...childrenProps })=>{
    const { getPopupContainer } = useDesignSystemContext();
    const { theme } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    return /*#__PURE__*/ jsx(ContextMenuPortal, {
        container: getPopupContainer && getPopupContainer(),
        children: /*#__PURE__*/ jsx(ContextMenuSubContent, {
            ...addDebugOutlineIfEnabled(),
            ...childrenProps,
            onWheel: (e)=>{
                e.stopPropagation();
            },
            onTouchMove: (e)=>{
                e.stopPropagation();
            },
            css: [
                dropdownContentStyles(theme, useNewBorderColors),
                {
                    minWidth
                }
            ],
            onKeyDown: (e)=>{
                if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                    e.stopPropagation();
                    handleKeyboardNavigation(e);
                }
                childrenProps.onKeyDown?.(e);
            },
            children: children
        })
    });
};
const Item$1 = ({ children, disabledReason, onClick, componentId, analyticsEvents, asChild, ...props })=>{
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.contextMenu', false);
    const { theme } = useDesignSystemTheme();
    const ref = useRef(null);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnClick,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnClick
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.ContextMenuItem,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents
    });
    const { elementRef: contextMenuItemRef } = useNotifyOnFirstView({
        onView: !asChild ? eventContext.onView : ()=>{}
    });
    const mergedRef = useMergeRefs([
        ref,
        contextMenuItemRef
    ]);
    const onClickWrapper = useCallback((e)=>{
        if (!asChild) {
            eventContext.onClick(e);
        }
        onClick?.(e);
    }, [
        asChild,
        eventContext,
        onClick
    ]);
    return /*#__PURE__*/ jsx(ContextMenuItem, {
        ...props,
        asChild: asChild,
        onClick: onClickWrapper,
        css: dropdownItemStyles(theme),
        ref: mergedRef,
        onKeyDown: (e)=>{
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        },
        ...eventContext.dataComponentProps,
        children: getNewChildren(children, props, disabledReason, ref)
    });
};
const CheckboxItem = ({ children, disabledReason, onCheckedChange, componentId, analyticsEvents, ...props })=>{
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.contextMenu', false);
    const { theme } = useDesignSystemTheme();
    const ref = useRef(null);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.ContextMenuCheckboxItem,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true
    });
    const { elementRef: contextMenuCheckboxItemRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.checked ?? props.defaultChecked
    });
    const mergedRef = useMergeRefs([
        ref,
        contextMenuCheckboxItemRef
    ]);
    const onCheckedChangeWrapper = useCallback((checked)=>{
        eventContext.onValueChange(checked);
        onCheckedChange?.(checked);
    }, [
        eventContext,
        onCheckedChange
    ]);
    return /*#__PURE__*/ jsxs(ContextMenuCheckboxItem, {
        ...props,
        onCheckedChange: onCheckedChangeWrapper,
        css: dropdownItemStyles(theme),
        ref: mergedRef,
        onKeyDown: (e)=>{
            if (e.key === 'Tab' || e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
            }
            props.onKeyDown?.(e);
        },
        ...eventContext.dataComponentProps,
        children: [
            /*#__PURE__*/ jsx(ContextMenuItemIndicator, {
                css: itemIndicatorStyles(theme),
                children: /*#__PURE__*/ jsx(CheckIcon, {})
            }),
            !props.checked && /*#__PURE__*/ jsx("div", {
                style: {
                    width: theme.general.iconFontSize + theme.spacing.xs
                }
            }),
            getNewChildren(children, props, disabledReason, ref)
        ]
    });
};
const RadioGroup = ({ onValueChange, componentId, analyticsEvents, valueHasNoPii, ...props })=>{
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.contextMenu', false);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.ContextMenuRadioGroup,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    const { elementRef: contextMenuRadioGroupRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.value ?? props.defaultValue
    });
    const onValueChangeWrapper = useCallback((value)=>{
        eventContext.onValueChange(value);
        onValueChange?.(value);
    }, [
        eventContext,
        onValueChange
    ]);
    return /*#__PURE__*/ jsx(ContextMenuRadioGroup, {
        ref: contextMenuRadioGroupRef,
        ...props,
        onValueChange: onValueChangeWrapper,
        ...eventContext.dataComponentProps
    });
};
const RadioItem = ({ children, disabledReason, ...props })=>{
    const { theme } = useDesignSystemTheme();
    const ref = useRef(null);
    return /*#__PURE__*/ jsxs(ContextMenuRadioItem, {
        ...props,
        css: [
            dropdownItemStyles(theme),
            {
                '&[data-state="unchecked"]': {
                    paddingLeft: theme.general.iconFontSize + theme.spacing.xs + theme.spacing.sm
                }
            }
        ],
        ref: ref,
        children: [
            /*#__PURE__*/ jsx(ContextMenuItemIndicator, {
                css: itemIndicatorStyles(theme),
                children: /*#__PURE__*/ jsx(CheckIcon, {})
            }),
            getNewChildren(children, props, disabledReason, ref)
        ]
    });
};
const Label$1 = ({ children, ...props })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(ContextMenuLabel, {
        ...props,
        css: {
            color: theme.colors.textSecondary,
            padding: `${theme.spacing.sm - 2}px ${theme.spacing.sm}px`
        },
        children: children
    });
};
const Hint$1 = ({ children })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("span", {
        css: {
            display: 'inline-flex',
            marginLeft: 'auto',
            paddingLeft: theme.spacing.sm
        },
        children: children
    });
};
const Separator = ()=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(ContextMenuSeparator, {
        css: dropdownSeparatorStyles(theme)
    });
};
const itemIndicatorStyles = (theme)=>/*#__PURE__*/ css({
        display: 'inline-flex',
        paddingRight: theme.spacing.xs
    });
const ContextMenu = {
    Root: Root$6,
    Trigger: Trigger$2,
    Label: Label$1,
    Item: Item$1,
    Group: Group$1,
    RadioGroup,
    CheckboxItem,
    RadioItem,
    Arrow,
    Separator,
    Sub,
    SubTrigger,
    SubContent,
    Content: Content$3,
    Hint: Hint$1
};

var ContextMenu$1 = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Arrow: Arrow,
  CheckboxItem: CheckboxItem,
  Content: Content$3,
  ContextMenu: ContextMenu,
  Group: Group$1,
  Hint: Hint$1,
  Item: Item$1,
  ItemIndicator: ItemIndicator,
  Label: Label$1,
  RadioGroup: RadioGroup,
  RadioItem: RadioItem,
  Root: Root$6,
  Separator: Separator,
  Sub: Sub,
  SubContent: SubContent,
  SubTrigger: SubTrigger,
  Trigger: Trigger$2,
  itemIndicatorStyles: itemIndicatorStyles
});

function getEmotionStyles(clsPrefix, theme) {
    const classFocused = `.${clsPrefix}-focused`;
    const classActiveBar = `.${clsPrefix}-active-bar`;
    const classSeparator = `.${clsPrefix}-separator`;
    const classSuffix = `.${clsPrefix}-suffix`;
    const styles = {
        height: 32,
        borderRadius: theme.borders.borderRadiusSm,
        borderColor: theme.colors.border,
        color: theme.colors.textPrimary,
        transition: 'border 0s, box-shadow 0s',
        [`&${classFocused},:hover`]: {
            borderColor: theme.colors.actionDefaultBorderHover
        },
        '&:active': {
            borderColor: theme.colors.actionDefaultBorderPress
        },
        [`&${classFocused}`]: {
            boxShadow: `none !important`,
            outline: `${theme.colors.actionDefaultBorderFocus} solid 2px !important`,
            outlineOffset: '-2px !important',
            borderColor: 'transparent !important'
        },
        [`& ${classActiveBar}`]: {
            background: `${theme.colors.actionDefaultBorderPress} !important`
        },
        [`& input::placeholder, & ${classSeparator}, & ${classSuffix}`]: {
            color: theme.colors.textPrimary
        }
    };
    return /*#__PURE__*/ css(styles);
}
const getDropdownStyles$1 = (theme)=>{
    return {
        zIndex: theme.options.zIndexBase + 50
    };
};
function useDatePickerStyles() {
    const { theme, getPrefixedClassName } = useDesignSystemTheme();
    const clsPrefix = getPrefixedClassName('picker');
    return getEmotionStyles(clsPrefix, theme);
}
const AccessibilityWrapper = ({ children, ariaLive = 'assertive', ...restProps })=>{
    const { theme } = useDesignSystemTheme();
    const ref = useRef(null);
    useEffect(()=>{
        if (ref.current) {
            const inputs = theme.isDarkMode ? ref.current.querySelectorAll('.du-bois-dark-picker-input > input') : ref.current.querySelectorAll('.du-bois-light-picker-input > input');
            inputs.forEach((input)=>input.setAttribute('aria-live', ariaLive));
        }
    }, [
        ref,
        ariaLive,
        theme.isDarkMode
    ]);
    return /*#__PURE__*/ jsx("div", {
        ...restProps,
        ref: ref,
        children: children
    });
};
const DuboisDatePicker = /*#__PURE__*/ forwardRef((props, ref)=>{
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(AccessibilityWrapper, {
            ...addDebugOutlineIfEnabled(),
            ...wrapperDivProps,
            ariaLive: ariaLive,
            children: /*#__PURE__*/ jsx(DatePicker, {
                css: styles,
                ref: ref,
                ...restProps,
                popupStyle: {
                    ...getDropdownStyles$1(theme),
                    ...props.popupStyle || {}
                }
            })
        })
    });
});
const RangePicker = /*#__PURE__*/ forwardRef((props, ref)=>{
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(AccessibilityWrapper, {
            ...wrapperDivProps,
            ariaLive: ariaLive,
            children: /*#__PURE__*/ jsx(DatePicker.RangePicker, {
                ...addDebugOutlineIfEnabled(),
                css: styles,
                ...restProps,
                ref: ref,
                popupStyle: {
                    ...getDropdownStyles$1(theme),
                    ...props.popupStyle || {}
                }
            })
        })
    });
});
const TimePicker = /*#__PURE__*/ forwardRef((props, ref)=>{
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(AccessibilityWrapper, {
            ...addDebugOutlineIfEnabled(),
            ...wrapperDivProps,
            ariaLive: ariaLive,
            children: /*#__PURE__*/ jsx(DatePicker.TimePicker, {
                css: styles,
                ...restProps,
                ref: ref,
                popupStyle: {
                    ...getDropdownStyles$1(theme),
                    ...props.popupStyle || {}
                }
            })
        })
    });
});
const QuarterPicker = /*#__PURE__*/ forwardRef((props, ref)=>{
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(AccessibilityWrapper, {
            ...addDebugOutlineIfEnabled(),
            ...wrapperDivProps,
            ariaLive: ariaLive,
            children: /*#__PURE__*/ jsx(DatePicker.QuarterPicker, {
                css: styles,
                ...restProps,
                ref: ref,
                popupStyle: {
                    ...getDropdownStyles$1(theme),
                    ...props.popupStyle || {}
                }
            })
        })
    });
});
const WeekPicker = /*#__PURE__*/ forwardRef((props, ref)=>{
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(AccessibilityWrapper, {
            ...addDebugOutlineIfEnabled(),
            ...wrapperDivProps,
            ariaLive: ariaLive,
            children: /*#__PURE__*/ jsx(DatePicker.WeekPicker, {
                css: styles,
                ...restProps,
                ref: ref,
                popupStyle: {
                    ...getDropdownStyles$1(theme),
                    ...props.popupStyle || {}
                }
            })
        })
    });
});
const MonthPicker = /*#__PURE__*/ forwardRef((props, ref)=>{
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(AccessibilityWrapper, {
            ...addDebugOutlineIfEnabled(),
            ...wrapperDivProps,
            ariaLive: ariaLive,
            children: /*#__PURE__*/ jsx(DatePicker.MonthPicker, {
                css: styles,
                ...restProps,
                ref: ref,
                popupStyle: {
                    ...getDropdownStyles$1(theme),
                    ...props.popupStyle || {}
                }
            })
        })
    });
});
const YearPicker = /*#__PURE__*/ forwardRef((props, ref)=>{
    const styles = useDatePickerStyles();
    const { theme } = useDesignSystemTheme();
    const { ariaLive, wrapperDivProps, ...restProps } = props;
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(AccessibilityWrapper, {
            ...addDebugOutlineIfEnabled(),
            ...wrapperDivProps,
            ariaLive: ariaLive,
            children: /*#__PURE__*/ jsx(DatePicker.YearPicker, {
                css: styles,
                ...restProps,
                ref: ref,
                popupStyle: {
                    ...getDropdownStyles$1(theme),
                    ...props.popupStyle || {}
                }
            })
        })
    });
});
/**
 * `LegacyDatePicker` was added as a temporary solution pending an
 * official Du Bois replacement. Use with caution.
 * @deprecated
 */ const LegacyDatePicker = /* #__PURE__ */ Object.assign(DuboisDatePicker, {
    /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */ RangePicker,
    /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */ TimePicker,
    /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */ QuarterPicker,
    /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */ WeekPicker,
    /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */ MonthPicker,
    /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */ YearPicker
});

const DialogCombobox = ({ children, label, id, value = [], open, emptyText, scrollToSelectedElement = true, rememberLastScrollPosition = false, componentId, analyticsEvents, valueHasNoPii, onOpenChange, ...props })=>{
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.dialogCombobox', false);
    // Used to avoid infinite loop when value is controlled from within the component (DialogComboboxOptionControlledList)
    // We can't remove setValue altogether because uncontrolled component users need to be able to set the value from root for trigger to update
    const [isControlled, setIsControlled] = useState(false);
    const [selectedValue, setSelectedValue] = useState(value);
    const [isOpen, setIsOpenState] = useState(Boolean(open));
    const setIsOpen = useCallback((isOpen)=>{
        setIsOpenState(isOpen);
        onOpenChange?.(isOpen);
    }, [
        setIsOpenState,
        onOpenChange
    ]);
    const [contentWidth, setContentWidth] = useState();
    const [textOverflowMode, setTextOverflowMode] = useState('multiline');
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const [disableMouseOver, setDisableMouseOver] = useState(false);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.DialogCombobox,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    const setSelectedValueWrapper = useCallback((newValue)=>{
        eventContext.onValueChange(JSON.stringify(newValue));
        setSelectedValue(newValue);
    }, [
        eventContext
    ]);
    useEffect(()=>{
        if ((!Array.isArray(selectedValue) || !Array.isArray(value)) && selectedValue !== value || selectedValue && value && selectedValue.length === value.length && selectedValue.every((v, i)=>v === value[i])) {
            return;
        }
        if (!isControlled) {
            setSelectedValueWrapper(value);
        }
    }, [
        value,
        isControlled,
        selectedValue,
        setSelectedValueWrapper
    ]);
    return /*#__PURE__*/ jsx(DialogComboboxContextProvider, {
        value: {
            id,
            label,
            value: selectedValue,
            setValue: setSelectedValueWrapper,
            setIsControlled,
            contentWidth,
            setContentWidth,
            textOverflowMode,
            setTextOverflowMode,
            isInsideDialogCombobox: true,
            multiSelect: props.multiSelect,
            stayOpenOnSelection: props.stayOpenOnSelection,
            isOpen,
            setIsOpen,
            emptyText,
            scrollToSelectedElement,
            rememberLastScrollPosition,
            componentId,
            analyticsEvents,
            valueHasNoPii,
            disableMouseOver,
            setDisableMouseOver,
            onView: eventContext.onView
        },
        children: /*#__PURE__*/ jsx(Root$5, {
            open: open !== undefined ? open : isOpen,
            ...props,
            children: /*#__PURE__*/ jsx(ComponentFinderContext.Provider, {
                value: {
                    dataComponentProps: eventContext.dataComponentProps
                },
                children: children
            })
        })
    });
};
const Root$5 = (props)=>{
    const { children, stayOpenOnSelection, multiSelect, ...restProps } = props;
    const { value, setIsOpen, onView } = useDialogComboboxContext();
    const firstView = useRef(true);
    useEffect(()=>{
        if (firstView.current) {
            onView(value);
            firstView.current = false;
        }
    }, [
        onView,
        value
    ]);
    const handleOpenChange = (open)=>{
        setIsOpen(open);
    };
    useEffect(()=>{
        if (!stayOpenOnSelection && (typeof stayOpenOnSelection === 'boolean' || !multiSelect)) {
            setIsOpen(false);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
        value,
        stayOpenOnSelection,
        multiSelect
    ]); // Don't trigger when setIsOpen changes.
    return /*#__PURE__*/ jsx(Popover.Root, {
        onOpenChange: handleOpenChange,
        ...restProps,
        children: children
    });
};

const DialogComboboxAddButton = ({ children, ...restProps })=>{
    const { theme } = useDesignSystemTheme();
    const { isInsideDialogCombobox, componentId } = useDialogComboboxContext();
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxAddButton` must be used within `DialogCombobox`');
    }
    return /*#__PURE__*/ jsx(Button, {
        // eslint-disable-next-line @databricks/no-dynamic-property-value -- http://go/static-frontend-log-property-strings exempt:fc0846cb-b73b-401b-834e-75bb8b6d4c20
        componentId: `${componentId ? componentId : 'design_system.dialogcombobox'}.add_option`,
        ...restProps,
        type: "tertiary",
        className: "combobox-footer-add-button",
        css: {
            ...getComboboxOptionItemWrapperStyles(theme),
            .../*#__PURE__*/ css(importantify({
                width: '100%',
                padding: 0,
                display: 'flex',
                alignItems: 'center',
                borderRadius: 0,
                '&:focus': {
                    background: theme.colors.actionTertiaryBackgroundHover,
                    outline: 'none'
                }
            }))
        },
        icon: /*#__PURE__*/ jsx(PlusIcon, {}),
        children: children
    });
};

const defaultMaxHeight = 'var(--radix-popover-content-available-height)';
const DialogComboboxContent = /*#__PURE__*/ forwardRef(({ children, loading, loadingDescription = 'DialogComboboxContent', matchTriggerWidth, textOverflowMode, maxHeight, maxWidth, minHeight, minWidth = 240, width, align = 'start', side = 'bottom', sideOffset = 4, onEscapeKeyDown, onKeyDown, forceCloseOnEscape, ...restProps }, forwardedRef)=>{
    const { theme } = useDesignSystemTheme();
    const { label, isInsideDialogCombobox, contentWidth, setContentWidth, textOverflowMode: contextTextOverflowMode, setTextOverflowMode, multiSelect, isOpen, rememberLastScrollPosition, setIsOpen } = useDialogComboboxContext();
    const { isInsideModal } = useModalContext();
    const { getPopupContainer } = useDesignSystemContext();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const [lastScrollPosition, setLastScrollPosition] = useState(0);
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxContent` must be used within `DialogCombobox`');
    }
    const contentRef = useRef(null);
    useImperativeHandle(forwardedRef, ()=>contentRef.current);
    const realContentWidth = matchTriggerWidth ? 'var(--radix-popover-trigger-width)' : width;
    useEffect(()=>{
        if (rememberLastScrollPosition) {
            if (!isOpen && contentRef.current) {
                setLastScrollPosition(contentRef.current.scrollTop);
            } else {
                // Wait for the popover to render and scroll to the last scrolled position
                const interval = setInterval(()=>{
                    if (contentRef.current) {
                        // Verify if the popover's content can be scrolled to the last scrolled position
                        if (lastScrollPosition && contentRef.current.scrollHeight >= lastScrollPosition) {
                            contentRef.current.scrollTo({
                                top: lastScrollPosition,
                                behavior: 'smooth'
                            });
                        }
                        clearInterval(interval);
                    }
                }, 50);
                return ()=>clearInterval(interval);
            }
        }
        return;
    }, [
        isOpen,
        rememberLastScrollPosition,
        lastScrollPosition
    ]);
    useEffect(()=>{
        if (contentWidth !== realContentWidth) {
            setContentWidth(realContentWidth);
        }
    }, [
        realContentWidth,
        contentWidth,
        setContentWidth
    ]);
    useEffect(()=>{
        if (textOverflowMode !== contextTextOverflowMode) {
            setTextOverflowMode(textOverflowMode ? textOverflowMode : 'multiline');
        }
    }, [
        textOverflowMode,
        contextTextOverflowMode,
        setTextOverflowMode
    ]);
    return /*#__PURE__*/ jsx(Popover.Portal, {
        container: getPopupContainer && getPopupContainer(),
        children: /*#__PURE__*/ jsx(Popover.Content, {
            ...addDebugOutlineIfEnabled(),
            "aria-label": `${label} options`,
            "aria-busy": loading,
            role: "listbox",
            "aria-multiselectable": multiSelect,
            css: getComboboxContentWrapperStyles(theme, {
                maxHeight: maxHeight ? `min(${maxHeight}px, ${defaultMaxHeight})` : defaultMaxHeight,
                maxWidth,
                minHeight,
                minWidth,
                width: realContentWidth,
                useNewBorderColors
            }),
            align: align,
            side: side,
            sideOffset: sideOffset,
            onKeyDown: (e)=>{
                // This is a workaround for Radix's DialogCombobox.Content not receiving Escape key events
                // when nested inside a modal. We need to stop propagation of the event so that the modal
                // doesn't close when the DropdownMenu should.
                if (e.key === 'Escape') {
                    if (isInsideModal || forceCloseOnEscape) {
                        e.stopPropagation();
                        setIsOpen(false);
                    }
                    onEscapeKeyDown?.(e.nativeEvent);
                }
                onKeyDown?.(e);
            },
            ...restProps,
            ref: contentRef,
            children: /*#__PURE__*/ jsx("div", {
                css: {
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-start',
                    justifyContent: 'center'
                },
                children: loading ? /*#__PURE__*/ jsx(LoadingSpinner, {
                    label: "Loading",
                    alt: "Loading spinner",
                    loadingDescription: loadingDescription
                }) : children ? children : /*#__PURE__*/ jsx(EmptyResults, {})
            })
        })
    });
});

const getCountBadgeStyles = (theme)=>/*#__PURE__*/ css(importantify({
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxSizing: 'border-box',
        padding: `${theme.spacing.xs / 2}px ${theme.spacing.xs}px`,
        background: theme.colors.tagDefault,
        borderRadius: theme.general.borderRadiusBase,
        fontSize: theme.typography.fontSizeBase,
        height: 20
    }));
const DialogComboboxCountBadge = (props)=>{
    const { countStartAt, ...restOfProps } = props;
    const { theme } = useDesignSystemTheme();
    const { value } = useDialogComboboxContext();
    return /*#__PURE__*/ jsx("div", {
        ...restOfProps,
        css: getCountBadgeStyles(theme),
        children: Array.isArray(value) ? countStartAt ? `+${value.length - countStartAt}` : value.length : value ? 1 : 0
    });
};

const DialogComboboxFooter = ({ children, ...restProps })=>{
    const { theme } = useDesignSystemTheme();
    const { isInsideDialogCombobox } = useDialogComboboxContext();
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxFooter` must be used within `DialogCombobox`');
    }
    return /*#__PURE__*/ jsx("div", {
        ...restProps,
        css: getFooterStyles(theme),
        children: children
    });
};

const DialogComboboxHintRow = ({ children })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        css: {
            minWidth: '100%',
            color: theme.colors.textSecondary,
            fontSize: theme.typography.fontSizeSm,
            '[data-disabled] &': {
                color: theme.colors.actionDisabledText
            }
        },
        children: children
    });
};

const DialogComboboxOptionListContext = /*#__PURE__*/ createContext({
    isInsideDialogComboboxOptionList: false,
    lookAhead: '',
    setLookAhead: ()=>{}
});
const DialogComboboxOptionListContextProvider = ({ children, value })=>{
    return /*#__PURE__*/ jsx(DialogComboboxOptionListContext.Provider, {
        value: value,
        children: children
    });
};

const DialogComboboxOptionList = /*#__PURE__*/ forwardRef(({ children, loading, loadingDescription = 'DialogComboboxOptionList', withProgressiveLoading, ...restProps }, forwardedRef)=>{
    const { isInsideDialogCombobox } = useDialogComboboxContext();
    const ref = useRef(null);
    useImperativeHandle(forwardedRef, ()=>ref.current);
    const [lookAhead, setLookAhead] = useState('');
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxOptionList` must be used within `DialogCombobox`');
    }
    const lookAheadTimeout = useRef(null);
    useEffect(()=>{
        if (lookAheadTimeout.current) {
            clearTimeout(lookAheadTimeout.current);
        }
        lookAheadTimeout.current = setTimeout(()=>{
            setLookAhead('');
        }, 1500);
        return ()=>{
            if (lookAheadTimeout.current) {
                clearTimeout(lookAheadTimeout.current);
            }
        };
    }, [
        lookAhead
    ]);
    useEffect(()=>{
        if (loading && !withProgressiveLoading) {
            return;
        }
        const optionItems = ref.current?.querySelectorAll('[role="option"]');
        const hasTabIndexedOption = Array.from(optionItems ?? []).some((optionItem)=>{
            return optionItem.getAttribute('tabindex') === '0';
        });
        if (!hasTabIndexedOption) {
            const firstOptionItem = optionItems?.[0];
            if (firstOptionItem) {
                highlightFirstNonDisabledOption(firstOptionItem, 'start');
            }
        }
    }, [
        loading,
        withProgressiveLoading
    ]);
    const handleOnMouseEnter = (event)=>{
        const target = event.target;
        if (target) {
            const options = target.hasAttribute('data-combobox-option-list') ? target.querySelectorAll('[role="option"]') : target?.closest('[data-combobox-option-list="true"]')?.querySelectorAll('[role="option"]');
            if (options) {
                options.forEach((option)=>option.removeAttribute('data-highlighted'));
            }
        }
    };
    return /*#__PURE__*/ jsx("div", {
        ref: ref,
        "aria-busy": loading,
        "data-combobox-option-list": "true",
        css: {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-start',
            width: '100%'
        },
        onMouseEnter: handleOnMouseEnter,
        ...restProps,
        children: /*#__PURE__*/ jsx(DialogComboboxOptionListContextProvider, {
            value: {
                isInsideDialogComboboxOptionList: true,
                lookAhead,
                setLookAhead
            },
            children: loading ? withProgressiveLoading ? /*#__PURE__*/ jsxs(Fragment, {
                children: [
                    children,
                    /*#__PURE__*/ jsx(LoadingSpinner, {
                        "aria-label": "Loading",
                        alt: "Loading spinner",
                        loadingDescription: loadingDescription
                    })
                ]
            }) : /*#__PURE__*/ jsx(LoadingSpinner, {
                "aria-label": "Loading",
                alt: "Loading spinner",
                loadingDescription: loadingDescription
            }) : children && Children.toArray(children).some((child)=>/*#__PURE__*/ React__default.isValidElement(child)) ? children : /*#__PURE__*/ jsx(EmptyResults, {})
        })
    });
});

const useDialogComboboxOptionListContext = ()=>{
    return useContext(DialogComboboxOptionListContext);
};

const InfoTooltip = ({ content, iconTitle = 'More information', ...props })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(Tooltip$1, {
        content: content,
        ...props,
        children: /*#__PURE__*/ jsx(InfoSmallIcon, {
            tabIndex: 0,
            "aria-hidden": "false",
            "aria-label": iconTitle,
            alt: iconTitle,
            css: {
                color: theme.colors.textSecondary
            }
        })
    });
};

const DuboisDialogComboboxOptionListCheckboxItem = /*#__PURE__*/ forwardRef(({ value, checked, indeterminate, onChange, children, disabledReason, _TYPE, ...props }, ref)=>{
    const { theme } = useDesignSystemTheme();
    const { textOverflowMode, contentWidth, disableMouseOver, setDisableMouseOver } = useDialogComboboxContext();
    const { isInsideDialogComboboxOptionList, setLookAhead, lookAhead } = useDialogComboboxOptionListContext();
    if (!isInsideDialogComboboxOptionList) {
        throw new Error('`DialogComboboxOptionListCheckboxItem` must be used within `DialogComboboxOptionList`');
    }
    const handleSelect = (e)=>{
        if (onChange) {
            onChange(value, e);
        }
    };
    let content = children ?? value;
    if (props.disabled && disabledReason) {
        content = /*#__PURE__*/ jsxs("div", {
            css: {
                display: 'flex'
            },
            children: [
                /*#__PURE__*/ jsx("div", {
                    children: content
                }),
                /*#__PURE__*/ jsx("div", {
                    css: {
                        display: 'flex'
                    },
                    children: /*#__PURE__*/ jsx(Tooltip$1, {
                        componentId: "dialog-combobox-option-list-checkbox-item-disabled-reason-tooltip",
                        content: disabledReason,
                        side: "right",
                        children: /*#__PURE__*/ jsx("span", {
                            css: [
                                getInfoIconStyles(theme),
                                {
                                    display: 'flex',
                                    alignItems: 'center',
                                    alignSelf: 'flex-start',
                                    marginTop: theme.spacing.xs / 2
                                }
                            ],
                            children: /*#__PURE__*/ jsx(InfoSmallIcon, {
                                "aria-label": "Disabled status information",
                                "aria-hidden": "false"
                            })
                        })
                    })
                })
            ]
        });
    }
    return /*#__PURE__*/ jsx("div", {
        ref: ref,
        role: "option",
        // Using aria-selected instead of aria-checked because the parent listbox
        "aria-selected": indeterminate ? false : checked,
        css: [
            getComboboxOptionItemWrapperStyles(theme)
        ],
        ...props,
        onClick: (e)=>{
            if (props.disabled) {
                e.preventDefault();
            } else {
                handleSelect(e);
            }
        },
        tabIndex: -1,
        ...getKeyboardNavigationFunctions(handleSelect, {
            onKeyDown: props.onKeyDown,
            onMouseEnter: props.onMouseEnter,
            onDefaultKeyDown: (e)=>dialogComboboxLookAheadKeyDown(e, setLookAhead, lookAhead),
            disableMouseOver,
            setDisableMouseOver
        }),
        children: /*#__PURE__*/ jsx(Checkbox, {
            componentId: "codegen_design-system_src_design-system_dialogcombobox_dialogcomboboxoptionlistcheckboxitem.tsx_86",
            disabled: props.disabled,
            isChecked: indeterminate ? null : checked,
            css: [
                getCheckboxStyles(theme, textOverflowMode),
                contentWidth ? {
                    '& > span:last-of-type': {
                        width: getDialogComboboxOptionLabelWidth(theme, contentWidth)
                    }
                } : {}
            ],
            tabIndex: -1,
            // Needed because Antd handles keyboard inputs as clicks
            onClick: (e)=>{
                e.stopPropagation();
                handleSelect(e);
            },
            children: /*#__PURE__*/ jsx("div", {
                css: {
                    maxWidth: '100%'
                },
                children: content
            })
        })
    });
});
DuboisDialogComboboxOptionListCheckboxItem.defaultProps = {
    _TYPE: 'DialogComboboxOptionListCheckboxItem'
};
const DialogComboboxOptionListCheckboxItem = DuboisDialogComboboxOptionListCheckboxItem;

const extractTextContent = (node)=>{
    if (typeof node === 'string' || typeof node === 'number') {
        return node.toString();
    }
    if (/*#__PURE__*/ React__default.isValidElement(node) && node.props.children) {
        return React__default.Children.toArray(node.props.children).map(extractTextContent).join(' ');
    }
    return '';
};
const filterChildren = (children, searchValue)=>{
    const lowerCaseSearchValue = searchValue.toLowerCase();
    return React__default.Children.map(children, (child)=>{
        if (/*#__PURE__*/ React__default.isValidElement(child)) {
            const childType = child.props['__EMOTION_TYPE_PLEASE_DO_NOT_USE__']?.defaultProps._TYPE ?? child.props._TYPE;
            if (childType === 'DialogComboboxOptionListSelectItem' || childType === 'DialogComboboxOptionListCheckboxItem') {
                const childTextContent = extractTextContent(child).toLowerCase();
                const childValue = child.props.value?.toLowerCase() ?? '';
                return childTextContent.includes(lowerCaseSearchValue) || childValue.includes(lowerCaseSearchValue) ? child : null;
            }
        }
        return child;
    })?.filter((child)=>child);
};
const DialogComboboxOptionListSearch = /*#__PURE__*/ forwardRef(({ onChange, onSearch, virtualized, children, hasWrapper, controlledValue, setControlledValue, rightSearchControls, ...restProps }, forwardedRef)=>{
    const { theme } = useDesignSystemTheme();
    const { componentId } = useDialogComboboxContext();
    const { isInsideDialogComboboxOptionList } = useDialogComboboxOptionListContext();
    const noResultId = `no-result-${generateUuidV4()}`;
    const inputRef = useRef(null);
    useImperativeHandle(forwardedRef, ()=>inputRef.current);
    const [searchValue, setSearchValue] = React__default.useState();
    if (!isInsideDialogComboboxOptionList) {
        throw new Error('`DialogComboboxOptionListSearch` must be used within `DialogComboboxOptionList`');
    }
    const handleOnChange = (event)=>{
        if (!virtualized) {
            setSearchValue(event.target.value);
        }
        setControlledValue?.(event.target.value);
        onSearch?.(event.target.value);
    };
    let filteredChildren = children;
    if (searchValue && !virtualized && controlledValue === undefined) {
        filteredChildren = filterChildren(hasWrapper ? children.props.children : children, searchValue);
        if (hasWrapper) {
            filteredChildren = /*#__PURE__*/ React__default.cloneElement(children, {}, filteredChildren);
        }
    }
    const inputWrapperRef = useRef(null);
    // When the search value changes, highlight the first option
    useEffect(()=>{
        if (!inputWrapperRef.current) {
            return;
        }
        const optionItems = getContentOptions(inputWrapperRef.current);
        if (optionItems) {
            // Reset previous highlights
            const highlightedOption = findHighlightedOption(optionItems);
            const firstOptionItem = optionItems?.[0];
            if (firstOptionItem) {
                highlightOption(firstOptionItem, highlightedOption, false);
            }
        }
    }, [
        searchValue
    ]);
    const handleOnKeyDown = (event)=>{
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || event.key === 'Enter') {
            event.preventDefault();
        } else {
            return;
        }
        // Find closest parent of type DialogComboboxOptionList and get all options within it
        const options = getContentOptions(event.target);
        if (!options) {
            return;
        }
        const highlightedOption = findHighlightedOption(options);
        // If the user is navigating the option is highlighted without focusing in order to avoid losing focus on the input
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            if (highlightedOption) {
                const nextOption = findClosestOptionSibling(highlightedOption, event.key === 'ArrowDown' ? 'next' : 'previous');
                if (nextOption) {
                    highlightOption(nextOption, highlightedOption, false);
                } else if (event.key === 'ArrowDown') {
                    // If there is no next option, highlight the first option
                    const firstOption = options[0];
                    highlightOption(firstOption, highlightedOption, false);
                } else if (event.key === 'ArrowUp') {
                    // If there is no previous option, highlight the last option
                    const lastOption = options[options.length - 1];
                    highlightOption(lastOption, highlightedOption, false);
                }
            } else {
                // In case there is no highlighted option, highlight the first / last option depending on key
                const nextOption = event.key === 'ArrowDown' ? options[0] : options[options.length - 1];
                if (nextOption) {
                    highlightOption(nextOption, undefined, false);
                }
            }
        // On Enter trigger a click event on the highlighted option
        } else if (event.key === 'Enter' && highlightedOption) {
            highlightedOption.click();
        }
    };
    const childrenIsNotEmpty = Children.toArray(hasWrapper ? children.props.children : children).some((child)=>/*#__PURE__*/ React__default.isValidElement(child));
    const hasFilteredResults = hasWrapper && filteredChildren?.props.children?.length || !hasWrapper && filteredChildren?.length;
    useEffect(()=>{
        if (!hasFilteredResults) {
            inputRef.current?.input?.setAttribute('aria-activedescendant', noResultId);
        }
    }, [
        hasFilteredResults,
        noResultId,
        inputRef.current?.input?.value
    ]);
    return /*#__PURE__*/ jsxs(Fragment, {
        children: [
            /*#__PURE__*/ jsx("div", {
                ref: inputWrapperRef,
                css: {
                    padding: `${theme.spacing.sm}px ${theme.spacing.lg / 2}px ${theme.spacing.sm}px`,
                    width: '100%',
                    boxSizing: 'border-box',
                    position: 'sticky',
                    top: 0,
                    background: theme.colors.backgroundPrimary,
                    zIndex: theme.options.zIndexBase + 1
                },
                children: /*#__PURE__*/ jsxs("div", {
                    css: {
                        display: 'flex',
                        flexDirection: 'row',
                        gap: theme.spacing.sm
                    },
                    children: [
                        /*#__PURE__*/ jsx(Input, {
                            // eslint-disable-next-line @databricks/no-dynamic-property-value -- http://go/static-frontend-log-property-strings exempt:a992c761-9f18-43b0-af9d-dc9ee629260a
                            componentId: componentId ? `${componentId}.search` : 'codegen_design_system_src_design_system_dialogcombobox_dialogcomboboxoptionlistsearch.tsx_173',
                            type: "search",
                            name: "search",
                            ref: inputRef,
                            prefix: /*#__PURE__*/ jsx(SearchIcon, {}),
                            placeholder: "Search",
                            onChange: handleOnChange,
                            onKeyDown: (event)=>{
                                handleOnKeyDown(event);
                                restProps.onKeyDown?.(event);
                            },
                            value: controlledValue ?? searchValue,
                            shouldPreventFormSubmission: true,
                            ...restProps
                        }),
                        rightSearchControls
                    ]
                })
            }),
            virtualized ? children : hasFilteredResults && childrenIsNotEmpty ? /*#__PURE__*/ jsx("div", {
                "aria-live": "polite",
                css: {
                    width: '100%'
                },
                children: filteredChildren
            }) : /*#__PURE__*/ jsx(EmptyResults, {
                id: noResultId
            })
        ]
    });
});

const selectContextDefaults = {
    isSelect: false
};
const SelectContext = /*#__PURE__*/ createContext(selectContextDefaults);
const SelectContextProvider = ({ children, value })=>{
    return /*#__PURE__*/ jsx(SelectContext.Provider, {
        value: value,
        children: children
    });
};

const useSelectContext = ()=>{
    return useContext(SelectContext);
};

const DuboisDialogComboboxOptionListSelectItem = /*#__PURE__*/ forwardRef(({ value, checked, disabledReason, onChange, hintColumn, hintColumnWidthPercent = 50, children, _TYPE, icon, dangerouslyHideCheck, ...props }, ref)=>{
    const { theme } = useDesignSystemTheme();
    const { stayOpenOnSelection, isOpen, setIsOpen, value: existingValue, contentWidth, textOverflowMode, scrollToSelectedElement, disableMouseOver, setDisableMouseOver } = useDialogComboboxContext();
    const { isInsideDialogComboboxOptionList, lookAhead, setLookAhead } = useDialogComboboxOptionListContext();
    const { isSelect } = useSelectContext();
    if (!isInsideDialogComboboxOptionList) {
        throw new Error('`DialogComboboxOptionListSelectItem` must be used within `DialogComboboxOptionList`');
    }
    const itemRef = useRef(null);
    const prevCheckedRef = useRef(checked);
    useImperativeHandle(ref, ()=>itemRef.current);
    useEffect(()=>{
        if (scrollToSelectedElement && isOpen) {
            // Check if checked didn't change since the last update, otherwise the popover is still open and we don't need to scroll
            if (checked && prevCheckedRef.current === checked) {
                // Wait for the popover to render and scroll to the selected element's position
                const interval = setInterval(()=>{
                    if (itemRef.current) {
                        itemRef.current?.scrollIntoView?.({
                            behavior: 'smooth',
                            block: 'center'
                        });
                        clearInterval(interval);
                    }
                }, 50);
                return ()=>clearInterval(interval);
            }
            prevCheckedRef.current = checked;
        }
        return;
    }, [
        isOpen,
        scrollToSelectedElement,
        checked
    ]);
    const handleSelect = (e)=>{
        if (onChange) {
            if (isSelect) {
                onChange({
                    value,
                    label: typeof children === 'string' ? children : value
                }, e);
                if (existingValue?.includes(value)) {
                    setIsOpen(false);
                }
                return;
            }
            onChange(value, e);
            // On selecting a previously selected value, manually close the popup, top level logic will not be triggered
            if (!stayOpenOnSelection && existingValue?.includes(value)) {
                setIsOpen(false);
            }
        }
    };
    let content = children ?? value;
    if (props.disabled && disabledReason) {
        content = /*#__PURE__*/ jsxs("div", {
            css: {
                display: 'flex',
                alignItems: 'center'
            },
            children: [
                /*#__PURE__*/ jsx("div", {
                    css: {
                        display: 'flex'
                    },
                    children: content
                }),
                /*#__PURE__*/ jsx(Tooltip$1, {
                    componentId: "dialog-combobox-option-list-select-item-disabled-reason-tooltip",
                    content: disabledReason,
                    side: "right",
                    children: /*#__PURE__*/ jsx("span", {
                        css: [
                            getInfoIconStyles(theme),
                            {
                                display: 'flex',
                                alignItems: 'center',
                                alignSelf: 'flex-start',
                                marginTop: theme.spacing.xs / 2
                            }
                        ],
                        children: /*#__PURE__*/ jsx(InfoSmallIcon, {
                            "aria-label": "Disabled status information",
                            "aria-hidden": "false"
                        })
                    })
                })
            ]
        });
    }
    return /*#__PURE__*/ jsxs("div", {
        ref: itemRef,
        css: [
            getComboboxOptionItemWrapperStyles(theme),
            {
                '&:hover': {
                    background: theme.colors.actionTertiaryBackgroundHover
                },
                '&:focus': {
                    background: theme.colors.actionTertiaryBackgroundHover,
                    outline: 'none'
                },
                '&:focus-visible': {
                    '&:not(:hover)': {
                        outlineStyle: 'solid',
                        outlineWidth: 2,
                        outlineOffset: -2,
                        outlineColor: theme.colors.actionDefaultBorderFocus
                    }
                }
            }
        ],
        ...props,
        onClick: (e)=>{
            if (props.disabled) {
                e.preventDefault();
            } else {
                handleSelect(e);
            }
        },
        tabIndex: -1,
        ...getKeyboardNavigationFunctions(handleSelect, {
            onKeyDown: props.onKeyDown,
            onMouseEnter: props.onMouseEnter,
            onDefaultKeyDown: (e)=>dialogComboboxLookAheadKeyDown(e, setLookAhead, lookAhead),
            disableMouseOver,
            setDisableMouseOver
        }),
        role: "option",
        "aria-selected": checked,
        children: [
            !dangerouslyHideCheck && (checked ? /*#__PURE__*/ jsx(CheckIcon, {
                css: {
                    marginTop: 'auto',
                    marginBottom: 'auto',
                    color: theme.colors.textSecondary
                }
            }) : /*#__PURE__*/ jsx("div", {
                style: {
                    width: 16,
                    flexShrink: 0
                }
            })),
            /*#__PURE__*/ jsxs("label", {
                css: getComboboxOptionLabelStyles({
                    theme,
                    dangerouslyHideCheck,
                    textOverflowMode,
                    contentWidth,
                    hasHintColumn: Boolean(hintColumn),
                    hasIcon: Boolean(icon),
                    hasDisabledReason: Boolean(disabledReason)
                }),
                children: [
                    icon && /*#__PURE__*/ jsx("span", {
                        style: {
                            position: 'relative',
                            top: 1,
                            marginRight: theme.spacing.sm,
                            color: theme.colors.textSecondary
                        },
                        children: icon
                    }),
                    hintColumn ? /*#__PURE__*/ jsxs("span", {
                        css: getSelectItemWithHintColumnStyles(hintColumnWidthPercent),
                        children: [
                            content,
                            /*#__PURE__*/ jsx("span", {
                                css: getHintColumnStyles(theme, Boolean(props.disabled), textOverflowMode),
                                children: hintColumn
                            })
                        ]
                    }) : content
                ]
            })
        ]
    });
});
DuboisDialogComboboxOptionListSelectItem.defaultProps = {
    _TYPE: 'DialogComboboxOptionListSelectItem'
};
const DialogComboboxOptionListSelectItem = DuboisDialogComboboxOptionListSelectItem;

const DialogComboboxOptionControlledList = /*#__PURE__*/ forwardRef(({ options, onChange, loading, loadingDescription = 'DialogComboboxOptionControlledList', withProgressiveLoading, withSearch, showAllOption, allOptionLabel = 'All', ...restProps }, forwardedRef)=>{
    const { isInsideDialogCombobox, multiSelect, value, setValue, setIsControlled } = useDialogComboboxContext();
    const [lookAhead, setLookAhead] = useState('');
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxOptionControlledList` must be used within `DialogCombobox`');
    }
    const lookAheadTimeout = useRef(null);
    const ref = useRef(null);
    useImperativeHandle(forwardedRef, ()=>ref.current);
    useEffect(()=>{
        if (lookAheadTimeout.current) {
            clearTimeout(lookAheadTimeout.current);
        }
        lookAheadTimeout.current = setTimeout(()=>{
            setLookAhead('');
        }, 1500);
        return ()=>{
            if (lookAheadTimeout.current) {
                clearTimeout(lookAheadTimeout.current);
            }
        };
    }, [
        lookAhead
    ]);
    useEffect(()=>{
        if (loading && !withProgressiveLoading) {
            return;
        }
        const optionItems = ref.current?.querySelectorAll('[role="option"]');
        const hasTabIndexedOption = Array.from(optionItems ?? []).some((optionItem)=>{
            return optionItem.getAttribute('tabindex') === '0';
        });
        if (!hasTabIndexedOption) {
            const firstOptionItem = optionItems?.[0];
            if (firstOptionItem) {
                highlightOption(firstOptionItem, undefined, false);
            }
        }
    }, [
        loading,
        withProgressiveLoading
    ]);
    const isOptionChecked = options.reduce((acc, option)=>{
        acc[option] = value?.includes(option);
        return acc;
    }, {});
    const handleUpdate = (updatedValue)=>{
        setIsControlled(true);
        let newValue = [];
        if (multiSelect) {
            if (value.find((item)=>item === updatedValue)) {
                newValue = value.filter((item)=>item !== updatedValue);
            } else {
                newValue = [
                    ...value,
                    updatedValue
                ];
            }
        } else {
            newValue = [
                updatedValue
            ];
        }
        setValue(newValue);
        isOptionChecked[updatedValue] = !isOptionChecked[updatedValue];
        if (onChange) {
            onChange(newValue);
        }
    };
    const handleSelectAll = ()=>{
        setIsControlled(true);
        if (value.length === options.length) {
            setValue([]);
            options.forEach((option)=>{
                isOptionChecked[option] = false;
            });
            if (onChange) {
                onChange([]);
            }
        } else {
            setValue(options);
            options.forEach((option)=>{
                isOptionChecked[option] = true;
            });
            if (onChange) {
                onChange(options);
            }
        }
    };
    const renderedOptions = /*#__PURE__*/ jsxs(Fragment, {
        children: [
            showAllOption && multiSelect && /*#__PURE__*/ jsx(DialogComboboxOptionListCheckboxItem, {
                value: "all",
                onChange: handleSelectAll,
                checked: value.length === options.length,
                indeterminate: Boolean(value.length) && value.length !== options.length,
                children: allOptionLabel
            }),
            options && options.length > 0 ? options.map((option, key)=>multiSelect ? /*#__PURE__*/ jsx(DialogComboboxOptionListCheckboxItem, {
                    value: option,
                    checked: isOptionChecked[option],
                    onChange: handleUpdate,
                    children: option
                }, key) : /*#__PURE__*/ jsx(DialogComboboxOptionListSelectItem, {
                    value: option,
                    checked: isOptionChecked[option],
                    onChange: handleUpdate,
                    children: option
                }, key)) : /*#__PURE__*/ jsx(EmptyResults, {})
        ]
    });
    const optionList = /*#__PURE__*/ jsx(DialogComboboxOptionList, {
        children: withSearch ? /*#__PURE__*/ jsx(DialogComboboxOptionListSearch, {
            hasWrapper: true,
            children: renderedOptions
        }) : renderedOptions
    });
    return /*#__PURE__*/ jsx("div", {
        ref: ref,
        "aria-busy": loading,
        css: {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-start',
            width: '100%'
        },
        ...restProps,
        children: /*#__PURE__*/ jsx(DialogComboboxOptionListContextProvider, {
            value: {
                isInsideDialogComboboxOptionList: true,
                lookAhead,
                setLookAhead
            },
            children: /*#__PURE__*/ jsx(Fragment, {
                children: loading ? withProgressiveLoading ? /*#__PURE__*/ jsxs(Fragment, {
                    children: [
                        optionList,
                        /*#__PURE__*/ jsx(LoadingSpinner, {
                            "aria-label": "Loading",
                            alt: "Loading spinner",
                            loadingDescription: loadingDescription
                        })
                    ]
                }) : /*#__PURE__*/ jsx(LoadingSpinner, {
                    "aria-label": "Loading",
                    alt: "Loading spinner",
                    loadingDescription: loadingDescription
                }) : optionList
            })
        })
    });
});

const DialogComboboxSectionHeader = ({ children, ...props })=>{
    const { isInsideDialogCombobox } = useDialogComboboxContext();
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxSectionHeader` must be used within `DialogCombobox`');
    }
    return /*#__PURE__*/ jsx(SectionHeader, {
        ...props,
        children: children
    });
};

const DialogComboboxSeparator = (props)=>{
    const { isInsideDialogCombobox } = useDialogComboboxContext();
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxSeparator` must be used within `DialogCombobox`');
    }
    return /*#__PURE__*/ jsx(Separator$2, {
        ...props
    });
};

const getTriggerWrapperStyles = (theme, clsPrefix, removable, disabled, width, useNewFormUISpacing)=>/*#__PURE__*/ css(importantify({
        display: 'inline-flex',
        alignItems: 'center',
        ...useNewFormUISpacing && {
            [`& + .${clsPrefix}-form-message`]: {
                marginTop: theme.spacing.sm
            }
        },
        ...width && {
            width: width
        },
        ...removable && {
            '& > button:last-of-type': importantify({
                borderBottomLeftRadius: 0,
                borderTopLeftRadius: 0,
                marginLeft: -1
            })
        },
        ...disabled && {
            cursor: 'not-allowed'
        }
    }));
const getTriggerStyles = (theme, disabled = false, maxWidth, minWidth, removable, width, validationState, isBare, isSelect, triggerSize)=>{
    const removeButtonInteractionStyles = {
        ...removable && {
            zIndex: theme.options.zIndexBase + 2,
            '&& + button': {
                marginLeft: -1,
                zIndex: theme.options.zIndexBase + 1
            }
        }
    };
    const validationColor = getValidationStateColor(theme, validationState);
    return /*#__PURE__*/ css(importantify({
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        maxWidth,
        minWidth,
        justifyContent: 'flex-start',
        background: 'transparent',
        padding: isBare ? 0 : triggerSize === 'small' ? '0 8px' : '6px 8px 6px 12px',
        boxSizing: 'border-box',
        height: isBare ? theme.typography.lineHeightBase : triggerSize === 'small' ? SMALL_BUTTON_HEIGHT$2 : theme.general.heightSm,
        border: isBare ? 'none' : `1px solid ${theme.colors.actionDefaultBorderDefault}`,
        ...!isBare && {
            boxShadow: theme.shadows.xs
        },
        borderRadius: theme.borders.borderRadiusSm,
        color: theme.colors.textPrimary,
        lineHeight: theme.typography.lineHeightBase,
        fontSize: theme.typography.fontSizeBase,
        cursor: 'pointer',
        ...width && {
            width: width,
            // Only set flex: 1 to items with width, otherwise in flex containers the trigger will take up all the space and break current usages that depend on content for width
            flex: 1
        },
        ...removable && {
            borderBottomRightRadius: 0,
            borderTopRightRadius: 0,
            borderRightColor: 'transparent'
        },
        '&:hover': {
            background: isBare ? 'transparent' : theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionDefaultBorderHover,
            ...removeButtonInteractionStyles
        },
        '&:focus': {
            borderColor: theme.colors.actionDefaultBorderFocus,
            outline: 'none',
            ...removeButtonInteractionStyles
        },
        '&:focus, &[data-state="open"]': {
            outlineColor: theme.colors.actionDefaultBorderFocus,
            outlineWidth: 2,
            outlineOffset: -2,
            outlineStyle: 'solid'
        },
        ...validationState && {
            borderColor: validationColor,
            '&:hover': {
                borderColor: validationColor
            },
            '&:focus': {
                outlineColor: validationColor,
                outlineOffset: -2
            }
        },
        ...isSelect && !disabled && {
            '&&, &&:hover, &&:focus': {
                background: 'transparent'
            },
            '&&:hover': {
                borderColor: theme.colors.actionDefaultBorderHover
            },
            '&&:focus, &[data-state="open"]': {
                borderColor: 'transparent'
            }
        },
        [`&[disabled]`]: {
            background: theme.colors.actionDisabledBackground,
            color: theme.colors.actionDisabledText,
            pointerEvents: 'none',
            userSelect: 'none',
            borderColor: theme.colors.actionDisabledBorder
        }
    }));
};
const DialogComboboxTrigger = /*#__PURE__*/ forwardRef(({ removable = false, onRemove, children, minWidth = 0, maxWidth = 9999, showTagAfterValueCount = 3, allowClear = true, controlled, onClear, wrapperProps, width, withChevronIcon = true, validationState, withInlineLabel = true, placeholder, id: legacyId, isBare = false, triggerSize = 'middle', renderDisplayedValue: formatDisplayedValue = (value)=>value, ...restProps }, forwardedRef)=>{
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { label, id: topLevelId, value, isInsideDialogCombobox, multiSelect, setValue } = useDialogComboboxContext();
    const { isSelect, placeholder: selectPlaceholder } = useSelectContext();
    const useNewFormUISpacing = safex('databricks.fe.designsystem.useNewFormUISpacing', false);
    const id = topLevelId ?? legacyId;
    if (!isInsideDialogCombobox) {
        throw new Error('`DialogComboboxTrigger` must be used within `DialogCombobox`');
    }
    const handleRemove = ()=>{
        if (!onRemove) {
            // eslint-disable-next-line no-console -- TODO(FEINF-3587)
            console.warn('DialogCombobox.Trigger: Attempted remove without providing onRemove handler');
        } else {
            onRemove();
        }
    };
    const handleClear = (e)=>{
        e.stopPropagation();
        if (controlled) {
            setValue([]);
            onClear?.();
        } else if (!onClear) {
            // eslint-disable-next-line no-console -- TODO(FEINF-3587)
            console.warn('DialogCombobox.Trigger: Attempted clear without providing onClear handler');
        } else {
            onClear();
        }
    };
    const [showTooltip, setShowTooltip] = React__default.useState();
    const triggerContentRef = React__default.useRef(null);
    useEffect(()=>{
        if (value?.length > showTagAfterValueCount) {
            setShowTooltip(true);
        } else if (triggerContentRef.current) {
            const { clientWidth, scrollWidth } = triggerContentRef.current;
            setShowTooltip(clientWidth < scrollWidth);
        }
    }, [
        showTagAfterValueCount,
        value
    ]);
    const renderFormattedValue = (v, index)=>{
        const formattedValue = formatDisplayedValue(v);
        return /*#__PURE__*/ jsxs(React__default.Fragment, {
            children: [
                index > 0 && ', ',
                typeof formattedValue === 'string' ? formattedValue : /*#__PURE__*/ jsx("span", {
                    children: formattedValue
                })
            ]
        }, index);
    };
    const getStringValue = (v)=>{
        const formattedValue = formatDisplayedValue(v);
        return typeof formattedValue === 'string' ? formattedValue : v;
    };
    const numValues = Array.isArray(value) ? value.length : 1;
    const concatenatedValues = Array.isArray(value) ? /*#__PURE__*/ jsxs(Fragment, {
        children: [
            value.slice(0, numValues > 10 ? 10 : undefined).map(renderFormattedValue),
            numValues > 10 && ` + ${numValues - 10}`
        ]
    }) : renderFormattedValue(value, 0);
    const displayedValues = /*#__PURE__*/ jsx("span", {
        children: concatenatedValues
    });
    const valuesBeforeBadge = Array.isArray(value) ? /*#__PURE__*/ jsx(Fragment, {
        children: value.slice(0, showTagAfterValueCount).map(renderFormattedValue)
    }) : renderFormattedValue(value, 0);
    let ariaLabel = '';
    if (!isSelect && !id && label) {
        ariaLabel = /*#__PURE__*/ React__default.isValidElement(label) ? 'Dialog Combobox' : `${label}`;
        if (value?.length) {
            const stringValues = Array.isArray(value) ? value.map(getStringValue).join(', ') : getStringValue(value);
            ariaLabel += multiSelect ? `, multiselectable, ${value.length} options selected: ${stringValues}` : `, selected option: ${stringValues}`;
        } else {
            ariaLabel += multiSelect ? ', multiselectable, 0 options selected' : ', no option selected';
        }
    } else if (isSelect) {
        ariaLabel = ((typeof label === 'string' ? label : '') || restProps['aria-label']) ?? '';
    }
    const customSelectContent = isSelect && children ? children : null;
    const dialogComboboxClassname = !isSelect ? `${classNamePrefix}-dialogcombobox` : '';
    const selectV2Classname = isSelect ? `${classNamePrefix}-selectv2` : '';
    const triggerContent = isSelect ? /*#__PURE__*/ jsxs(Popover.Trigger, {
        ...ariaLabel && {
            'aria-label': ariaLabel
        },
        ref: forwardedRef,
        role: "combobox",
        "aria-haspopup": "listbox",
        "aria-invalid": validationState === 'error',
        id: id,
        ...restProps,
        css: getTriggerStyles(theme, restProps.disabled, maxWidth, minWidth, removable, width, validationState, isBare, isSelect, triggerSize),
        children: [
            /*#__PURE__*/ jsx("span", {
                css: {
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    height: theme.typography.lineHeightBase,
                    marginRight: 'auto'
                },
                ref: triggerContentRef,
                children: value?.length ? customSelectContent ?? displayedValues : /*#__PURE__*/ jsx("span", {
                    css: {
                        color: theme.colors.textPlaceholder
                    },
                    children: selectPlaceholder
                })
            }),
            allowClear && value?.length ? /*#__PURE__*/ jsx(ClearSelectionButton, {
                onClick: handleClear
            }) : null,
            /*#__PURE__*/ jsx(ChevronDownIcon, {
                css: {
                    color: restProps.disabled ? theme.colors.actionDisabledText : theme.colors.textSecondary,
                    marginLeft: theme.spacing.xs
                }
            })
        ]
    }) : /*#__PURE__*/ jsxs(Popover.Trigger, {
        id: id,
        ...ariaLabel && {
            'aria-label': ariaLabel
        },
        ref: forwardedRef,
        role: "combobox",
        "aria-haspopup": "listbox",
        "aria-invalid": validationState === 'error',
        ...restProps,
        css: getTriggerStyles(theme, restProps.disabled, maxWidth, minWidth, removable, width, validationState, isBare, isSelect, triggerSize),
        children: [
            /*#__PURE__*/ jsxs("span", {
                css: {
                    display: 'flex',
                    alignItems: 'center',
                    height: theme.typography.lineHeightBase,
                    marginRight: 'auto',
                    '&, & > *': {
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                    }
                },
                ref: triggerContentRef,
                children: [
                    withInlineLabel ? /*#__PURE__*/ jsxs("span", {
                        css: {
                            height: theme.typography.lineHeightBase,
                            marginRight: theme.spacing.xs,
                            whiteSpace: 'unset',
                            overflow: 'unset',
                            textOverflow: 'unset'
                        },
                        children: [
                            label,
                            value?.length ? ':' : null
                        ]
                    }) : !value?.length && /*#__PURE__*/ jsx("span", {
                        css: {
                            color: theme.colors.textPlaceholder
                        },
                        children: placeholder
                    }),
                    value?.length > showTagAfterValueCount ? /*#__PURE__*/ jsxs(Fragment, {
                        children: [
                            /*#__PURE__*/ jsx("span", {
                                style: {
                                    marginRight: theme.spacing.xs
                                },
                                children: valuesBeforeBadge
                            }),
                            /*#__PURE__*/ jsx(DialogComboboxCountBadge, {
                                countStartAt: showTagAfterValueCount,
                                role: "status",
                                "aria-label": "Selected options count"
                            })
                        ]
                    }) : displayedValues
                ]
            }),
            allowClear && value?.length ? /*#__PURE__*/ jsx(ClearSelectionButton, {
                onClick: handleClear
            }) : null,
            withChevronIcon ? /*#__PURE__*/ jsx(ChevronDownIcon, {
                css: {
                    color: restProps.disabled ? theme.colors.actionDisabledText : theme.colors.textSecondary,
                    justifySelf: 'flex-end',
                    marginLeft: theme.spacing.xs
                }
            }) : null
        ]
    });
    const dataComponentProps = useComponentFinderContext(DesignSystemEventProviderComponentTypes.DialogCombobox);
    return /*#__PURE__*/ jsxs("div", {
        ...wrapperProps,
        className: `${restProps?.className ?? ''} ${dialogComboboxClassname} ${selectV2Classname}`.trim(),
        css: [
            getTriggerWrapperStyles(theme, classNamePrefix, removable, restProps.disabled, width, useNewFormUISpacing),
            wrapperProps?.css
        ],
        ...addDebugOutlineIfEnabled(),
        ...dataComponentProps,
        children: [
            showTooltip && value?.length ? /*#__PURE__*/ jsx(LegacyTooltip, {
                title: customSelectContent ?? displayedValues,
                children: triggerContent
            }) : triggerContent,
            removable && /*#__PURE__*/ jsx(Button, {
                componentId: "codegen_design-system_src_design-system_dialogcombobox_dialogcomboboxtrigger.tsx_355",
                "aria-label": `Remove ${label}`,
                onClick: handleRemove,
                dangerouslySetForceIconStyles: true,
                children: /*#__PURE__*/ jsx(CloseIcon, {
                    "aria-label": `Remove ${label}`,
                    "aria-hidden": "false"
                })
            })
        ]
    });
});
/**
 * A custom button trigger that can be wrapped around any button.
 */ const DialogComboboxCustomButtonTriggerWrapper = ({ children })=>{
    return /*#__PURE__*/ jsx(Popover.Trigger, {
        asChild: true,
        children: children
    });
};

const DEFAULT_WIDTH = 320;
const MIN_WIDTH = 320;
const MAX_WIDTH = '90vw';
const DEFAULT_POSITION = 'right';
const ZINDEX_OVERLAY = 1;
const ZINDEX_CONTENT = ZINDEX_OVERLAY + 1;
/** Context to track if drawer is nested within a parent drawer */ const DrawerContext = /*#__PURE__*/ React__default.createContext({
    isParentDrawerOpen: false
});
const Content$2 = ({ children, footer, title, width, position: positionOverride, useCustomScrollBehavior, expandContentToFullHeight, disableOpenAutoFocus, onInteractOutside, seeThrough, hideClose, closeOnClickOutside, onCloseClick, componentId = 'design_system.drawer.content', analyticsEvents = [
    DesignSystemEventProviderAnalyticsEventTypes.OnView
], size = 'default', ...props })=>{
    const { getPopupContainer } = useDesignSystemContext();
    const { theme } = useDesignSystemTheme();
    const horizontalContentPadding = size === 'small' ? theme.spacing.md : theme.spacing.lg;
    const [shouldContentBeFocusable, setShouldContentBeFocusable] = useState(false);
    const contentContainerRef = useRef(null);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents, [
        analyticsEvents
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Drawer,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents
    });
    const { isParentDrawerOpen } = React__default.useContext(DrawerContext);
    const { elementRef: onViewRef } = useNotifyOnFirstView({
        onView: eventContext.onView
    });
    const contentRef = useCallback((node)=>{
        if (!node || !node.clientHeight) return;
        setShouldContentBeFocusable(node.scrollHeight > node.clientHeight);
    }, []);
    const mergedContentRef = useMergeRefs([
        contentRef,
        onViewRef
    ]);
    const position = positionOverride ?? DEFAULT_POSITION;
    const overlayShow = position === 'right' ? /*#__PURE__*/ keyframes({
        '0%': {
            transform: 'translate(100%, 0)'
        },
        '100%': {
            transform: 'translate(0, 0)'
        }
    }) : /*#__PURE__*/ keyframes({
        '0%': {
            transform: 'translate(-100%, 0)'
        },
        '100%': {
            transform: 'translate(0, 0)'
        }
    });
    const dialogPrimitiveContentStyle = /*#__PURE__*/ css({
        color: theme.colors.textPrimary,
        backgroundColor: theme.colors.backgroundPrimary,
        boxShadow: theme.shadows.xl,
        position: 'fixed',
        top: 0,
        left: position === 'left' ? 0 : undefined,
        right: position === 'right' ? 0 : undefined,
        boxSizing: 'border-box',
        width: width ?? DEFAULT_WIDTH,
        minWidth: MIN_WIDTH,
        maxWidth: MAX_WIDTH,
        zIndex: theme.options.zIndexBase + ZINDEX_CONTENT,
        height: '100vh',
        paddingTop: size === 'small' ? theme.spacing.sm : theme.spacing.md,
        paddingLeft: 0,
        paddingBottom: 0,
        paddingRight: 0,
        overflow: 'hidden',
        '&:focus': {
            outline: 'none'
        },
        ...isParentDrawerOpen ? {} : {
            '@media (prefers-reduced-motion: no-preference)': {
                animation: `${overlayShow} 350ms cubic-bezier(0.16, 1, 0.3, 1)`
            }
        }
    });
    return /*#__PURE__*/ jsxs(DialogPrimitive.Portal, {
        container: getPopupContainer && getPopupContainer(),
        children: [
            /*#__PURE__*/ jsx(DialogPrimitive.Overlay, {
                "data-testid": "drawer-overlay",
                css: {
                    backgroundColor: theme.colors.overlayOverlay,
                    position: 'fixed',
                    inset: 0,
                    // needed so that it covers the PersonaNavSidebar
                    zIndex: theme.options.zIndexBase + ZINDEX_OVERLAY,
                    opacity: seeThrough || isParentDrawerOpen ? 0 : 1
                },
                onClick: closeOnClickOutside ? onCloseClick : undefined
            }),
            /*#__PURE__*/ jsx(DialogPrimitive.DialogContent, {
                ...addDebugOutlineIfEnabled(),
                css: dialogPrimitiveContentStyle,
                style: {
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'flex-start',
                    opacity: seeThrough ? 0 : 1,
                    ...theme.isDarkMode && {
                        borderLeft: `1px solid ${theme.colors.borderDecorative}`
                    }
                },
                onWheel: (e)=>{
                    e.stopPropagation();
                },
                onTouchMove: (e)=>{
                    e.stopPropagation();
                },
                "aria-hidden": seeThrough,
                ref: contentContainerRef,
                onOpenAutoFocus: (event)=>{
                    if (disableOpenAutoFocus) {
                        event.preventDefault();
                    }
                },
                onInteractOutside: onInteractOutside,
                ...props,
                ...eventContext.dataComponentProps,
                children: /*#__PURE__*/ jsxs(ApplyDesignSystemContextOverrides, {
                    getPopupContainer: ()=>contentContainerRef.current ?? document.body,
                    children: [
                        (title || !hideClose) && /*#__PURE__*/ jsxs("div", {
                            css: {
                                flexGrow: 0,
                                flexShrink: 1,
                                display: 'flex',
                                flexDirection: 'row',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                paddingRight: horizontalContentPadding,
                                paddingLeft: horizontalContentPadding,
                                marginBottom: theme.spacing.sm
                            },
                            children: [
                                /*#__PURE__*/ jsx(DialogPrimitive.Title, {
                                    title: typeof title === 'string' ? title : undefined,
                                    asChild: typeof title === 'string',
                                    css: {
                                        flexGrow: 1,
                                        marginBottom: 0,
                                        marginTop: 0,
                                        whiteSpace: 'nowrap',
                                        overflow: 'hidden'
                                    },
                                    children: typeof title === 'string' ? /*#__PURE__*/ jsx(Typography.Title, {
                                        elementLevel: 2,
                                        level: size === 'small' ? 3 : 2,
                                        withoutMargins: true,
                                        ellipsis: true,
                                        children: title
                                    }) : title
                                }),
                                !hideClose && /*#__PURE__*/ jsx(DialogPrimitive.Close, {
                                    asChild: true,
                                    css: {
                                        flexShrink: 1,
                                        marginLeft: theme.spacing.xs
                                    },
                                    onClick: onCloseClick,
                                    children: /*#__PURE__*/ jsx(Button, {
                                        componentId: `${componentId}.close`,
                                        "aria-label": "Close",
                                        icon: /*#__PURE__*/ jsx(CloseIcon, {}),
                                        size: size === 'small' ? 'small' : undefined
                                    })
                                })
                            ]
                        }),
                        /*#__PURE__*/ jsxs("div", {
                            ref: mergedContentRef,
                            // Needed to make drawer content focusable when scrollable for keyboard-only users to be able to focus & scroll
                            // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
                            tabIndex: shouldContentBeFocusable ? 0 : -1,
                            css: {
                                // in order to have specific content in the drawer scroll with fixed title
                                // hide overflow here and remove padding on the right side; content will be responsible for setting right padding
                                // so that the scrollbar will appear in the padding right gutter
                                paddingRight: useCustomScrollBehavior ? 0 : horizontalContentPadding,
                                paddingLeft: horizontalContentPadding,
                                overflowY: useCustomScrollBehavior ? 'hidden' : 'auto',
                                height: expandContentToFullHeight ? '100%' : undefined,
                                ...!useCustomScrollBehavior ? getShadowScrollStyles(theme) : {}
                            },
                            children: [
                                /*#__PURE__*/ jsx(DrawerContext.Provider, {
                                    value: {
                                        isParentDrawerOpen: true
                                    },
                                    children: children
                                }),
                                !footer && /*#__PURE__*/ jsx(Spacer, {
                                    size: size === 'small' ? 'md' : 'lg'
                                })
                            ]
                        }),
                        footer && /*#__PURE__*/ jsx("div", {
                            style: {
                                paddingTop: theme.spacing.md,
                                paddingRight: horizontalContentPadding,
                                paddingLeft: horizontalContentPadding,
                                paddingBottom: size === 'small' ? theme.spacing.md : theme.spacing.lg,
                                flexGrow: 0,
                                flexShrink: 1
                            },
                            children: footer
                        })
                    ]
                })
            })
        ]
    });
};
function Root$4(props) {
    return /*#__PURE__*/ jsx(DialogPrimitive.Root, {
        ...props
    });
}
function Trigger$1(props) {
    return /*#__PURE__*/ jsx(DialogPrimitive.Trigger, {
        asChild: true,
        ...props
    });
}

var Drawer = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Content: Content$2,
  Root: Root$4,
  Trigger: Trigger$1
});

/**
 * @deprecated Use `DropdownMenu` instead.
 */ const Dropdown = ({ dangerouslySetAntdProps, ...props })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Dropdown$1, {
            ...addDebugOutlineIfEnabled(),
            mouseLeaveDelay: 0.25,
            ...props,
            overlayStyle: {
                zIndex: theme.options.zIndexBase + 50,
                ...props.overlayStyle
            },
            ...dangerouslySetAntdProps
        })
    });
};

const { Title: Title$1, Paragraph } = Typography;
function getEmptyStyles(theme) {
    const styles = {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        textAlign: 'center',
        maxWidth: 600,
        wordBreak: 'break-word',
        // TODO: This isn't ideal, but migrating to a safer selector would require a SAFE flag / careful migration.
        '> [role="img"]': {
            // Set size of image to 64px
            fontSize: 64,
            color: theme.colors.actionDisabledText,
            marginBottom: theme.spacing.md
        }
    };
    return /*#__PURE__*/ css(styles);
}
function getEmptyTitleStyles(theme, clsPrefix) {
    const styles = {
        [`&.${clsPrefix}-typography`]: {
            color: theme.colors.textSecondary,
            marginTop: 0,
            marginBottom: 0
        }
    };
    return /*#__PURE__*/ css(styles);
}
function getEmptyDescriptionStyles(theme, clsPrefix) {
    const styles = {
        [`&.${clsPrefix}-typography`]: {
            color: theme.colors.textSecondary,
            marginBottom: theme.spacing.md
        }
    };
    return /*#__PURE__*/ css(styles);
}
const Empty = (props)=>{
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { title, description, image = /*#__PURE__*/ jsx(ListIcon, {}), button, dangerouslyAppendEmotionCSS, ...dataProps } = props;
    return /*#__PURE__*/ jsx("div", {
        ...dataProps,
        ...addDebugOutlineIfEnabled(),
        css: {
            display: 'flex',
            justifyContent: 'center'
        },
        children: /*#__PURE__*/ jsxs("div", {
            css: [
                getEmptyStyles(theme),
                dangerouslyAppendEmotionCSS
            ],
            children: [
                image,
                title && /*#__PURE__*/ jsx(Title$1, {
                    level: 3,
                    css: getEmptyTitleStyles(theme, classNamePrefix),
                    children: title
                }),
                /*#__PURE__*/ jsx(Paragraph, {
                    css: getEmptyDescriptionStyles(theme, classNamePrefix),
                    children: description
                }),
                button
            ]
        })
    });
};

const getMessageStyles = (clsPrefix, theme)=>{
    const errorClass = `.${clsPrefix}-form-error-message`;
    const infoClass = `.${clsPrefix}-form-info-message`;
    const successClass = `.${clsPrefix}-form-success-message`;
    const warningClass = `.${clsPrefix}-form-warning-message`;
    const styles = {
        '&&': {
            lineHeight: theme.typography.lineHeightSm,
            fontSize: theme.typography.fontSizeSm,
            marginTop: theme.spacing.sm,
            display: 'flex',
            alignItems: 'start'
        },
        [`&${errorClass}`]: {
            color: theme.colors.actionDangerPrimaryBackgroundDefault
        },
        [`&${infoClass}`]: {
            color: theme.colors.textPrimary
        },
        [`&${successClass}`]: {
            color: theme.colors.textValidationSuccess
        },
        [`&${warningClass}`]: {
            color: theme.colors.textValidationWarning
        },
        ...getAnimationCss(theme.options.enableAnimation)
    };
    return /*#__PURE__*/ css(styles);
};
const VALIDATION_STATE_ICONS = {
    error: DangerIcon,
    success: CheckCircleIcon,
    warning: WarningIcon,
    info: InfoSmallIcon
};
function FormMessage({ id, message, type = 'error', className = '', css }) {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const stateClass = `${classNamePrefix}-form-${type}-message`;
    const StateIcon = VALIDATION_STATE_ICONS[type];
    const wrapperClass = `${classNamePrefix}-form-message ${className} ${stateClass}`.trim();
    return /*#__PURE__*/ jsxs("div", {
        ...id && {
            id
        },
        className: wrapperClass,
        ...addDebugOutlineIfEnabled(),
        css: [
            getMessageStyles(classNamePrefix, theme),
            css
        ],
        role: "alert",
        children: [
            /*#__PURE__*/ jsx(StateIcon, {}),
            /*#__PURE__*/ jsx("div", {
                style: {
                    paddingLeft: theme.spacing.xs
                },
                children: message
            })
        ]
    });
}

const getHintStyles = (classNamePrefix, theme)=>{
    const styles = {
        display: 'block',
        color: theme.colors.textSecondary,
        lineHeight: theme.typography.lineHeightSm,
        fontSize: theme.typography.fontSizeSm,
        [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-input-affix-wrapper, && + .${classNamePrefix}-select, && + .${classNamePrefix}-selectv2, && + .${classNamePrefix}-dialogcombobox, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group, && + .${classNamePrefix}-typeahead-combobox, && + .${classNamePrefix}-datepicker, && + .${classNamePrefix}-rangepicker`]: {
            marginTop: theme.spacing.sm
        }
    };
    return /*#__PURE__*/ css(styles);
};
const Hint = (props)=>{
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const { className, ...restProps } = props;
    return /*#__PURE__*/ jsx("span", {
        ...addDebugOutlineIfEnabled(),
        className: classnames(`${classNamePrefix}-hint`, className),
        css: getHintStyles(classNamePrefix, theme),
        ...restProps
    });
};

const getLabelStyles$1 = (theme, { inline })=>{
    const styles = {
        '&&': {
            color: theme.colors.textPrimary,
            fontWeight: theme.typography.typographyBoldFontWeight,
            display: inline ? 'inline' : 'block',
            lineHeight: theme.typography.lineHeightBase
        }
    };
    return /*#__PURE__*/ css(styles);
};
const getLabelWrapperStyles = (classNamePrefix, theme)=>{
    const styles = {
        display: 'flex',
        gap: theme.spacing.xs,
        alignItems: 'center',
        [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-input-affix-wrapper, && + .${classNamePrefix}-select, && + .${classNamePrefix}-selectv2, && + .${classNamePrefix}-dialogcombobox, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group, && + .${classNamePrefix}-typeahead-combobox, && + .${classNamePrefix}-datepicker, && + .${classNamePrefix}-rangepicker`]: {
            marginTop: theme.spacing.sm
        }
    };
    return /*#__PURE__*/ css(styles);
};
const Label = (props)=>{
    const { children, className, inline, required, infoPopoverContents, infoPopoverProps = {}, labelTopRightElement, ...restProps } = props; // Destructure the new prop
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const label = /*#__PURE__*/ jsx("label", {
        ...addDebugOutlineIfEnabled(),
        css: [
            getLabelStyles$1(theme, {
                inline
            }),
            ...!infoPopoverContents ? [
                getLabelWrapperStyles(classNamePrefix, theme)
            ] : []
        ],
        className: classnames(`${classNamePrefix}-label`, className),
        ...restProps,
        children: /*#__PURE__*/ jsxs("span", {
            css: {
                display: 'flex',
                alignItems: 'center'
            },
            children: [
                children,
                required && /*#__PURE__*/ jsx("span", {
                    "aria-hidden": "true",
                    children: "*"
                }),
                labelTopRightElement && /*#__PURE__*/ jsx("span", {
                    css: {
                        marginLeft: 'auto'
                    },
                    children: labelTopRightElement
                })
            ]
        })
    });
    return infoPopoverContents ? /*#__PURE__*/ jsxs("div", {
        css: getLabelWrapperStyles(classNamePrefix, theme),
        children: [
            label,
            /*#__PURE__*/ jsx(InfoPopover, {
                ...infoPopoverProps,
                children: infoPopoverContents
            })
        ]
    }) : label;
};

function getSelectEmotionStyles({ clsPrefix, theme, validationState, useNewFormUISpacing }) {
    const classFocused = `.${clsPrefix}-focused`;
    const classOpen = `.${clsPrefix}-open`;
    const classSingle = `.${clsPrefix}-single`;
    const classSelector = `.${clsPrefix}-selector`;
    const classDisabled = `.${clsPrefix}-disabled`;
    const classMultiple = `.${clsPrefix}-multiple`;
    const classItem = `.${clsPrefix}-selection-item`;
    const classItemOverflowContainer = `.${clsPrefix}-selection-overflow`;
    const classItemOverflowItem = `.${clsPrefix}-selection-overflow-item`;
    const classItemOverflowSuffix = `.${clsPrefix}-selection-overflow-item-suffix`;
    const classArrow = `.${clsPrefix}-arrow`;
    const classArrowLoading = `.${clsPrefix}-arrow-loading`;
    const classPlaceholder = `.${clsPrefix}-selection-placeholder`;
    const classCloseButton = `.${clsPrefix}-selection-item-remove`;
    const classSearch = `.${clsPrefix}-selection-search`;
    const classShowSearch = `.${clsPrefix}-show-search`;
    const classSearchClear = `.${clsPrefix}-clear`;
    const classAllowClear = `.${clsPrefix}-allow-clear`;
    const classSearchInput = `.${clsPrefix}-selection-search-input`;
    const classFormMessage = `.${clsPrefix.replace('-select', '')}-form-message`;
    const validationColor = getValidationStateColor(theme, validationState);
    const styles = {
        ...addDebugOutlineStylesIfEnabled(theme),
        ...useNewFormUISpacing && {
            [`& + ${classFormMessage}`]: {
                marginTop: theme.spacing.sm
            }
        },
        '&:hover': {
            [classSelector]: {
                borderColor: theme.colors.actionDefaultBorderHover
            }
        },
        [classSelector]: {
            paddingLeft: 12,
            // Only the select _item_ is clickable, so we need to have zero padding here, and add it on the item itself,
            // to make sure the whole select is clickable.
            paddingRight: 0,
            color: theme.colors.textPrimary,
            backgroundColor: 'transparent',
            height: theme.general.heightSm,
            borderColor: theme.colors.actionDefaultBorderDefault,
            '::after': {
                lineHeight: theme.typography.lineHeightBase
            },
            '::before': {
                lineHeight: theme.typography.lineHeightBase
            }
        },
        [classSingle]: {
            [`&${classSelector}`]: {
                height: theme.general.heightSm
            }
        },
        [classItem]: {
            color: theme.colors.textPrimary,
            paddingRight: 32,
            lineHeight: theme.typography.lineHeightBase,
            paddingTop: 5,
            paddingBottom: 5
        },
        // Note: This supports search, which we don't support. The styles here support legacy usages.
        [classSearch]: {
            right: 24,
            left: 8,
            marginInlineStart: 4,
            [classSearchInput]: {
                color: theme.colors.actionDefaultTextDefault,
                height: 24
            }
        },
        [`&${classSingle}`]: {
            [classSearchInput]: {
                height: theme.general.heightSm
            }
        },
        // Note: This supports search, which we don't support. The styles here support legacy usages.
        [`&${classShowSearch}${classOpen}${classSingle}`]: {
            [classItem]: {
                color: theme.colors.actionDisabledText
            }
        },
        // Note: This supports search, which we don't support. The styles here support legacy usages.
        [classSearchClear]: {
            right: 24,
            backgroundColor: 'transparent'
        },
        [`&${classFocused}`]: {
            [classSelector]: {
                outlineColor: theme.colors.actionDefaultBorderFocus,
                outlineWidth: 2,
                outlineOffset: -2,
                outlineStyle: 'solid',
                borderColor: 'transparent',
                boxShadow: 'none'
            }
        },
        [`&${classDisabled}`]: {
            [classSelector]: {
                backgroundColor: theme.colors.actionDisabledBackground,
                color: theme.colors.actionDisabledText,
                border: 'transparent'
            },
            [classItem]: {
                color: theme.colors.actionDisabledText
            },
            [classArrow]: {
                color: theme.colors.actionDisabledText
            }
        },
        [classArrow]: {
            height: theme.general.iconFontSize,
            width: theme.general.iconFontSize,
            top: (theme.general.heightSm - theme.general.iconFontSize) / 2,
            marginTop: 0,
            color: theme.colors.textSecondary,
            fontSize: theme.general.iconFontSize,
            '.anticon': {
                // For some reason ant sets this to 'auto'. Need to set it back to 'none' to allow the element below to receive
                // the click event.
                pointerEvents: 'none',
                // anticon default line height is 0 and that wrongly shifts the icon down
                lineHeight: 1
            },
            [`&${classArrowLoading}`]: {
                top: (theme.general.heightSm - theme.general.iconFontSize) / 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: theme.general.iconFontSize
            }
        },
        [classPlaceholder]: {
            color: theme.colors.textPlaceholder,
            right: 'auto',
            left: 'auto',
            width: '100%',
            paddingRight: 32,
            lineHeight: theme.typography.lineHeightBase,
            alignSelf: 'center'
        },
        [`&${classMultiple}`]: {
            [classSelector]: {
                paddingTop: 3,
                paddingBottom: 3,
                paddingLeft: 8,
                paddingRight: 30,
                minHeight: theme.general.heightSm,
                height: 'auto',
                '&::after': {
                    margin: 0
                }
            },
            [classItem]: {
                backgroundColor: theme.colors.tagDefault,
                color: theme.colors.textPrimary,
                border: 'none',
                height: 20,
                lineHeight: theme.typography.lineHeightBase,
                fontSize: theme.typography.fontSizeBase,
                marginInlineEnd: 4,
                marginTop: 2,
                marginBottom: 2,
                paddingRight: 0,
                paddingTop: 0,
                paddingBottom: 0
            },
            [classItemOverflowContainer]: {
                minHeight: 24
            },
            [classItemOverflowItem]: {
                alignSelf: 'auto',
                height: 24,
                lineHeight: theme.typography.lineHeightBase
            },
            [classSearch]: {
                marginTop: 0,
                left: 0,
                right: 0
            },
            [`&${classDisabled}`]: {
                [classItem]: {
                    paddingRight: 2
                }
            },
            [classArrow]: {
                top: (theme.general.heightSm - theme.general.iconFontSize) / 2
            },
            [`&${classAllowClear}`]: {
                [classSearchClear]: {
                    top: (theme.general.heightSm - theme.general.iconFontSize + 4) / 2
                }
            },
            [classPlaceholder]: {
                // Compensate for the caret placeholder width
                paddingLeft: 4,
                color: theme.colors.textPlaceholder
            },
            [`&:not(${classFocused})`]: {
                [classItemOverflowSuffix]: {
                    // Do not keep the caret's placeholder at full height when not focused,
                    // because it introduces a new line even when not focused. Using display: none would break the caret
                    height: 0
                }
            }
        },
        [`&${classMultiple}${classDisabled}`]: {
            [classItem]: {
                color: theme.colors.actionDisabledText
            }
        },
        [`&${classAllowClear}`]: {
            [classItem]: {
                paddingRight: 0
            },
            [classSelector]: {
                paddingRight: 52
            },
            [classSearchClear]: {
                top: (theme.general.heightSm - theme.general.iconFontSize + 4) / 2,
                opacity: 100,
                width: theme.general.iconFontSize,
                height: theme.general.iconFontSize,
                marginTop: 0
            }
        },
        [classCloseButton]: {
            color: theme.colors.textPrimary,
            borderTopRightRadius: theme.legacyBorders.borderRadiusMd,
            borderBottomRightRadius: theme.legacyBorders.borderRadiusMd,
            height: theme.general.iconFontSize,
            width: theme.general.iconFontSize,
            lineHeight: theme.typography.lineHeightBase,
            paddingInlineEnd: 0,
            marginInlineEnd: 0,
            '& > .anticon': {
                height: theme.general.iconFontSize - 4,
                fontSize: theme.general.iconFontSize - 4
            },
            '&:hover': {
                color: theme.colors.actionTertiaryTextHover,
                backgroundColor: theme.colors.tagHover
            },
            '&:active': {
                color: theme.colors.actionTertiaryTextPress,
                backgroundColor: theme.colors.tagPress
            }
        },
        ...validationState && {
            [`& > ${classSelector}`]: {
                borderColor: validationColor,
                '&:hover': {
                    borderColor: validationColor
                }
            },
            [`&${classFocused} > ${classSelector}`]: {
                outlineColor: validationColor,
                outlineOffset: -2
            }
        },
        ...getAnimationCss(theme.options.enableAnimation)
    };
    const importantStyles = importantify(styles);
    return /*#__PURE__*/ css(importantStyles);
}
function getDropdownStyles(clsPrefix, theme, useNewBorderColors) {
    const classItem = `.${clsPrefix}-item-option`;
    const classItemActive = `.${clsPrefix}-item-option-active`;
    const classItemSelected = `.${clsPrefix}-item-option-selected`;
    const classItemState = `.${clsPrefix}-item-option-state`;
    const styles = {
        borderColor: useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative,
        borderWidth: 1,
        borderStyle: 'solid',
        zIndex: theme.options.zIndexBase + 50,
        boxShadow: theme.general.shadowLow,
        ...addDebugOutlineStylesIfEnabled(theme),
        [classItem]: {
            height: theme.general.heightSm
        },
        [classItemActive]: {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
            height: theme.general.heightSm,
            '&:hover': {
                backgroundColor: theme.colors.actionTertiaryBackgroundHover
            }
        },
        [classItemSelected]: {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
            fontWeight: 'normal',
            '&:hover': {
                backgroundColor: theme.colors.actionTertiaryBackgroundHover
            }
        },
        [classItemState]: {
            color: theme.colors.textSecondary,
            '& > span': {
                verticalAlign: 'middle'
            }
        },
        [`.${clsPrefix}-loading-options`]: {
            pointerEvents: 'none',
            margin: '0 auto',
            height: theme.general.heightSm,
            display: 'block'
        },
        ...getAnimationCss(theme.options.enableAnimation),
        ...getDarkModePortalStyles(theme, useNewBorderColors)
    };
    const importantStyles = importantify(styles);
    return /*#__PURE__*/ css(importantStyles);
}
function getLoadingIconStyles(theme) {
    return /*#__PURE__*/ css({
        fontSize: 20,
        color: theme.colors.textSecondary,
        lineHeight: '20px'
    });
}
const scrollbarVisibleItemsCount = 8;
const getIconSizeStyle = (theme, newIconDefault)=>importantify({
        fontSize: newIconDefault ?? theme.general.iconFontSize
    });
function DuboisSelect({ children, validationState, loading, loadingDescription = 'Select', mode, options, notFoundContent, optionFilterProp, dangerouslySetAntdProps, virtual, dropdownClassName, id, onDropdownVisibleChange, maxHeight, ...restProps }, ref) {
    const { theme, getPrefixedClassName } = useDesignSystemTheme();
    const useNewFormUISpacing = safex('databricks.fe.designsystem.useNewFormUISpacing', false);
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const clsPrefix = getPrefixedClassName('select');
    const [isOpen, setIsOpen] = useState(false);
    const [uniqueId$1, setUniqueId] = useState('');
    // Antd's default is 256, to show half an extra item when scrolling we add 0.5 height extra
    // Reducing to 5.5 as it's default with other components is not an option here because it would break existing usages relying on 8 items being shown by default
    const MAX_HEIGHT = maxHeight ?? theme.general.heightSm * 8.5;
    useEffect(()=>{
        setUniqueId(id || uniqueId('dubois-select-'));
    }, [
        id
    ]);
    useEffect(()=>{
        // Ant doesn't populate aria-expanded on init (only on user interaction) so we need to do it ourselves
        // in order to pass accessibility tests (Microsoft Accessibility Insights). See: JOBS-11125
        document.getElementById(uniqueId$1)?.setAttribute('aria-expanded', 'false');
    }, [
        uniqueId$1
    ]);
    return /*#__PURE__*/ jsx(ClassNames, {
        children: ({ css })=>{
            return /*#__PURE__*/ jsxs(DesignSystemAntDConfigProvider, {
                children: [
                    loading && /*#__PURE__*/ jsx(LoadingState, {
                        description: loadingDescription
                    }),
                    /*#__PURE__*/ jsx(Select$1, {
                        onDropdownVisibleChange: (visible)=>{
                            onDropdownVisibleChange?.(visible);
                            setIsOpen(visible);
                        },
                        ...!isOpen ? {
                            'aria-owns': undefined,
                            'aria-controls': undefined,
                            'aria-activedescendant': undefined
                        } : {},
                        id: uniqueId$1,
                        css: getSelectEmotionStyles({
                            clsPrefix,
                            theme,
                            validationState,
                            useNewFormUISpacing
                        }),
                        removeIcon: /*#__PURE__*/ jsx(CloseIcon, {
                            "aria-hidden": "false",
                            css: getIconSizeStyle(theme)
                        }),
                        clearIcon: /*#__PURE__*/ jsx(XCircleFillIcon, {
                            "aria-hidden": "false",
                            css: getIconSizeStyle(theme, 12),
                            "aria-label": "close-circle"
                        }),
                        ref: ref,
                        suffixIcon: loading && mode === 'tags' ? /*#__PURE__*/ jsx(LoadingIcon, {
                            spin: true,
                            "aria-label": "loading",
                            "aria-hidden": "false",
                            css: getIconSizeStyle(theme, 12)
                        }) : /*#__PURE__*/ jsx(ChevronDownIcon, {
                            css: getIconSizeStyle(theme)
                        }),
                        menuItemSelectedIcon: /*#__PURE__*/ jsx(CheckIcon, {
                            css: getIconSizeStyle(theme)
                        }),
                        showArrow: true,
                        dropdownMatchSelectWidth: true,
                        notFoundContent: notFoundContent ?? /*#__PURE__*/ jsx("div", {
                            css: {
                                color: theme.colors.textSecondary,
                                textAlign: 'center'
                            },
                            children: "No results found"
                        }),
                        dropdownClassName: css([
                            getDropdownStyles(clsPrefix, theme, useNewBorderColors),
                            dropdownClassName
                        ]),
                        listHeight: MAX_HEIGHT,
                        maxTagPlaceholder: (items)=>`+ ${items.length} more`,
                        mode: mode,
                        options: options,
                        loading: loading,
                        filterOption: true,
                        // NOTE(FEINF-1102): This is needed to avoid ghost scrollbar that generates error when clicked on exactly 8 elements
                        // Because by default AntD uses true for virtual, we want to replicate the same even if there are no children
                        virtual: virtual ?? (children && Array.isArray(children) && children.length !== scrollbarVisibleItemsCount || options && options.length !== scrollbarVisibleItemsCount || !children && !options),
                        optionFilterProp: optionFilterProp ?? 'children',
                        ...restProps,
                        ...dangerouslySetAntdProps,
                        children: loading && mode !== 'tags' ? /*#__PURE__*/ jsxs(Fragment, {
                            children: [
                                children,
                                /*#__PURE__*/ jsx(LegacyOption, {
                                    disabled: true,
                                    value: "select-loading-options",
                                    className: `${clsPrefix}-loading-options`,
                                    children: /*#__PURE__*/ jsx(LoadingIcon, {
                                        "aria-hidden": "false",
                                        spin: true,
                                        css: getLoadingIconStyles(theme),
                                        "aria-label": "loading"
                                    })
                                })
                            ]
                        }) : children
                    })
                ]
            });
        }
    });
}
const LegacySelectOption = /*#__PURE__*/ forwardRef(function Option(props, ref) {
    const { dangerouslySetAntdProps, ...restProps } = props;
    return /*#__PURE__*/ jsx(Select$1.Option, {
        ...restProps,
        ref: ref,
        ...dangerouslySetAntdProps
    });
});
// Needed for rc-select to not throw warning about our component not being Select.Option
LegacySelectOption.isSelectOption = true;
/**
 * @deprecated use LegacySelect.Option instead
 */ const LegacyOption = LegacySelectOption;
const LegacySelectOptGroup = /* #__PURE__ */ (()=>{
    const OptGroup = /*#__PURE__*/ forwardRef(function OptGroup(props, ref) {
        return /*#__PURE__*/ jsx(Select$1.OptGroup, {
            ...props,
            isSelectOptGroup: true,
            ref: ref
        });
    });
    // Needed for antd to work properly and for rc-select to not throw warning about our component not being Select.OptGroup
    OptGroup.isSelectOptGroup = true;
    return OptGroup;
})();
/**
 * @deprecated use LegacySelect.OptGroup instead
 */ const LegacyOptGroup = LegacySelectOptGroup;
/**
 * @deprecated Use Select, TypeaheadCombobox, or DialogCombobox depending on your use-case. See http://go/deprecate-ant-select for more information
 */ const LegacySelect = /* #__PURE__ */ (()=>{
    const DuboisRefForwardedSelect = /*#__PURE__*/ forwardRef(DuboisSelect);
    DuboisRefForwardedSelect.Option = LegacySelectOption;
    DuboisRefForwardedSelect.OptGroup = LegacySelectOptGroup;
    return DuboisRefForwardedSelect;
})();

const RadioGroupContext = /*#__PURE__*/ React__default.createContext(undefined);
const useRadioGroupContext = ()=>{
    const context = React__default.useContext(RadioGroupContext);
    if (!context) {
        throw new Error('Radio components are only allowed within a Radio.Group');
    }
    return context;
};
const getRadioInputStyles = ({ clsPrefix, theme })=>({
        [`.${clsPrefix}`]: {
            alignSelf: 'start',
            // Unchecked Styles
            [`> .${clsPrefix}-input + .${clsPrefix}-inner`]: {
                width: theme.spacing.md,
                height: theme.spacing.md,
                background: theme.colors.actionDefaultBackgroundDefault,
                borderStyle: 'solid',
                borderColor: theme.colors.actionDefaultBorderDefault,
                boxShadow: 'unset',
                transform: 'unset',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                borderRadius: theme.borders.borderRadiusFull,
                '&:after': {
                    all: 'unset'
                }
            },
            // Hover
            [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:hover + .${clsPrefix}-inner`]: {
                borderColor: theme.colors.actionPrimaryBackgroundHover,
                background: theme.colors.actionDefaultBackgroundHover
            },
            // Focus
            [`&:not(.${clsPrefix}-disabled)> .${clsPrefix}-input:focus + .${clsPrefix}-inner`]: {
                borderColor: theme.colors.actionPrimaryBackgroundDefault
            },
            // Active
            [`&:not(.${clsPrefix}-disabled)> .${clsPrefix}-input:active + .${clsPrefix}-inner`]: {
                borderColor: theme.colors.actionPrimaryBackgroundPress,
                background: theme.colors.actionDefaultBackgroundPress
            },
            // Disabled
            [`&.${clsPrefix}-disabled > .${clsPrefix}-input + .${clsPrefix}-inner`]: {
                borderColor: `${theme.colors.actionDisabledBorder} !important`,
                background: theme.colors.actionDisabledBackground,
                '@media (forced-colors: active)': {
                    borderColor: 'GrayText !important'
                }
            },
            // Checked Styles
            [`&.${clsPrefix}-checked`]: {
                '&:after': {
                    border: 'unset'
                },
                [`> .${clsPrefix}-input + .${clsPrefix}-inner`]: {
                    background: theme.colors.actionPrimaryBackgroundDefault,
                    borderColor: theme.colors.primary,
                    boxShadow: theme.shadows.xs,
                    '&:after': {
                        content: `""`,
                        borderRadius: theme.spacing.xs,
                        backgroundColor: theme.colors.white,
                        width: theme.spacing.xs,
                        height: theme.spacing.xs
                    },
                    '@media (forced-colors: active)': {
                        borderColor: 'Highlight !important',
                        backgroundColor: 'Highlight !important'
                    }
                },
                // Hover
                [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:hover + .${clsPrefix}-inner`]: {
                    background: theme.colors.actionPrimaryBackgroundHover,
                    borderColor: theme.colors.actionPrimaryBackgroundPress
                },
                // Focus
                [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:focus-visible + .${clsPrefix}-inner`]: {
                    background: theme.colors.actionPrimaryBackgroundDefault,
                    borderColor: theme.colors.actionDefaultBorderFocus,
                    outline: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                    outlineOffset: 1
                },
                // Active
                [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:active + .${clsPrefix}-inner`]: {
                    background: theme.colors.actionDefaultBackgroundPress,
                    borderColor: theme.colors.actionDefaultBorderPress
                },
                // Disabled
                [`&.${clsPrefix}-disabled > .${clsPrefix}-input + .${clsPrefix}-inner`]: {
                    background: theme.colors.actionDisabledBackground,
                    border: `1px solid ${theme.colors.actionDisabledBorder} !important`,
                    '&:after': {
                        backgroundColor: theme.colors.actionDisabledText
                    },
                    '@media (forced-colors: active)': {
                        borderColor: 'GrayText !important',
                        backgroundColor: 'GrayText !important'
                    }
                }
            }
        }
    });
const getCommonRadioGroupStyles = ({ theme, clsPrefix, classNamePrefix })=>/*#__PURE__*/ css({
        '& > label': {
            [`&.${classNamePrefix}-radio-wrapper-disabled > span`]: {
                color: theme.colors.actionDisabledText
            }
        },
        [`& > label + .${classNamePrefix}-hint`]: {
            paddingLeft: theme.spacing.lg
        },
        ...getRadioInputStyles({
            theme,
            clsPrefix
        }),
        ...getAnimationCss(theme.options.enableAnimation)
    });
const getHorizontalRadioGroupStyles = ({ theme, classNamePrefix, useEqualColumnWidths, useNewFormUISpacing })=>/*#__PURE__*/ css({
        '&&': {
            display: 'grid',
            gridTemplateRows: '[label] auto [hint] auto',
            gridAutoColumns: useEqualColumnWidths ? 'minmax(0, 1fr)' : 'max-content',
            gridColumnGap: theme.spacing.md,
            ...useNewFormUISpacing && {
                [`& + .${classNamePrefix}-form-message`]: {
                    marginTop: theme.spacing.sm
                }
            }
        },
        ...useNewFormUISpacing && {
            [`:has(> .${classNamePrefix}-hint)`]: {
                marginTop: theme.spacing.sm
            }
        },
        [`& > label, & > .${classNamePrefix}-radio-tile`]: {
            gridRow: 'label / label',
            marginRight: 0
        },
        [`& > label + .${classNamePrefix}-hint`]: {
            display: 'inline-block',
            gridRow: 'hint / hint'
        }
    });
const getVerticalRadioGroupStyles = ({ theme, classNamePrefix, useNewFormUISpacing })=>/*#__PURE__*/ css({
        display: 'flex',
        flexDirection: 'column',
        flexWrap: 'wrap',
        ...useNewFormUISpacing && {
            [`& + .${classNamePrefix}-form-message`]: {
                marginTop: theme.spacing.sm
            },
            [`~ .${classNamePrefix}-label)`]: {
                marginTop: theme.spacing.sm,
                background: 'red'
            }
        },
        [`.${classNamePrefix}-radio-tile + .${classNamePrefix}-radio-tile`]: {
            marginTop: theme.spacing.md
        },
        '& > label': {
            fontWeight: 'normal',
            paddingBottom: theme.spacing.sm
        },
        [`& > label:last-of-type`]: {
            paddingBottom: 0
        },
        [`& > label + .${classNamePrefix}-hint`]: {
            marginBottom: theme.spacing.sm,
            paddingLeft: theme.spacing.lg,
            marginTop: `-${theme.spacing.sm}px`
        },
        [`& > label:last-of-type + .${classNamePrefix}-hint`]: {
            marginTop: 0
        }
    });
const getRadioStyles = ({ theme, clsPrefix })=>{
    // Default as bold for standalone radios
    const fontWeight = 'normal';
    const styles = {
        fontWeight
    };
    return /*#__PURE__*/ css({
        ...getRadioInputStyles({
            theme,
            clsPrefix
        }),
        ...styles
    });
};
const DuboisRadio = /*#__PURE__*/ forwardRef(function Radio({ children, dangerouslySetAntdProps, __INTERNAL_DISABLE_RADIO_ROLE, componentId, analyticsEvents, valueHasNoPii, onChange, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.radio', false);
    const { theme, getPrefixedClassName } = useDesignSystemTheme();
    const { componentId: contextualComponentId } = React__default.useContext(RadioGroupContext) ?? {};
    const clsPrefix = getPrefixedClassName('radio');
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Radio,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    const firstView = useRef(true);
    useEffect(()=>{
        // Only call the onView callback if the Radio is standalone and not part of a RadioGroup
        if (componentId && contextualComponentId === undefined && firstView.current) {
            eventContext.onView(props.value);
            firstView.current = false;
        }
    }, [
        eventContext,
        componentId,
        contextualComponentId,
        props.value
    ]);
    const onChangeWrapper = useCallback((e)=>{
        // Only call the onValueChange callback if the Radio is standalone and not part of a RadioGroup
        if (contextualComponentId === undefined) {
            eventContext.onValueChange?.(e.target.value);
        }
        onChange?.(e);
    }, [
        contextualComponentId,
        eventContext,
        onChange
    ]);
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Radio$1, {
            ...addDebugOutlineIfEnabled(),
            css: getRadioStyles({
                theme,
                clsPrefix
            }),
            ...props,
            ...dangerouslySetAntdProps,
            ...__INTERNAL_DISABLE_RADIO_ROLE ? {
                role: 'none'
            } : {},
            onChange: onChangeWrapper,
            ref: ref,
            "data-component-type": contextualComponentId ? DesignSystemEventProviderComponentTypes.RadioGroup : DesignSystemEventProviderComponentTypes.Radio,
            // eslint-disable-next-line @databricks/no-dynamic-property-value -- http://go/static-frontend-log-property-strings exempt:8b3795a8-d0ff-4b21-888c-a6fbd0f3d7f2
            "data-component-id": contextualComponentId ?? componentId,
            children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                children: children
            })
        })
    });
});
const StyledRadioGroup = /*#__PURE__*/ forwardRef(function StyledRadioGroup({ children, dangerouslySetAntdProps, componentId, analyticsEvents, valueHasNoPii, onChange, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.radio', false);
    const { theme, getPrefixedClassName, classNamePrefix } = useDesignSystemTheme();
    const uniqueId = useUniqueId();
    const clsPrefix = getPrefixedClassName('radio');
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const [value, setValue] = React__default.useState(props.defaultValue ?? '');
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.RadioGroup,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    const internalRef = useRef();
    useImperativeHandle(ref, ()=>internalRef.current);
    const { elementRef: radioGroupRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.value ?? props.defaultValue
    });
    const mergedRef = useMergeRefs([
        internalRef,
        radioGroupRef
    ]);
    const onChangeWrapper = useCallback((e)=>{
        eventContext.onValueChange?.(e.target.value);
        setValue(e.target.value);
        onChange?.(e);
    }, [
        eventContext,
        onChange
    ]);
    useEffect(()=>{
        if (props.value !== undefined) {
            setValue(props.value);
            // Antd's Radio (rc-checkbox) is not updating checked state correctly even though state is managed appropriately on our end
            // Manually add and remove checked attribute to the radio input to ensure it is checked and A11y tools and tests can rely on this for validation
            if (internalRef?.current) {
                // Remove checked attribute from old radio input
                const checkedInput = internalRef.current.querySelector('input[checked]');
                if (checkedInput) {
                    checkedInput.removeAttribute('checked');
                }
                // Add checked attribute to new radio input
                const toBeCheckedInput = internalRef.current.querySelector(`input[value="${props.value}"]`);
                if (toBeCheckedInput) {
                    toBeCheckedInput.setAttribute('checked', 'checked');
                }
            }
        }
    }, [
        props.value
    ]);
    const ariaLabelledby = props['aria-labelledby'];
    // A11y helper (FIT-1649): Effect helping to set the aria-labelledby attribute on the radio group if it is not provided
    useEffect(()=>{
        if (ariaLabelledby) {
            return;
        }
        if (props.id) {
            // look for the label of the group pointing to the id
            const label = document.querySelector(`label[for="${props.id}"]`);
            if (label) {
                // If the label already has an id, map the radio group to it
                if (label.hasAttribute('id')) {
                    internalRef.current?.setAttribute('aria-labelledby', label.getAttribute('id') ?? '');
                } else {
                    label.setAttribute('id', `${props.id}${uniqueId}-label`);
                    internalRef.current?.setAttribute('aria-labelledby', `${props.id}${uniqueId}-label`);
                }
            }
        }
    }, [
        ariaLabelledby,
        props.id,
        uniqueId
    ]);
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(RadioGroupContext.Provider, {
            value: {
                componentId,
                value,
                onChange: onChangeWrapper
            },
            children: /*#__PURE__*/ jsx(Radio$1.Group, {
                /* @ts-expect-error - role is not a valid prop for RadioGroup in Antd but it applies it correctly to the group */ role: "radiogroup",
                ...addDebugOutlineIfEnabled(),
                ...props,
                css: getCommonRadioGroupStyles({
                    theme,
                    clsPrefix,
                    classNamePrefix
                }),
                value: value,
                onChange: onChangeWrapper,
                ...dangerouslySetAntdProps,
                ref: mergedRef,
                children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                    children: children
                })
            })
        })
    });
});
const HorizontalGroup = /*#__PURE__*/ forwardRef(function HorizontalGroup({ dangerouslySetAntdProps, useEqualColumnWidths, ...props }, ref) {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const useNewFormUISpacing = safex('databricks.fe.designsystem.useNewFormUISpacing', false);
    return /*#__PURE__*/ jsx(StyledRadioGroup, {
        css: getHorizontalRadioGroupStyles({
            theme,
            classNamePrefix,
            useEqualColumnWidths,
            useNewFormUISpacing
        }),
        ...props,
        ref: ref,
        ...dangerouslySetAntdProps
    });
});
const Group = /*#__PURE__*/ forwardRef(function HorizontalGroup({ dangerouslySetAntdProps, layout = 'vertical', useEqualColumnWidths, ...props }, ref) {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const useNewFormUISpacing = safex('databricks.fe.designsystem.useNewFormUISpacing', false);
    return /*#__PURE__*/ jsx(StyledRadioGroup, {
        css: layout === 'horizontal' ? getHorizontalRadioGroupStyles({
            theme,
            classNamePrefix,
            useEqualColumnWidths,
            useNewFormUISpacing
        }) : getVerticalRadioGroupStyles({
            theme,
            classNamePrefix,
            useNewFormUISpacing
        }),
        ...props,
        ref: ref,
        ...dangerouslySetAntdProps
    });
});
// Note: We are overriding ant's default "Group" with our own.
const RadioNamespace = /* #__PURE__ */ Object.assign(DuboisRadio, {
    Group,
    HorizontalGroup
});
const Radio = RadioNamespace;
// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
// We should ideally be using __Group instead of __VerticalGroup, but that exists under Checkbox too and conflicts, therefore
// we show a wrong component name in "Show code" in docs, fix included in story to replace this with correct name
const __INTERNAL_DO_NOT_USE__VerticalGroup = Group;
const __INTERNAL_DO_NOT_USE__HorizontalGroup = HorizontalGroup;

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */ const Select = (props)=>{
    const { children, placeholder, value, label, ...restProps } = props;
    return /*#__PURE__*/ jsx(SelectContextProvider, {
        value: {
            isSelect: true,
            placeholder
        },
        children: /*#__PURE__*/ jsx(DialogCombobox, {
            label: label,
            value: value ? [
                value
            ] : [],
            ...restProps,
            children: children
        })
    });
};

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */ const SelectContent = /*#__PURE__*/ forwardRef(({ children, minWidth = 150, ...restProps }, ref)=>{
    return /*#__PURE__*/ jsx(DialogComboboxContent, {
        minWidth: minWidth,
        ...restProps,
        ref: ref,
        children: /*#__PURE__*/ jsx(DialogComboboxOptionList, {
            children: children
        })
    });
});

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */ const SelectOption = /*#__PURE__*/ forwardRef((props, ref)=>{
    const { value } = useDialogComboboxContext();
    return /*#__PURE__*/ jsx(DialogComboboxOptionListSelectItem, {
        checked: value && value[0] === props.value,
        ...props,
        ref: ref
    });
});

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */ const SelectOptionGroup = (props)=>{
    const { name, children, ...restProps } = props;
    return /*#__PURE__*/ jsxs(Fragment, {
        children: [
            /*#__PURE__*/ jsx(DialogComboboxSectionHeader, {
                ...restProps,
                children: name
            }),
            children
        ]
    });
};

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */ const SelectTrigger = /*#__PURE__*/ forwardRef((props, ref)=>{
    const { children, ...restProps } = props;
    return /*#__PURE__*/ jsx(DialogComboboxTrigger, {
        allowClear: false,
        ...restProps,
        ref: ref,
        children: children
    });
});

const SimpleSelectContext = /*#__PURE__*/ createContext(undefined);
const getSelectedOption = (children, value)=>{
    const childArray = React__default.Children.toArray(children);
    for (const child of childArray){
        if (/*#__PURE__*/ React__default.isValidElement(child)) {
            if (child.type === SimpleSelectOption && child.props.value === value) {
                return child;
            }
            if (child.props.children) {
                const nestedOption = getSelectedOption(child.props.children, value);
                if (nestedOption) {
                    return nestedOption;
                }
            }
        }
    }
    return undefined;
};
const getSelectedOptionLabel = (children, value)=>{
    const selectedOption = getSelectedOption(children, value);
    if (/*#__PURE__*/ React__default.isValidElement(selectedOption)) {
        return selectedOption.props.children;
    }
    return '';
};
/**
 * This is the future `Select` component which simplifies the API of the current Select primitives.
 * It is temporarily named `SimpleSelect` pending cleanup.
 */ const SimpleSelect = /*#__PURE__*/ forwardRef(({ defaultValue, name, placeholder, children, contentProps, onChange, onOpenChange, id, label, value, validationState, forceCloseOnEscape, componentId, analyticsEvents, valueHasNoPii, ...rest }, ref)=>{
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.simpleSelect', false);
    const [defaultLabel] = useState(()=>{
        if (value) {
            return getSelectedOptionLabel(children, value);
        }
        return '';
    });
    const innerRef = useRef(null);
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- TODO(FEINF-3982)
    useImperativeHandle(ref, ()=>innerRef.current, []);
    const previousExternalValue = useRef(value);
    const [internalValue, setInternalValue] = useState(value);
    const [selectedLabel, setSelectedLabel] = useState(defaultLabel);
    const isControlled = value !== undefined;
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.SimpleSelect,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    const { elementRef: simpleSelectRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: value ?? defaultValue
    });
    // Controlled state setup.
    useEffect(()=>{
        if (value !== undefined && value !== previousExternalValue.current) {
            setInternalValue(value);
            previousExternalValue.current = value;
        }
    }, [
        value
    ]);
    // Uncontrolled state setup.
    useEffect(()=>{
        if (isControlled) {
            return;
        }
        // Set initial state.
        const element = innerRef.current;
        const initialValue = defaultValue || element?.value || '';
        setInternalValue(initialValue);
        previousExternalValue.current = initialValue;
    }, [
        isControlled,
        defaultValue,
        value
    ]);
    // Separately update the label when the value changes; this responds
    // to either the controlled or uncontrolled useEffects above.
    useEffect(()=>{
        setSelectedLabel(getSelectedOptionLabel(children, internalValue || ''));
    }, [
        internalValue,
        children
    ]);
    // Handles controlled state, and propagates changes to the input element.
    const handleChange = (newValue)=>{
        eventContext.onValueChange(newValue);
        innerRef.current?.setAttribute('value', newValue || '');
        setInternalValue(newValue);
        setSelectedLabel(getSelectedOptionLabel(children, newValue));
        if (onChange) {
            onChange({
                target: {
                    name,
                    type: 'select',
                    value: newValue
                },
                type: 'change'
            });
        }
    };
    return /*#__PURE__*/ jsx(SimpleSelectContext.Provider, {
        value: {
            value: internalValue,
            onChange: handleChange
        },
        children: /*#__PURE__*/ jsx(Select, {
            // SimpleSelect emits its own value change events rather than emitting them from the underlying
            // DialogCombobox due to how SimpleSelect sets its initial state. The Select componentId is explicitly
            // set to undefined to prevent it from emitting events if the componentId prop is required in the future.
            // eslint-disable-next-line @databricks/no-dynamic-property-value -- http://go/static-frontend-log-property-strings exempt:e5d37a4d-10cd-4a8f-a964-2223405cc738
            componentId: undefined,
            value: internalValue,
            placeholder: placeholder,
            label: label ?? rest['aria-label'],
            id: id,
            children: /*#__PURE__*/ jsxs(SimpleSelectContentWrapper, {
                onOpenChange: onOpenChange,
                children: [
                    /*#__PURE__*/ jsx(SelectTrigger, {
                        ref: simpleSelectRef,
                        ...rest,
                        validationState: validationState,
                        onClear: ()=>{
                            handleChange('');
                        },
                        id: id,
                        value: internalValue,
                        ...eventContext.dataComponentProps,
                        children: selectedLabel || placeholder
                    }),
                    /*#__PURE__*/ jsx("input", {
                        type: "hidden",
                        ref: innerRef
                    }),
                    /*#__PURE__*/ jsx(SelectContent, {
                        forceCloseOnEscape: forceCloseOnEscape,
                        ...contentProps,
                        children: children
                    })
                ]
            })
        })
    });
});
// This component is used to propagate the open state of the DialogCombobox to the SimpleSelect.
// We don't directly pass through `onOpenChange` since it's tied into the actual state; `SimpleSelect` merely
// needs to communicate via the optional prop if the dropdown is open or not and doesn't need to control it.
const SimpleSelectContentWrapper = ({ children, onOpenChange })=>{
    const { isOpen } = useDialogComboboxContext();
    useEffect(()=>{
        if (onOpenChange) {
            onOpenChange(Boolean(isOpen));
        }
    }, [
        isOpen,
        onOpenChange
    ]);
    return /*#__PURE__*/ jsx(Fragment, {
        children: children
    });
};
const SimpleSelectOption = /*#__PURE__*/ forwardRef(({ value, children, ...rest }, ref)=>{
    const context = useContext(SimpleSelectContext);
    if (!context) {
        throw new Error('SimpleSelectOption must be used within a SimpleSelect');
    }
    const { onChange } = context;
    return /*#__PURE__*/ jsx(SelectOption, {
        ...rest,
        ref: ref,
        value: value,
        onChange: ({ value })=>{
            onChange(value);
        },
        children: children
    });
});
const SimpleSelectOptionGroup = ({ children, label, ...props })=>{
    const context = useContext(SimpleSelectContext);
    if (!context) {
        throw new Error('SimpleSelectOptionGroup must be used within a SimpleSelect');
    }
    return /*#__PURE__*/ jsx(SelectOptionGroup, {
        ...props,
        name: label,
        children: children
    });
};

const getSwitchWithLabelStyles = ({ clsPrefix, theme, disabled })=>{
    // Default value
    const SWITCH_WIDTH = 28;
    const styles = {
        display: 'flex',
        alignItems: 'center',
        ...disabled && {
            '&&, label': {
                color: theme.colors.actionDisabledText
            }
        },
        [`&.${clsPrefix}-switch, &.${clsPrefix}-switch-checked`]: {
            [`.${clsPrefix}-switch-handle`]: {
                top: -1
            },
            [`.${clsPrefix}-switch-handle, .${clsPrefix}-switch-handle:before`]: {
                width: 16,
                height: 16,
                borderRadius: theme.borders.borderRadiusFull
            }
        },
        // Switch is Off
        [`&.${clsPrefix}-switch`]: {
            backgroundColor: theme.colors.backgroundPrimary,
            border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
            borderRadius: theme.borders.borderRadiusFull,
            [`.${clsPrefix}-switch-handle:before`]: {
                border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
                boxShadow: theme.shadows.xs,
                left: -1,
                transition: 'none',
                borderRadius: theme.borders.borderRadiusFull
            },
            [`&:hover:not(.${clsPrefix}-switch-disabled)`]: {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
                [`.${clsPrefix}-switch-handle:before`]: {
                    boxShadow: theme.shadows.xs,
                    border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`
                }
            },
            [`&:active:not(.${clsPrefix}-switch-disabled)`]: {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,
                [`.${clsPrefix}-switch-handle:before`]: {
                    boxShadow: 'none',
                    border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`
                }
            },
            [`&.${clsPrefix}-switch-disabled`]: {
                backgroundColor: theme.colors.actionDisabledBackground,
                border: `1px solid ${theme.colors.actionDisabledBorder}`,
                [`.${clsPrefix}-switch-handle:before`]: {
                    boxShadow: 'none',
                    border: `1px solid ${theme.colors.actionDisabledBorder}`
                }
            },
            [`&:focus-visible`]: {
                border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                boxShadow: 'none',
                outlineStyle: 'solid',
                outlineWidth: '1px',
                outlineColor: theme.colors.actionDefaultBorderFocus,
                [`.${clsPrefix}-switch-handle:before`]: {
                    boxShadow: theme.shadows.xs,
                    border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`
                }
            },
            [`&:focus`]: {
                boxShadow: 'none'
            }
        },
        // Switch is On
        [`&.${clsPrefix}-switch-checked`]: {
            backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
            border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
            [`&:hover:not(.${clsPrefix}-switch-disabled)`]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundHover,
                border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
                [`.${clsPrefix}-switch-handle:before`]: {
                    border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`
                }
            },
            [`&:active:not(.${clsPrefix}-switch-disabled)`]: {
                backgroundColor: theme.colors.actionPrimaryBackgroundPress,
                border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,
                [`.${clsPrefix}-switch-handle:before`]: {
                    border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`
                }
            },
            [`.${clsPrefix}-switch-handle:before`]: {
                boxShadow: theme.shadows.xs,
                border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
                right: -1
            },
            [`&.${clsPrefix}-switch-disabled`]: {
                backgroundColor: theme.colors.actionDisabledText,
                border: `1px solid ${theme.colors.actionDisabledText}`,
                [`.${clsPrefix}-switch-handle:before`]: {
                    boxShadow: 'none',
                    border: `1px solid ${theme.colors.actionDisabledText}`
                }
            },
            [`&:focus-visible`]: {
                outlineOffset: '1px'
            }
        },
        [`.${clsPrefix}-switch-handle:before`]: {
            backgroundColor: theme.colors.backgroundPrimary
        },
        [`&& + .${clsPrefix}-hint, && + .${clsPrefix}-form-message`]: {
            paddingLeft: theme.spacing.sm + SWITCH_WIDTH
        },
        [`&& + .${clsPrefix}-form-message`]: {
            marginTop: 0
        },
        [`.${clsPrefix}-click-animating-node`]: {
            animation: 'none'
        },
        opacity: 1
    };
    const importantStyles = importantify(styles);
    return /*#__PURE__*/ css(importantStyles);
};
const Switch = ({ dangerouslySetAntdProps, label, labelProps, activeLabel, inactiveLabel, disabledLabel, componentId, analyticsEvents, onChange, ...props })=>{
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.switch', false);
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const duboisId = useUniqueId('dubois-switch');
    const uniqueId = props.id ?? duboisId;
    const [isChecked, setIsChecked] = useState(props.checked || props.defaultChecked);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Switch,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true
    });
    const { elementRef: viewRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: isChecked
    });
    const handleToggle = (newState, event)=>{
        eventContext.onValueChange(newState);
        if (onChange) {
            onChange(newState, event);
        } else {
            setIsChecked(newState);
        }
    };
    const onChangeHandler = (newState, event)=>{
        eventContext.onValueChange(newState);
        onChange?.(newState, event);
    };
    useEffect(()=>{
        setIsChecked(props.checked);
    }, [
        props.checked
    ]);
    const hasNewLabels = activeLabel && inactiveLabel && disabledLabel;
    const stateMessage = isChecked ? activeLabel : inactiveLabel;
    // AntDSwitch's interface does not include `id` even though it passes it through and works as expected
    // We are using this to bypass that check
    const idPropObj = {
        id: uniqueId
    };
    const switchComponent = /*#__PURE__*/ jsx(Switch$1, {
        ...addDebugOutlineIfEnabled(),
        ...props,
        ...dangerouslySetAntdProps,
        onChange: handleToggle,
        ...idPropObj,
        css: {
            .../*#__PURE__*/ css(getAnimationCss(theme.options.enableAnimation)),
            ...getSwitchWithLabelStyles({
                clsPrefix: classNamePrefix,
                theme,
                disabled: props.disabled
            })
        },
        ref: viewRef
    });
    const labelComponent = /*#__PURE__*/ jsx(Label, {
        inline: true,
        ...labelProps,
        htmlFor: uniqueId,
        style: {
            ...hasNewLabels && {
                marginRight: theme.spacing.sm
            }
        },
        children: label
    });
    return label ? /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx("div", {
            ...addDebugOutlineIfEnabled(),
            css: getSwitchWithLabelStyles({
                clsPrefix: classNamePrefix,
                theme,
                disabled: props.disabled
            }),
            ...eventContext.dataComponentProps,
            children: hasNewLabels ? /*#__PURE__*/ jsxs(Fragment, {
                children: [
                    labelComponent,
                    /*#__PURE__*/ jsx("span", {
                        style: {
                            marginLeft: 'auto',
                            marginRight: theme.spacing.sm,
                            color: theme.colors.textPrimary
                        },
                        children: `${stateMessage}${props.disabled ? ` (${disabledLabel})` : ''}`
                    }),
                    switchComponent
                ]
            }) : /*#__PURE__*/ jsxs(Fragment, {
                children: [
                    switchComponent,
                    labelComponent
                ]
            })
        })
    }) : /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Switch$1, {
            onChange: onChangeHandler,
            ...addDebugOutlineIfEnabled(),
            ...props,
            ...dangerouslySetAntdProps,
            ...idPropObj,
            css: {
                .../*#__PURE__*/ css(getAnimationCss(theme.options.enableAnimation)),
                ...getSwitchWithLabelStyles({
                    clsPrefix: classNamePrefix,
                    theme,
                    disabled: props.disabled
                })
            },
            ...eventContext.dataComponentProps,
            ref: viewRef
        })
    });
};

const typeaheadComboboxContextDefaults = {
    componentId: 'codegen_design-system_src_design-system_typeaheadcombobox_providers_typeaheadcomboboxcontext.tsx_17',
    isInsideTypeaheadCombobox: false,
    multiSelect: false
};
const TypeaheadComboboxContext = /*#__PURE__*/ createContext(typeaheadComboboxContextDefaults);
const TypeaheadComboboxContextProvider = ({ children, value })=>{
    const [inputWidth, setInputWidth] = useState();
    return /*#__PURE__*/ jsx(TypeaheadComboboxContext.Provider, {
        value: {
            ...value,
            setInputWidth,
            inputWidth
        },
        children: children
    });
};

const TypeaheadComboboxRoot = /*#__PURE__*/ forwardRef(({ comboboxState, multiSelect = false, children, ...props }, ref)=>{
    const { classNamePrefix } = useDesignSystemTheme();
    const { refs, floatingStyles } = useFloating({
        whileElementsMounted: autoUpdate,
        middleware: [
            offset(4),
            flip(),
            shift()
        ],
        placement: 'bottom-start'
    });
    const { elementRef: typeaheadComboboxRootRef } = useNotifyOnFirstView({
        onView: comboboxState.onView,
        value: comboboxState.firstOnViewValue
    });
    const mergedRef = useMergeRefs([
        ref,
        typeaheadComboboxRootRef
    ]);
    return /*#__PURE__*/ jsx(TypeaheadComboboxContextProvider, {
        value: {
            componentId: comboboxState.componentId,
            multiSelect,
            isInsideTypeaheadCombobox: true,
            floatingUiRefs: refs,
            floatingStyles: floatingStyles
        },
        children: /*#__PURE__*/ jsx("div", {
            ...comboboxState.getComboboxProps({}, {
                suppressRefError: true
            }),
            className: `${classNamePrefix}-typeahead-combobox`,
            css: {
                display: 'inline-block',
                width: '100%'
            },
            ...props,
            ref: mergedRef,
            "data-component-type": DesignSystemEventProviderComponentTypes.TypeaheadCombobox,
            "data-component-id": comboboxState.componentId,
            children: children
        })
    });
});

const mapItemsToString = (items, itemToString)=>{
    return JSON.stringify(items.map(itemToString));
};
const TypeaheadComboboxStateChangeTypes = useCombobox.stateChangeTypes;
const TypeaheadComboboxMultiSelectStateChangeTypes = useMultipleSelection.stateChangeTypes;
function useComboboxState({ allItems, items, itemToString, onIsOpenChange, allowNewValue = false, formValue, formOnChange, formOnBlur, componentId, valueHasNoPii, analyticsEvents, matcher, preventUnsetOnBlur = false, selectedMatcher, ...props }) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.typeaheadCombobox', false);
    const getFilteredItems = useCallback((inputValue)=>{
        const lowerCasedInputValue = inputValue?.toLowerCase() ?? '';
        // If the input is empty or if there is no matcher supplied, do not filter
        return allItems.filter((item)=>!inputValue || !matcher || matcher(item, lowerCasedInputValue));
    }, [
        allItems,
        matcher
    ]);
    const [inputValue, setInputValue] = useState();
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.TypeaheadCombobox,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    const itemToStringWithDefaultToString = useCallback((item)=>{
        return item ? itemToString ? itemToString(item) : item.toString() : '';
    }, [
        itemToString
    ]);
    const prevAllItems = useRef(allItems);
    useEffect(()=>{
        // When allItems changes, re-apply filter so users don't see stale items values in the dropdown box
        if (!props.multiSelect && (allItems.length !== prevAllItems.current.length || // Avoid redundant or endless updates by checking individual elements as allItems may have an unstable reference.
        allItems.some((item, index)=>itemToStringWithDefaultToString(item) !== itemToStringWithDefaultToString(prevAllItems.current[index])))) {
            props.setItems(getFilteredItems(inputValue));
            prevAllItems.current = allItems;
        }
    }, [
        allItems,
        inputValue,
        getFilteredItems,
        props,
        itemToStringWithDefaultToString
    ]);
    const comboboxState = {
        ...useCombobox({
            onIsOpenChange: onIsOpenChange,
            onInputValueChange: ({ inputValue })=>{
                if (inputValue !== undefined) {
                    setInputValue(inputValue);
                    props.setInputValue?.(inputValue);
                }
                if (!props.multiSelect) {
                    props.setItems(getFilteredItems(inputValue));
                }
            },
            items: items,
            itemToString: itemToStringWithDefaultToString,
            defaultHighlightedIndex: props.multiSelect ? 0 : undefined,
            scrollIntoView: ()=>{},
            selectedItem: props.multiSelect ? null : formValue,
            stateReducer (state, actionAndChanges) {
                const { changes, type } = actionAndChanges;
                switch(type){
                    case useCombobox.stateChangeTypes.InputBlur:
                        if (preventUnsetOnBlur) {
                            return changes;
                        }
                        if (!props.multiSelect) {
                            // If allowNewValue is true, register the input's current value on blur
                            if (allowNewValue) {
                                const newInputValue = state.inputValue === '' ? null : state.inputValue;
                                formOnChange?.(newInputValue);
                                formOnBlur?.(newInputValue);
                            } else {
                                // If allowNewValue is false, clear value on blur
                                formOnChange?.(null);
                                formOnBlur?.(null);
                            }
                        } else {
                            formOnBlur?.(state.selectedItem);
                        }
                        return changes;
                    case useCombobox.stateChangeTypes.InputKeyDownEnter:
                    case useCombobox.stateChangeTypes.ItemClick:
                        formOnChange?.(changes.selectedItem);
                        return {
                            ...changes,
                            highlightedIndex: props.multiSelect ? state.highlightedIndex : 0,
                            isOpen: props.multiSelect ? true : false
                        };
                    default:
                        return changes;
                }
            },
            onStateChange: (args)=>{
                const { type, selectedItem: newSelectedItem, inputValue: newInputValue } = args;
                const isNewSelectedItemNullish = newSelectedItem === undefined || newSelectedItem === null;
                props.onStateChange?.(args);
                if (props.multiSelect) {
                    switch(type){
                        case useCombobox.stateChangeTypes.InputKeyDownEnter:
                        case useCombobox.stateChangeTypes.ItemClick:
                        case useCombobox.stateChangeTypes.InputBlur:
                            if (!isNewSelectedItemNullish) {
                                props.setSelectedItems([
                                    ...props.selectedItems,
                                    newSelectedItem
                                ]);
                                props.setInputValue('');
                                formOnBlur?.([
                                    ...props.selectedItems,
                                    newSelectedItem
                                ]);
                            }
                            break;
                        case useCombobox.stateChangeTypes.InputChange:
                            props.setInputValue(newInputValue ?? '');
                            break;
                        case useCombobox.stateChangeTypes.FunctionReset:
                            eventContext.onValueChange?.('[]');
                            break;
                    }
                    // Unselect when clicking selected item
                    if (!isNewSelectedItemNullish && (selectedMatcher ? props.selectedItems.some((item)=>selectedMatcher(item, newSelectedItem)) : props.selectedItems.includes(newSelectedItem))) {
                        const newSelectedItems = props.selectedItems.filter((item)=>selectedMatcher ? !selectedMatcher(item, newSelectedItem) : item !== newSelectedItem);
                        props.setSelectedItems(newSelectedItems);
                        eventContext.onValueChange?.(mapItemsToString(newSelectedItems, itemToStringWithDefaultToString));
                    } else if (!isNewSelectedItemNullish) {
                        eventContext.onValueChange?.(mapItemsToString([
                            ...props.selectedItems,
                            newSelectedItem
                        ], itemToStringWithDefaultToString));
                    }
                } else if (!isNewSelectedItemNullish) {
                    eventContext.onValueChange?.(itemToStringWithDefaultToString(newSelectedItem));
                } else if (type === useCombobox.stateChangeTypes.FunctionReset) {
                    eventContext.onValueChange?.('');
                }
            },
            initialInputValue: props.initialInputValue,
            initialSelectedItem: props.initialSelectedItem
        }),
        componentId,
        analyticsEvents,
        valueHasNoPii,
        onView: eventContext.onView,
        firstOnViewValue: props.multiSelect ? mapItemsToString(props.selectedItems, itemToStringWithDefaultToString) : itemToStringWithDefaultToString(props.initialSelectedItem ?? null)
    };
    return comboboxState;
}
function useMultipleSelectionState(selectedItems, setSelectedItems, { componentId, analyticsEvents = [
    DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
], valueHasNoPii, itemToString }) {
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents, [
        analyticsEvents
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.TypeaheadCombobox,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    return useMultipleSelection({
        selectedItems,
        onStateChange ({ selectedItems: newSelectedItems, type }) {
            switch(type){
                case useMultipleSelection.stateChangeTypes.SelectedItemKeyDownBackspace:
                case useMultipleSelection.stateChangeTypes.SelectedItemKeyDownDelete:
                case useMultipleSelection.stateChangeTypes.DropdownKeyDownBackspace:
                case useMultipleSelection.stateChangeTypes.FunctionRemoveSelectedItem:
                    setSelectedItems(newSelectedItems || []);
                    break;
            }
            const itemToStringWithDefaultToString = itemToString ?? ((item)=>item?.toString() ?? '');
            eventContext.onValueChange?.(mapItemsToString(newSelectedItems || [], itemToStringWithDefaultToString));
        }
    });
}

const useTypeaheadComboboxContext = ()=>{
    return useContext(TypeaheadComboboxContext);
};

const getTypeaheadComboboxMenuStyles = ()=>{
    return /*#__PURE__*/ css({
        padding: 0,
        margin: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        position: 'absolute'
    });
};
const TypeaheadComboboxMenu = /*#__PURE__*/ forwardRef(({ comboboxState, loading, emptyText, width, minWidth = 240, maxWidth, minHeight, maxHeight, listWrapperHeight, virtualizerRef, children, matchTriggerWidth, ...restProps }, ref)=>{
    const { getMenuProps, isOpen } = comboboxState;
    const { ref: downshiftRef, ...downshiftProps } = getMenuProps({}, {
        suppressRefError: true
    });
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const [viewPortMaxHeight, setViewPortMaxHeight] = useState(undefined);
    const { floatingUiRefs, floatingStyles, isInsideTypeaheadCombobox, inputWidth } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxMenu` must be used within `TypeaheadCombobox`');
    }
    const mergedRef = useMergeRefs([
        ref,
        downshiftRef,
        floatingUiRefs?.setFloating,
        virtualizerRef
    ]);
    const { theme } = useDesignSystemTheme();
    const { getPopupContainer } = useDesignSystemContext();
    const recalculateMaxHeight = useCallback(()=>{
        if (isOpen && floatingUiRefs?.floating && floatingUiRefs.reference.current && floatingUiRefs?.reference && floatingUiRefs.floating.current) {
            computePosition(floatingUiRefs.reference.current, floatingUiRefs.floating.current, {
                middleware: [
                    flip$1(),
                    size({
                        padding: theme.spacing.sm,
                        apply ({ availableHeight }) {
                            setViewPortMaxHeight(availableHeight);
                        }
                    })
                ]
            });
        }
    }, [
        isOpen,
        floatingUiRefs,
        theme.spacing.sm
    ]);
    useEffect(()=>{
        if (!isOpen || maxHeight) {
            return;
        }
        recalculateMaxHeight();
        window.addEventListener('scroll', recalculateMaxHeight);
        return ()=>{
            window.removeEventListener('scroll', recalculateMaxHeight);
        };
    }, [
        isOpen,
        maxHeight,
        recalculateMaxHeight
    ]);
    if (!isOpen) return null;
    const hasFragmentWrapper = children && !Array.isArray(children) && children.type === Fragment$1;
    const filterableChildren = hasFragmentWrapper ? children.props.children : children;
    const hasResults = filterableChildren && Children.toArray(filterableChildren).some((child)=>{
        if (/*#__PURE__*/ React__default.isValidElement(child)) {
            const childType = child.props['__EMOTION_TYPE_PLEASE_DO_NOT_USE__']?.defaultProps._type ?? child.props._type;
            return [
                'TypeaheadComboboxMenuItem',
                'TypeaheadComboboxCheckboxItem'
            ].includes(childType);
        }
        return false;
    });
    const [menuItemChildren, footer] = Children.toArray(filterableChildren).reduce((acc, child)=>{
        const isFooter = /*#__PURE__*/ React__default.isValidElement(child) && child.props._type === 'TypeaheadComboboxFooter';
        if (isFooter) {
            acc[1].push(child);
        } else {
            acc[0].push(child);
        }
        return acc;
    }, [
        [],
        []
    ]);
    return /*#__PURE__*/ createPortal(/*#__PURE__*/ jsx("ul", {
        ...addDebugOutlineIfEnabled(),
        "aria-busy": loading,
        ...downshiftProps,
        ref: mergedRef,
        css: [
            getComboboxContentWrapperStyles(theme, {
                maxHeight: maxHeight ?? viewPortMaxHeight,
                maxWidth,
                minHeight,
                minWidth,
                width,
                useNewBorderColors
            }),
            getTypeaheadComboboxMenuStyles(),
            matchTriggerWidth && inputWidth && {
                width: inputWidth
            }
        ],
        style: {
            ...floatingStyles
        },
        ...restProps,
        children: loading ? /*#__PURE__*/ jsx(LoadingSpinner, {
            "aria-label": "Loading",
            alt: "Loading spinner"
        }) : hasResults ? /*#__PURE__*/ jsxs(Fragment, {
            children: [
                /*#__PURE__*/ jsx("div", {
                    style: {
                        position: 'relative',
                        width: '100%',
                        ...listWrapperHeight && {
                            height: listWrapperHeight,
                            flexShrink: 0
                        }
                    },
                    children: menuItemChildren
                }),
                footer
            ]
        }) : /*#__PURE__*/ jsxs(Fragment, {
            children: [
                /*#__PURE__*/ jsx(EmptyResults, {
                    emptyText: emptyText
                }),
                footer
            ]
        })
    }), getPopupContainer ? getPopupContainer() : document.body);
});

const getMenuItemStyles = (theme, isHighlighted, disabled)=>{
    return /*#__PURE__*/ css({
        ...disabled && {
            pointerEvents: 'none',
            color: theme.colors.actionDisabledText
        },
        ...isHighlighted && {
            background: theme.colors.actionTertiaryBackgroundHover
        }
    });
};
const getLabelStyles = (theme, textOverflowMode)=>{
    return /*#__PURE__*/ css({
        marginLeft: theme.spacing.sm,
        fontSize: theme.typography.fontSizeBase,
        fontStyle: 'normal',
        fontWeight: 400,
        cursor: 'pointer',
        overflow: 'hidden',
        wordBreak: 'break-word',
        ...textOverflowMode === 'ellipsis' && {
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap'
        }
    });
};
const TypeaheadComboboxMenuItem = /*#__PURE__*/ forwardRef(({ item, index, comboboxState, textOverflowMode = 'multiline', isDisabled, disabledReason, hintContent, onClick: onClickProp, children, ...restProps }, ref)=>{
    const { selectedItem, highlightedIndex, getItemProps, isOpen } = comboboxState;
    const isSelected = isEqual(selectedItem, item);
    const isHighlighted = highlightedIndex === index;
    const { theme } = useDesignSystemTheme();
    const listItemRef = useRef(null);
    useImperativeHandle(ref, ()=>listItemRef.current);
    const { onClick, ...downshiftItemProps } = getItemProps({
        item,
        index,
        disabled: isDisabled,
        onMouseUp: (e)=>{
            e.stopPropagation();
            restProps.onMouseUp?.(e);
        },
        ref: listItemRef
    });
    const handleClick = (e)=>{
        onClickProp?.(e);
        onClick?.(e);
    };
    // Scroll to the highlighted item if it is not in the viewport
    useEffect(()=>{
        if (isOpen && highlightedIndex === index && listItemRef.current) {
            const parentContainer = listItemRef.current.closest('ul');
            if (!parentContainer) {
                return;
            }
            const parentTop = parentContainer.scrollTop;
            const parentBottom = parentContainer.scrollTop + parentContainer.clientHeight;
            const itemTop = listItemRef.current.offsetTop;
            const itemBottom = listItemRef.current.offsetTop + listItemRef.current.clientHeight;
            // Check if item is visible in the viewport before scrolling
            if (itemTop < parentTop || itemBottom > parentBottom) {
                listItemRef.current?.scrollIntoView({
                    block: 'nearest'
                });
            }
        }
    }, [
        highlightedIndex,
        index,
        isOpen,
        listItemRef
    ]);
    return /*#__PURE__*/ jsxs("li", {
        role: "option",
        "aria-selected": isSelected,
        "aria-disabled": isDisabled,
        onClick: handleClick,
        css: [
            getComboboxOptionItemWrapperStyles(theme),
            getMenuItemStyles(theme, isHighlighted, isDisabled)
        ],
        ...downshiftItemProps,
        ...restProps,
        children: [
            isSelected ? /*#__PURE__*/ jsx(CheckIcon, {
                css: {
                    paddingTop: 2
                }
            }) : /*#__PURE__*/ jsx("div", {
                style: {
                    width: 16,
                    flexShrink: 0
                }
            }),
            /*#__PURE__*/ jsxs("label", {
                css: getLabelStyles(theme, textOverflowMode),
                children: [
                    isDisabled && disabledReason ? /*#__PURE__*/ jsxs("div", {
                        css: {
                            display: 'flex'
                        },
                        children: [
                            /*#__PURE__*/ jsx("div", {
                                children: children
                            }),
                            /*#__PURE__*/ jsx("div", {
                                css: getInfoIconStyles(theme),
                                children: /*#__PURE__*/ jsx(InfoTooltip, {
                                    componentId: "typeahead-combobox-menu-item-disabled-reason-info-tooltip",
                                    side: "right",
                                    content: disabledReason
                                })
                            })
                        ]
                    }) : children,
                    /*#__PURE__*/ jsx(HintRow$1, {
                        disabled: isDisabled,
                        children: hintContent
                    })
                ]
            })
        ]
    });
});
TypeaheadComboboxMenuItem.defaultProps = {
    _type: 'TypeaheadComboboxMenuItem'
};

const TypeaheadComboboxCheckboxItem = /*#__PURE__*/ forwardRef(({ item, index, comboboxState, selectedItems, selectedMatcher, textOverflowMode = 'multiline', isDisabled, disabledReason, hintContent, onClick: onClickProp, children, ...restProps }, ref)=>{
    const { highlightedIndex, getItemProps, isOpen } = comboboxState;
    const isHighlighted = highlightedIndex === index;
    const { theme } = useDesignSystemTheme();
    const isSelected = selectedMatcher ? selectedItems.some((selectedItem)=>selectedMatcher(selectedItem, item)) : selectedItems.includes(item);
    const listItemRef = useRef(null);
    useImperativeHandle(ref, ()=>listItemRef.current);
    const { onClick, ...downshiftItemProps } = getItemProps({
        item,
        index,
        disabled: isDisabled,
        onMouseUp: (e)=>{
            e.stopPropagation();
            restProps.onMouseUp?.(e);
        },
        ref: listItemRef
    });
    const handleClick = (e)=>{
        onClickProp?.(e);
        onClick(e);
    };
    // Scroll to the highlighted item if it is not in the viewport
    useEffect(()=>{
        if (isOpen && highlightedIndex === index && listItemRef.current) {
            const parentContainer = listItemRef.current.closest('ul');
            if (!parentContainer) {
                return;
            }
            const parentTop = parentContainer.scrollTop;
            const parentBottom = parentContainer.scrollTop + parentContainer.clientHeight;
            const itemTop = listItemRef.current.offsetTop;
            const itemBottom = listItemRef.current.offsetTop + listItemRef.current.clientHeight;
            // Check if item is visible in the viewport before scrolling
            if (itemTop < parentTop || itemBottom > parentBottom) {
                listItemRef.current?.scrollIntoView({
                    block: 'nearest'
                });
            }
        }
    }, [
        highlightedIndex,
        index,
        isOpen,
        listItemRef
    ]);
    return /*#__PURE__*/ jsx("li", {
        role: "option",
        "aria-selected": isSelected,
        disabled: isDisabled,
        onClick: handleClick,
        css: [
            getComboboxOptionItemWrapperStyles(theme),
            getMenuItemStyles(theme, isHighlighted, isDisabled)
        ],
        ...downshiftItemProps,
        ...restProps,
        children: /*#__PURE__*/ jsx(Checkbox, {
            componentId: "codegen_design-system_src_design-system_typeaheadcombobox_typeaheadcomboboxcheckboxitem.tsx_92",
            disabled: isDisabled,
            isChecked: isSelected,
            css: getCheckboxStyles(theme, textOverflowMode),
            tabIndex: -1,
            // Needed because Antd handles keyboard inputs as clicks
            onClick: (e)=>{
                e.stopPropagation();
            },
            children: /*#__PURE__*/ jsxs("label", {
                children: [
                    isDisabled && disabledReason ? /*#__PURE__*/ jsxs("div", {
                        css: {
                            display: 'flex'
                        },
                        children: [
                            /*#__PURE__*/ jsx("div", {
                                children: children
                            }),
                            /*#__PURE__*/ jsx("div", {
                                css: getInfoIconStyles(theme),
                                children: /*#__PURE__*/ jsx(InfoTooltip, {
                                    componentId: "typeahead-combobox-checkbox-item-disabled-reason-info-tooltip",
                                    content: disabledReason
                                })
                            })
                        ]
                    }) : children,
                    /*#__PURE__*/ jsx(HintRow$1, {
                        disabled: isDisabled,
                        children: hintContent
                    })
                ]
            })
        })
    });
});
TypeaheadComboboxCheckboxItem.defaultProps = {
    _type: 'TypeaheadComboboxCheckboxItem'
};

const getToggleButtonStyles = (theme, disabled)=>{
    return /*#__PURE__*/ css({
        cursor: 'pointer',
        userSelect: 'none',
        color: theme.colors.textSecondary,
        backgroundColor: 'transparent',
        border: 'none',
        padding: 0,
        marginLeft: theme.spacing.xs,
        height: 16,
        width: 16,
        ...disabled && {
            pointerEvents: 'none',
            color: theme.colors.actionDisabledText
        }
    });
};
const TypeaheadComboboxToggleButton = /*#__PURE__*/ React__default.forwardRef(({ disabled, ...restProps }, ref)=>{
    const { theme } = useDesignSystemTheme();
    const { onClick } = restProps;
    function handleClick(e) {
        e.stopPropagation();
        onClick(e);
    }
    return /*#__PURE__*/ jsx("button", {
        type: "button",
        "aria-label": "toggle menu",
        ref: ref,
        css: getToggleButtonStyles(theme, disabled),
        ...restProps,
        onClick: handleClick,
        children: /*#__PURE__*/ jsx(ChevronDownIcon, {})
    });
});

const TypeaheadComboboxControls = ({ getDownshiftToggleButtonProps, showClearSelectionButton, showComboboxToggleButton = true, handleClear, disabled })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsxs("div", {
        css: {
            position: 'absolute',
            top: theme.spacing.sm,
            right: 7,
            height: 16,
            zIndex: 1,
            cursor: disabled ? 'not-allowed' : undefined
        },
        children: [
            showClearSelectionButton && /*#__PURE__*/ jsx(ClearSelectionButton, {
                onClick: handleClear,
                css: {
                    pointerEvents: 'all',
                    verticalAlign: 'text-top'
                }
            }),
            showComboboxToggleButton && /*#__PURE__*/ jsx(TypeaheadComboboxToggleButton, {
                ...getDownshiftToggleButtonProps(),
                disabled: disabled
            })
        ]
    });
};

const getContainerStyles$1 = ()=>{
    return /*#__PURE__*/ css({
        display: 'flex',
        position: 'relative'
    });
};
const getInputStyles$1 = (theme, showComboboxToggleButton, useNewBorderRadii, useNewBorderColors)=>/*#__PURE__*/ css({
        paddingRight: showComboboxToggleButton ? 52 : 26,
        width: '100%',
        minWidth: 72,
        ...useNewBorderRadii && {
            borderRadius: theme.borders.borderRadiusSm
        },
        ...useNewBorderColors && {
            borderColor: theme.colors.actionDefaultBorderDefault
        },
        '&:disabled': {
            borderColor: theme.colors.actionDisabledBorder,
            backgroundColor: theme.colors.actionDisabledBackground,
            color: theme.colors.actionDisabledText
        },
        '&:not(:disabled)': {
            backgroundColor: 'transparent'
        }
    });
const TypeaheadComboboxInput = /*#__PURE__*/ forwardRef(({ comboboxState, allowClear = true, showComboboxToggleButton = true, formOnChange, onClick, clearInputValueOnFocus = false, ...restProps }, ref)=>{
    const { isInsideTypeaheadCombobox, floatingUiRefs, setInputWidth, inputWidth } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxInput` must be used within `TypeaheadCombobox`');
    }
    const { getInputProps, getToggleButtonProps, toggleMenu, inputValue, setInputValue, reset, isOpen, selectedItem, componentId } = comboboxState;
    const { ref: downshiftRef, ...downshiftProps } = getInputProps({}, {
        suppressRefError: true
    });
    const mergedRef = useMergeRefs([
        ref,
        downshiftRef
    ]);
    const { theme } = useDesignSystemTheme();
    const useNewBorderRadii = safex('databricks.fe.designsystem.useNewBorderRadii', false);
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const handleClick = (e)=>{
        onClick?.(e);
        toggleMenu();
    };
    const previousInputValue = useRef(null);
    useEffect(()=>{
        if (!clearInputValueOnFocus) {
            return;
        }
        // If the input is open and has value, clear the input value
        if (isOpen && !previousInputValue.current) {
            previousInputValue.current = {
                selectedItem: selectedItem,
                inputValue: inputValue
            };
            setInputValue('');
        }
        // If the input is closed and the input value was cleared, restore the input value
        if (!isOpen && previousInputValue.current) {
            // Only restore the input value if the selected item is the same as the previous selected item
            if (previousInputValue.current.selectedItem === selectedItem) {
                setInputValue(previousInputValue.current.inputValue);
            }
            previousInputValue.current = null;
        }
    }, [
        isOpen,
        inputValue,
        setInputValue,
        previousInputValue,
        clearInputValueOnFocus,
        selectedItem
    ]);
    const handleClear = ()=>{
        setInputValue('');
        reset();
        formOnChange?.(null);
    };
    // Gets the width of the input and sets it inside the context for rendering the dropdown when `matchTriggerWidth` is true on the menu
    useEffect(()=>{
        // Use the DOM reference of the TypeaheadComboboxInput container div to get the width of the input
        if (floatingUiRefs?.domReference) {
            const width = floatingUiRefs.domReference.current?.getBoundingClientRect().width ?? 0;
            // Only update context width when the input width updated
            if (width !== inputWidth) {
                setInputWidth?.(width);
            }
        }
    }, [
        floatingUiRefs?.domReference,
        setInputWidth,
        inputWidth
    ]);
    return /*#__PURE__*/ jsxs("div", {
        ref: floatingUiRefs?.setReference,
        css: getContainerStyles$1(),
        className: restProps.className,
        ...addDebugOutlineIfEnabled(),
        children: [
            /*#__PURE__*/ jsx(Input, {
                componentId: componentId ? `${componentId}.input` : 'design_system.typeahead_combobox.input',
                ref: mergedRef,
                ...downshiftProps,
                "aria-controls": comboboxState.isOpen ? downshiftProps['aria-controls'] : undefined,
                onClick: handleClick,
                css: getInputStyles$1(theme, showComboboxToggleButton, useNewBorderRadii, useNewBorderColors),
                ...restProps
            }),
            /*#__PURE__*/ jsx(TypeaheadComboboxControls, {
                getDownshiftToggleButtonProps: getToggleButtonProps,
                showClearSelectionButton: allowClear && Boolean(inputValue) && !restProps.disabled,
                showComboboxToggleButton: showComboboxToggleButton,
                handleClear: handleClear,
                disabled: restProps.disabled
            })
        ]
    });
});

const getSelectedItemStyles = (theme, disabled)=>{
    return /*#__PURE__*/ css({
        backgroundColor: theme.colors.tagDefault,
        borderRadius: theme.general.borderRadiusBase,
        color: theme.colors.textPrimary,
        lineHeight: theme.typography.lineHeightBase,
        fontSize: theme.typography.fontSizeBase,
        marginTop: 2,
        marginBottom: 2,
        marginInlineEnd: theme.spacing.xs,
        paddingRight: 0,
        paddingTop: 0,
        paddingBottom: 0,
        paddingInlineStart: theme.spacing.xs,
        position: 'relative',
        flex: 'none',
        maxWidth: '100%',
        wordBreak: 'break-word',
        ...disabled && {
            pointerEvents: 'none'
        }
    });
};
const getIconContainerStyles = (theme, disabled)=>{
    return /*#__PURE__*/ css({
        width: 16,
        height: 16,
        ':hover': {
            color: theme.colors.actionTertiaryTextHover,
            backgroundColor: theme.colors.tagHover
        },
        ...disabled && {
            pointerEvents: 'none',
            color: theme.colors.actionDisabledText
        }
    });
};
const getXIconStyles = (theme)=>{
    return /*#__PURE__*/ css({
        fontSize: theme.typography.fontSizeSm,
        verticalAlign: '-1px',
        paddingLeft: theme.spacing.xs / 2,
        paddingRight: theme.spacing.xs / 2
    });
};
const TypeaheadComboboxSelectedItem = /*#__PURE__*/ forwardRef(({ label, item, getSelectedItemProps, removeSelectedItem, disabled, ...restProps }, ref)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsxs("span", {
        ...getSelectedItemProps({
            selectedItem: item
        }),
        css: getSelectedItemStyles(theme, disabled),
        ref: ref,
        ...restProps,
        children: [
            /*#__PURE__*/ jsx("span", {
                css: {
                    marginRight: 2,
                    ...disabled && {
                        color: theme.colors.actionDisabledText
                    }
                },
                children: label
            }),
            /*#__PURE__*/ jsx("span", {
                css: getIconContainerStyles(theme, disabled),
                children: /*#__PURE__*/ jsx(CloseIcon, {
                    "aria-hidden": "false",
                    onClick: (e)=>{
                        if (!disabled) {
                            e.stopPropagation();
                            removeSelectedItem(item);
                        }
                    },
                    css: getXIconStyles(theme),
                    role: "button",
                    "aria-label": "Remove selected item"
                })
            })
        ]
    });
});

const CountBadge = ({ countStartAt, totalCount, disabled })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        css: [
            getSelectedItemStyles(theme),
            {
                paddingInlineEnd: theme.spacing.xs,
                ...disabled && {
                    color: theme.colors.actionDisabledText
                }
            }
        ],
        children: countStartAt ? `+${totalCount - countStartAt}` : totalCount
    });
};

const getContainerStyles = (theme, validationState, width, maxHeight, disabled, useNewBorderColors)=>{
    const validationColor = getValidationStateColor(theme, validationState);
    return /*#__PURE__*/ css({
        cursor: 'text',
        display: 'inline-block',
        verticalAlign: 'top',
        border: `1px solid ${useNewBorderColors ? theme.colors.actionDefaultBorderDefault : theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        minHeight: 32,
        height: 'auto',
        minWidth: 0,
        ...width ? {
            width
        } : {},
        ...maxHeight ? {
            maxHeight
        } : {},
        padding: '5px 52px 5px 12px',
        position: 'relative',
        overflow: 'auto',
        textOverflow: 'ellipsis',
        '&:hover': {
            border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`
        },
        '&:focus-within': {
            outlineColor: theme.colors.actionDefaultBorderFocus,
            outlineWidth: 2,
            outlineOffset: -2,
            outlineStyle: 'solid',
            boxShadow: 'none',
            borderColor: 'transparent'
        },
        '&&': {
            ...validationState && {
                borderColor: validationColor
            },
            '&:hover': {
                borderColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundHover
            },
            '&:focus': {
                outlineColor: validationState ? validationColor : theme.colors.actionDefaultBorderFocus,
                outlineWidth: 2,
                outlineOffset: -2,
                outlineStyle: 'solid',
                boxShadow: 'none',
                borderColor: 'transparent'
            },
            ...disabled && {
                borderColor: theme.colors.actionDisabledBorder,
                backgroundColor: theme.colors.actionDisabledBackground,
                cursor: 'not-allowed',
                outline: 'none',
                '&:hover': {
                    border: `1px solid ${theme.colors.actionDisabledBorder}`
                },
                '&:focus-within': {
                    outline: 'none',
                    borderColor: theme.colors.actionDisabledBorder
                }
            }
        }
    });
};
const getContentWrapperStyles = ()=>{
    return /*#__PURE__*/ css({
        display: 'flex',
        flex: 'auto',
        flexWrap: 'wrap',
        maxWidth: '100%',
        position: 'relative'
    });
};
const getInputWrapperStyles = ()=>{
    return /*#__PURE__*/ css({
        display: 'inline-flex',
        position: 'relative',
        maxWidth: '100%',
        alignSelf: 'auto',
        flex: 'none'
    });
};
const getInputStyles = (theme)=>{
    return /*#__PURE__*/ css({
        lineHeight: 20,
        height: 24,
        margin: 0,
        padding: 0,
        appearance: 'none',
        cursor: 'auto',
        width: '100%',
        backgroundColor: 'transparent',
        color: theme.colors.textPrimary,
        '&, &:hover, &:focus-visible': {
            border: 'none',
            outline: 'none'
        },
        '&::placeholder': {
            color: theme.colors.textPlaceholder
        }
    });
};
const TypeaheadComboboxMultiSelectInput = /*#__PURE__*/ forwardRef(({ comboboxState, multipleSelectionState, selectedItems, setSelectedItems, getSelectedItemLabel, allowClear = true, showTagAfterValueCount = 20, width, maxHeight, placeholder, validationState, showComboboxToggleButton, disableTooltip = false, ...restProps }, ref)=>{
    const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxMultiSelectInput` must be used within `TypeaheadCombobox`');
    }
    const { getInputProps, getToggleButtonProps, toggleMenu, inputValue, setInputValue } = comboboxState;
    const { getSelectedItemProps, getDropdownProps, reset, removeSelectedItem } = multipleSelectionState;
    const { ref: downshiftRef, ...downshiftProps } = getInputProps(getDropdownProps({}, {
        suppressRefError: true
    }));
    const { floatingUiRefs, setInputWidth: setContextInputWidth, inputWidth: contextInputWidth } = useTypeaheadComboboxContext();
    const containerRef = useRef(null);
    const mergedContainerRef = useMergeRefs([
        containerRef,
        floatingUiRefs?.setReference
    ]);
    const itemsRef = useRef(null);
    const measureRef = useRef(null);
    const innerRef = useRef(null);
    const mergedInputRef = useMergeRefs([
        ref,
        innerRef,
        downshiftRef
    ]);
    const { theme } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const [inputWidth, setInputWidth] = useState(0);
    const shouldShowCountBadge = selectedItems.length > showTagAfterValueCount;
    const [showTooltip, setShowTooltip] = useState(shouldShowCountBadge);
    const selectedItemsToRender = selectedItems.slice(0, showTagAfterValueCount);
    const handleClick = ()=>{
        if (!restProps.disabled) {
            innerRef.current?.focus();
            toggleMenu();
        }
    };
    const handleClear = ()=>{
        setInputValue('');
        reset();
        setSelectedItems([]);
    };
    // We measure width and set to the input immediately
    useLayoutEffect(()=>{
        if (measureRef?.current) {
            const measuredWidth = measureRef.current.scrollWidth;
            setInputWidth(measuredWidth);
        }
    }, [
        measureRef?.current?.scrollWidth,
        selectedItems?.length
    ]);
    // Gets the width of the input and sets it inside the context for rendering the dropdown when `matchTriggerWidth` is true on the menu
    useEffect(()=>{
        // Use the DOM reference of the TypeaheadComboboxInput container div to get the width of the input
        if (floatingUiRefs?.domReference) {
            const width = floatingUiRefs.domReference.current?.getBoundingClientRect().width ?? 0;
            // Only update context width when the input width updated
            if (width !== contextInputWidth) {
                setContextInputWidth?.(width);
            }
        }
    }, [
        floatingUiRefs?.domReference,
        setContextInputWidth,
        contextInputWidth
    ]);
    // Determine whether to show tooltip
    useEffect(()=>{
        let isPartiallyHidden = false;
        if (itemsRef.current && containerRef.current) {
            const { clientHeight: innerHeight } = itemsRef.current;
            const { clientHeight: outerHeight } = containerRef.current;
            isPartiallyHidden = innerHeight > outerHeight;
        }
        setShowTooltip(!disableTooltip && (shouldShowCountBadge || isPartiallyHidden));
    }, [
        shouldShowCountBadge,
        itemsRef.current?.clientHeight,
        containerRef.current?.clientHeight,
        disableTooltip
    ]);
    const content = /*#__PURE__*/ jsxs("div", {
        ...addDebugOutlineIfEnabled(),
        onClick: handleClick,
        ref: mergedContainerRef,
        css: getContainerStyles(theme, validationState, width, maxHeight, restProps.disabled, useNewBorderColors),
        tabIndex: restProps.disabled ? -1 : 0,
        children: [
            /*#__PURE__*/ jsxs("div", {
                ref: itemsRef,
                css: getContentWrapperStyles(),
                children: [
                    selectedItemsToRender?.map((selectedItemForRender, index)=>/*#__PURE__*/ jsx(TypeaheadComboboxSelectedItem, {
                            label: getSelectedItemLabel(selectedItemForRender),
                            item: selectedItemForRender,
                            getSelectedItemProps: getSelectedItemProps,
                            removeSelectedItem: removeSelectedItem,
                            disabled: restProps.disabled
                        }, `selected-item-${index}`)),
                    shouldShowCountBadge && /*#__PURE__*/ jsx(CountBadge, {
                        countStartAt: showTagAfterValueCount,
                        totalCount: selectedItems.length,
                        role: "status",
                        "aria-label": "Selected options count",
                        disabled: restProps.disabled
                    }),
                    /*#__PURE__*/ jsxs("div", {
                        css: getInputWrapperStyles(),
                        children: [
                            /*#__PURE__*/ jsx("input", {
                                ...downshiftProps,
                                ref: mergedInputRef,
                                css: [
                                    getInputStyles(theme),
                                    {
                                        width: inputWidth
                                    }
                                ],
                                placeholder: selectedItems?.length ? undefined : placeholder,
                                "aria-controls": comboboxState.isOpen ? downshiftProps['aria-controls'] : undefined,
                                ...restProps
                            }),
                            /*#__PURE__*/ jsxs("span", {
                                ref: measureRef,
                                "aria-hidden": true,
                                css: {
                                    visibility: 'hidden',
                                    whiteSpace: 'pre',
                                    position: 'absolute'
                                },
                                children: [
                                    innerRef.current?.value ? innerRef.current.value : placeholder,
                                    "\xa0"
                                ]
                            })
                        ]
                    })
                ]
            }),
            /*#__PURE__*/ jsx(TypeaheadComboboxControls, {
                getDownshiftToggleButtonProps: getToggleButtonProps,
                showComboboxToggleButton: showComboboxToggleButton,
                showClearSelectionButton: allowClear && (Boolean(inputValue) || selectedItems && selectedItems.length > 0) && !restProps.disabled,
                handleClear: handleClear,
                disabled: restProps.disabled
            })
        ]
    });
    if (showTooltip && selectedItems.length > 0) {
        return /*#__PURE__*/ jsx(Tooltip$1, {
            componentId: "typeahead-combobox-multi-select-input-tooltip",
            side: "right",
            content: selectedItems.map((item)=>getSelectedItemLabel(item)).join(', '),
            children: content
        });
    }
    return content;
});

const TypeaheadComboboxSectionHeader = ({ children, ...props })=>{
    const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxSectionHeader` must be used within `TypeaheadComboboxMenu`');
    }
    return /*#__PURE__*/ jsx(SectionHeader, {
        ...props,
        children: children
    });
};

const TypeaheadComboboxSeparator = (props)=>{
    const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxSeparator` must be used within `TypeaheadComboboxMenu`');
    }
    return /*#__PURE__*/ jsx(Separator$2, {
        ...props
    });
};

const DuboisTypeaheadComboboxFooter = ({ children, ...restProps })=>{
    const { theme } = useDesignSystemTheme();
    const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxFooter` must be used within `TypeaheadComboboxMenu`');
    }
    return /*#__PURE__*/ jsx("div", {
        ...restProps,
        css: getFooterStyles(theme),
        children: children
    });
};
DuboisTypeaheadComboboxFooter.defaultProps = {
    _type: 'TypeaheadComboboxFooter'
};
const TypeaheadComboboxFooter = DuboisTypeaheadComboboxFooter;

const TypeaheadComboboxAddButton = /*#__PURE__*/ forwardRef(({ children, ...restProps }, forwardedRef)=>{
    const { theme } = useDesignSystemTheme();
    const { isInsideTypeaheadCombobox, componentId } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxAddButton` must be used within `TypeaheadCombobox`');
    }
    return /*#__PURE__*/ jsx(Button, {
        ...restProps,
        // eslint-disable-next-line @databricks/no-dynamic-property-value -- http://go/static-frontend-log-property-strings exempt:844c1e83-84b8-41ba-95af-bae4799834aa
        componentId: `${componentId}.add_option`,
        type: "tertiary",
        onClick: (event)=>{
            event.stopPropagation();
            restProps.onClick?.(event);
        },
        onMouseUp: (event)=>{
            event.stopPropagation();
            restProps.onMouseUp?.(event);
        },
        className: "combobox-footer-add-button",
        css: {
            ...getComboboxOptionItemWrapperStyles(theme),
            .../*#__PURE__*/ css(importantify({
                width: '100%',
                padding: 0,
                display: 'flex',
                alignItems: 'center',
                borderRadius: 0,
                '&:focus': {
                    background: theme.colors.actionTertiaryBackgroundHover,
                    outline: 'none'
                }
            }))
        },
        icon: /*#__PURE__*/ jsx(PlusIcon, {}),
        ref: forwardedRef,
        children: children
    });
});

function RHFControlledInput({ name, control, rules, inputRef, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules
    });
    const mergedRef = useMergeRefs([
        field.ref,
        inputRef
    ]);
    return /*#__PURE__*/ jsx(Input, {
        ...restProps,
        ...field,
        ref: mergedRef,
        value: field.value,
        defaultValue: restProps.defaultValue
    });
}
function RHFControlledPasswordInput({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules
    });
    return /*#__PURE__*/ jsx(Input.Password, {
        ...restProps,
        ...field,
        value: field.value,
        defaultValue: restProps.defaultValue
    });
}
function RHFControlledTextArea({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules
    });
    return /*#__PURE__*/ jsx(Input.TextArea, {
        ...restProps,
        ...field,
        value: field.value,
        defaultValue: restProps.defaultValue
    });
}
/**
 * @deprecated Use `RHFControlledSelect` instead.
 */ function RHFControlledLegacySelect({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules
    });
    return /*#__PURE__*/ jsx(LegacySelect, {
        ...restProps,
        ...field,
        value: field.value,
        defaultValue: field.value
    });
}
/**
 * @deprecated This component is no longer necessary as `SimpleSelect` can be used uncontrolled by RHF.
 * Please consult the Forms Guide on go/dubois.
 */ function RHFControlledSelect({ name, control, rules, options, validationState, children, width, triggerProps, contentProps, optionProps, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules
    });
    const [selectedValueLabel, setSelectedValueLabel] = useState(field.value ? field.value.label ? field.value.label : field.value : '');
    const handleOnChange = (option)=>{
        field.onChange(typeof option === 'object' ? option.value : option);
    };
    useEffect(()=>{
        if (!field.value) {
            return;
        }
        // Find the appropriate label for the selected value
        if (!options?.length && children) {
            const renderedChildren = children({
                onChange: handleOnChange
            });
            const child = (Array.isArray(renderedChildren) ? renderedChildren : Children.toArray(renderedChildren.props.children)).find((child)=>/*#__PURE__*/ React__default.isValidElement(child) && child.props.value === field.value);
            if (child) {
                if (child.props?.children !== field.value) {
                    setSelectedValueLabel(child.props.children);
                } else {
                    setSelectedValueLabel(field.value);
                }
            }
        } else if (options?.length) {
            const option = options.find((option)=>option.value === field.value);
            setSelectedValueLabel(option?.label);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
        field.value
    ]);
    return /*#__PURE__*/ jsxs(Select, {
        ...restProps,
        value: field.value,
        children: [
            /*#__PURE__*/ jsx(SelectTrigger, {
                ...triggerProps,
                width: width,
                onBlur: field.onBlur,
                validationState: validationState,
                ref: field.ref,
                children: selectedValueLabel
            }),
            /*#__PURE__*/ jsx(SelectContent, {
                ...contentProps,
                side: "bottom",
                children: options && options.length > 0 ? options.map((option)=>/*#__PURE__*/ createElement(SelectOption, {
                        ...optionProps,
                        key: option.value,
                        value: option.value,
                        onChange: handleOnChange
                    }, option.label)) : // We expose onChange through a children renderer function to let users pass this down to SelectOption
                children?.({
                    onChange: handleOnChange
                })
            })
        ]
    });
}
function RHFControlledDialogCombobox({ name, control, rules, children, allowClear, validationState, placeholder, width, triggerProps, contentProps, optionListProps, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules
    });
    const [valueMap, setValueMap] = useState({});
    const updateValueMap = useCallback((updatedValue)=>{
        if (updatedValue) {
            if (Array.isArray(updatedValue)) {
                setValueMap(updatedValue.reduce((acc, value)=>{
                    acc[value] = true;
                    return acc;
                }, {}));
            } else {
                setValueMap({
                    [updatedValue]: true
                });
            }
        } else {
            setValueMap({});
        }
    }, []);
    useEffect(()=>{
        updateValueMap(field.value);
    }, [
        field.value,
        updateValueMap
    ]);
    const handleOnChangeSingleSelect = (option)=>{
        let updatedValue = field.value;
        if (field.value === option) {
            updatedValue = undefined;
        } else {
            updatedValue = option;
        }
        field.onChange(updatedValue);
        updateValueMap(updatedValue);
    };
    const handleOnChangeMultiSelect = (option)=>{
        let updatedValue;
        if (field.value?.includes(option)) {
            updatedValue = field.value.filter((value)=>value !== option);
        } else if (!field.value) {
            updatedValue = [
                option
            ];
        } else {
            updatedValue = [
                ...field.value,
                option
            ];
        }
        field.onChange(updatedValue);
        updateValueMap(updatedValue);
    };
    const handleOnChange = (option)=>{
        if (restProps.multiSelect) {
            handleOnChangeMultiSelect(option);
        } else {
            handleOnChangeSingleSelect(option);
        }
    };
    const isChecked = (option)=>{
        return valueMap[option];
    };
    const handleOnClear = ()=>{
        field.onChange(Array.isArray(field.value) ? [] : '');
        setValueMap({});
    };
    return /*#__PURE__*/ jsxs(DialogCombobox, {
        ...restProps,
        value: field.value ? Array.isArray(field.value) ? field.value : [
            field.value
        ] : undefined,
        children: [
            /*#__PURE__*/ jsx(DialogComboboxTrigger, {
                ...triggerProps,
                onBlur: field.onBlur,
                allowClear: allowClear,
                validationState: validationState,
                onClear: handleOnClear,
                withInlineLabel: false,
                placeholder: placeholder,
                width: width,
                ref: field.ref
            }),
            /*#__PURE__*/ jsx(DialogComboboxContent, {
                ...contentProps,
                side: "bottom",
                width: width,
                children: /*#__PURE__*/ jsx(DialogComboboxOptionList, {
                    ...optionListProps,
                    children: children?.({
                        onChange: handleOnChange,
                        value: field.value,
                        isChecked
                    })
                })
            })
        ]
    });
}
function RHFControlledTypeaheadCombobox({ name, control, rules, allItems, itemToString, matcher, allowNewValue, children, validationState, inputProps, menuProps, onInputChange, componentId, analyticsEvents, valueHasNoPii, preventUnsetOnBlur = false, ...props }) {
    const { field } = useController({
        name,
        control,
        rules
    });
    const [items, setItems] = useState(allItems);
    const comboboxState = useComboboxState({
        allItems,
        items,
        setItems,
        itemToString,
        matcher,
        allowNewValue,
        formValue: field.value,
        formOnChange: field.onChange,
        formOnBlur: field.onBlur,
        componentId,
        analyticsEvents,
        valueHasNoPii,
        preventUnsetOnBlur
    });
    const lastEmmitedInputValue = useRef(inputProps?.value);
    useEffect(()=>{
        setItems(allItems);
    }, [
        allItems
    ]);
    useEffect(()=>{
        if (onInputChange && lastEmmitedInputValue.current !== comboboxState.inputValue) {
            onInputChange(comboboxState.inputValue);
            lastEmmitedInputValue.current = comboboxState.inputValue;
        }
    }, [
        comboboxState.inputValue,
        onInputChange
    ]);
    return /*#__PURE__*/ jsxs(TypeaheadComboboxRoot, {
        ...props,
        comboboxState: comboboxState,
        children: [
            /*#__PURE__*/ jsx(TypeaheadComboboxInput, {
                ...inputProps,
                validationState: validationState,
                formOnChange: field.onChange,
                comboboxState: comboboxState,
                ref: field.ref
            }),
            /*#__PURE__*/ jsx(TypeaheadComboboxMenu, {
                ...menuProps,
                comboboxState: comboboxState,
                children: children({
                    comboboxState,
                    items
                })
            })
        ]
    });
}
function RHFControlledMultiSelectTypeaheadCombobox({ name, control, rules, allItems, itemToString, matcher, children, validationState, inputProps, menuProps, onInputChange, componentId, analyticsEvents, valueHasNoPii, ...props }) {
    const { field } = useController({
        name,
        control,
        rules
    });
    const [inputValue, setInputValue] = useState('');
    const [selectedItems, setSelectedItems] = useState(field.value || []);
    useEffect(()=>{
        setSelectedItems(field.value || []);
    }, [
        field.value
    ]);
    const items = React__default.useMemo(()=>allItems.filter((item)=>matcher(item, inputValue)), [
        inputValue,
        matcher,
        allItems
    ]);
    const handleItemUpdate = (item)=>{
        field.onChange(item);
        setSelectedItems(item);
    };
    const comboboxState = useComboboxState({
        allItems,
        items,
        setInputValue,
        matcher,
        itemToString,
        multiSelect: true,
        selectedItems,
        setSelectedItems: handleItemUpdate,
        formValue: field.value,
        formOnChange: field.onChange,
        formOnBlur: field.onBlur,
        componentId,
        analyticsEvents,
        valueHasNoPii
    });
    const multipleSelectionState = useMultipleSelectionState(selectedItems, handleItemUpdate, comboboxState);
    const lastEmmitedInputValue = useRef(inputProps?.value);
    useEffect(()=>{
        if (onInputChange && lastEmmitedInputValue.current !== comboboxState.inputValue) {
            onInputChange(comboboxState.inputValue);
            lastEmmitedInputValue.current = comboboxState.inputValue;
        }
    }, [
        comboboxState.inputValue,
        onInputChange
    ]);
    return /*#__PURE__*/ jsxs(TypeaheadComboboxRoot, {
        ...props,
        comboboxState: comboboxState,
        children: [
            /*#__PURE__*/ jsx(TypeaheadComboboxMultiSelectInput, {
                ...inputProps,
                multipleSelectionState: multipleSelectionState,
                selectedItems: selectedItems,
                setSelectedItems: handleItemUpdate,
                getSelectedItemLabel: itemToString,
                comboboxState: comboboxState,
                validationState: validationState,
                ref: field.ref
            }),
            /*#__PURE__*/ jsx(TypeaheadComboboxMenu, {
                ...menuProps,
                comboboxState: comboboxState,
                children: children({
                    comboboxState,
                    items,
                    selectedItems
                })
            })
        ]
    });
}
function RHFControlledCheckboxGroup({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules
    });
    return /*#__PURE__*/ jsx(Checkbox.Group, {
        ...restProps,
        ...field,
        value: field.value
    });
}
function RHFControlledCheckbox({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules
    });
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        css: {
            marginTop: theme.spacing.sm
        },
        children: /*#__PURE__*/ jsx(Checkbox, {
            ...restProps,
            ...field,
            isChecked: field.value
        })
    });
}
function RHFControlledRadioGroup({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules
    });
    return /*#__PURE__*/ jsx(Radio.Group, {
        ...restProps,
        ...field
    });
}
function RHFControlledSwitch({ name, control, rules, ...restProps }) {
    const { field } = useController({
        name: name,
        control: control,
        rules: rules
    });
    return /*#__PURE__*/ jsx(Switch, {
        ...restProps,
        ...field,
        checked: field.value
    });
}
const RHFControlledComponents = {
    Input: RHFControlledInput,
    Password: RHFControlledPasswordInput,
    TextArea: RHFControlledTextArea,
    LegacySelect: RHFControlledLegacySelect,
    Select: RHFControlledSelect,
    DialogCombobox: RHFControlledDialogCombobox,
    Checkbox: RHFControlledCheckbox,
    CheckboxGroup: RHFControlledCheckboxGroup,
    RadioGroup: RHFControlledRadioGroup,
    TypeaheadCombobox: RHFControlledTypeaheadCombobox,
    MultiSelectTypeaheadCombobox: RHFControlledMultiSelectTypeaheadCombobox,
    Switch: RHFControlledSwitch
};

const getHorizontalInputStyles = (theme, labelColWidth, inputColWidth)=>{
    return /*#__PURE__*/ css({
        display: 'flex',
        gap: theme.spacing.sm,
        '& > input, & > textarea, & > select': {
            marginTop: '0 !important'
        },
        '& > div:nth-of-type(1)': {
            width: labelColWidth
        },
        '& > div:nth-of-type(2)': {
            width: inputColWidth
        }
    });
};
const HorizontalFormRow = ({ children, labelColWidth = '33%', inputColWidth = '66%', ...restProps })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        css: getHorizontalInputStyles(theme, labelColWidth, inputColWidth),
        ...restProps,
        children: children
    });
};
const FormUI = {
    Message: FormMessage,
    Label: Label,
    Hint: Hint,
    HorizontalFormRow
};

const Col = ({ dangerouslySetAntdProps, ...props })=>/*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Col$1, {
            ...props,
            ...dangerouslySetAntdProps
        })
    });

const ROW_GUTTER_SIZE = 8;
const Row = ({ gutter = ROW_GUTTER_SIZE, ...props })=>{
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Row$1, {
            gutter: gutter,
            ...props
        })
    });
};

const Space = ({ dangerouslySetAntdProps, ...props })=>{
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Space$1, {
            ...props,
            ...dangerouslySetAntdProps
        })
    });
};

const getHeaderStyles = (clsPrefix, theme)=>{
    const breadcrumbClass = `.${clsPrefix}-breadcrumb`;
    const styles = {
        [breadcrumbClass]: {
            lineHeight: theme.typography.lineHeightBase
        }
    };
    return /*#__PURE__*/ css(importantify(styles));
};
const Header$1 = ({ breadcrumbs, title, titleAddOns, dangerouslyAppendEmotionCSS, buttons, children, titleElementLevel, allowTitleWrap = true, ...rest })=>{
    const { classNamePrefix: clsPrefix, theme } = useDesignSystemTheme();
    const buttonsArray = Array.isArray(buttons) ? buttons : buttons ? [
        buttons
    ] : [];
    // TODO: Move to getHeaderStyles for consistency, followup ticket: https://databricks.atlassian.net/browse/FEINF-1222
    const styles = {
        titleWrapper: /*#__PURE__*/ css({
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            flexWrap: allowTitleWrap ? 'wrap' : 'nowrap',
            rowGap: theme.spacing.sm,
            // Buttons have 32px height while Title level 2 elements used by this component have a height of 28px
            // These paddings enforce height to be the same without buttons too
            ...buttonsArray.length === 0 && {
                paddingTop: breadcrumbs ? 0 : theme.spacing.xs / 2,
                paddingBottom: theme.spacing.xs / 2
            }
        }),
        breadcrumbWrapper: /*#__PURE__*/ css({
            lineHeight: theme.typography.lineHeightBase,
            marginBottom: theme.spacing.xs
        }),
        title: /*#__PURE__*/ css({
            marginTop: 0,
            marginBottom: '0 !important',
            alignSelf: 'stretch',
            ...!allowTitleWrap && {
                // Allow the title to shrink if wrapping is disabled
                flex: 1,
                minWidth: 0
            }
        }),
        // TODO: Look into a more emotion-idomatic way of doing this.
        titleIfOtherElementsPresent: /*#__PURE__*/ css({
            marginTop: 2
        }),
        buttonContainer: /*#__PURE__*/ css({
            marginLeft: 8
        }),
        titleAddOnsWrapper: /*#__PURE__*/ css({
            display: 'inline-flex',
            verticalAlign: 'middle',
            alignItems: 'center',
            flexWrap: 'wrap',
            marginLeft: theme.spacing.sm,
            gap: theme.spacing.xs
        })
    };
    return /*#__PURE__*/ jsxs("div", {
        ...addDebugOutlineIfEnabled(),
        css: [
            getHeaderStyles(clsPrefix, theme),
            dangerouslyAppendEmotionCSS
        ],
        ...rest,
        children: [
            breadcrumbs && /*#__PURE__*/ jsx("div", {
                css: styles.breadcrumbWrapper,
                children: breadcrumbs
            }),
            /*#__PURE__*/ jsxs("div", {
                css: styles.titleWrapper,
                children: [
                    /*#__PURE__*/ jsxs(Title$2, {
                        level: 2,
                        elementLevel: titleElementLevel,
                        css: [
                            styles.title,
                            (buttons || breadcrumbs) && styles.titleIfOtherElementsPresent
                        ],
                        children: [
                            title,
                            titleAddOns && /*#__PURE__*/ jsx("span", {
                                css: styles.titleAddOnsWrapper,
                                children: titleAddOns
                            })
                        ]
                    }),
                    buttons && /*#__PURE__*/ jsx("div", {
                        css: styles.buttonContainer,
                        children: /*#__PURE__*/ jsx(Space, {
                            dangerouslySetAntdProps: {
                                wrap: true
                            },
                            size: 8,
                            children: buttonsArray.filter(Boolean).map((button, i)=>{
                                const defaultKey = `dubois-header-button-${i}`;
                                return /*#__PURE__*/ React__default.isValidElement(button) ? /*#__PURE__*/ React__default.cloneElement(button, {
                                    key: button.key || defaultKey
                                }) : /*#__PURE__*/ jsx(React__default.Fragment, {
                                    children: button
                                }, defaultKey);
                            })
                        })
                    })
                ]
            })
        ]
    });
};

/**
 * The HoverCard component combines Radix's HoverCard primitives into a single, easy-to-use component.
 * It handles the setup of the trigger, content, and arrow elements, as well as applying custom styles
 * using Emotion CSS
 */ const HoverCard = ({ trigger, content, side = 'top', sideOffset = 4, align = 'center', minWidth = 220, maxWidth, backgroundColor, withArrow = true, ...props })=>{
    const { getPopupContainer } = useDesignSystemContext();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const hoverCardStyles = useHoverCardStyles({
        minWidth,
        maxWidth,
        useNewBorderColors,
        backgroundColor
    });
    return /*#__PURE__*/ jsxs(RadixHoverCard.Root, {
        ...props,
        children: [
            /*#__PURE__*/ jsx(RadixHoverCard.Trigger, {
                asChild: true,
                children: trigger
            }),
            /*#__PURE__*/ jsx(RadixHoverCard.Portal, {
                container: getPopupContainer && getPopupContainer(),
                children: /*#__PURE__*/ jsxs(RadixHoverCard.Content, {
                    side: side,
                    sideOffset: sideOffset,
                    align: align,
                    css: hoverCardStyles['content'],
                    children: [
                        content,
                        withArrow && /*#__PURE__*/ jsx(RadixHoverCard.Arrow, {
                            css: hoverCardStyles['arrow']
                        })
                    ]
                })
            })
        ]
    });
};
// CONSTANTS used for defining the Arrow's appearance and behavior
const CONSTANTS = {
    arrowWidth: 12,
    arrowHeight: 6,
    arrowBottomLength () {
        // The built in arrow is a polygon: 0,0 30,0 15,10
        return 30;
    },
    arrowSide () {
        return 2 * (this.arrowHeight ** 2 * 2) ** 0.5;
    },
    arrowStrokeWidth () {
        // This is eyeballed b/c relative to the svg viewbox coordinate system
        return 2;
    }
};
/**
 * A custom hook to generate CSS styles for the HoverCard's content and arrow.
 * These styles are dynamically generated based on the theme and optional min/max width props.
 * The hook also applies necessary dark mode adjustments
 */ const useHoverCardStyles = ({ minWidth, maxWidth, useNewBorderColors, backgroundColor })=>{
    const { theme } = useDesignSystemTheme();
    return {
        content: {
            backgroundColor: backgroundColor ?? theme.colors.backgroundPrimary,
            color: theme.colors.textPrimary,
            lineHeight: theme.typography.lineHeightBase,
            border: `1px solid ${useNewBorderColors ? theme.colors.border : theme.colors.borderDecorative}`,
            borderRadius: theme.borders.borderRadiusSm,
            padding: `${theme.spacing.sm}px`,
            boxShadow: theme.shadows.lg,
            userSelect: 'none',
            zIndex: theme.options.zIndexBase + 30,
            minWidth,
            maxWidth,
            ...getDarkModePortalStyles(theme, useNewBorderColors),
            a: importantify({
                color: theme.colors.actionTertiaryTextDefault,
                cursor: 'default',
                '&:hover, &:focus': {
                    color: theme.colors.actionTertiaryTextHover
                }
            }),
            '&:focus-visible': {
                outlineStyle: 'solid',
                outlineWidth: '2px',
                outlineOffset: '1px',
                outlineColor: theme.colors.actionDefaultBorderFocus
            }
        },
        arrow: {
            fill: backgroundColor ?? theme.colors.backgroundPrimary,
            height: CONSTANTS.arrowHeight,
            stroke: theme.colors.borderDecorative,
            strokeDashoffset: -CONSTANTS.arrowBottomLength(),
            strokeDasharray: CONSTANTS.arrowBottomLength() + 2 * CONSTANTS.arrowSide(),
            strokeWidth: CONSTANTS.arrowStrokeWidth(),
            width: CONSTANTS.arrowWidth,
            position: 'relative',
            top: -1,
            zIndex: theme.options.zIndexBase + 30
        }
    };
};

const { Header, Footer, Sider, Content: Content$1 } = Layout$1;
/**
 * @deprecated Use PageWrapper instead
 */ const Layout = /* #__PURE__ */ (()=>{
    const Layout = ({ children, dangerouslySetAntdProps, ...props })=>{
        return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsx(Layout$1, {
                ...addDebugOutlineIfEnabled(),
                ...props,
                ...dangerouslySetAntdProps,
                children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                    children: children
                })
            })
        });
    };
    Layout.Header = ({ children, ...props })=>/*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsx(Header, {
                ...addDebugOutlineIfEnabled(),
                ...props,
                children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                    children: children
                })
            })
        });
    Layout.Footer = ({ children, ...props })=>/*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsx(Footer, {
                ...addDebugOutlineIfEnabled(),
                ...props,
                children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                    children: children
                })
            })
        });
    Layout.Sider = /*#__PURE__*/ React__default.forwardRef(({ children, ...props }, ref)=>/*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsx(Sider, {
                ...addDebugOutlineIfEnabled(),
                ...props,
                ref: ref,
                children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                    children: children
                })
            })
        }));
    Layout.Content = ({ children, ...props })=>/*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsx(Content$1, {
                ...addDebugOutlineIfEnabled(),
                ...props,
                children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                    children: children
                })
            })
        });
    return Layout;
})();

const getFormItemEmotionStyles = ({ theme, clsPrefix })=>{
    const clsFormItemLabel = `.${clsPrefix}-form-item-label`;
    const clsFormItemInputControl = `.${clsPrefix}-form-item-control-input`;
    const clsFormItemExplain = `.${clsPrefix}-form-item-explain`;
    const clsHasError = `.${clsPrefix}-form-item-has-error`;
    return /*#__PURE__*/ css({
        [clsFormItemLabel]: {
            fontWeight: theme.typography.typographyBoldFontWeight,
            lineHeight: theme.typography.lineHeightBase,
            '.anticon': {
                fontSize: theme.general.iconFontSize
            }
        },
        [clsFormItemExplain]: {
            fontSize: theme.typography.fontSizeSm,
            margin: 0,
            [`&${clsFormItemExplain}-success`]: {
                color: theme.colors.textValidationSuccess
            },
            [`&${clsFormItemExplain}-warning`]: {
                color: theme.colors.textValidationDanger
            },
            [`&${clsFormItemExplain}-error`]: {
                color: theme.colors.textValidationDanger
            },
            [`&${clsFormItemExplain}-validating`]: {
                color: theme.colors.textSecondary
            }
        },
        [clsFormItemInputControl]: {
            minHeight: theme.general.heightSm
        },
        [`${clsFormItemInputControl} input[disabled]`]: {
            border: 'none'
        },
        [`&${clsHasError} input:focus`]: importantify({
            boxShadow: 'none'
        }),
        ...getAnimationCss(theme.options.enableAnimation)
    });
};
/**
 * @deprecated Use `Form` from `@databricks/design-system/development` instead.
 */ const LegacyFormDubois = /*#__PURE__*/ forwardRef(function Form$1({ dangerouslySetAntdProps, children, ...props }, ref) {
    const mergedProps = {
        ...props,
        layout: props.layout || 'vertical',
        requiredMark: props.requiredMark || false
    };
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Form, {
            ...addDebugOutlineIfEnabled(),
            ...mergedProps,
            colon: false,
            ref: ref,
            ...dangerouslySetAntdProps,
            children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                children: children
            })
        })
    });
});
const FormItem = ({ dangerouslySetAntdProps, children, ...props })=>{
    const { theme, classNamePrefix } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Form.Item, {
            ...addDebugOutlineIfEnabled(),
            ...props,
            css: getFormItemEmotionStyles({
                theme,
                clsPrefix: classNamePrefix
            }),
            ...dangerouslySetAntdProps,
            children: children
        })
    });
};
const FormNamespace = /* #__PURE__ */ Object.assign(LegacyFormDubois, {
    Item: FormItem,
    List: Form.List,
    useForm: Form.useForm
});
const LegacyForm = FormNamespace;
// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
const __INTERNAL_DO_NOT_USE__FormItem = FormItem;

// Note: AntD only exposes context to notifications via the `useNotification` hook, and we need context to apply themes
// to AntD. As such you can currently only use notifications from within functional components.
/**
 * `useLegacyNotification` is deprecated in favor of the new `Notification` component.
 * @deprecated
 */ function useLegacyNotification() {
    const [notificationInstance, contextHolder] = notification.useNotification();
    const { getPrefixedClassName, theme } = useDesignSystemTheme();
    const { getPopupContainer: getContainer } = useDesignSystemContext();
    const clsPrefix = getPrefixedClassName('notification');
    const open = useCallback((args)=>{
        const mergedArgs = {
            getContainer,
            ...defaultProps,
            ...args,
            style: {
                zIndex: theme.options.zIndexBase + 30,
                boxShadow: theme.general.shadowLow
            }
        };
        const iconClassName = `${clsPrefix}-notice-icon-${mergedArgs.type}`;
        mergedArgs.icon = /*#__PURE__*/ jsx(SeverityIcon, {
            severity: mergedArgs.type,
            className: iconClassName
        });
        mergedArgs.closeIcon = /*#__PURE__*/ jsx(CloseIcon, {
            "aria-hidden": "false",
            css: {
                cursor: 'pointer',
                fontSize: theme.general.iconSize
            },
            "aria-label": mergedArgs.closeLabel || 'Close notification'
        });
        notificationInstance.open(mergedArgs);
    }, [
        notificationInstance,
        getContainer,
        theme,
        clsPrefix
    ]);
    const wrappedNotificationAPI = useMemo(()=>{
        const error = (args)=>open({
                ...args,
                type: 'error'
            });
        const warning = (args)=>open({
                ...args,
                type: 'warning'
            });
        const info = (args)=>open({
                ...args,
                type: 'info'
            });
        const success = (args)=>open({
                ...args,
                type: 'success'
            });
        const close = (key)=>notification.close(key);
        return {
            open,
            close,
            error,
            warning,
            info,
            success
        };
    }, [
        open
    ]);
    // eslint-disable-next-line react/jsx-key -- TODO(FEINF-1756)
    return [
        wrappedNotificationAPI,
        /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: contextHolder
        })
    ];
}
const defaultProps = {
    type: 'info',
    duration: 3
};
/**
 * A higher-order component factory function, enables using notifications in
 * class components in a similar way to useNotification() hook. Wrapped component will have
 * additional "notificationAPI" and "notificationContextHolder" props injected containing
 * the notification API object and context holder react node respectively.
 *
 * The wrapped component can implement WithNotificationsHOCProps<OwnProps> type which
 * enriches the component's interface with the mentioned props.
 *
 * @deprecated Please migrate components to functional components and use useNotification() hook instead.
 */ const withNotifications = (Component)=>/*#__PURE__*/ forwardRef((props, ref)=>{
        const [notificationAPI, notificationContextHolder] = useLegacyNotification();
        return /*#__PURE__*/ jsx(Component, {
            ref: ref,
            notificationAPI: notificationAPI,
            notificationContextHolder: notificationContextHolder,
            ...props
        });
    });

/**
 * `LegacyPopover` is deprecated in favor of the new `Popover` component.
 * @deprecated
 */ const LegacyPopover = ({ content, dangerouslySetAntdProps, ...props })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Popover$1, {
            zIndex: theme.options.zIndexBase + 30,
            ...props,
            content: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                children: content
            })
        })
    });
};

/** @deprecated This component is deprecated. Use ParagraphSkeleton, TitleSkeleton, or GenericSkeleton instead. */ const LegacySkeleton = /* #__PURE__ */ (()=>{
    const LegacySkeleton = ({ dangerouslySetAntdProps, label, loadingDescription = 'LegacySkeleton', ...props })=>{
        // There is a conflict in how the 'loading' prop is handled here, so we can't default it to true in
        // props destructuring above like we do for 'loadingDescription'. The 'loading' param is used both
        // for <LoadingState> and in <AntDSkeleton>. The intent is for 'loading' to default to true in
        // <LoadingState>, but if we do that, <AntDSkeleton> will not render the children. The workaround
        // here is to default 'loading' to true only when considering whether to render a <LoadingState>.
        // Also, AntDSkeleton looks at the presence of 'loading' in props, so We cannot explicitly destructure
        // 'loading' in the constructor since we would no longer be able to differentiate between it not being
        // passed in at all and being passed undefined.
        const loadingStateLoading = props.loading ?? true;
        return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsxs(AccessibleContainer, {
                label: label,
                children: [
                    loadingStateLoading && /*#__PURE__*/ jsx(LoadingState, {
                        description: loadingDescription
                    }),
                    /*#__PURE__*/ jsx(Skeleton, {
                        ...props,
                        ...dangerouslySetAntdProps
                    })
                ]
            })
        });
    };
    LegacySkeleton.Button = Skeleton.Button;
    LegacySkeleton.Image = Skeleton.Image;
    LegacySkeleton.Input = Skeleton.Input;
    return LegacySkeleton;
})();

function getPaginationEmotionStyles(clsPrefix, theme, useNewBorderColors) {
    const classRoot = `.${clsPrefix}-pagination`;
    const classItem = `.${clsPrefix}-pagination-item`;
    const classLink = `.${clsPrefix}-pagination-item-link`;
    const classActive = `.${clsPrefix}-pagination-item-active`;
    const classEllipsis = `.${clsPrefix}-pagination-item-ellipsis`;
    const classNext = `.${clsPrefix}-pagination-next`;
    const classPrev = `.${clsPrefix}-pagination-prev`;
    const classJumpNext = `.${clsPrefix}-pagination-jump-next`;
    const classJumpPrev = `.${clsPrefix}-pagination-jump-prev`;
    const classQuickJumper = `.${clsPrefix}-pagination-options-quick-jumper`;
    const classSizeChanger = `.${clsPrefix}-pagination-options-size-changer`;
    const classOptions = `.${clsPrefix}-pagination-options`;
    const classDisabled = `.${clsPrefix}-pagination-disabled`;
    const classSelector = `.${clsPrefix}-select-selector`;
    const classDropdown = `.${clsPrefix}-select-dropdown`;
    const styles = {
        'span[role=img]': {
            color: theme.colors.textSecondary,
            '> *': {
                color: 'inherit'
            }
        },
        [classItem]: {
            backgroundColor: 'none',
            border: 'none',
            color: theme.colors.textSecondary,
            '&:focus-visible': {
                outline: 'auto'
            },
            '> a': {
                color: theme.colors.textSecondary,
                textDecoration: 'none',
                '&:hover': {
                    color: theme.colors.actionDefaultTextHover
                },
                '&:active': {
                    color: theme.colors.actionDefaultTextPress
                }
            },
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress
            }
        },
        [classActive]: {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
            color: theme.colors.actionDefaultTextPress,
            border: 'none',
            '> a': {
                color: theme.colors.actionDefaultTextPress
            },
            '&:focus-visible': {
                outline: 'auto'
            },
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                color: theme.colors.actionDefaultTextPress
            }
        },
        [classLink]: {
            border: 'none',
            color: theme.colors.textSecondary,
            '&[disabled]': {
                display: 'none'
            },
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress
            },
            '&:focus-visible': {
                outline: 'auto'
            }
        },
        [classEllipsis]: {
            color: 'inherit'
        },
        [`${classNext}, ${classPrev}, ${classJumpNext}, ${classJumpPrev}`]: {
            color: theme.colors.textSecondary,
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress
            },
            '&:focus-visible': {
                outline: 'auto'
            },
            [`&${classDisabled}`]: {
                pointerEvents: 'none'
            }
        },
        [`&${classRoot}.mini, ${classRoot}.mini`]: {
            [`${classItem}, ${classNext}, ${classPrev}, ${classJumpNext}, ${classJumpPrev}`]: {
                height: '32px',
                minWidth: '32px',
                width: 'auto',
                lineHeight: '32px'
            },
            [classSizeChanger]: {
                marginLeft: 4
            },
            [`input,  ${classOptions}`]: {
                height: '32px'
            },
            [`${classSelector}`]: {
                boxShadow: theme.shadows.xs,
                ...useNewBorderColors && {
                    borderColor: theme.colors.actionDefaultBorderDefault
                }
            }
        },
        ...useNewBorderColors && {
            [`${classQuickJumper} > input`]: {
                borderColor: theme.colors.actionDefaultBorderDefault
            },
            [`${classDropdown}`]: {
                borderColor: theme.colors.actionDefaultBorderDefault
            }
        }
    };
    const importantStyles = importantify(styles);
    return /*#__PURE__*/ css(importantStyles);
}
const Pagination = function Pagination({ currentPageIndex, pageSize = 10, numTotal, onChange, style, hideOnSinglePage, dangerouslySetAntdProps, componentId, analyticsEvents }) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.pagination', false);
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const { pageSizeSelectAriaLabel, pageQuickJumperAriaLabel, ...restDangerouslySetAntdProps } = dangerouslySetAntdProps ?? {};
    const ref = useRef(null);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Pagination,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true
    });
    const onChangeWrapper = useCallback((pageIndex, pageSize)=>{
        eventContext.onValueChange(pageIndex);
        onChange(pageIndex, pageSize);
    }, [
        eventContext,
        onChange
    ]);
    const { elementRef: paginationRef } = useNotifyOnFirstView({
        onView: eventContext.onView
    });
    const mergedRef = useMergeRefs([
        ref,
        paginationRef
    ]);
    useEffect(()=>{
        if (ref && ref.current) {
            const selectDropdown = ref.current.querySelector(`.${classNamePrefix}-select-selection-search-input`);
            if (selectDropdown) {
                selectDropdown.setAttribute('aria-label', pageSizeSelectAriaLabel ?? 'Select page size');
            }
            const pageQuickJumper = ref.current.querySelector(`.${classNamePrefix}-pagination-options-quick-jumper > input`);
            if (pageQuickJumper) {
                pageQuickJumper.setAttribute('aria-label', pageQuickJumperAriaLabel ?? 'Go to page');
            }
        }
    }, [
        pageQuickJumperAriaLabel,
        pageSizeSelectAriaLabel,
        classNamePrefix
    ]);
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx("div", {
            ref: mergedRef,
            children: /*#__PURE__*/ jsx(Pagination$1, {
                ...addDebugOutlineIfEnabled(),
                css: getPaginationEmotionStyles(classNamePrefix, theme, useNewBorderColors),
                current: currentPageIndex,
                pageSize: pageSize,
                responsive: false,
                total: numTotal,
                onChange: onChangeWrapper,
                showSizeChanger: false,
                showQuickJumper: false,
                size: "small",
                style: style,
                hideOnSinglePage: hideOnSinglePage,
                ...restDangerouslySetAntdProps,
                ...eventContext.dataComponentProps
            })
        })
    });
};
const CursorPagination = function CursorPagination({ onNextPage, onPreviousPage, hasNextPage, hasPreviousPage, nextPageText = 'Next', previousPageText = 'Previous', pageSizeSelect: { options: pageSizeOptions, default: defaultPageSize, getOptionText: getPageSizeOptionText, onChange: onPageSizeChange, ariaLabel = 'Select page size' } = {}, componentId = 'design_system.cursor_pagination', analyticsEvents = [
    DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
], valueHasNoPii }) {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const [pageSizeValue, setPageSizeValue] = useState(defaultPageSize);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents, [
        analyticsEvents
    ]);
    const pageSizeEventComponentId = `${componentId}.page_size`;
    const pageSizeEventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.LegacySelect,
        componentId: pageSizeEventComponentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    const getPageSizeOptionTextDefault = (pageSize)=>`${pageSize} / page`;
    return /*#__PURE__*/ jsxs("div", {
        css: {
            display: 'flex',
            flexDirection: 'row',
            gap: theme.spacing.sm,
            [`.${classNamePrefix}-select-selector::after`]: {
                content: 'none'
            }
        },
        ...pageSizeEventContext.dataComponentProps,
        children: [
            /*#__PURE__*/ jsx(Button, {
                componentId: `${componentId}.previous_page`,
                icon: /*#__PURE__*/ jsx(ChevronLeftIcon, {}),
                disabled: !hasPreviousPage,
                onClick: onPreviousPage,
                type: "tertiary",
                children: previousPageText
            }),
            /*#__PURE__*/ jsx(Button, {
                componentId: `${componentId}.next_page`,
                endIcon: /*#__PURE__*/ jsx(ChevronRightIcon, {}),
                disabled: !hasNextPage,
                onClick: onNextPage,
                type: "tertiary",
                children: nextPageText
            }),
            pageSizeOptions && /*#__PURE__*/ jsx(LegacySelect, {
                "aria-label": ariaLabel,
                value: String(pageSizeValue),
                css: {
                    width: 120
                },
                onChange: (pageSize)=>{
                    const updatedPageSize = Number(pageSize);
                    onPageSizeChange?.(updatedPageSize);
                    setPageSizeValue(updatedPageSize);
                    // When this usage of LegacySelect is migrated to Select, this call
                    // can be removed in favor of passing a componentId to Select
                    pageSizeEventContext.onValueChange(pageSize);
                },
                children: pageSizeOptions.map((pageSize)=>/*#__PURE__*/ jsx(LegacySelect.Option, {
                        value: String(pageSize),
                        children: (getPageSizeOptionText || getPageSizeOptionTextDefault)(pageSize)
                    }, pageSize))
            })
        ]
    });
};

const getTableEmotionStyles = (classNamePrefix, theme, scrollableInFlexibleContainer, useNewBorderColors)=>{
    const styles = [
        /*#__PURE__*/ css({
            [`.${classNamePrefix}-table-pagination`]: {
                ...getPaginationEmotionStyles(classNamePrefix, theme, useNewBorderColors)
            }
        })
    ];
    if (scrollableInFlexibleContainer) {
        styles.push(getScrollableInFlexibleContainerStyles(classNamePrefix));
    }
    return styles;
};
const getScrollableInFlexibleContainerStyles = (clsPrefix)=>{
    const styles = {
        minHeight: 0,
        [`.${clsPrefix}-spin-nested-loading`]: {
            height: '100%'
        },
        [`.${clsPrefix}-spin-container`]: {
            height: '100%',
            display: 'flex',
            flexDirection: 'column'
        },
        [`.${clsPrefix}-table-container`]: {
            height: '100%',
            display: 'flex',
            flexDirection: 'column'
        },
        [`.${clsPrefix}-table`]: {
            minHeight: 0
        },
        [`.${clsPrefix}-table-header`]: {
            flexShrink: 0
        },
        [`.${clsPrefix}-table-body`]: {
            minHeight: 0
        }
    };
    return /*#__PURE__*/ css(styles);
};
const DEFAULT_LOADING_SPIN_PROPS = {
    indicator: /*#__PURE__*/ jsx(Spinner, {})
};
/**
 * `LegacyTable` is deprecated in favor of the new `Table` component
 * For more information please join #dubois-table-early-adopters in Slack.
 * @deprecated
 */ // eslint-disable-next-line @typescript-eslint/no-restricted-types
const LegacyTable = (props)=>{
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const { loading, scrollableInFlexibleContainer, children, ...tableProps } = props;
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Table$1, {
            ...addDebugOutlineIfEnabled(),
            // NOTE(FEINF-1273): The default loading indicator from AntD does not animate
            // and the design system spinner is recommended over the AntD one. Therefore,
            // if `loading` is `true`, render the design system <Spinner/> component.
            loading: loading === true ? DEFAULT_LOADING_SPIN_PROPS : loading,
            scroll: scrollableInFlexibleContainer ? {
                y: 'auto'
            } : undefined,
            ...tableProps,
            css: getTableEmotionStyles(classNamePrefix, theme, Boolean(scrollableInFlexibleContainer), useNewBorderColors),
            // ES-902549 this allows column names of "children", using a name that is less likely to be hit
            expandable: {
                childrenColumnName: '__antdChildren'
            },
            children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                children: children
            })
        })
    });
};

const getLegacyTabEmotionStyles = (clsPrefix, theme)=>{
    const classTab = `.${clsPrefix}-tabs-tab`;
    const classButton = `.${clsPrefix}-tabs-tab-btn`;
    const classActive = `.${clsPrefix}-tabs-tab-active`;
    const classDisabled = `.${clsPrefix}-tabs-tab-disabled`;
    const classUnderline = `.${clsPrefix}-tabs-ink-bar`;
    const classClosable = `.${clsPrefix}-tabs-tab-with-remove`;
    const classNav = `.${clsPrefix}-tabs-nav`;
    const classCloseButton = `.${clsPrefix}-tabs-tab-remove`;
    const classAddButton = `.${clsPrefix}-tabs-nav-add`;
    const styles = {
        '&&': {
            overflow: 'unset'
        },
        [classTab]: {
            borderBottom: 'none',
            backgroundColor: 'transparent',
            border: 'none',
            paddingLeft: 0,
            paddingRight: 0,
            paddingTop: 6,
            paddingBottom: 6,
            marginRight: 24
        },
        [classButton]: {
            color: theme.colors.textSecondary,
            fontWeight: theme.typography.typographyBoldFontWeight,
            textShadow: 'none',
            fontSize: theme.typography.fontSizeMd,
            lineHeight: theme.typography.lineHeightBase,
            '&:hover': {
                color: theme.colors.actionDefaultTextHover
            },
            '&:active': {
                color: theme.colors.actionDefaultTextPress
            },
            outlineWidth: 2,
            outlineStyle: 'none',
            outlineColor: theme.colors.actionDefaultBorderFocus,
            outlineOffset: 2,
            '&:focus-visible': {
                outlineStyle: 'auto'
            }
        },
        [classActive]: {
            [classButton]: {
                color: theme.colors.textPrimary
            },
            // Use box-shadow instead of border to prevent it from affecting the size of the element, which results in visual
            // jumping when switching tabs.
            boxShadow: `inset 0 -3px 0 ${theme.colors.actionPrimaryBackgroundDefault}`
        },
        [classDisabled]: {
            [classButton]: {
                color: theme.colors.actionDisabledText,
                '&:hover': {
                    color: theme.colors.actionDisabledText
                },
                '&:active': {
                    color: theme.colors.actionDisabledText
                }
            }
        },
        [classUnderline]: {
            display: 'none'
        },
        [classClosable]: {
            borderTop: 'none',
            borderLeft: 'none',
            borderRight: 'none',
            background: 'none',
            paddingTop: 0,
            paddingBottom: 0,
            height: theme.general.heightSm
        },
        [classNav]: {
            height: theme.general.heightSm,
            '&::before': {
                borderColor: theme.colors.actionDefaultBorderDefault
            }
        },
        [classCloseButton]: {
            height: 24,
            width: 24,
            padding: 6,
            borderRadius: theme.legacyBorders.borderRadiusMd,
            marginTop: 0,
            marginRight: 0,
            marginBottom: 0,
            marginLeft: 4,
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                color: theme.colors.actionDefaultTextHover
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                color: theme.colors.actionDefaultTextPress
            },
            '&:focus-visible': {
                outlineWidth: 2,
                outlineStyle: 'solid',
                outlineColor: theme.colors.actionDefaultBorderFocus
            }
        },
        [classAddButton]: {
            backgroundColor: 'transparent',
            color: theme.colors.textValidationInfo,
            border: 'none',
            borderRadius: theme.legacyBorders.borderRadiusMd,
            margin: 4,
            height: 24,
            width: 24,
            padding: 0,
            minWidth: 'auto',
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                color: theme.colors.actionDefaultTextHover
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                color: theme.colors.actionDefaultTextPress
            },
            '&:focus-visible': {
                outlineWidth: 2,
                outlineStyle: 'solid',
                outlineColor: theme.colors.actionDefaultBorderFocus
            },
            '& > .anticon': {
                fontSize: 16
            }
        },
        ...getAnimationCss(theme.options.enableAnimation)
    };
    const importantStyles = importantify(styles);
    return importantStyles;
};
/**
 * `LegacyTabs` is deprecated in favor of the new `Tabs` component
 * @deprecated
 */ const LegacyTabPane = ({ children, ...props })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Tabs$1.TabPane, {
            closeIcon: /*#__PURE__*/ jsx(CloseIcon, {
                css: {
                    fontSize: theme.general.iconSize
                }
            }),
            ...props,
            ...props.dangerouslySetAntdProps,
            children: /*#__PURE__*/ jsx(RestoreAntDDefaultClsPrefix, {
                children: children
            })
        })
    });
};
/**
 * `LegacyTabs` is deprecated in favor of the new `Tabs` component
 * @deprecated
 */ const LegacyTabs = /* #__PURE__ */ (()=>{
    const LegacyTabs = ({ editable = false, activeKey, defaultActiveKey, onChange, onEdit, children, destroyInactiveTabPane = false, dangerouslySetAntdProps = {}, dangerouslyAppendEmotionCSS = {}, ...props })=>{
        const { theme, classNamePrefix } = useDesignSystemTheme();
        return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsx(Tabs$1, {
                ...addDebugOutlineIfEnabled(),
                activeKey: activeKey,
                defaultActiveKey: defaultActiveKey,
                onChange: onChange,
                onEdit: onEdit,
                destroyInactiveTabPane: destroyInactiveTabPane,
                type: editable ? 'editable-card' : 'card',
                addIcon: /*#__PURE__*/ jsx(PlusIcon, {
                    css: {
                        fontSize: theme.general.iconSize
                    }
                }),
                css: [
                    getLegacyTabEmotionStyles(classNamePrefix, theme),
                    importantify(dangerouslyAppendEmotionCSS)
                ],
                ...dangerouslySetAntdProps,
                ...props,
                children: children
            })
        });
    };
    LegacyTabs.TabPane = LegacyTabPane;
    return LegacyTabs;
})();

/**
 * @deprecated Use `DropdownMenu` instead.
 */ const Menu = /* #__PURE__ */ (()=>{
    const Menu = ({ dangerouslySetAntdProps, ...props })=>{
        return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsx(Menu$1, {
                ...addDebugOutlineIfEnabled(),
                ...props,
                ...dangerouslySetAntdProps
            })
        });
    };
    Menu.Item = Menu$1.Item;
    Menu.ItemGroup = Menu$1.ItemGroup;
    Menu.SubMenu = function SubMenu({ dangerouslySetAntdProps, ...props }) {
        const { theme } = useDesignSystemTheme();
        return /*#__PURE__*/ jsx(ClassNames, {
            children: ({ css })=>{
                return /*#__PURE__*/ jsx(Menu$1.SubMenu, {
                    ...addDebugOutlineIfEnabled(),
                    popupClassName: css({
                        zIndex: theme.options.zIndexBase + 50
                    }),
                    popupOffset: [
                        -6,
                        -10
                    ],
                    ...props,
                    ...dangerouslySetAntdProps
                });
            }
        });
    };
    return Menu;
})();

const Root$3 = /*#__PURE__*/ React__default.forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(RadixNavigationMenu.Root, {
        ...props,
        ref: forwardedRef
    });
});
const List$1 = /*#__PURE__*/ React__default.forwardRef((props, forwardedRef)=>{
    const { theme } = useDesignSystemTheme();
    const commonTabsListStyles = getCommonTabsListStyles(theme);
    return /*#__PURE__*/ jsx(RadixNavigationMenu.List, {
        css: {
            ...commonTabsListStyles,
            marginTop: 0,
            padding: 0,
            overflow: 'auto hidden',
            listStyle: 'none'
        },
        ...props,
        ref: forwardedRef
    });
});
const Item = /*#__PURE__*/ React__default.forwardRef(({ children, active, ...props }, forwardedRef)=>{
    const { theme } = useDesignSystemTheme();
    const commonTabsTriggerStyles = getCommonTabsTriggerStyles(theme);
    return /*#__PURE__*/ jsx(RadixNavigationMenu.Item, {
        css: {
            ...commonTabsTriggerStyles,
            height: theme.general.heightSm,
            minWidth: theme.spacing.lg,
            justifyContent: 'center',
            ...active && {
                // Use box-shadow instead of border to prevent it from affecting the size of the element, which results in visual
                // jumping when switching tabs.
                boxShadow: `inset 0 -4px 0 ${theme.colors.actionPrimaryBackgroundDefault}`
            }
        },
        ...props,
        ref: forwardedRef,
        children: /*#__PURE__*/ jsx(RadixNavigationMenu.Link, {
            asChild: true,
            active: active,
            css: {
                padding: `${theme.spacing.xs}px 0 ${theme.spacing.sm}px 0`,
                '&:focus': {
                    outline: `2px auto ${theme.colors.actionDefaultBorderFocus}`,
                    outlineOffset: '-1px'
                },
                '&&': {
                    color: active ? theme.colors.textPrimary : theme.colors.textSecondary,
                    textDecoration: 'none',
                    '&:hover': {
                        color: active ? theme.colors.textPrimary : theme.colors.actionDefaultTextHover,
                        textDecoration: 'none'
                    },
                    '&:focus': {
                        textDecoration: 'none'
                    },
                    '&:active': {
                        color: active ? theme.colors.textPrimary : theme.colors.actionDefaultTextPress
                    }
                }
            },
            children: children
        })
    });
});

var NavigationMenu = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Item: Item,
  List: List$1,
  Root: Root$3
});

const hideAnimation = /*#__PURE__*/ keyframes({
    from: {
        opacity: 1
    },
    to: {
        opacity: 0
    }
});
const slideInAnimation = /*#__PURE__*/ keyframes({
    from: {
        transform: 'translateX(calc(100% + 12px))'
    },
    to: {
        transform: 'translateX(0)'
    }
});
const swipeOutAnimation = /*#__PURE__*/ keyframes({
    from: {
        transform: 'translateX(var(--radix-toast-swipe-end-x))'
    },
    to: {
        transform: 'translateX(calc(100% + 12px))'
    }
});
const getToastRootStyle = (theme, classNamePrefix, useNewBorderColors)=>{
    return /*#__PURE__*/ css({
        '&&': {
            position: 'relative',
            display: 'grid',
            background: theme.colors.backgroundPrimary,
            padding: 12,
            columnGap: 4,
            boxShadow: theme.shadows.lg,
            borderRadius: theme.borders.borderRadiusSm,
            lineHeight: '20px',
            ...useNewBorderColors && {
                borderColor: `1px solid ${theme.colors.border}`
            },
            gridTemplateRows: '[header] auto [content] auto',
            gridTemplateColumns: '[icon] auto [content] 1fr [close] auto',
            ...getDarkModePortalStyles(theme, useNewBorderColors)
        },
        [`.${classNamePrefix}-notification-severity-icon`]: {
            gridRow: 'header / content',
            gridColumn: 'icon / icon',
            display: 'inline-flex',
            alignItems: 'center'
        },
        [`.${classNamePrefix}-btn`]: {
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center'
        },
        [`.${classNamePrefix}-notification-info-icon`]: {
            color: theme.colors.textSecondary
        },
        [`.${classNamePrefix}-notification-success-icon`]: {
            color: theme.colors.textValidationSuccess
        },
        [`.${classNamePrefix}-notification-warning-icon`]: {
            color: theme.colors.textValidationWarning
        },
        [`.${classNamePrefix}-notification-error-icon`]: {
            color: theme.colors.textValidationDanger
        },
        '&&[data-state="open"]': {
            animation: `${slideInAnimation} 300ms cubic-bezier(0.16, 1, 0.3, 1)`
        },
        '&[data-state="closed"]': {
            animation: `${hideAnimation} 100ms ease-in`
        },
        '&[data-swipe="move"]': {
            transform: 'translateX(var(--radix-toast-swipe-move-x))'
        },
        '&[data-swipe="cancel"]': {
            transform: 'translateX(0)',
            transition: 'transform 200ms ease-out'
        },
        '&[data-swipe="end"]': {
            animation: `${swipeOutAnimation} 100ms ease-out`
        }
    });
};
const Root$2 = /*#__PURE__*/ forwardRef(function({ children, severity = 'info', componentId, analyticsEvents = [
    DesignSystemEventProviderAnalyticsEventTypes.OnView
], ...props }, ref) {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents, [
        analyticsEvents
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Notification,
        componentId,
        componentSubType: DesignSystemEventProviderComponentSubTypeMap[severity],
        analyticsEvents: memoizedAnalyticsEvents,
        shouldStartInteraction: false
    });
    // A new ref was created rather than creating additional complexity of merging the refs, something to consider for the future to optimize
    const { elementRef } = useNotifyOnFirstView({
        onView: eventContext.onView
    });
    return /*#__PURE__*/ jsxs(Toast.Root, {
        ref: ref,
        css: getToastRootStyle(theme, classNamePrefix, useNewBorderColors),
        ...props,
        ...addDebugOutlineIfEnabled(),
        children: [
            /*#__PURE__*/ jsx(SeverityIcon, {
                className: `${classNamePrefix}-notification-severity-icon ${classNamePrefix}-notification-${severity}-icon`,
                severity: severity,
                ref: elementRef
            }),
            children
        ]
    });
});
// TODO: Support light and dark mode
const getViewportStyle = (theme)=>{
    return {
        position: 'fixed',
        top: 0,
        right: 0,
        display: 'flex',
        flexDirection: 'column',
        padding: 12,
        gap: 12,
        width: 440,
        listStyle: 'none',
        zIndex: theme.options.zIndexBase + 100,
        outline: 'none',
        maxWidth: `calc(100% - ${theme.spacing.lg}px)`
    };
};
const getTitleStyles = (theme)=>{
    return /*#__PURE__*/ css({
        fontWeight: theme.typography.typographyBoldFontWeight,
        color: theme.colors.textPrimary,
        gridRow: 'header / header',
        gridColumn: 'content / content',
        userSelect: 'text'
    });
};
const Title = /*#__PURE__*/ forwardRef(function({ children, ...props }, ref) {
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(Toast.Title, {
        ref: ref,
        css: getTitleStyles(theme),
        ...props,
        children: children
    });
});
const getDescriptionStyles = (theme)=>{
    return /*#__PURE__*/ css({
        marginTop: 4,
        color: theme.colors.textPrimary,
        gridRow: 'content / content',
        gridColumn: 'content / content',
        userSelect: 'text',
        wordBreak: 'break-word'
    });
};
const Description = /*#__PURE__*/ forwardRef(function({ children, ...props }, ref) {
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(Toast.Description, {
        ref: ref,
        css: getDescriptionStyles(theme),
        ...props,
        children: children
    });
});
const getCloseStyles = (theme)=>{
    return /*#__PURE__*/ css({
        color: theme.colors.textSecondary,
        position: 'absolute',
        // Offset close button position to align with the title, title uses 20px line height, button has 32px
        right: 6,
        top: 6
    });
};
const Close = /*#__PURE__*/ forwardRef(function(props, ref) {
    const { theme } = useDesignSystemTheme();
    const { closeLabel, componentId, analyticsEvents, ...restProps } = props;
    return(// Wrapper to keep close column width for content sizing, close button positioned absolute for alignment without affecting the grid's first row height (title)
    /*#__PURE__*/ jsx("div", {
        style: {
            gridColumn: 'close / close',
            gridRow: 'header / content',
            width: 20
        },
        children: /*#__PURE__*/ jsx(Toast.Close, {
            ref: ref,
            css: getCloseStyles(theme),
            ...restProps,
            asChild: true,
            children: /*#__PURE__*/ jsx(Button, {
                componentId: componentId ? componentId : 'codegen_design-system_src_design-system_notification_notification.tsx_224',
                analyticsEvents: analyticsEvents,
                icon: /*#__PURE__*/ jsx(CloseIcon, {}),
                "aria-label": closeLabel ?? restProps['aria-label'] ?? 'Close notification'
            })
        })
    }));
});
const Provider = ({ children, ...props })=>{
    return /*#__PURE__*/ jsx(Toast.Provider, {
        ...props,
        children: children
    });
};
const Viewport = (props)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(Toast.Viewport, {
        className: DU_BOIS_ENABLE_ANIMATION_CLASSNAME,
        style: getViewportStyle(theme),
        ...props
    });
};

var Notification = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Close: Close,
  Description: Description,
  Provider: Provider,
  Root: Root$2,
  Title: Title,
  Viewport: Viewport
});

const oldTagColorsMap = {
    default: 'tagDefault',
    brown: 'tagBrown',
    coral: 'tagCoral',
    charcoal: 'grey600',
    indigo: 'tagIndigo',
    lemon: 'tagLemon',
    lime: 'tagLime',
    pink: 'tagPink',
    purple: 'tagPurple',
    teal: 'tagTeal',
    turquoise: 'tagTurquoise'
};
function getTagEmotionStyles(theme, color = 'default', clickable = false, closable = false) {
    let textColor = theme.colors.tagText;
    let backgroundColor = theme.colors[oldTagColorsMap[color]];
    let iconColor = '';
    let outlineColor = theme.colors.actionDefaultBorderFocus;
    const capitalizedColor = color.charAt(0).toUpperCase() + color.slice(1);
    textColor = theme.DU_BOIS_INTERNAL_ONLY.colors[`tagText${capitalizedColor}`];
    backgroundColor = theme.DU_BOIS_INTERNAL_ONLY.colors[`tagBackground${capitalizedColor}`];
    iconColor = theme.DU_BOIS_INTERNAL_ONLY.colors[`tagIcon${capitalizedColor}`];
    if (color === 'charcoal') {
        outlineColor = theme.colors.white;
    }
    const iconHover = theme.colors.tagIconHover;
    const iconPress = theme.colors.tagIconPress;
    return {
        wrapper: {
            backgroundColor: backgroundColor,
            display: 'inline-flex',
            alignItems: 'center',
            marginRight: theme.spacing.sm,
            borderRadius: theme.borders.borderRadiusSm
        },
        tag: {
            border: 'none',
            color: textColor,
            padding: '',
            backgroundColor: 'transparent',
            borderRadius: theme.borders.borderRadiusSm,
            marginRight: theme.spacing.sm,
            display: 'inline-block',
            cursor: clickable ? 'pointer' : undefined,
            ...closable && {
                borderTopRightRadius: 0,
                borderBottomRightRadius: 0
            },
            ...clickable && {
                '&:hover': {
                    '& > div': {
                        backgroundColor: theme.colors.actionDefaultBackgroundHover
                    }
                },
                '&:active': {
                    '& > div': {
                        backgroundColor: theme.colors.actionDefaultBackgroundPress
                    }
                }
            }
        },
        content: {
            display: 'flex',
            alignItems: 'center',
            minWidth: 0,
            height: theme.typography.lineHeightBase
        },
        close: {
            height: theme.typography.lineHeightBase,
            width: theme.typography.lineHeightBase,
            lineHeight: `${theme.general.iconFontSize}px`,
            padding: 0,
            color: textColor,
            fontSize: theme.general.iconFontSize,
            borderTopRightRadius: theme.borders.borderRadiusSm,
            borderBottomRightRadius: theme.borders.borderRadiusSm,
            border: 'none',
            background: 'none',
            cursor: 'pointer',
            marginLeft: theme.spacing.xs,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: 0,
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                color: iconHover
            },
            '&:active': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                color: iconPress
            },
            '&:focus-visible': {
                outlineStyle: 'solid',
                outlineWidth: 1,
                outlineOffset: -2,
                outlineColor
            },
            '.anticon': {
                verticalAlign: 0,
                fontSize: 12
            }
        },
        text: {
            padding: 0,
            fontSize: theme.typography.fontSizeBase,
            fontWeight: theme.typography.typographyRegularFontWeight,
            lineHeight: theme.typography.lineHeightSm,
            '& .anticon': {
                verticalAlign: 'text-top'
            },
            whiteSpace: 'nowrap'
        },
        icon: {
            color: iconColor,
            paddingLeft: theme.spacing.xs,
            height: theme.typography.lineHeightBase,
            display: 'inline-flex',
            alignItems: 'center',
            borderTopLeftRadius: theme.borders.borderRadiusSm,
            borderBottomLeftRadius: theme.borders.borderRadiusSm,
            '& > span': {
                fontSize: 12
            },
            '& + div': {
                borderTopLeftRadius: 0,
                borderBottomLeftRadius: 0,
                ...closable && {
                    borderTopRightRadius: 0,
                    borderBottomRightRadius: 0
                }
            }
        },
        childrenWrapper: {
            paddingLeft: theme.spacing.xs,
            paddingRight: theme.spacing.xs,
            height: theme.typography.lineHeightBase,
            display: 'inline-flex',
            alignItems: 'center',
            borderRadius: theme.borders.borderRadiusSm,
            minWidth: 0
        }
    };
}
const Tag = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    const { theme } = useDesignSystemTheme();
    const { color, children, closable, onClose, role = 'status', closeButtonProps, analyticsEvents, componentId, icon, onClick, ...attributes } = props;
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.tag', false);
    const isClickable = Boolean(props.onClick);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnClick,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnClick
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Tag,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents
    });
    const { elementRef } = useNotifyOnFirstView({
        onView: eventContext.onView
    });
    const mergedRef = useMergeRefs([
        elementRef,
        forwardedRef
    ]);
    const closeButtonComponentId = componentId ? `${componentId}.close` : undefined;
    const closeButtonEventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentId: closeButtonComponentId,
        analyticsEvents: [
            DesignSystemEventProviderAnalyticsEventTypes.OnClick
        ]
    });
    const handleClick = useCallback((e)=>{
        if (onClick) {
            eventContext.onClick(e);
            onClick(e);
        }
    }, [
        eventContext,
        onClick
    ]);
    const handleCloseClick = useCallback((e)=>{
        closeButtonEventContext.onClick(e);
        e.stopPropagation();
        if (onClose) {
            onClose();
        }
    }, [
        closeButtonEventContext,
        onClose
    ]);
    const styles = getTagEmotionStyles(theme, color, isClickable, closable);
    return /*#__PURE__*/ jsxs("div", {
        ref: mergedRef,
        role: role,
        onClick: handleClick,
        css: [
            styles.wrapper
        ],
        ...attributes,
        ...addDebugOutlineIfEnabled(),
        ...eventContext.dataComponentProps,
        // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
        tabIndex: isClickable ? 0 : -1,
        children: [
            /*#__PURE__*/ jsxs("div", {
                css: [
                    styles.tag,
                    styles.content,
                    styles.text,
                    {
                        marginRight: 0
                    }
                ],
                ...eventContext.dataComponentProps,
                children: [
                    icon && /*#__PURE__*/ jsx("div", {
                        css: [
                            styles.icon
                        ],
                        children: icon
                    }),
                    /*#__PURE__*/ jsx("div", {
                        css: [
                            styles.childrenWrapper
                        ],
                        children: children
                    })
                ]
            }),
            closable && /*#__PURE__*/ jsx("button", {
                css: styles.close,
                tabIndex: 0,
                onClick: handleCloseClick,
                onMouseDown: (e)=>{
                    // Keeps dropdowns of any underlying select from opening.
                    e.stopPropagation();
                },
                ...closeButtonProps,
                ...closeButtonEventContext.dataComponentProps,
                children: /*#__PURE__*/ jsx(CloseIcon, {
                    css: {
                        fontSize: theme.general.iconFontSize - 4
                    }
                })
            })
        ]
    });
});

const Overflow = ({ children, noMargin = false, visibleItemsCount = 1, ...props })=>{
    const { theme } = useDesignSystemTheme();
    const childrenList = children && Children.toArray(children);
    if (!childrenList || childrenList.length === 0) {
        return /*#__PURE__*/ jsx(Fragment, {
            children: children
        });
    }
    const visibleItems = childrenList.slice(0, visibleItemsCount);
    const additionalItems = childrenList.slice(visibleItemsCount);
    const renderOverflowLabel = (label)=>/*#__PURE__*/ jsx(Tag, {
            componentId: "codegen_design-system_src_design-system_overflow_overflow.tsx_28",
            css: getTagStyles(theme),
            children: label
        });
    return additionalItems.length === 0 ? /*#__PURE__*/ jsx(Fragment, {
        children: visibleItems
    }) : /*#__PURE__*/ jsxs("div", {
        ...props,
        css: {
            display: 'inline-flex',
            alignItems: 'center',
            gap: noMargin ? 0 : theme.spacing.sm,
            maxWidth: '100%'
        },
        children: [
            visibleItems,
            additionalItems.length > 0 && /*#__PURE__*/ jsx(OverflowPopover, {
                items: additionalItems,
                renderLabel: renderOverflowLabel,
                ...props
            })
        ]
    });
};
const getTagStyles = (theme)=>{
    const styles = {
        marginRight: 0,
        color: theme.colors.actionTertiaryTextDefault,
        cursor: 'pointer',
        '&:focus': {
            color: theme.colors.actionTertiaryTextDefault
        },
        '&:hover': {
            color: theme.colors.actionTertiaryTextHover
        },
        '&:active': {
            color: theme.colors.actionTertiaryTextPress
        }
    };
    return /*#__PURE__*/ css(styles);
};

const PageWrapper = ({ children, ...props })=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        ...addDebugOutlineIfEnabled(),
        css: /*#__PURE__*/ css({
            paddingLeft: 16,
            paddingRight: 16,
            backgroundColor: theme.isDarkMode ? theme.colors.backgroundPrimary : 'transparent'
        }),
        ...props,
        children: children
    });
};

const PreviewCard = ({ icon, title, subtitle, titleActions, children, startActions, endActions, image, fullBleedImage = true, onClick, size = 'default', dangerouslyAppendEmotionCSS, componentId, analyticsEvents = [
    DesignSystemEventProviderAnalyticsEventTypes.OnClick
], disabled, selected, href, target, ...props })=>{
    const styles = usePreviewCardStyles({
        onClick,
        size,
        disabled,
        fullBleedImage,
        href
    });
    const tabIndex = onClick && !href ? 0 : undefined;
    const role = onClick && !href ? 'button' : undefined;
    const showFooter = startActions || endActions;
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents, [
        analyticsEvents
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.PreviewCard,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents
    });
    const { elementRef: previewCardRef } = useNotifyOnFirstView({
        onView: eventContext.onView
    });
    const onClickWrapper = useCallback((e)=>{
        if (onClick) {
            eventContext.onClick(e);
            onClick(e);
        }
    }, [
        eventContext,
        onClick
    ]);
    const onHrefClickWrapper = useCallback((e)=>{
        eventContext.onClick(e);
        if (onClick) {
            onClick(e);
        }
    }, [
        eventContext,
        onClick
    ]);
    const content = /*#__PURE__*/ jsxs("div", {
        ...addDebugOutlineIfEnabled(),
        css: [
            styles['container'],
            dangerouslyAppendEmotionCSS
        ],
        tabIndex: disabled ? -1 : tabIndex,
        onClick: onClickWrapper,
        onKeyDown: (e)=>{
            if (!onClick || disabled) {
                return;
            }
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClickWrapper(e);
            }
        },
        role: role,
        "aria-disabled": disabled,
        "aria-pressed": selected,
        ...props,
        ref: previewCardRef,
        children: [
            image && /*#__PURE__*/ jsx("div", {
                css: styles['image'],
                children: image
            }),
            /*#__PURE__*/ jsxs("div", {
                css: styles['header'],
                children: [
                    icon && /*#__PURE__*/ jsx("div", {
                        children: icon
                    }),
                    /*#__PURE__*/ jsxs("div", {
                        css: styles['titleWrapper'],
                        children: [
                            title && /*#__PURE__*/ jsx("div", {
                                css: styles['title'],
                                children: title
                            }),
                            subtitle && /*#__PURE__*/ jsx("div", {
                                css: styles['subTitle'],
                                children: subtitle
                            })
                        ]
                    }),
                    titleActions && /*#__PURE__*/ jsx("div", {
                        children: titleActions
                    })
                ]
            }),
            children && /*#__PURE__*/ jsx("div", {
                css: styles['childrenWrapper'],
                children: children
            }),
            showFooter && /*#__PURE__*/ jsxs("div", {
                css: styles['footer'],
                children: [
                    /*#__PURE__*/ jsx("div", {
                        css: styles['action'],
                        children: startActions
                    }),
                    /*#__PURE__*/ jsx("div", {
                        css: styles['action'],
                        children: endActions
                    })
                ]
            })
        ]
    });
    if (href) {
        return /*#__PURE__*/ jsx("a", {
            href: href,
            target: target,
            style: {
                textDecoration: 'none'
            },
            onClick: onHrefClickWrapper,
            children: content
        });
    }
    return content;
};
const usePreviewCardStyles = ({ onClick, size, disabled, fullBleedImage, href })=>{
    const { theme } = useDesignSystemTheme();
    const isInteractive = href !== undefined || onClick !== undefined;
    const paddingSize = size === 'large' ? theme.spacing.lg : theme.spacing.md;
    return {
        container: {
            overflow: 'hidden',
            borderRadius: theme.borders.borderRadiusMd,
            border: `1px solid ${theme.colors.border}`,
            padding: paddingSize,
            color: theme.colors.textSecondary,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            gap: size === 'large' ? theme.spacing.md : theme.spacing.sm,
            boxShadow: theme.shadows.sm,
            cursor: isInteractive ? 'pointer' : 'default',
            ...isInteractive && {
                transition: 'box-shadow 0.2s, background-color 0.2s, border-color 0.2s, color 0.2s',
                '&[aria-disabled="true"]': {
                    pointerEvents: 'none',
                    backgroundColor: theme.colors.actionDisabledBackground,
                    borderColor: theme.colors.actionDisabledBorder,
                    color: theme.colors.actionDisabledText
                },
                '&:hover, &:focus-within': {
                    boxShadow: theme.shadows.md
                },
                '&:active': {
                    background: theme.colors.actionTertiaryBackgroundPress,
                    borderColor: theme.colors.actionDefaultBorderHover,
                    boxShadow: theme.shadows.md
                },
                '&:focus, &[aria-pressed="true"]': {
                    outlineColor: theme.colors.actionDefaultBorderFocus,
                    outlineWidth: 2,
                    outlineOffset: -2,
                    outlineStyle: 'solid',
                    boxShadow: theme.shadows.md,
                    borderColor: theme.colors.actionDefaultBorderHover
                },
                '&:active:not(:focus):not(:focus-within)': {
                    background: 'transparent',
                    borderColor: theme.colors.border
                }
            }
        },
        image: {
            margin: fullBleedImage ? `-${paddingSize}px -${paddingSize}px 0` : 0,
            '& > *': {
                borderRadius: fullBleedImage ? 0 : theme.borders.borderRadiusSm
            }
        },
        header: {
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm
        },
        title: {
            fontWeight: theme.typography.typographyBoldFontWeight,
            color: disabled ? theme.colors.actionDisabledText : theme.colors.textPrimary,
            lineHeight: theme.typography.lineHeightSm
        },
        subTitle: {
            lineHeight: theme.typography.lineHeightSm
        },
        titleWrapper: {
            flexGrow: 1,
            overflow: 'hidden'
        },
        childrenWrapper: {
            flexGrow: 1
        },
        footer: {
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap'
        },
        action: {
            overflow: 'hidden',
            // to ensure focus ring is rendered
            margin: theme.spacing.md * -1,
            padding: theme.spacing.md
        }
    };
};

const getRadioTileStyles = (theme, classNamePrefix, maxWidth)=>{
    const radioWrapper = `.${classNamePrefix}-radio-wrapper`;
    return /*#__PURE__*/ css({
        '&&': {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
            padding: theme.spacing.md,
            gap: theme.spacing.xs,
            borderRadius: theme.borders.borderRadiusSm,
            background: 'transparent',
            cursor: 'pointer',
            ...maxWidth && {
                maxWidth
            },
            // Label, radio and icon container
            '& > div:first-of-type': {
                width: '100%',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: theme.spacing.sm
            },
            // Description container
            '& > div:nth-of-type(2)': {
                alignSelf: 'flex-start',
                textAlign: 'left',
                color: theme.colors.textSecondary,
                fontSize: theme.typography.fontSizeSm
            },
            '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                borderColor: theme.colors.actionDefaultBorderHover
            },
            '&:disabled': {
                borderColor: theme.colors.actionDisabledBorder,
                backgroundColor: 'transparent',
                cursor: 'not-allowed',
                '& > div:nth-of-type(2)': {
                    color: theme.colors.actionDisabledText
                }
            }
        },
        [radioWrapper]: {
            display: 'flex',
            flexDirection: 'row-reverse',
            justifyContent: 'space-between',
            flex: 1,
            margin: 0,
            '& > span': {
                padding: 0
            },
            '::after': {
                display: 'none'
            }
        }
    });
};
const RadioTile = (props)=>{
    const { description, icon, maxWidth, checked, defaultChecked, onChange, ...rest } = props;
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const { value: groupValue, onChange: groupOnChange } = useRadioGroupContext();
    return /*#__PURE__*/ jsxs("button", {
        role: "radio",
        type: "button",
        "aria-checked": groupValue === props.value,
        onClick: ()=>{
            if (props.disabled) {
                return;
            }
            onChange?.(props.value);
            groupOnChange?.({
                target: {
                    value: props.value
                }
            });
        },
        tabIndex: 0,
        className: `${classNamePrefix}-radio-tile`,
        css: getRadioTileStyles(theme, classNamePrefix, maxWidth),
        disabled: props.disabled,
        children: [
            /*#__PURE__*/ jsxs("div", {
                children: [
                    icon ? /*#__PURE__*/ jsx("span", {
                        css: {
                            color: props.disabled ? theme.colors.actionDisabledText : theme.colors.textSecondary
                        },
                        children: icon
                    }) : null,
                    /*#__PURE__*/ jsx(Radio, {
                        __INTERNAL_DISABLE_RADIO_ROLE: true,
                        ...rest,
                        tabIndex: -1
                    })
                ]
            }),
            description ? /*#__PURE__*/ jsx("div", {
                children: description
            }) : null
        ]
    });
};

const STATUS_TO_ICON = {
    online: ({ theme, style, ...props })=>/*#__PURE__*/ jsx(CircleIcon, {
            color: "success",
            css: {
                ...style
            },
            ...props
        }),
    disconnected: ({ theme, style, ...props })=>/*#__PURE__*/ jsx(CircleOutlineIcon, {
            css: {
                color: theme.colors.grey500,
                ...style
            },
            ...props
        }),
    offline: ({ theme, style, ...props })=>/*#__PURE__*/ jsx(CircleOffIcon, {
            css: {
                color: theme.colors.grey500,
                ...style
            },
            ...props
        })
};
const ResourceStatusIndicator = (props)=>{
    const { status, style, ...restProps } = props;
    const { theme } = useDesignSystemTheme();
    const StatusIcon = STATUS_TO_ICON[status];
    return /*#__PURE__*/ jsx(StatusIcon, {
        theme: theme,
        style: style,
        ...restProps
    });
};

// TODO(GP): Add this to common spacing vars; I didn't want to make a decision on the value right now,
// so copied it from `Button`.
const SMALL_BUTTON_HEIGHT$1 = 24;
function getSegmentedControlGroupEmotionStyles(clsPrefix, theme, spaced = false, truncateButtons, useSegmentedSliderStyle = false) {
    const classGroup = `.${clsPrefix}-radio-group`;
    const classSmallGroup = `.${clsPrefix}-radio-group-small`;
    const classButtonWrapper = `.${clsPrefix}-radio-button-wrapper`;
    const styles = {
        ...truncateButtons && {
            display: 'flex',
            maxWidth: '100%'
        },
        [`&${classGroup}`]: spaced ? {
            display: 'flex',
            gap: 8,
            flexWrap: 'wrap'
        } : {},
        [`&${classSmallGroup} ${classButtonWrapper}`]: {
            padding: '0 12px'
        }
    };
    const sliderStyles = {
        height: 'min-content',
        width: 'min-content',
        background: theme.colors.backgroundSecondary,
        borderRadius: theme.borders.borderRadiusSm,
        display: 'flex',
        gap: theme.spacing.xs,
        ...truncateButtons && {
            maxWidth: '100%',
            overflow: 'auto'
        }
    };
    const importantStyles = importantify(useSegmentedSliderStyle && !spaced ? sliderStyles : styles);
    return /*#__PURE__*/ css(importantStyles);
}
function getSegmentedControlButtonEmotionStyles(clsPrefix, theme, size, spaced = false, truncateButtons, onlyIcon, useSegmentedSliderStyle = false) {
    const classWrapperChecked = `.${clsPrefix}-radio-button-wrapper-checked`;
    const classWrapper = `.${clsPrefix}-radio-button-wrapper`;
    const classWrapperDisabled = `.${clsPrefix}-radio-button-wrapper-disabled`;
    const classButton = `.${clsPrefix}-radio-button`;
    // Note: Ant radio button uses a 1px-wide `before` pseudo-element to recreate the left border of the button.
    // This is because the actual left border is disabled to avoid a double-border effect with the adjacent button's
    // right border.
    // We must override the background colour of this pseudo-border to be the same as the real border above.
    const styles = {
        backgroundColor: theme.colors.actionDefaultBackgroundDefault,
        borderColor: theme.colors.actionDefaultBorderDefault,
        color: theme.colors.actionDefaultTextDefault,
        boxShadow: theme.shadows.xs,
        // This handles the left border of the button when they're adjacent
        '::before': {
            display: spaced ? 'none' : 'block',
            backgroundColor: theme.colors.actionDefaultBorderDefault
        },
        '&:hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionDefaultBorderHover,
            color: theme.colors.actionDefaultTextHover,
            '::before': {
                backgroundColor: theme.colors.actionDefaultBorderHover
            },
            // Also target the same pseudo-element on the next sibling, because this is used to create the right border
            [`& + ${classWrapper}::before`]: {
                backgroundColor: theme.colors.actionDefaultBorderPress
            }
        },
        '&:active': {
            backgroundColor: theme.colors.actionTertiaryBackgroundPress,
            borderColor: theme.colors.actionDefaultBorderPress,
            color: theme.colors.actionTertiaryTextPress
        },
        [`&${classWrapperChecked}`]: {
            backgroundColor: theme.colors.actionTertiaryBackgroundPress,
            borderColor: theme.colors.actionDefaultBorderPress,
            color: theme.colors.actionTertiaryTextPress,
            '::before': {
                backgroundColor: theme.colors.actionDefaultBorderPress
            },
            [`& + ${classWrapper}::before`]: {
                backgroundColor: theme.colors.actionDefaultBorderPress
            }
        },
        [`&${classWrapperChecked}:focus-within`]: {
            '::before': {
                width: 0
            }
        },
        [`&${classWrapper}`]: {
            padding: size === 'middle' ? `0 16px` : '0 8px',
            display: 'inline-flex',
            ...onlyIcon ? {
                '& > span': {
                    display: 'inline-flex'
                }
            } : {},
            verticalAlign: 'middle',
            ...truncateButtons && {
                flexShrink: 1,
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
            },
            '&:first-of-type': {
                borderTopLeftRadius: theme.borders.borderRadiusSm,
                borderBottomLeftRadius: theme.borders.borderRadiusSm
            },
            '&:last-of-type': {
                borderTopRightRadius: theme.borders.borderRadiusSm,
                borderBottomRightRadius: theme.borders.borderRadiusSm
            },
            ...spaced ? {
                borderWidth: 1,
                borderRadius: theme.borders.borderRadiusSm
            } : {},
            '&:focus-within': {
                outlineStyle: 'solid',
                outlineWidth: '2px',
                outlineOffset: '-2px',
                outlineColor: theme.colors.actionDefaultBorderFocus
            },
            ...truncateButtons && {
                'span:last-of-type': {
                    textOverflow: 'ellipsis',
                    overflow: 'hidden',
                    whiteSpace: 'nowrap'
                }
            }
        },
        [`&${classWrapper}, ${classButton}`]: {
            height: size === 'middle' ? theme.general.heightSm : SMALL_BUTTON_HEIGHT$1,
            lineHeight: theme.typography.lineHeightBase,
            alignItems: 'center'
        },
        [`&${classWrapperDisabled}, &${classWrapperDisabled} + ${classWrapperDisabled}`]: {
            color: theme.colors.actionDisabledText,
            backgroundColor: 'transparent',
            borderColor: theme.colors.actionDisabledBorder,
            '&:hover': {
                color: theme.colors.actionDisabledText,
                borderColor: theme.colors.actionDisabledBorder,
                backgroundColor: 'transparent'
            },
            '&:active': {
                color: theme.colors.actionDisabledText,
                borderColor: theme.colors.actionDisabledBorder,
                backgroundColor: 'transparent'
            },
            '::before': {
                backgroundColor: theme.colors.actionDisabledBorder
            },
            [`&${classWrapperChecked}`]: {
                borderColor: theme.colors.actionDefaultBorderPress,
                '::before': {
                    backgroundColor: theme.colors.actionDefaultBorderPress
                }
            },
            [`&${classWrapperChecked} + ${classWrapper}`]: {
                '::before': {
                    backgroundColor: theme.colors.actionDefaultBorderPress
                }
            }
        },
        ...getAnimationCss(theme.options.enableAnimation)
    };
    const sliderStyles = {
        minWidth: 'fit-content',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        border: 'none',
        display: 'inline-flex',
        alignItems: 'center',
        color: theme.colors.textSecondary,
        borderRadius: theme.borders.borderRadiusSm,
        backgroundColor: 'transparent',
        '::before': {
            display: 'none'
        },
        '&:hover': {
            backgroundColor: theme.colors.tableRowHover
        },
        [`&${classWrapper}`]: {
            padding: size === 'middle' ? onlyIcon ? `0 ${theme.spacing.sm}px` : `0 12px` : `0 ${theme.spacing.sm}px`,
            verticalAlign: 'middle',
            ...truncateButtons && {
                flexShrink: 1,
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                '& > span:last-of-type': {
                    display: 'inline-block',
                    textOverflow: 'ellipsis',
                    overflow: 'hidden',
                    whiteSpace: 'nowrap',
                    '& > *': {
                        display: 'flex',
                        alignItems: 'center'
                    }
                }
            },
            '& > span': {
                display: 'inline-flex'
            }
        },
        [`&${classWrapper}, ${classButton}`]: {
            height: size === 'middle' ? theme.general.heightSm : SMALL_BUTTON_HEIGHT$1,
            lineHeight: theme.typography.lineHeightBase,
            alignItems: 'center'
        },
        [`&${classWrapperChecked}`]: {
            color: theme.colors.actionDefaultTextDefault,
            backgroundColor: theme.colors.backgroundPrimary,
            boxShadow: `inset 0 0 0 1px ${theme.colors.actionDefaultBorderDefault}, ${theme.shadows.sm}`
        },
        [`&${classWrapperDisabled}, &${classWrapperDisabled} + ${classWrapperDisabled}`]: {
            color: theme.colors.actionDisabledText,
            backgroundColor: 'transparent'
        },
        ...getAnimationCss(theme.options.enableAnimation)
    };
    const importantStyles = importantify(useSegmentedSliderStyle && !spaced ? sliderStyles : styles);
    return /*#__PURE__*/ css(importantStyles);
}
const SegmentedControlGroupContext = /*#__PURE__*/ createContext({
    size: 'middle',
    spaced: false,
    useSegmentedSliderStyle: false
});
const SegmentedControlButton = /*#__PURE__*/ forwardRef(function SegmentedControlButton({ dangerouslySetAntdProps, ...props }, ref) {
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const truncateButtons = safex('databricks.fe.designsystem.truncateSegmentedControlText', false);
    const { size, spaced, useSegmentedSliderStyle } = useContext(SegmentedControlGroupContext);
    const buttonRef = useRef(null);
    useImperativeHandle(ref, ()=>buttonRef.current);
    const onlyIcon = Boolean(props.icon && !props.children);
    const getLabelFromChildren = useCallback(()=>{
        let label = '';
        React__default.Children.map(props.children, (child)=>{
            if (typeof child === 'string') {
                label += child;
            }
        });
        return label;
    }, [
        props.children
    ]);
    useEffect(()=>{
        if (buttonRef.current) {
            // Using `as any` because Antd uses a `Checkbox` type that's not exported
            const labelParent = buttonRef.current.input.closest('label');
            if (labelParent) {
                labelParent.setAttribute('title', getLabelFromChildren());
                if (truncateButtons) {
                    const labelWidth = labelParent.scrollWidth;
                    const threshold = size === 'small' ? 58 : 68;
                    if (labelWidth > threshold) {
                        labelParent.style.setProperty('min-width', `${threshold}px`, 'important');
                    }
                }
            }
        }
    }, [
        buttonRef,
        getLabelFromChildren,
        size,
        truncateButtons
    ]);
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Radio$1.Button, {
            css: getSegmentedControlButtonEmotionStyles(classNamePrefix, theme, size, spaced, truncateButtons, onlyIcon, useSegmentedSliderStyle),
            ...props,
            ...dangerouslySetAntdProps,
            ref: buttonRef,
            children: props.icon ? /*#__PURE__*/ jsxs("div", {
                css: {
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: theme.spacing.xs
                },
                children: [
                    props.icon,
                    props.children && truncateButtons ? /*#__PURE__*/ jsx("span", {
                        css: {
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap'
                        },
                        children: props.children
                    }) : props.children
                ]
            }) : props.children
        })
    });
});
const SegmentedControlGroup = /*#__PURE__*/ forwardRef(function SegmentedControlGroup({ dangerouslySetAntdProps, size = 'middle', spaced = false, onChange, componentId, analyticsEvents, valueHasNoPii, newStyleFlagOverride, ...props }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.segmentedControlGroup', false);
    const { classNamePrefix, theme } = useDesignSystemTheme();
    const truncateButtons = safex('databricks.fe.designsystem.truncateSegmentedControlText', false);
    const useSegmentedSliderStyle = newStyleFlagOverride ?? safex('databricks.fe.designsystem.useNewSegmentedControlStyles', false);
    const uniqueId = useUniqueId();
    const internalRef = useRef();
    useImperativeHandle(ref, ()=>internalRef.current);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.SegmentedControlGroup,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii
    });
    const { elementRef: segmentedControlGroupRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: props.value ?? props.defaultValue
    });
    const mergedRef = useMergeRefs([
        internalRef,
        segmentedControlGroupRef
    ]);
    const onChangeWrapper = useCallback((e)=>{
        eventContext.onValueChange(e.target.value);
        onChange?.(e);
    }, [
        eventContext,
        onChange
    ]);
    const ariaLabelledby = props['aria-labelledby'];
    // A11y helper (FIT-1649): Effect helping to set the aria-labelledby attribute on the radio group if it is not provided
    useEffect(()=>{
        if (ariaLabelledby) {
            return;
        }
        if (props.id) {
            // look for the label of the group pointing to the id
            const label = document.querySelector(`label[for="${props.id}"]`);
            if (label) {
                // If the label already has an id, map the radio group to it
                if (label.hasAttribute('id')) {
                    internalRef.current?.setAttribute('aria-labelledby', label.getAttribute('id') ?? '');
                } else {
                    label.setAttribute('id', `${props.id}${uniqueId}-label`);
                    internalRef.current?.setAttribute('aria-labelledby', `${props.id}${uniqueId}-label`);
                }
            }
        }
    }, [
        ariaLabelledby,
        props.id,
        uniqueId
    ]);
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(SegmentedControlGroupContext.Provider, {
            value: {
                size,
                spaced,
                useSegmentedSliderStyle
            },
            children: /*#__PURE__*/ jsx(Radio$1.Group, {
                ...addDebugOutlineIfEnabled(),
                ...props,
                css: getSegmentedControlGroupEmotionStyles(classNamePrefix, theme, spaced, truncateButtons, useSegmentedSliderStyle),
                // @ts-expect-error - role is not a valid prop for RadioGroup in Antd but it applies it correctly to the group
                role: "radiogroup",
                onChange: onChangeWrapper,
                ...dangerouslySetAntdProps,
                ref: mergedRef,
                ...eventContext.dataComponentProps
            })
        })
    });
});

const getRootStyles = ()=>{
    return /*#__PURE__*/ css({
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        '&[data-orientation="vertical"]': {
            flexDirection: 'column',
            width: 20,
            height: 100
        },
        '&[data-orientation="horizontal"]': {
            height: 20,
            width: 200
        }
    });
};
const Root$1 = /*#__PURE__*/ forwardRef((props, ref)=>{
    return /*#__PURE__*/ jsx(RadixSlider.Root, {
        ...addDebugOutlineIfEnabled(),
        css: getRootStyles(),
        ...props,
        ref: ref
    });
});
const getTrackStyles = (theme)=>{
    return /*#__PURE__*/ css({
        backgroundColor: theme.colors.grey100,
        position: 'relative',
        flexGrow: 1,
        borderRadius: theme.borders.borderRadiusFull,
        '&[data-orientation="vertical"]': {
            width: 3
        },
        '&[data-orientation="horizontal"]': {
            height: 3
        }
    });
};
const Track = /*#__PURE__*/ forwardRef((props, ref)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(RadixSlider.Track, {
        css: getTrackStyles(theme),
        ...props,
        ref: ref
    });
});
const getRangeStyles = (theme)=>{
    return /*#__PURE__*/ css({
        backgroundColor: theme.colors.primary,
        position: 'absolute',
        borderRadius: theme.borders.borderRadiusFull,
        height: '100%',
        '&[data-disabled]': {
            backgroundColor: theme.colors.grey100
        }
    });
};
const Range = /*#__PURE__*/ forwardRef((props, ref)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(RadixSlider.Range, {
        css: getRangeStyles(theme),
        ...props,
        ref: ref
    });
});
const getThumbStyles = (theme)=>{
    return /*#__PURE__*/ css({
        display: 'block',
        width: 20,
        height: 20,
        backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
        boxShadow: theme.shadows.xs,
        borderRadius: theme.borders.borderRadiusFull,
        outline: 'none',
        '&:hover': {
            backgroundColor: theme.colors.actionPrimaryBackgroundHover
        },
        '&:focus': {
            backgroundColor: theme.colors.actionPrimaryBackgroundPress
        },
        '&[data-disabled]': {
            backgroundColor: theme.colors.grey200,
            boxShadow: 'none'
        }
    });
};
const Thumb = /*#__PURE__*/ forwardRef((props, ref)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(RadixSlider.Thumb, {
        css: getThumbStyles(theme),
        "aria-label": "Slider thumb",
        ...props,
        ref: ref
    });
});

var Slider = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Range: Range,
  Root: Root$1,
  Thumb: Thumb,
  Track: Track
});

const ButtonGroup = Button$1.Group;
const DropdownButton = (props)=>{
    const { theme } = useDesignSystemTheme();
    const { getPopupContainer: getContextPopupContainer, getPrefixCls } = useDesignSystemContext();
    const useNewBorderRadii = safex('databricks.fe.designsystem.useNewBorderRadii', false);
    const { type, danger, disabled, loading, onClick, htmlType, children, className, overlay, trigger, align, open, onOpenChange, placement, getPopupContainer, href, icon = /*#__PURE__*/ jsx(AntDIcon, {}), title, buttonsRender = (buttons)=>buttons, mouseEnterDelay, mouseLeaveDelay, overlayClassName, overlayStyle, destroyPopupOnHide, menuButtonLabel = 'Open dropdown', menu, leftButtonIcon, dropdownMenuRootProps, 'aria-label': ariaLabel, componentId, analyticsEvents, form, ...restProps } = props;
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
        destroyPopupOnHide
    };
    if ('open' in props) {
        dropdownProps.open = open;
    }
    if ('placement' in props) {
        dropdownProps.placement = placement;
    } else {
        dropdownProps.placement = 'bottomRight';
    }
    const leftButton = /*#__PURE__*/ jsxs(Button, {
        componentId: componentId ? `${componentId}.primary_button` : 'codegen_design-system_src_design-system_splitbutton_dropdown_dropdownbutton.tsx_148',
        type: type,
        form: form,
        danger: danger,
        disabled: disabled,
        loading: loading,
        onClick: onClick,
        htmlType: htmlType,
        href: href,
        title: title,
        icon: children && leftButtonIcon ? leftButtonIcon : undefined,
        "aria-label": ariaLabel,
        size: props.size,
        css: {
            ...useNewBorderRadii ? {
                borderTopRightRadius: '0 !important',
                borderBottomRightRadius: '0 !important'
            } : {}
        },
        children: [
            leftButtonIcon && !children ? leftButtonIcon : undefined,
            children
        ]
    });
    const rightButton = /*#__PURE__*/ jsx(Button, {
        componentId: componentId ? `${componentId}.dropdown_button` : 'codegen_design-system_src_design-system_splitbutton_dropdown_dropdownbutton.tsx_166',
        type: type,
        danger: danger,
        disabled: disabled,
        "aria-label": menuButtonLabel,
        size: props.size,
        css: {
            ...useNewBorderRadii ? {
                borderTopLeftRadius: '0 !important',
                borderBottomLeftRadius: '0 !important'
            } : {},
            ...props.size === 'small' ? {
                '&&&': {
                    paddingLeft: `${theme.spacing.xs}px !important`,
                    paddingRight: `${theme.spacing.xs}px !important`,
                    width: '24px !important'
                }
            } : {}
        },
        children: icon ? icon : /*#__PURE__*/ jsx(ChevronDownIcon, {})
    });
    const [leftButtonToRender, rightButtonToRender] = buttonsRender([
        leftButton,
        rightButton
    ]);
    return /*#__PURE__*/ jsxs(ButtonGroup, {
        ...restProps,
        className: classnames(prefixCls, className),
        children: [
            leftButtonToRender,
            overlay !== undefined ? /*#__PURE__*/ jsx(Dropdown$1, {
                ...dropdownProps,
                overlay: overlay,
                children: rightButtonToRender
            }) : /*#__PURE__*/ jsxs(Root$7, {
                ...dropdownMenuRootProps,
                itemHtmlType: htmlType === 'submit' ? 'submit' : undefined,
                children: [
                    /*#__PURE__*/ jsx(Trigger$3, {
                        disabled: disabled,
                        asChild: true,
                        children: rightButtonToRender
                    }),
                    menu && /*#__PURE__*/ React.cloneElement(menu, {
                        align: menu.props.align || 'end'
                    })
                ]
            })
        ]
    });
};

const BUTTON_HORIZONTAL_PADDING = 12;
function getSplitButtonEmotionStyles(classNamePrefix, theme, useNewBorderRadii, size) {
    const classDefault = `.${classNamePrefix}-btn`;
    const classPrimary = `.${classNamePrefix}-btn-primary`;
    const classDropdownTrigger = `.${classNamePrefix}-dropdown-trigger`;
    const classSmall = `.${classNamePrefix}-btn-group-sm`;
    const styles = {
        [classDefault]: {
            ...getDefaultStyles(theme),
            boxShadow: theme.shadows.xs,
            height: size === 'small' ? theme.general.iconSize : theme.general.heightSm,
            padding: `4px ${BUTTON_HORIZONTAL_PADDING}px`,
            '&:focus-visible': {
                outlineStyle: 'solid',
                outlineWidth: '2px',
                outlineOffset: '-2px',
                outlineColor: theme.colors.actionDefaultBorderFocus
            },
            '.anticon, &:focus-visible .anticon': {
                color: theme.colors.textSecondary
            },
            '&:hover .anticon': {
                color: theme.colors.actionDefaultIconHover
            },
            '&:active .anticon': {
                color: theme.colors.actionDefaultIconPress
            }
        },
        ...useNewBorderRadii && {
            [`${classDefault}:first-of-type`]: {
                borderTopRightRadius: '0px !important',
                borderBottomRightRadius: '0px !important'
            }
        },
        [classPrimary]: {
            ...getPrimaryStyles(theme),
            boxShadow: theme.shadows.xs,
            [`&:first-of-type`]: {
                borderRight: `1px solid ${theme.colors.actionPrimaryTextDefault}`,
                marginRight: 1
            },
            [classDropdownTrigger]: {
                borderLeft: `1px solid ${theme.colors.actionPrimaryTextDefault}`
            },
            '&:focus-visible': {
                outlineStyle: 'solid',
                outlineWidth: '1px',
                outlineOffset: '-3px',
                outlineColor: theme.colors.white
            },
            '.anticon, &:hover .anticon, &:active .anticon, &:focus-visible .anticon': {
                color: theme.colors.actionPrimaryIcon
            }
        },
        [classDropdownTrigger]: {
            // Needs to be 1px less than our standard 8px to allow for the off-by-one border handling in this component.
            padding: 3,
            borderLeftColor: 'transparent',
            width: theme.general.heightSm
        },
        [`&${classSmall}`]: {
            [classDropdownTrigger]: {
                padding: 5
            }
        },
        '&&': {
            [`[disabled], ${classPrimary}[disabled]`]: {
                ...getDisabledSplitButtonStyles(theme),
                boxShadow: 'none',
                [`&:first-of-type`]: {
                    borderRight: `1px solid ${theme.colors.actionPrimaryIcon}`,
                    marginRight: 1
                },
                [classDropdownTrigger]: {
                    borderLeft: `1px solid ${theme.colors.actionPrimaryIcon}`
                },
                '.anticon, &:hover .anticon, &:active .anticon, &:focus-visible .anticon': {
                    color: theme.colors.actionDisabledText
                }
            },
            [`${classPrimary}[disabled]`]: {
                ...getDisabledPrimarySplitButtonStyles(theme),
                '.anticon, &:hover .anticon, &:active .anticon, &:focus-visible .anticon': {
                    color: theme.colors.actionPrimaryTextDefault
                }
            }
        },
        [`${classDefault}:not(:first-of-type)`]: {
            width: theme.general.heightSm,
            padding: '3px !important',
            ...useNewBorderRadii && {
                borderTopLeftRadius: '0px !important',
                borderBottomLeftRadius: '0px !important'
            }
        },
        ...getAnimationCss(theme.options.enableAnimation)
    };
    const importantStyles = importantify(styles);
    return /*#__PURE__*/ css(importantStyles);
}
const SplitButton = (props)=>{
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const useNewBorderRadii = safex('databricks.fe.designsystem.useNewBorderRadii', false);
    const { children, icon, deprecatedMenu, type, loading, loadingButtonStyles, placement, dangerouslySetAntdProps, size, ...dropdownButtonProps } = props;
    // Size of button when loading only icon is shown
    const LOADING_BUTTON_SIZE = theme.general.iconFontSize + 2 * BUTTON_HORIZONTAL_PADDING + 2 * theme.general.borderWidth;
    const [width, setWidth] = useState(LOADING_BUTTON_SIZE);
    // Set the width to the button's width in regular state to later use when in loading state
    // We do this to have just a loading icon in loading state at the normal width to avoid flicker and width changes in page
    const ref = useCallback((node)=>{
        // Skip getBoundingClientRect if the consumer does not intend to use the loading state. Getting the bounding box
        // causes a layout which negatively impacts rendering performance.
        if (loading === undefined) {
            return;
        }
        if (node && !loading) {
            setWidth(node.getBoundingClientRect().width);
        }
    }, [
        loading
    ]);
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx("div", {
            ref: ref,
            css: {
                display: 'inline-flex',
                position: 'relative',
                verticalAlign: 'middle'
            },
            children: loading ? /*#__PURE__*/ jsx(Button, {
                componentId: "codegen_design-system_src_design-system_splitbutton_splitbutton.tsx_163",
                type: type === 'default' ? undefined : type,
                style: {
                    width: width,
                    fontSize: theme.general.iconFontSize,
                    ...loadingButtonStyles
                },
                loading: true,
                htmlType: props.htmlType,
                title: props.title,
                className: props.className,
                size: props.size,
                children: children
            }) : /*#__PURE__*/ jsx(DropdownButton, {
                ...dropdownButtonProps,
                size: props.size,
                overlay: deprecatedMenu,
                trigger: [
                    'click'
                ],
                css: getSplitButtonEmotionStyles(classNamePrefix, theme, useNewBorderRadii, size),
                icon: /*#__PURE__*/ jsx(ChevronDownIcon, {
                    css: {
                        fontSize: theme.general.iconFontSize
                    },
                    "aria-hidden": "true"
                }),
                placement: placement || 'bottomRight',
                type: type === 'default' ? undefined : type,
                leftButtonIcon: icon,
                ...dangerouslySetAntdProps,
                children: children
            })
        })
    });
};

/** @deprecated Please use the supported Stepper widget instead. See https://ui-infra.dev.databricks.com/storybook/js/packages/du-bois/index.html?path=/docs/primitives-stepper--docs */ const Steps = /* #__PURE__ */ (()=>{
    function Steps({ dangerouslySetAntdProps, ...props }) {
        return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
            children: /*#__PURE__*/ jsx(Steps$1, {
                ...addDebugOutlineIfEnabled(),
                ...props,
                ...dangerouslySetAntdProps
            })
        });
    }
    Steps.Step = Steps$1.Step;
    return Steps;
})();

const getTableFilterInputStyles = (theme, defaultWidth)=>{
    return /*#__PURE__*/ css({
        [theme.responsive.mediaQueries.sm]: {
            width: 'auto'
        },
        [theme.responsive.mediaQueries.lg]: {
            width: '30%'
        },
        [theme.responsive.mediaQueries.xxl]: {
            width: defaultWidth
        }
    });
};
const TableFilterInput = /*#__PURE__*/ forwardRef(function SearchInput({ onSubmit, showSearchButton, className, containerProps, searchButtonProps, ignoreFilterMediaSizing = false, ...inputProps }, ref) {
    const { theme } = useDesignSystemTheme();
    const DEFAULT_WIDTH = 400;
    let component = /*#__PURE__*/ jsx(Input, {
        prefix: /*#__PURE__*/ jsx(SearchIcon, {}),
        allowClear: true,
        ...inputProps,
        className: className,
        ref: ref
    });
    if (showSearchButton) {
        component = /*#__PURE__*/ jsxs(Input.Group, {
            css: {
                display: 'flex',
                width: '100%'
            },
            className: className,
            children: [
                /*#__PURE__*/ jsx(Input, {
                    allowClear: true,
                    ...inputProps,
                    ref: ref,
                    css: {
                        flex: 1
                    }
                }),
                /*#__PURE__*/ jsx(Button, {
                    componentId: inputProps.componentId ? `${inputProps.componentId}.search_submit` : 'codegen_design-system_src_design-system_tableui_tablefilterinput.tsx_65',
                    htmlType: "submit",
                    "aria-label": "Search",
                    ...searchButtonProps,
                    children: /*#__PURE__*/ jsx(SearchIcon, {})
                })
            ]
        });
    }
    return /*#__PURE__*/ jsx("div", {
        style: {
            height: theme.general.heightSm
        },
        css: ignoreFilterMediaSizing ? {} : getTableFilterInputStyles(theme, DEFAULT_WIDTH),
        ...containerProps,
        children: onSubmit ? /*#__PURE__*/ jsx("form", {
            onSubmit: (e)=>{
                e.preventDefault();
                onSubmit();
            },
            children: component
        }) : component
    });
});

const TableFilterLayout = /*#__PURE__*/ forwardRef(function TableFilterLayout({ children, style, className, actions, ...rest }, ref) {
    const { theme } = useDesignSystemTheme();
    const tableFilterLayoutStyles = {
        layout: /*#__PURE__*/ css({
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
            marginBottom: 'var(--table-filter-layout-group-margin)',
            columnGap: 'var(--table-filter-layout-group-margin)',
            rowGap: 'var(--table-filter-layout-item-gap)',
            flexWrap: 'wrap'
        }),
        filters: /*#__PURE__*/ css({
            display: 'flex',
            flexWrap: 'wrap',
            flexDirection: 'row',
            alignItems: 'center',
            gap: 'var(--table-filter-layout-item-gap)',
            marginRight: 'var(--table-filter-layout-group-margin)',
            flex: 1
        }),
        filterActions: /*#__PURE__*/ css({
            display: 'flex',
            flexWrap: 'wrap',
            gap: 'var(--table-filter-layout-item-gap)',
            alignSelf: 'flex-start'
        })
    };
    return /*#__PURE__*/ jsxs("div", {
        ...rest,
        ref: ref,
        style: {
            ['--table-filter-layout-item-gap']: `${theme.spacing.sm}px`,
            ['--table-filter-layout-group-margin']: `${theme.spacing.md}px`,
            ...style
        },
        css: tableFilterLayoutStyles.layout,
        className: className,
        children: [
            /*#__PURE__*/ jsx("div", {
                css: tableFilterLayoutStyles.filters,
                children: children
            }),
            actions && /*#__PURE__*/ jsx("div", {
                css: tableFilterLayoutStyles.filterActions,
                children: actions
            })
        ]
    });
});

const TableHeaderResizeHandle = /*#__PURE__*/ forwardRef(function TableHeaderResizeHandle({ style, resizeHandler, increaseWidthHandler, decreaseWidthHandler, children, ...rest }, ref) {
    const { isHeader } = useContext(TableRowContext);
    if (!isHeader) {
        throw new Error('`TableHeaderResizeHandle` must be used within a `TableRow` with `isHeader` set to true.');
    }
    const [isPopoverOpen, setIsPopoverOpen] = useState(false);
    const dragStartPosRef = useRef(null);
    const initialEventRef = useRef(null);
    const initialRenderRef = useRef(true);
    const isDragging = useRef(false);
    const MAX_DRAG_DISTANCE = 2;
    const { theme } = useDesignSystemTheme();
    const handlePointerDown = useCallback((event)=>{
        if (!increaseWidthHandler || !decreaseWidthHandler) {
            resizeHandler?.(event);
            return;
        }
        if (isPopoverOpen && !initialRenderRef.current) return;
        else initialRenderRef.current = false;
        dragStartPosRef.current = {
            x: event.clientX,
            y: event.clientY
        };
        initialEventRef.current = event;
        isDragging.current = false;
        const handlePointerMove = (event)=>{
            if (dragStartPosRef.current) {
                const dx = event.clientX - dragStartPosRef.current.x;
                if (Math.abs(dx) > MAX_DRAG_DISTANCE && initialEventRef.current) {
                    isDragging.current = true;
                    resizeHandler?.(initialEventRef.current);
                    document.removeEventListener('pointermove', handlePointerMove);
                }
            }
        };
        const handlePointerUp = ()=>{
            dragStartPosRef.current = null;
            document.removeEventListener('pointermove', handlePointerMove);
            document.removeEventListener('pointerup', handlePointerUp);
        };
        document.addEventListener('pointermove', handlePointerMove);
        document.addEventListener('pointerup', handlePointerUp);
    }, [
        isPopoverOpen,
        resizeHandler,
        increaseWidthHandler,
        decreaseWidthHandler
    ]);
    const handleClick = useCallback((event)=>{
        if (isDragging.current) {
            event.preventDefault();
            event.stopPropagation();
            isDragging.current = false;
            return;
        }
    }, []);
    const result = /*#__PURE__*/ jsx("div", {
        ...rest,
        ref: ref,
        onPointerDown: handlePointerDown,
        onClick: handleClick,
        css: tableStyles.resizeHandleContainer,
        style: style,
        role: "button",
        "aria-label": "Resize Column",
        children: /*#__PURE__*/ jsx("div", {
            css: tableStyles.resizeHandle
        })
    });
    return increaseWidthHandler && decreaseWidthHandler ? /*#__PURE__*/ jsxs(Root$8, {
        componentId: "codegen_design-system_src_design-system_tableui_tableheader.tsx_114",
        onOpenChange: setIsPopoverOpen,
        children: [
            /*#__PURE__*/ jsx(Trigger$4, {
                asChild: true,
                children: result
            }),
            /*#__PURE__*/ jsxs(Content$5, {
                side: "top",
                align: "center",
                sideOffset: 0,
                minWidth: 135,
                style: {
                    padding: `${theme.spacing.sm} ${theme.spacing.md} ${theme.spacing.md} ${theme.spacing.sm}`
                },
                children: [
                    /*#__PURE__*/ jsxs("div", {
                        style: {
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center'
                        },
                        children: [
                            /*#__PURE__*/ jsx(Typography.Title, {
                                style: {
                                    marginBottom: 0,
                                    marginTop: 0
                                },
                                children: "Resize Column"
                            }),
                            /*#__PURE__*/ jsxs("div", {
                                style: {
                                    display: 'flex',
                                    flexDirection: 'row',
                                    alignItems: 'center'
                                },
                                children: [
                                    /*#__PURE__*/ jsx(Button, {
                                        onClick: ()=>{
                                            decreaseWidthHandler();
                                        },
                                        size: "small",
                                        componentId: "design_system.adjustable_width_header.decrease_width_button",
                                        icon: /*#__PURE__*/ jsx(MinusSquareIcon, {}),
                                        style: {
                                            backgroundColor: theme.colors.actionTertiaryBackgroundHover
                                        }
                                    }),
                                    /*#__PURE__*/ jsx(Button, {
                                        onClick: ()=>{
                                            increaseWidthHandler();
                                        },
                                        size: "small",
                                        componentId: "design_system.adjustable_width_header.increase_width_button",
                                        icon: /*#__PURE__*/ jsx(PlusSquareIcon, {})
                                    })
                                ]
                            })
                        ]
                    }),
                    /*#__PURE__*/ jsx(Arrow$2, {})
                ]
            })
        ]
    }) : result;
});
const TableHeader = /*#__PURE__*/ forwardRef(function TableHeader({ children, ellipsis = false, multiline = false, sortable, sortDirection, onToggleSort, style, className, isResizing = false, align = 'left', wrapContent = true, column, header, setColumnSizing, componentId, analyticsEvents, 'aria-label': ariaLabel, ...rest }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.tableHeader', false);
    // Pulling props from the column and header props with deprecated fallbacks.
    // Doing this to avoid breaking changes + have a cleaner mechanism for testing removal of deprecated props.
    const resizable = column?.getCanResize() || rest.resizable || false;
    const resizeHandler = header?.getResizeHandler() || rest.resizeHandler;
    const supportsColumnPopover = column && header && setColumnSizing;
    const { size, grid } = useContext(TableContext);
    const { isHeader } = useContext(TableRowContext);
    const [currentSortDirection, setCurrentSortDirection] = useState(sortDirection);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.TableHeader,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true
    });
    const { elementRef: tableHeaderRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: currentSortDirection
    });
    const mergedRef = useMergeRefs([
        ref,
        tableHeaderRef
    ]);
    if (!isHeader) {
        throw new Error('`TableHeader` a must be used within a `TableRow` with `isHeader` set to true.');
    }
    let sortIcon = /*#__PURE__*/ jsx(Fragment, {});
    // While most libaries use `asc` and `desc` for the sort value, the ARIA spec
    // uses `ascending` and `descending`.
    let ariaSort;
    if (sortable) {
        if (sortDirection === 'asc') {
            sortIcon = /*#__PURE__*/ jsx(SortAscendingIcon, {});
            ariaSort = 'ascending';
        } else if (sortDirection === 'desc') {
            sortIcon = /*#__PURE__*/ jsx(SortDescendingIcon, {});
            ariaSort = 'descending';
        } else if (sortDirection === 'none') {
            sortIcon = /*#__PURE__*/ jsx(SortUnsortedIcon, {});
            ariaSort = 'none';
        }
    }
    useEffect(()=>{
        if (sortDirection !== currentSortDirection) {
            setCurrentSortDirection(sortDirection);
            eventContext.onValueChange(sortDirection);
        }
    }, [
        sortDirection,
        currentSortDirection,
        eventContext
    ]);
    const sortIconOnLeft = align === 'right';
    let typographySize = 'md';
    if (size === 'small') {
        typographySize = 'sm';
    }
    const content = wrapContent ? /*#__PURE__*/ jsx(Typography.Text, {
        className: "table-header-text",
        ellipsis: !multiline,
        size: typographySize,
        title: !multiline && typeof children === 'string' && children || undefined,
        bold: true,
        children: children
    }) : children;
    const getColumnResizeHandler = useCallback((newSize)=>()=>{
            if (column && setColumnSizing) {
                setColumnSizing((old)=>({
                        ...old,
                        [column.id]: newSize
                    }));
            }
        }, [
        column,
        setColumnSizing
    ]);
    const increaseWidthHandler = useCallback(()=>{
        if (column && setColumnSizing) {
            const currentSize = column.getSize();
            getColumnResizeHandler(currentSize + 10)();
        }
    }, [
        column,
        setColumnSizing,
        getColumnResizeHandler
    ]);
    const decreaseWidthHandler = useCallback(()=>{
        if (column && setColumnSizing) {
            const currentSize = column.getSize();
            getColumnResizeHandler(currentSize - 10)();
        }
    }, [
        column,
        setColumnSizing,
        getColumnResizeHandler
    ]);
    const renderResizeHandle = resizable && resizeHandler ? /*#__PURE__*/ jsx(TableHeaderResizeHandle, {
        style: {
            height: size === 'default' ? '20px' : '16px'
        },
        resizeHandler: resizeHandler,
        increaseWidthHandler: supportsColumnPopover ? increaseWidthHandler : undefined,
        decreaseWidthHandler: supportsColumnPopover ? decreaseWidthHandler : undefined
    }) : null;
    const isSortButtonVisible = sortable && !isResizing;
    return /*#__PURE__*/ jsxs("div", {
        ...rest,
        ref: mergedRef,
        // PE-259 Use more performance className for grid but keep css= for compatibility.
        css: !grid ? [
            repeatingElementsStyles.cell,
            repeatingElementsStyles.header
        ] : undefined,
        className: classnames(grid && tableClassNames.cell, grid && tableClassNames.header, {
            'table-header-isGrid': grid
        }, className),
        role: "columnheader",
        "aria-sort": sortable && ariaSort || undefined,
        style: {
            justifyContent: align,
            textAlign: align,
            ...style
        },
        "aria-label": isSortButtonVisible ? undefined : ariaLabel,
        ...eventContext.dataComponentProps,
        children: [
            isSortButtonVisible ? /*#__PURE__*/ jsxs("div", {
                css: [
                    tableStyles.headerButtonTarget
                ],
                role: "button",
                tabIndex: 0,
                onClick: onToggleSort,
                onKeyDown: (event)=>{
                    if (sortable && (event.key === 'Enter' || event.key === ' ')) {
                        event.preventDefault();
                        return onToggleSort?.(event);
                    }
                },
                "aria-label": isSortButtonVisible ? ariaLabel : undefined,
                children: [
                    sortIconOnLeft ? /*#__PURE__*/ jsx("span", {
                        className: "table-header-icon-container",
                        css: [
                            tableStyles.sortHeaderIconOnLeft
                        ],
                        children: sortIcon
                    }) : null,
                    content,
                    !sortIconOnLeft ? /*#__PURE__*/ jsx("span", {
                        className: "table-header-icon-container",
                        css: [
                            tableStyles.sortHeaderIconOnRight
                        ],
                        children: sortIcon
                    }) : null
                ]
            }) : content,
            renderResizeHandle
        ]
    });
});

const TableRowActionHeader = ({ children })=>{
    return /*#__PURE__*/ jsx(TableRowAction, {
        children: /*#__PURE__*/ jsx("span", {
            css: visuallyHidden,
            children: children
        })
    });
};

/**
 * A component that renders a multi-action row in a table. Similar to TableRowAction, but with a gap between the actions.
 * TableRowMultiAction also allows for individual buttons to apply the `skipHideIconButtonClassName` classname and be
 * always-visible when necessary.
 *
 * Child buttons can also apply the `skipHideIconButtonClassName` classname and be always-visible when necessary.
 *
 * @param children - The child nodes for the table row. Should contain one or more small-sized buttons.
 * @param style - The style property.
 * @param className - The class name property.
 * @param css - The CSS property.
 * @param rest - The rest of the props.
 */ const TableRowMultiAction = /*#__PURE__*/ forwardRef(function TableRowMultiAction({ children, style, className, css: emotionCss, ...rest }, ref) {
    const { size } = useContext(TableContext);
    const { isHeader } = useContext(TableRowContext);
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx("div", {
        ...rest,
        ref: ref,
        role: isHeader ? 'columnheader' : 'cell',
        style: {
            paddingTop: size === 'default' ? theme.spacing.xs : 0,
            paddingBottom: size === 'default' ? theme.spacing.xs : 0,
            paddingLeft: size === 'default' ? theme.spacing.xs : 0,
            paddingRight: size === 'default' ? theme.spacing.xs : 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'flex-end',
            gap: theme.spacing.xs,
            ...style
        },
        css: emotionCss,
        className: classnames(className, !isHeader && hideIconButtonActionCellClassName),
        children: children
    });
});

const TableRowSelectCell = /*#__PURE__*/ forwardRef(function TableRowSelectCell({ onChange, checked, indeterminate, noCheckbox, children, isDisabled, checkboxLabel, componentId, analyticsEvents, ...rest }, ref) {
    const { theme } = useDesignSystemTheme();
    const { isHeader } = useContext(TableRowContext);
    const { someRowsSelected } = useContext(TableContext);
    if (typeof someRowsSelected === 'undefined') {
        throw new Error('`TableRowSelectCell` cannot be used unless `someRowsSelected` has been provided to the `Table` component, see documentation.');
    }
    if (!isHeader && indeterminate) {
        throw new Error('`TableRowSelectCell` cannot be used with `indeterminate` in a non-header row.');
    }
    return /*#__PURE__*/ jsx("div", {
        ...rest,
        ref: ref,
        css: tableStyles.checkboxCell,
        style: {
            ['--row-checkbox-opacity']: someRowsSelected ? 1 : 0,
            zIndex: theme.options.zIndexBase
        },
        role: isHeader ? 'columnheader' : 'cell',
        // TODO: Ideally we shouldn't need to specify this `className`, but it allows for row-hovering to reveal
        // the checkbox in `TableRow`'s CSS without extra JS pointerin/out events.
        className: "table-row-select-cell",
        children: !noCheckbox && /*#__PURE__*/ jsx(Checkbox, {
            componentId: componentId,
            analyticsEvents: analyticsEvents,
            isChecked: checked || indeterminate && null,
            onChange: (_checked, event)=>onChange?.(event.nativeEvent),
            isDisabled: isDisabled,
            "aria-label": checkboxLabel
        })
    });
});

const TabsRootContext = /*#__PURE__*/ React__default.createContext({
    activeValue: undefined,
    dataComponentProps: {
        'data-component-id': 'design_system.tabs.default_component_id',
        'data-component-type': DesignSystemEventProviderComponentTypes.Tabs
    }
});
const TabsListContext = /*#__PURE__*/ React__default.createContext({
    viewportRef: {
        current: null
    }
});
const Root = /*#__PURE__*/ React__default.forwardRef(({ value, defaultValue, onValueChange, componentId, analyticsEvents, valueHasNoPii, ...props }, forwardedRef)=>{
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.tabs', false);
    const isControlled = value !== undefined;
    const [uncontrolledActiveValue, setUncontrolledActiveValue] = React__default.useState(defaultValue);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Tabs,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii,
        shouldStartInteraction: true
    });
    const { elementRef: tabsRootRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: value ?? defaultValue
    });
    const mergedRef = useMergeRefs([
        forwardedRef,
        tabsRootRef
    ]);
    const onValueChangeWrapper = (value)=>{
        eventContext.onValueChange(value);
        if (onValueChange) {
            onValueChange(value);
        }
        if (!isControlled) {
            setUncontrolledActiveValue(value);
        }
    };
    return /*#__PURE__*/ jsx(TabsRootContext.Provider, {
        value: {
            activeValue: isControlled ? value : uncontrolledActiveValue,
            dataComponentProps: eventContext.dataComponentProps
        },
        children: /*#__PURE__*/ jsx(RadixTabs.Root, {
            value: value,
            defaultValue: defaultValue,
            onValueChange: onValueChangeWrapper,
            ...props,
            ref: mergedRef
        })
    });
});
const List = /*#__PURE__*/ React__default.forwardRef(({ addButtonProps, scrollAreaViewportCss, tabListCss, children, dangerouslyAppendEmotionCSS, shadowScrollStylesBackgroundColor, scrollbarHeight, getScrollAreaViewportRef, ...props }, forwardedRef)=>{
    const viewportRef = React__default.useRef(null);
    const { dataComponentProps } = React__default.useContext(TabsRootContext);
    const css = useListStyles(shadowScrollStylesBackgroundColor, scrollbarHeight);
    React__default.useEffect(()=>{
        if (getScrollAreaViewportRef) {
            getScrollAreaViewportRef(viewportRef.current);
        }
    }, [
        getScrollAreaViewportRef
    ]);
    return /*#__PURE__*/ jsx(TabsListContext.Provider, {
        value: {
            viewportRef
        },
        children: /*#__PURE__*/ jsxs("div", {
            css: [
                css['container'],
                dangerouslyAppendEmotionCSS
            ],
            children: [
                /*#__PURE__*/ jsxs(ScrollArea.Root, {
                    type: "hover",
                    css: [
                        css['root']
                    ],
                    children: [
                        /*#__PURE__*/ jsx(ScrollArea.Viewport, {
                            css: [
                                css['viewport'],
                                scrollAreaViewportCss,
                                {
                                    // Added to prevent adding and removing tabs from leaving extra empty spaces between existing tabs and the "+" button
                                    '& > div': {
                                        display: 'inline-block !important'
                                    }
                                }
                            ],
                            ref: viewportRef,
                            children: /*#__PURE__*/ jsx(RadixTabs.List, {
                                css: [
                                    css['list'],
                                    tabListCss
                                ],
                                ...props,
                                ref: forwardedRef,
                                ...dataComponentProps,
                                children: children
                            })
                        }),
                        /*#__PURE__*/ jsx(ScrollArea.Scrollbar, {
                            orientation: "horizontal",
                            css: css['scrollbar'],
                            children: /*#__PURE__*/ jsx(ScrollArea.Thumb, {
                                css: css['thumb']
                            })
                        })
                    ]
                }),
                addButtonProps && /*#__PURE__*/ jsx("div", {
                    css: [
                        css['addButtonContainer'],
                        addButtonProps.dangerouslyAppendEmotionCSS
                    ],
                    children: /*#__PURE__*/ jsx(Button, {
                        icon: /*#__PURE__*/ jsx(PlusIcon, {}),
                        size: "small",
                        "aria-label": "Add tab",
                        css: css['addButton'],
                        onClick: addButtonProps.onClick,
                        // eslint-disable-next-line @databricks/no-dynamic-property-value -- http://go/static-frontend-log-property-strings exempt:2a2973ea-6130-4b74-82c2-9f5f6b4c7c62
                        componentId: `${dataComponentProps['data-component-id']}.add_tab`,
                        className: addButtonProps.className
                    })
                })
            ]
        })
    });
});
const Trigger = /*#__PURE__*/ React__default.forwardRef(({ onClose, suppressDeleteClose, customizedCloseAriaLabel, value, disabled, children, ...props }, forwardedRef)=>{
    const triggerRef = React__default.useRef(null);
    const mergedRef = useMergeRefs([
        forwardedRef,
        triggerRef
    ]);
    const { activeValue, dataComponentProps } = React__default.useContext(TabsRootContext);
    const componentId = dataComponentProps['data-component-id'];
    const { viewportRef } = React__default.useContext(TabsListContext);
    const isClosable = onClose !== undefined && !disabled;
    const css = useTriggerStyles(isClosable);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.Button,
        componentId: `${componentId}.close_tab`,
        analyticsEvents: [
            DesignSystemEventProviderAnalyticsEventTypes.OnClick
        ]
    });
    const scrollActiveTabIntoView = React__default.useCallback(()=>{
        if (triggerRef.current && viewportRef.current && activeValue === value) {
            const viewportPosition = viewportRef.current.getBoundingClientRect();
            const triggerPosition = triggerRef.current.getBoundingClientRect();
            if (triggerPosition.left < viewportPosition.left) {
                viewportRef.current.scrollLeft -= viewportPosition.left - triggerPosition.left;
            } else if (triggerPosition.right > viewportPosition.right) {
                viewportRef.current.scrollLeft += triggerPosition.right - viewportPosition.right;
            }
        }
    }, [
        viewportRef,
        activeValue,
        value
    ]);
    const debouncedScrollActiveTabIntoView = React__default.useMemo(()=>debounce(scrollActiveTabIntoView, 10), [
        scrollActiveTabIntoView
    ]);
    React__default.useEffect(()=>{
        scrollActiveTabIntoView();
    }, [
        scrollActiveTabIntoView
    ]);
    React__default.useEffect(()=>{
        if (!viewportRef.current || !triggerRef.current) {
            return;
        }
        const resizeObserver = new ResizeObserver(debouncedScrollActiveTabIntoView);
        resizeObserver.observe(viewportRef.current);
        resizeObserver.observe(triggerRef.current);
        return ()=>{
            resizeObserver.disconnect();
            debouncedScrollActiveTabIntoView.cancel();
        };
    }, [
        debouncedScrollActiveTabIntoView,
        viewportRef
    ]);
    return /*#__PURE__*/ jsxs(RadixTabs.Trigger, {
        css: css['trigger'],
        value: value,
        disabled: disabled,
        // The close icon cannot be focused within the trigger button
        // Instead, we close the tab when the Delete key is pressed
        onKeyDown: (e)=>{
            if (isClosable && !suppressDeleteClose && e.key === 'Delete') {
                eventContext.onClick(e);
                e.stopPropagation();
                e.preventDefault();
                onClose(value);
            }
        },
        // Middle click also closes the tab
        // The Radix Tabs implementation uses onMouseDown for handling clicking tabs so we use it here as well
        onMouseDown: (e)=>{
            if (isClosable && e.button === 1) {
                eventContext.onClick(e);
                e.stopPropagation();
                e.preventDefault();
                onClose(value);
            }
        },
        ...props,
        ref: mergedRef,
        children: [
            children,
            isClosable && // An icon is used instead of a button to prevent nesting a button within a button
            /*#__PURE__*/ jsx(CloseSmallIcon, {
                onMouseDown: (e)=>{
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
                },
                css: css['closeSmallIcon'],
                "aria-hidden": "false",
                "aria-label": suppressDeleteClose ? customizedCloseAriaLabel : 'Press delete to close the tab'
            })
        ]
    });
});
const Content = /*#__PURE__*/ React__default.forwardRef(({ mountMode = 'active', children, ...props }, forwardedRef)=>{
    const { theme } = useDesignSystemTheme();
    const css = useContentStyles(theme);
    const { activeValue } = React__default.useContext(TabsRootContext);
    const shouldSetActivated = props.value === activeValue;
    const [hasBeenActivated, setHasBeenActivated] = React__default.useState(shouldSetActivated);
    if (shouldSetActivated && !hasBeenActivated) {
        setHasBeenActivated(true);
    }
    return /*#__PURE__*/ jsx(RadixTabs.Content, {
        css: css,
        ...props,
        ref: forwardedRef,
        forceMount: mountMode === 'force' || mountMode === 'preserve' || undefined,
        // `force` mode is implemented natively in Radix, but for `preserve` we
        // need to control rendering ourselves
        children: mountMode === 'preserve' && !hasBeenActivated ? undefined : children
    });
});
const useListStyles = (shadowScrollStylesBackgroundColor, scrollbarHeight)=>{
    const { theme } = useDesignSystemTheme();
    const containerStyles = getCommonTabsListStyles(theme);
    return {
        container: containerStyles,
        root: {
            overflow: 'hidden'
        },
        viewport: {
            ...getHorizontalTabShadowStyles(theme, {
                backgroundColor: shadowScrollStylesBackgroundColor
            })
        },
        list: {
            display: 'flex',
            alignItems: 'center'
        },
        scrollbar: {
            display: 'flex',
            flexDirection: 'column',
            userSelect: 'none',
            /* Disable browser handling of all panning and zooming gestures on touch devices */ touchAction: 'none',
            height: scrollbarHeight ?? 3
        },
        thumb: {
            flex: 1,
            background: theme.isDarkMode ? 'rgba(255, 255, 255, 0.2)' : 'rgba(17, 23, 28, 0.2)',
            '&:hover': {
                background: theme.isDarkMode ? 'rgba(255, 255, 255, 0.3)' : 'rgba(17, 23, 28, 0.3)'
            },
            borderRadius: theme.borders.borderRadiusSm,
            position: 'relative'
        },
        addButtonContainer: {
            flex: 1
        },
        addButton: {
            margin: '2px 0 6px 0'
        }
    };
};
const useTriggerStyles = (isClosable)=>{
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
                visibility: 'hidden'
            },
            '&:hover': {
                cursor: 'pointer',
                color: theme.colors.actionDefaultTextHover,
                [`& > .anticon:last-of-type`]: {
                    visibility: 'visible'
                }
            },
            '&:active': {
                color: theme.colors.actionDefaultTextPress
            },
            outlineStyle: 'none',
            outlineColor: theme.colors.actionDefaultBorderFocus,
            '&:focus-visible': {
                outlineStyle: 'auto'
            },
            '&[data-state="active"]': {
                color: theme.colors.textPrimary,
                // Use box-shadow instead of border to prevent it from affecting the size of the element, which results in visual
                // jumping when switching tabs.
                boxShadow: `inset 0 -4px 0 ${theme.colors.actionPrimaryBackgroundDefault}`,
                // The close icon is always visible on active tabs
                [`& > .anticon:last-of-type`]: {
                    visibility: 'visible'
                }
            },
            '&[data-disabled]': {
                color: theme.colors.actionDisabledText,
                '&:hover': {
                    cursor: 'not-allowed'
                }
            }
        },
        closeSmallIcon: {
            marginLeft: theme.spacing.xs,
            color: theme.colors.textSecondary,
            '&:hover': {
                color: theme.colors.actionDefaultTextHover
            },
            '&:active': {
                color: theme.colors.actionDefaultTextPress
            }
        }
    };
};
const useContentStyles = (theme)=>{
    // This is needed so force mounted content is not displayed when the tab is inactive
    return {
        color: theme.colors.textPrimary,
        '&[data-state="inactive"]': {
            display: 'none'
        }
    };
};

var Tabs = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Content: Content,
  List: List,
  Root: Root,
  Trigger: Trigger
});

const SMALL_BUTTON_HEIGHT = 24;
const getStyles = (theme, size, onlyIcon)=>{
    return /*#__PURE__*/ css({
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        whiteSpace: 'nowrap',
        ...!onlyIcon && {
            boxShadow: theme.shadows.xs
        },
        border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
        borderRadius: theme.borders.borderRadiusSm,
        backgroundColor: 'transparent',
        color: theme.colors.actionDefaultTextDefault,
        height: theme.general.heightSm,
        padding: '0 12px',
        fontSize: theme.typography.fontSizeBase,
        lineHeight: `${theme.typography.lineHeightBase}px`,
        '&[data-state="off"] .togglebutton-icon-wrapper': {
            color: theme.colors.textSecondary
        },
        '&[data-state="off"]:hover .togglebutton-icon-wrapper': {
            color: theme.colors.actionDefaultTextHover
        },
        '&[data-state="on"]': {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
            color: theme.colors.actionDefaultTextPress,
            borderColor: theme.colors.actionDefaultBorderPress
        },
        '&:hover': {
            cursor: 'pointer',
            color: theme.colors.actionDefaultTextHover,
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionDefaultBorderHover,
            '& > svg': {
                stroke: theme.colors.actionDefaultBorderHover
            }
        },
        '&:disabled': {
            cursor: 'default',
            borderColor: theme.colors.actionDisabledBorder,
            color: theme.colors.actionDisabledText,
            backgroundColor: 'transparent',
            boxShadow: 'none',
            '& > svg': {
                stroke: theme.colors.border
            }
        },
        ...!onlyIcon && {
            '&&': {
                padding: '4px 12px',
                ...size === 'small' && {
                    padding: '0 8px'
                }
            }
        },
        ...onlyIcon && {
            width: theme.general.heightSm,
            border: 'none'
        },
        ...size === 'small' && {
            height: SMALL_BUTTON_HEIGHT,
            lineHeight: theme.typography.lineHeightBase,
            ...onlyIcon && {
                width: SMALL_BUTTON_HEIGHT,
                paddingTop: 0,
                paddingBottom: 0,
                verticalAlign: 'middle'
            }
        }
    });
};
const RectangleSvg = (props)=>/*#__PURE__*/ jsx("svg", {
        width: "16",
        height: "16",
        viewBox: "0 0 16 16",
        fill: "none",
        xmlns: "http://www.w3.org/2000/svg",
        ...props,
        children: /*#__PURE__*/ jsx("rect", {
            x: "0.5",
            y: "0.5",
            width: "15",
            height: "15",
            rx: "3.5"
        })
    });
const RectangleIcon = /*#__PURE__*/ forwardRef((props, forwardedRef)=>{
    return /*#__PURE__*/ jsx(Icon, {
        ref: forwardedRef,
        ...props,
        component: RectangleSvg
    });
});
const ToggleButton = /*#__PURE__*/ forwardRef(({ children, pressed, defaultPressed, icon, size = 'middle', componentId, analyticsEvents, ...props }, ref)=>{
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.toggleButton', false);
    const { theme } = useDesignSystemTheme();
    const [isPressed, setIsPressed] = React__default.useState(defaultPressed);
    const memoizedAnalyticsEvents = useMemo(()=>analyticsEvents ?? (emitOnView ? [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
            DesignSystemEventProviderAnalyticsEventTypes.OnView
        ] : [
            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange
        ]), [
        analyticsEvents,
        emitOnView
    ]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.ToggleButton,
        componentId: componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true
    });
    const { elementRef: toggleButtonRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: pressed ?? defaultPressed
    });
    const mergedRef = useMergeRefs([
        ref,
        toggleButtonRef
    ]);
    const handleOnPressedChange = useCallback((pressed)=>{
        eventContext.onValueChange(pressed);
        props.onPressedChange?.(pressed);
        setIsPressed(pressed);
    }, [
        eventContext,
        props
    ]);
    useEffect(()=>{
        setIsPressed(pressed);
    }, [
        pressed
    ]);
    const iconOnly = !children && Boolean(icon);
    const iconStyle = iconOnly ? {} : {
        marginRight: theme.spacing.xs
    };
    const checkboxIcon = isPressed ? /*#__PURE__*/ jsx(CheckIcon, {}) : /*#__PURE__*/ jsx(RectangleIcon, {
        css: {
            stroke: theme.colors.border
        }
    });
    return /*#__PURE__*/ jsxs(Toggle.Root, {
        ...addDebugOutlineIfEnabled(),
        css: getStyles(theme, size, iconOnly),
        ...props,
        pressed: isPressed,
        onPressedChange: handleOnPressedChange,
        ref: mergedRef,
        ...eventContext.dataComponentProps,
        children: [
            /*#__PURE__*/ jsx("span", {
                className: "togglebutton-icon-wrapper",
                style: {
                    display: 'flex',
                    ...iconStyle
                },
                children: icon ? icon : checkboxIcon
            }),
            children
        ]
    });
});

const hideLinesForSizes = [
    'x-small',
    'xx-small'
];
const sizeMap = {
    default: {
        nodeSize: 32,
        indent: 28
    },
    small: {
        nodeSize: 24,
        indent: 24
    },
    'x-small': {
        nodeSize: 24,
        indent: 16
    },
    'xx-small': {
        nodeSize: 24,
        indent: 8
    }
};
/**
 * These styles share some aspects with the styles in the main `Checkbox.tsx` component.
 * However, due to significant differences in the internal implementation and DOM structure of the Tree Checkbox and the
 * main Checkbox, we have forked the styles here.
 * Some notable differences are:
 * 1. Tree checkbox does not have a wrapper div
 * 2. Tree checkbox does not use a hidden input element
 * 3. Tree checkbox does not support the disabled state.
 * 4. Tree checkbox does not support keyboard focus
 */ function getTreeCheckboxEmotionStyles(clsPrefix, theme) {
    const classRoot = `.${clsPrefix}`;
    const classInner = `.${clsPrefix}-inner`;
    const classIndeterminate = `.${clsPrefix}-indeterminate`;
    const classChecked = `.${clsPrefix}-checked`;
    const classDisabled = `.${clsPrefix}-disabled`;
    const styles = {
        [`${classRoot} > ${classInner}`]: {
            backgroundColor: theme.colors.actionDefaultBackgroundDefault,
            borderColor: theme.colors.actionDefaultBorderDefault
        },
        // Hover
        [`${classRoot}:hover > ${classInner}`]: {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            borderColor: theme.colors.actionDefaultBorderHover
        },
        // Mouse pressed
        [`${classRoot}:active > ${classInner}`]: {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
            borderColor: theme.colors.actionDefaultBorderPress
        },
        // Checked state
        [`${classChecked} > ${classInner}`]: {
            backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
            borderColor: 'transparent'
        },
        // Checked hover
        [`${classChecked}:hover > ${classInner}`]: {
            backgroundColor: theme.colors.actionPrimaryBackgroundHover,
            borderColor: 'transparent'
        },
        // Checked and mouse pressed
        [`${classChecked}:active > ${classInner}`]: {
            backgroundColor: theme.colors.actionPrimaryBackgroundPress,
            borderColor: 'transparent'
        },
        // Indeterminate
        [`${classIndeterminate} > ${classInner}`]: {
            backgroundColor: theme.colors.primary,
            borderColor: theme.colors.primary,
            // The after pseudo-element is used for the check image itself
            '&:after': {
                backgroundColor: theme.colors.white,
                height: '3px',
                width: '8px',
                borderRadius: '4px'
            }
        },
        // Indeterminate hover
        [`${classIndeterminate}:hover > ${classInner}`]: {
            backgroundColor: theme.colors.actionPrimaryBackgroundHover,
            borderColor: 'transparent'
        },
        // Indeterminate and mouse pressed
        [`${classIndeterminate}:active > ${classInner}`]: {
            backgroundColor: theme.colors.actionPrimaryBackgroundPress
        },
        // Disabled
        [`${classDisabled} > ${classInner}, ${classDisabled}:hover > ${classInner}, ${classDisabled}:active > ${classInner}`]: {
            backgroundColor: theme.colors.actionDisabledBackground
        },
        ...getAnimationCss(theme.options.enableAnimation)
    };
    return styles;
}
function getTreeEmotionStyles(clsPrefix, theme, size, useNewBorderColors) {
    const classNode = `.${clsPrefix}-tree-treenode`;
    const classNodeSelected = `.${clsPrefix}-tree-treenode-selected`;
    const classNodeActive = `.${clsPrefix}-tree-treenode-active`;
    const classNodeDisabled = `.${clsPrefix}-tree-treenode-disabled`;
    const classContent = `.${clsPrefix}-tree-node-content-wrapper`;
    const classContentTitle = `.${clsPrefix}-tree-title`;
    const classSelected = `.${clsPrefix}-tree-node-selected`;
    const classSwitcher = `.${clsPrefix}-tree-switcher`;
    const classSwitcherNoop = `.${clsPrefix}-tree-switcher-noop`;
    const classFocused = `.${clsPrefix}-tree-focused`;
    const classCheckbox = `.${clsPrefix}-tree-checkbox`;
    const classUnselectable = `.${clsPrefix}-tree-unselectable`;
    const classIndent = `.${clsPrefix}-tree-indent-unit`;
    const classTreeList = `.${clsPrefix}-tree-list`;
    const classScrollbar = `.${clsPrefix}-tree-list-scrollbar`;
    const classScrollbarThumb = `.${clsPrefix}-tree-list-scrollbar-thumb`;
    const classIcon = `.${clsPrefix}-tree-iconEle`;
    const classAntMotion = `.${clsPrefix}-tree-treenode-motion, .ant-motion-collapse-appear, .ant-motion-collapse-appear-active, .ant-motion-collapse`;
    const NODE_SIZE = sizeMap[size].nodeSize;
    const ICON_FONT_SIZE = 16;
    const BORDER_WIDTH = 4;
    const baselineAligned = {
        alignSelf: 'baseline',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
    };
    const styles = {
        // Basic node
        [classNode]: {
            minHeight: NODE_SIZE,
            width: '100%',
            padding: 0,
            paddingLeft: BORDER_WIDTH,
            display: 'flex',
            alignItems: 'center',
            // Ant tree renders some hidden tree nodes (presumably for internal purposes). Setting these to width: 100% causes
            // overflow, so we need to reset here.
            '&[aria-hidden=true]': {
                width: 'auto'
            },
            '&:hover': {
                backgroundColor: theme.colors.actionTertiaryBackgroundHover
            },
            '&:active': {
                backgroundColor: theme.colors.actionTertiaryBackgroundPress
            }
        },
        [`&${classUnselectable}`]: {
            // Remove hover and press styles if tree nodes are not selectable
            [classNode]: {
                '&:hover': {
                    backgroundColor: 'transparent'
                },
                '&:active': {
                    backgroundColor: 'transparent'
                }
            },
            [classContent]: {
                cursor: 'default'
            },
            // Unselectable nodes don't have any background, so the switcher looks better with rounded corners.
            [classSwitcher]: {
                borderRadius: theme.legacyBorders.borderRadiusMd
            }
        },
        // The "active" node is the one that is currently focused via keyboard navigation. We give it the same visual
        // treatment as the mouse hover style.
        [classNodeActive]: {
            backgroundColor: theme.colors.actionTertiaryBackgroundHover
        },
        // The "selected" node is one that has either been clicked on, or selected via pressing enter on the keyboard.
        [classNodeSelected]: {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
            borderLeft: `${BORDER_WIDTH}px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
            paddingLeft: 0,
            // When hovering over a selected node, we still want it to look selected
            '&:hover': {
                backgroundColor: theme.colors.actionTertiaryBackgroundPress
            }
        },
        [classSelected]: {
            background: 'none'
        },
        [classNodeDisabled]: {
            '&:hover': {
                backgroundColor: 'transparent'
            },
            '&:active': {
                backgroundColor: 'transparent'
            }
        },
        [classContent]: {
            lineHeight: `${NODE_SIZE}px`,
            // The content label is the interactive element, so we want it to fill the node to maximise the click area.
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            '&:hover': {
                backgroundColor: 'transparent'
            },
            '&:active': {
                backgroundColor: 'transparent'
            }
        },
        [classContentTitle]: {
            lineHeight: theme.typography.lineHeightBase,
            padding: `${(NODE_SIZE - parseInt(theme.typography.lineHeightBase, 10)) / 2}px 0`,
            // The content inside 'classContent' is wrapped in the title class, which is the actual interactive element.
            width: '100%'
        },
        // TODO(FEINF-1595): Temporary style for now
        [`${classSwitcherNoop} + ${classContent}, ${classSwitcherNoop} + ${classCheckbox}`]: {
            marginLeft: NODE_SIZE + 4
        },
        [classSwitcher]: {
            height: NODE_SIZE,
            width: NODE_SIZE,
            paddingTop: (NODE_SIZE - ICON_FONT_SIZE) / 2,
            marginRight: theme.spacing.xs,
            color: theme.colors.textSecondary,
            backgroundColor: 'transparent',
            // Keyboard navigation only allows moving between entire nodes, not between the switcher and label directly.
            // However, under mouse control, the two can still be clicked separately. We apply hover and press treatment
            // here to indicate to mouse users that the switcher is clickable.
            '&:hover': {
                backgroundColor: theme.colors.actionTertiaryBackgroundHover
            },
            '&:active': {
                backgroundColor: theme.colors.actionTertiaryBackgroundPress
            }
        },
        [classSwitcherNoop]: {
            display: 'none',
            '&:hover': {
                backgroundColor: 'transparent'
            },
            '&:active': {
                backgroundColor: 'transparent'
            }
        },
        [`&${classFocused}`]: {
            backgroundColor: 'transparent',
            outlineWidth: 2,
            outlineOffset: 1,
            outlineColor: theme.colors.actionDefaultBorderFocus,
            outlineStyle: 'solid'
        },
        [classCheckbox]: {
            marginTop: size === 'default' ? theme.spacing.sm : theme.spacing.xs,
            marginBottom: 0,
            marginRight: size === 'default' ? theme.spacing.sm : theme.spacing.xs,
            ...baselineAligned
        },
        [classScrollbarThumb]: {
            background: chroma(theme.isDarkMode ? '#ffffff' : '#000000').alpha(0.5).hex()
        },
        [`${classIcon}:has(*)`]: {
            ...baselineAligned,
            height: NODE_SIZE,
            color: theme.colors.textSecondary,
            marginRight: size === 'default' ? theme.spacing.sm : theme.spacing.xs
        },
        // Needed to avoid flickering when has icon and expanding
        [classAntMotion]: {
            ...getAnimationCss(theme.options.enableAnimation),
            visibility: 'hidden'
        },
        // Vertical line
        [classIndent]: {
            width: sizeMap[size].indent
        },
        [`${classIndent}::before`]: {
            height: '100%',
            ...useNewBorderColors && {
                borderColor: theme.colors.border
            }
        },
        [classTreeList]: {
            [`&:hover ${classScrollbar}`]: {
                display: 'block !important'
            },
            [`&:active ${classScrollbar}`]: {
                display: 'block !important'
            }
        },
        ...getTreeCheckboxEmotionStyles(`${clsPrefix}-tree-checkbox`, theme),
        ...getAnimationCss(theme.options.enableAnimation)
    };
    const importantStyles = importantify(styles);
    return /*#__PURE__*/ css(importantStyles);
}
const SHOW_LINE_DEFAULT = {
    showLeafIcon: false
};
// @ts-expect-error: Tree doesn't expose a proper type
const Tree = /*#__PURE__*/ forwardRef(function Tree({ treeData, defaultExpandedKeys, defaultSelectedKeys, defaultCheckedKeys, disabled = false, mode = 'default', size = 'default', showLine, dangerouslySetAntdProps, ...props }, ref) {
    const { theme, classNamePrefix } = useDesignSystemTheme();
    const useNewBorderColors = safex('databricks.fe.designsystem.useNewBorderColors', false);
    let calculatedShowLine = showLine ?? false;
    if (hideLinesForSizes.includes(size)) {
        calculatedShowLine = false;
    } else {
        calculatedShowLine = showLine ?? SHOW_LINE_DEFAULT;
    }
    return /*#__PURE__*/ jsx(DesignSystemAntDConfigProvider, {
        children: /*#__PURE__*/ jsx(Tree$1, {
            ...addDebugOutlineIfEnabled(),
            treeData: treeData,
            defaultExpandedKeys: defaultExpandedKeys,
            defaultSelectedKeys: defaultSelectedKeys,
            defaultCheckedKeys: defaultCheckedKeys,
            disabled: disabled,
            css: getTreeEmotionStyles(classNamePrefix, theme, size, useNewBorderColors),
            switcherIcon: /*#__PURE__*/ jsx(ChevronDownIcon, {
                css: {
                    fontSize: '16px !important'
                }
            }),
            tabIndex: 0,
            selectable: mode === 'selectable' || mode === 'multiselectable',
            checkable: mode === 'checkable',
            multiple: mode === 'multiselectable',
            // With the library flag, defaults to showLine = true. The status quo default is showLine = false.
            showLine: calculatedShowLine,
            ...dangerouslySetAntdProps,
            ...props,
            ref: ref
        })
    });
});

export { AccessibleContainer, Accordion, AccordionPanel, Alert, AlignCenterIcon, AlignJustifyIcon, AlignLeftIcon, AlignRightIcon, AppIcon, ApplyDesignSystemContextOverrides, ApplyGlobalStyles, ArrowDownDotIcon, ArrowDownFillIcon, ArrowDownIcon, ArrowInIcon, ArrowLeftIcon, ArrowOverIcon, ArrowRightIcon, ArrowUpDotIcon, ArrowUpFillIcon, ArrowUpIcon, ArrowsConnectIcon, ArrowsUpDownIcon, AssistantAvatar, AssistantIcon, AtIcon, AutoComplete, Avatar, AvatarGroup, AzHorizontalIcon, AzVerticalIcon, BackupIcon, BadgeCodeIcon, BadgeCodeOffIcon, BarChartIcon, BarGroupedIcon, BarStackedIcon, BarStackedPercentageIcon, BarsAscendingHorizontalIcon, BarsAscendingVerticalIcon, BarsDescendingHorizontalIcon, BarsDescendingVerticalIcon, BeakerIcon, BinaryIcon, BlockQuoteIcon, BoldIcon, BookIcon, BookmarkFillIcon, BookmarkIcon, BooksIcon, BracketsCheckIcon, BracketsCurlyIcon, BracketsErrorIcon, BracketsSquareIcon, BracketsXIcon, BranchCheckIcon, BranchIcon, BranchResetIcon, Breadcrumb, BriefcaseFillIcon, BriefcaseIcon, BrushIcon, BugIcon, Button, CalendarClockIcon, CalendarEventIcon, CalendarIcon, CalendarRangeIcon, CalendarSyncIcon, CameraIcon, Card, CaretDownSquareIcon, CaretUpSquareIcon, CatalogCloudIcon, CatalogGearIcon, CatalogHomeIcon, CatalogIcon, CatalogOffIcon, CatalogSharedIcon, CellsSquareIcon, CertifiedFillIcon, CertifiedFillSmallIcon, CertifiedIcon, ChainIcon, ChartLineIcon, CheckCircleBadgeIcon, CheckCircleFillIcon, CheckCircleIcon, CheckIcon, CheckLineIcon, CheckSmallIcon, Checkbox, CheckboxIcon, ChecklistIcon, ChevronDoubleDownIcon, ChevronDoubleLeftIcon, ChevronDoubleLeftOffIcon, ChevronDoubleRightIcon, ChevronDoubleRightOffIcon, ChevronDoubleUpIcon, ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon, ChevronUpIcon, ChipIcon, CircleIcon, CircleOffIcon, CircleOutlineIcon, ClipboardIcon, ClockKeyIcon, ClockOffIcon, CloseIcon, CloseSmallIcon, CloudDatabaseIcon, CloudDownloadIcon, CloudIcon, CloudKeyIcon, CloudModelIcon, CloudOffIcon, CloudUploadIcon, CodeIcon, Col, ColorFillIcon, ColumnIcon, ColumnSplitIcon, ColumnsIcon, CommandIcon, CommandPaletteIcon, ComponentFinderContext, ConnectIcon, ContextMenu$1 as ContextMenu, CopyIcon, CreditCardIcon, CursorClickIcon, CursorIcon, CursorPagination, CursorTypeIcon, CustomAppIcon, DBAssistantAvatar, DU_BOIS_ENABLE_ANIMATION_CLASSNAME, DagHorizontalIcon, DagIcon, DagVerticalIcon, DangerFillIcon, DangerIcon, DashIcon, DashboardIcon, DataIcon, DatabaseClockIcon, DatabaseIcon, DatabaseImportIcon, DecimalIcon, DeprecatedIcon, DeprecatedSmallIcon, DesignSystemAntDConfigProvider, DesignSystemEventProvider, DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentSubTypeMap, DesignSystemEventProviderComponentTypes, DesignSystemEventSuppressInteractionProviderContext, DesignSystemEventSuppressInteractionTrueContextValue, DialogCombobox, DialogComboboxAddButton, DialogComboboxContent, DialogComboboxCountBadge, DialogComboboxCustomButtonTriggerWrapper, EmptyResults as DialogComboboxEmpty, DialogComboboxFooter, DialogComboboxHintRow, DialogComboboxOptionControlledList, DialogComboboxOptionList, DialogComboboxOptionListCheckboxItem, DialogComboboxOptionListSearch, DialogComboboxOptionListSelectItem, DialogComboboxSectionHeader, DialogComboboxSeparator, DialogComboboxTrigger, DollarIcon, DomainCirclesThree, DotsCircleIcon, DownloadIcon, DragIcon, Drawer, Dropdown, DropdownMenu, DuboisDatePicker, Empty, ErdIcon, ExpandLessIcon, ExpandMoreIcon, FaceFrownIcon, FaceNeutralIcon, FaceSmileIcon, FileCodeIcon, FileCubeIcon, FileDocumentIcon, FileIcon, FileImageIcon, FileLockIcon, FileModelIcon, FileNewIcon, FilePipelineIcon, FilterIcon, FlagPointerIcon, FloatIcon, FlowIcon, FolderBranchFillIcon, FolderBranchIcon, FolderCloudFilledIcon, FolderCloudIcon, FolderCubeIcon, FolderCubeOutlineIcon, FolderFillIcon, FolderHomeIcon, FolderIcon, FolderNewIcon, FolderNodeIcon, FolderOpenBranchIcon, FolderOpenCloudIcon, FolderOpenCubeIcon, FolderOpenIcon, FolderOpenPipelineIcon, FolderOutlinePipelineIcon, FolderSolidPipelineIcon, FontIcon, ForkHorizontalIcon, ForkIcon, FormUI, FullscreenExitIcon, FullscreenIcon, FunctionIcon, GavelIcon, GearFillIcon, GearIcon, GenericSkeleton, GenieDeepResearchIcon, GiftIcon, GitCommitIcon, GlobeIcon, GridDashIcon, GridIcon, H1Icon, H2Icon, H3Icon, H4Icon, H5Icon, H6Icon, HashIcon, Header$1 as Header, HistoryIcon, HomeIcon, HoverCard, Icon, ImageIcon, IndentDecreaseIcon, IndentIncreaseIcon, InfinityIcon, InfoBookIcon, InfoFillIcon, InfoIcon, InfoPopover, InfoSmallIcon, InfoTooltip, IngestionIcon, Input, ItalicIcon, JoinOperatorIcon, KeyIcon, KeyboardIcon, LayerGraphIcon, LayerIcon, Layout, LeafIcon, LegacyDatePicker, LegacyForm, LegacyFormDubois, LegacyInfoTooltip, LegacyOptGroup, LegacyOption, LegacyPopover, LegacySelect, LegacySelectOptGroup, LegacySelectOption, LegacySkeleton, LegacyTabPane, LegacyTable, LegacyTabs, LegacyTooltip, LetterFormatIcon, LettersIcon, LettersNumbersIcon, LibrariesIcon, LifesaverIcon, LightbulbIcon, LightningIcon, LinkIcon, LinkOffIcon, ListBorderIcon, ListClearIcon, ListIcon, ListNumberIcon, LoadingIcon, LoadingState, LockFillIcon, LockShareIcon, LockUnlockedIcon, LoopIcon, MailIcon, MapIcon, MarkdownIcon, McpIcon, MeasureIcon, Menu, MenuIcon, MinusCircleFillIcon, MinusCircleIcon, MinusSquareIcon, Modal, ModelsIcon, MoonIcon, NavigationMenu, NeonProjectIcon, NewChatIcon, NoIcon, NotebookIcon, NotebookPipelineIcon, Notification, NotificationIcon, NotificationOffIcon, NumberFormatIcon, NumbersIcon, OfficeIcon, Overflow, OverflowIcon, PageBottomIcon, PageFirstIcon, PageLastIcon, PageTopIcon, PageWrapper, Pagination, PanelDockedIcon, PanelFloatingIcon, PaperclipIcon, ParagraphSkeleton, PauseIcon, PencilFillIcon, PencilIcon, PencilSparkleIcon, PieChartIcon, PinCancelIcon, PinFillIcon, PinIcon, PipelineCodeIcon, PipelineCubeIcon, PipelineIcon, PlayCircleFillIcon, PlayCircleIcon, PlayDoubleIcon, PlayIcon, PlayMultipleIcon, PlugIcon, PlusCircleFillIcon, PlusCircleIcon, PlusIcon, PlusMinusSquareIcon, PlusSquareIcon, PreviewCard, PullRequestIcon, PuzzleIcon, QueryEditorIcon, QueryIcon, QuestionMarkFillIcon, QuestionMarkIcon, RHFControlledComponents, ROW_GUTTER_SIZE, Radio, RadioIcon, RadioTile, ReaderModeIcon, RedoIcon, RefreshIcon, RefreshPlayIcon, RefreshXIcon, ReplyIcon, ResizeIcon, ResourceStatusIndicator, RestoreAntDDefaultClsPrefix, RichTextIcon, RobotIcon, RocketIcon, Row, RowsIcon, RunIcon, RunningIcon, SMALL_BUTTON_HEIGHT$2 as SMALL_BUTTON_HEIGHT, SaveClockIcon, SaveIcon, SchemaIcon, SchoolIcon, SearchDataIcon, SearchIcon, SegmentedControlButton, SegmentedControlGroup, Select, SelectContent, SelectContext, SelectContextProvider, SelectOption, SelectOptionGroup, SelectTrigger, SendIcon, ShareIcon, ShieldCheckIcon, ShieldIcon, ShieldOffIcon, ShortcutIcon, SidebarAutoIcon, SidebarCollapseIcon, SidebarExpandIcon, SidebarIcon, SidebarSyncIcon, SimpleSelect, SimpleSelectOption, SimpleSelectOptionGroup, SlashSquareIcon, Slider, SlidersIcon, SortAscendingIcon, SortCustomHorizontalIcon, SortCustomVerticalIcon, SortDescendingIcon, SortHorizontalAscendingIcon, SortHorizontalDescendingIcon, SortLetterHorizontalAscendingIcon, SortLetterHorizontalDescendingIcon, SortLetterUnsortedIcon, SortLetterVerticalAscendingIcon, SortLetterVerticalDescendingIcon, SortUnsortedIcon, SortVerticalAscendingIcon, SortVerticalDescendingIcon, Space, Spacer, SparkleDoubleFillIcon, SparkleDoubleIcon, SparkleFillIcon, SparkleIcon, SparkleRectangleIcon, SpeechBubbleIcon, SpeechBubblePlusIcon, SpeechBubbleQuestionMarkFillIcon, SpeechBubbleQuestionMarkIcon, SpeechBubbleStarIcon, SpeedometerIcon, Spinner, SplitButton, StarFillIcon, StarIcon, Steps, StopCircleFillIcon, StopCircleIcon, StopIcon, StoredProcedureIcon, StorefrontIcon, StreamIcon, StrikeThroughIcon, SunIcon, Switch, SyncIcon, SyncToFileIcon, Table, TableCell, TableClockIcon, TableCombineIcon, TableContext, TableFilterInput, TableFilterLayout, TableGlassesIcon, TableGlobeIcon, TableHeader, TableIcon, TableLightningIcon, TableMeasureIcon, TableModelIcon, TableRow, TableRowAction, TableRowActionHeader, TableRowContext, TableRowMenuContainer, TableRowMultiAction, TableRowSelectCell, TableSkeleton, TableSkeletonRows, TableStreamIcon, TableVectorIcon, TableViewIcon, Tabs, Tag, TagIcon, TargetIcon, TerminalIcon, TextBoxIcon, TextIcon, TextJustifyIcon, TextUnderlineIcon, ThreeDotsIcon, ThumbsDownIcon, ThumbsUpIcon, TitleSkeleton, ToggleButton, TokenIcon, Tooltip$1 as Tooltip, TrashIcon, Tree, TreeIcon, TrendingIcon, TriangleIcon, TypeaheadComboboxAddButton, TypeaheadComboboxCheckboxItem, TypeaheadComboboxFooter, TypeaheadComboboxInput, TypeaheadComboboxMenu, TypeaheadComboboxMenuItem, TypeaheadComboboxMultiSelectInput, TypeaheadComboboxMultiSelectStateChangeTypes, TypeaheadComboboxRoot, TypeaheadComboboxSectionHeader, TypeaheadComboboxSelectedItem, TypeaheadComboboxSeparator, TypeaheadComboboxStateChangeTypes, TypeaheadComboboxToggleButton, Typography, UnderlineIcon, UndoIcon, UploadIcon, UsbIcon, UserBadgeIcon, UserCircleIcon, UserGroupFillIcon, UserGroupIcon, UserIcon, UserKeyIconIcon, UserSparkleIcon, UserTeamIcon, VisibleFillIcon, VisibleIcon, VisibleOffIcon, WarningFillIcon, WarningIcon, WorkflowCodeIcon, WorkflowCubeIcon, WorkflowsIcon, WorkspacesIcon, WrenchIcon, WrenchSparkleIcon, XCircleFillIcon, XCircleIcon, ZaHorizontalIcon, ZaVerticalIcon, ZoomInIcon, ZoomMarqueeSelection, ZoomOutIcon, ZoomToFitIcon, __INTERNAL_DO_NOT_USE__FormItem, __INTERNAL_DO_NOT_USE__Group, __INTERNAL_DO_NOT_USE__HorizontalGroup, __INTERNAL_DO_NOT_USE__VerticalGroup, dialogComboboxLookAheadKeyDown, findClosestOptionSibling, findHighlightedOption, getAnimationCss, getComboboxOptionItemWrapperStyles, getComboboxOptionLabelStyles, getContentOptions, getDarkModePortalStyles, getDialogComboboxOptionLabelWidth, getGlobalStyles, getHorizontalTabShadowStyles, getKeyboardNavigationFunctions, getLegacyTabEmotionStyles, getPaginationEmotionStyles, getRadioStyles, getShadowScrollStyles, getValidationStateColor, getWrapperStyle, hideIconButtonRowStyles, highlightFirstNonDisabledOption, highlightOption, importantify, setupDesignSystemEventProviderForTesting, skipHideIconButtonActionClassName, useComboboxState, useComponentFinderContext, useDesignSystemEventComponentCallbacks, useDesignSystemFlags, useDesignSystemTheme, useLegacyNotification, useModalContext, useMultipleSelectionState, useNotifyOnFirstView, useRadioGroupContext, useThemedStyles, useTypeaheadComboboxContext, visuallyHidden, withNotifications };
//# sourceMappingURL=index.js.map

import { u as useDesignSystemContext, a as useDesignSystemTheme, I as Icon, D as DesignSystemAntDConfigProvider, R as RestoreAntDDefaultClsPrefix, b as useDesignSystemEventComponentCallbacks, c as DesignSystemEventProviderComponentTypes, d as DesignSystemEventProviderAnalyticsEventTypes, e as addDebugOutlineStylesIfEnabled, g as getAnimationCss, f as useNotifyOnFirstView, h as addDebugOutlineIfEnabled, C as CloseIcon, i as getDarkModePortalStyles, j as useUniqueId, k as Root$5, T as Trigger$3, l as Content$4, A as Arrow$2, B as Button, m as ChevronRightIcon, L as LoadingState, v as visuallyHidden, n as DesignSystemEventSuppressInteractionProviderContext, o as DesignSystemEventSuppressInteractionTrueContextValue, p as Typography, q as importantify, s as safex, r as CheckIcon, S as Spinner, t as getValidationStateColor, w as ApplyDesignSystemContextOverrides, x as getShadowScrollStyles, y as DangerIcon, W as WarningIcon, z as LoadingIcon, E as Title$2, F as AccessibleContainer, G as ChevronLeftIcon, H as DU_BOIS_ENABLE_ANIMATION_CLASSNAME, J as lightColorList, K as getDefaultStyles, M as getPrimaryStyles, N as getDisabledSplitButtonStyles } from './Typography-a18b0186.js';
export { Z as ApplyDesignSystemFlags, a5 as ColorVars, V as DesignSystemContext, P as DesignSystemEventProvider, Y as DesignSystemProvider, U as DesignSystemThemeContext, X as DesignSystemThemeProvider, a1 as LoadingStateContext, a0 as NewWindowIcon, a2 as Popover, $ as WithDesignSystemThemeHoc, a4 as getBottomOnlyShadowScrollStyles, O as getButtonEmotionStyles, a3 as getTypographyColor, _ as useAntDConfigProviderContext, Q as useDesignSystemEventSuppressInteractionContext } from './Typography-a18b0186.js';
import { css, Global, keyframes, ClassNames, createElement } from '@emotion/react';
import { Collapse, Alert as Alert$1, AutoComplete as AutoComplete$1, Tooltip, Breadcrumb as Breadcrumb$1, Checkbox as Checkbox$1, DatePicker, Dropdown as Dropdown$1, Form as Form$1, Select as Select$1, Radio as Radio$1, Switch as Switch$1, Col as Col$1, Row as Row$1, Space as Space$1, Layout as Layout$1, notification, Popover as Popover$1, Skeleton, Pagination as Pagination$1, Table as Table$1, Menu as Menu$1, Button as Button$1, Steps as Steps$1, Tabs as Tabs$1, Tree as Tree$1 } from 'antd';
import * as React from 'react';
import React__default, { useRef, useMemo, forwardRef, useCallback, useEffect, useState, createContext, useImperativeHandle, useContext, Children, Fragment as Fragment$1, useLayoutEffect } from 'react';
import { jsx, jsxs, Fragment } from '@emotion/react/jsx-runtime';
import classnames from 'classnames';
import _isNil from 'lodash/isNil';
import _isUndefined from 'lodash/isUndefined';
import { ContextMenuTrigger, ContextMenuItemIndicator, ContextMenuGroup, ContextMenuArrow, ContextMenuSub, ContextMenu as ContextMenu$2, ContextMenuSubTrigger, ContextMenuPortal, ContextMenuContent, ContextMenuSubContent, ContextMenuItem, ContextMenuCheckboxItem, ContextMenuRadioGroup, ContextMenuRadioItem, ContextMenuLabel, ContextMenuSeparator } from '@radix-ui/react-context-menu';
import * as DropdownMenu$1 from '@radix-ui/react-dropdown-menu';
import { I as InfoIcon, u as useModalContext, S as Spacer, L as ListIcon, T as Tooltip$1 } from './Tooltip-cba3a3a6.js';
export { C as Content, D as DangerModal, M as Modal, N as Nav, a as NavButton, P as Panel, e as PanelBody, b as PanelHeader, d as PanelHeaderButtons, c as PanelHeaderTitle, f as Sidebar } from './Tooltip-cba3a3a6.js';
import * as Popover from '@radix-ui/react-popover';
import { P as PlusIcon, I as Input } from './index-9b7de1ae.js';
export { C as ClockIcon, a as CloseSmallIcon, M as MegaphoneIcon, g as getInputStyles } from './index-9b7de1ae.js';
import { useMergeRefs, useFloating, autoUpdate, offset, flip, shift } from '@floating-ui/react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { useController } from 'react-hook-form';
import _uniqueId from 'lodash/uniqueId';
import { useCombobox, useMultipleSelection } from 'downshift';
import { createPortal } from 'react-dom';
import AntDIcon, { InfoCircleOutlined } from '@ant-design/icons';
import * as RadixHoverCard from '@radix-ui/react-hover-card';
import * as Toast from '@radix-ui/react-toast';
import _times from 'lodash/times';
import _random from 'lodash/random';
import * as Toggle from '@radix-ui/react-toggle';
import chroma from 'chroma-js';
import _isEqual from 'lodash/isEqual';
import '@radix-ui/react-tooltip';
import 'lodash/endsWith';
import 'lodash/isBoolean';
import 'lodash/isNumber';
import 'lodash/isString';
import 'lodash/mapValues';
import 'lodash/memoize';
import '@emotion/unitless';
import 'react-resizable';

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
 */
const useThemedStyles = styleFactory => {
  const {
    theme
  } = useDesignSystemTheme();

  // We can assume that the factory function won't change and we're
  // observing theme changes only.
  const styleFactoryRef = useRef(styleFactory);
  return useMemo(() => styleFactoryRef.current(theme), [theme]);
};

function SvgAlignCenterIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M1 2.5h14V1H1zM11.5 5.75h-7v-1.5h7zM15 8.75H1v-1.5h14zM15 15H1v-1.5h14zM4.5 11.75h7v-1.5h-7z"
    })
  });
}
const AlignCenterIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgAlignCenterIcon
  });
});
AlignCenterIcon.displayName = 'AlignCenterIcon';
var AlignCenterIcon$1 = AlignCenterIcon;

function SvgAlignLeftIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M1 2.5h14V1H1zM8 5.75H1v-1.5h7zM1 8.75v-1.5h14v1.5zM1 15v-1.5h14V15zM1 11.75h7v-1.5H1z"
    })
  });
}
const AlignLeftIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgAlignLeftIcon
  });
});
AlignLeftIcon.displayName = 'AlignLeftIcon';
var AlignLeftIcon$1 = AlignLeftIcon;

function SvgAlignRightIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M1 2.5h14V1H1zM15 5.75H8v-1.5h7zM1 8.75v-1.5h14v1.5zM1 15v-1.5h14V15zM8 11.75h7v-1.5H8z"
    })
  });
}
const AlignRightIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgAlignRightIcon
  });
});
AlignRightIcon.displayName = 'AlignRightIcon';
var AlignRightIcon$1 = AlignRightIcon;

function SvgAppIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.75 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M8 1a1.75 1.75 0 1 0 0 3.5A1.75 1.75 0 0 0 8 1m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M2.75 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M2.75 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5m5.25 0A1.75 1.75 0 1 0 8 15a1.75 1.75 0 0 0 0-3.5m5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5",
      clipRule: "evenodd"
    })
  });
}
const AppIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgAppIcon
  });
});
AppIcon.displayName = 'AppIcon';
var AppIcon$1 = AppIcon;

function SvgArrowDownDotIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M8 15a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3M3.47 6.53 8 11.06l4.53-4.53-1.06-1.06-2.72 2.72V1h-1.5v7.19L4.53 5.47z"
    })
  });
}
const ArrowDownDotIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgArrowDownDotIcon
  });
});
ArrowDownDotIcon.displayName = 'ArrowDownDotIcon';
var ArrowDownDotIcon$1 = ArrowDownDotIcon;

function SvgArrowDownIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8.03 15.06 1 8.03l1.06-1.06 5.22 5.22V1h1.5v11.19L14 6.97l1.06 1.06z",
      clipRule: "evenodd"
    })
  });
}
const ArrowDownIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgArrowDownIcon
  });
});
ArrowDownIcon.displayName = 'ArrowDownIcon';
var ArrowDownIcon$1 = ArrowDownIcon;

function SvgArrowInIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M4.5 2.5h9v11h-9V11H3v3.25c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H3.75a.75.75 0 0 0-.75.75V5h1.5z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M12.06 8 8.03 3.97 6.97 5.03l2.22 2.22H1v1.5h8.19l-2.22 2.22 1.06 1.06z"
    })]
  });
}
const ArrowInIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgArrowInIcon
  });
});
ArrowInIcon.displayName = 'ArrowInIcon';
var ArrowInIcon$1 = ArrowInIcon;

function SvgArrowLeftIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 8.03 8.03 1l1.061 1.06-5.22 5.22h11.19v1.5H3.87L9.091 14l-1.06 1.06z",
      clipRule: "evenodd"
    })
  });
}
const ArrowLeftIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgArrowLeftIcon
  });
});
ArrowLeftIcon.displayName = 'ArrowLeftIcon';
var ArrowLeftIcon$1 = ArrowLeftIcon;

function SvgArrowOverIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M8 2.5a5.48 5.48 0 0 1 3.817 1.54l.009.009.5.451H11V6h4V2h-1.5v1.539l-.651-.588A7.003 7.003 0 0 0 1.367 5.76l1.42.48A5.5 5.5 0 0 1 8 2.5M8 11a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3"
    })
  });
}
const ArrowOverIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgArrowOverIcon
  });
});
ArrowOverIcon.displayName = 'ArrowOverIcon';
var ArrowOverIcon$1 = ArrowOverIcon;

function SvgArrowRightIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m15.06 8.03-7.03 7.03L6.97 14l5.22-5.22H1v-1.5h11.19L6.97 2.06 8.03 1z",
      clipRule: "evenodd"
    })
  });
}
const ArrowRightIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgArrowRightIcon
  });
});
ArrowRightIcon.displayName = 'ArrowRightIcon';
var ArrowRightIcon$1 = ArrowRightIcon;

function SvgArrowUpDotIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M8 1a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3M12.53 9.47 8 4.94 3.47 9.47l1.06 1.06 2.72-2.72V15h1.5V7.81l2.72 2.72z"
    })
  });
}
const ArrowUpDotIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgArrowUpDotIcon
  });
});
ArrowUpDotIcon.displayName = 'ArrowUpDotIcon';
var ArrowUpDotIcon$1 = ArrowUpDotIcon;

function SvgArrowUpIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m8.03 1 7.03 7.03L14 9.091l-5.22-5.22v11.19h-1.5V3.87l-5.22 5.22L1 8.031z",
      clipRule: "evenodd"
    })
  });
}
const ArrowUpIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgArrowUpIcon
  });
});
ArrowUpIcon.displayName = 'ArrowUpIcon';
var ArrowUpIcon$1 = ArrowUpIcon;

function SvgArrowsUpDownIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M5.03 1 1 5.03l1.06 1.061 2.22-2.22v6.19h1.5V3.87L8 6.091l1.06-1.06zM11.03 15.121l4.03-4.03-1.06-1.06-2.22 2.219V6.06h-1.5v6.19l-2.22-2.22L7 11.091z"
    })
  });
}
const ArrowsUpDownIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgArrowsUpDownIcon
  });
});
ArrowsUpDownIcon.displayName = 'ArrowsUpDownIcon';
var ArrowsUpDownIcon$1 = ArrowsUpDownIcon;

function SvgAssistantIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M11.28 11.03H4.73v-1.7h6.55zm-2.8-4.7H4.73v1.7h3.75zM15.79 8h-1.7a6.09 6.09 0 0 1-6.08 6.08H3.12l.58-.58c.33-.33.33-.87 0-1.2A6.04 6.04 0 0 1 1.92 8 6.09 6.09 0 0 1 8 1.92V.22C3.71.22.22 3.71.22 8c0 1.79.6 3.49 1.71 4.87L.47 14.33c-.24.24-.32.61-.18.93.13.32.44.52.79.52h6.93c4.29 0 7.78-3.49 7.78-7.78m-.62-3.47c.4-.15.4-.72 0-.88l-1.02-.38c-.73-.28-1.31-.85-1.58-1.58L12.19.67c-.08-.2-.26-.3-.44-.3s-.36.1-.44.3l-.38 1.02c-.28.73-.85 1.31-1.58 1.58l-1.02.38c-.4.15-.4.72 0 .88l1.02.38c.73.28 1.31.85 1.58 1.58l.38 1.02c.08.2.26.3.44.3s.36-.1.44-.3l.38-1.02c.28-.73.85-1.31 1.58-1.58z"
    })
  });
}
const AssistantIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgAssistantIcon
  });
});
AssistantIcon.displayName = 'AssistantIcon';
var AssistantIcon$1 = AssistantIcon;

function SvgBadgeCodeIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "m5.56 8.53 1.97 1.97-1.06 1.06-3.03-3.03L6.47 5.5l1.06 1.06zM10.49 8.53 8.52 6.56 9.58 5.5l3.03 3.03-3.03 3.03-1.06-1.06z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 0a3.25 3.25 0 0 0-3 2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H11a3.25 3.25 0 0 0-3-2M6.285 2.9a1.75 1.75 0 0 1 3.43 0c.07.349.378.6.735.6h3.05v10h-11v-10h3.05a.75.75 0 0 0 .735-.6",
      clipRule: "evenodd"
    })]
  });
}
const BadgeCodeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBadgeCodeIcon
  });
});
BadgeCodeIcon.displayName = 'BadgeCodeIcon';
var BadgeCodeIcon$1 = BadgeCodeIcon;

function SvgBadgeCodeOffIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 17 17",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M16 2.75v11.19l-1.5-1.5V3.5h-3.05a.75.75 0 0 1-.735-.6 1.75 1.75 0 0 0-3.43 0 .75.75 0 0 1-.735.6h-.99L4.06 2H6a3.25 3.25 0 0 1 6 0h3.25a.75.75 0 0 1 .75.75"
    }), jsx("path", {
      fill: "currentColor",
      d: "m12.1 10.04-1.06-1.06.48-.48-1.97-1.97 1.06-1.06 3.031 3.03z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m12.94 15 1.03 1.03 1.06-1.06-13-13L.97 3.03 2 4.06v10.19c0 .414.336.75.75.75zm-4.455-4.454L7.47 11.56 4.44 8.53l1.015-1.016L3.5 5.561V13.5h7.94z",
      clipRule: "evenodd"
    })]
  });
}
const BadgeCodeOffIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBadgeCodeOffIcon
  });
});
BadgeCodeOffIcon.displayName = 'BadgeCodeOffIcon';
var BadgeCodeOffIcon$1 = BadgeCodeOffIcon;

function SvgBarChartIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M1 1v13.25c0 .414.336.75.75.75H15v-1.5H2.5V1z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M7 1v11h1.5V1zM10 5v7h1.5V5zM4 5v7h1.5V5zM13 12V8h1.5v4z"
    })]
  });
}
const BarChartIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBarChartIcon
  });
});
BarChartIcon.displayName = 'BarChartIcon';
var BarChartIcon$1 = BarChartIcon;

function SvgBarGroupedIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M12.25 2a.75.75 0 0 0-.75.75V7H9.25a.75.75 0 0 0-.75.75v5.5c0 .414.336.75.75.75h6a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75zm-.75 10.5v-4H10v4zm1.5 0h1.5v-9H13zM3.75 5a.75.75 0 0 0-.75.75V9H.75a.75.75 0 0 0-.75.75v3.5c0 .414.336.75.75.75h6a.75.75 0 0 0 .75-.75v-7.5A.75.75 0 0 0 6.75 5zM3 12.5v-2H1.5v2zm1.5 0H6v-6H4.5z",
      clipRule: "evenodd"
    })
  });
}
const BarGroupedIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBarGroupedIcon
  });
});
BarGroupedIcon.displayName = 'BarGroupedIcon';
var BarGroupedIcon$1 = BarGroupedIcon;

function SvgBarStackedIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6.25 1a.75.75 0 0 0-.75.75V7H2.75a.75.75 0 0 0-.75.75v6.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-9.5a.75.75 0 0 0-.75-.75H10.5V1.75A.75.75 0 0 0 9.75 1zM9 8.5v5H7v-5zM9 7V2.5H7V7zm3.5 6.5h-2v-1.75h2zm-2-8v4.75h2V5.5zm-5 4.75V8.5h-2v1.75zm0 3.25v-1.75h-2v1.75z",
      clipRule: "evenodd"
    })
  });
}
const BarStackedIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBarStackedIcon
  });
});
BarStackedIcon.displayName = 'BarStackedIcon';
var BarStackedIcon$1 = BarStackedIcon;

function SvgBarStackedPercentageIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zM9 8.5v5H7v-5zM9 7V2.5H7V7zm3.5 6.5h-2v-1.75h2zm-2-11v7.75h2V2.5zm-5 0h-2v7.75h2zm0 11v-1.75h-2v1.75z",
      clipRule: "evenodd"
    })
  });
}
const BarStackedPercentageIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBarStackedPercentageIcon
  });
});
BarStackedPercentageIcon.displayName = 'BarStackedPercentageIcon';
var BarStackedPercentageIcon$1 = BarStackedPercentageIcon;

function SvgBeakerIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M5.75 1a.75.75 0 0 0-.75.75v6.089c0 .38-.173.739-.47.976l-2.678 2.143A2.27 2.27 0 0 0 3.27 15h9.46a2.27 2.27 0 0 0 1.418-4.042L11.47 8.815A1.25 1.25 0 0 1 11 7.839V1.75a.75.75 0 0 0-.75-.75zm.75 6.839V2.5h3v5.339c0 .606.2 1.188.559 1.661H5.942A2.75 2.75 0 0 0 6.5 7.839M4.2 11 2.79 12.13a.77.77 0 0 0 .48 1.37h9.461a.77.77 0 0 0 .481-1.37L11.8 11z",
      clipRule: "evenodd"
    })
  });
}
const BeakerIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBeakerIcon
  });
});
BeakerIcon.displayName = 'BeakerIcon';
var BeakerIcon$1 = BeakerIcon;

function SvgBinaryIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 3a2 2 0 1 1 4 0v2a2 2 0 1 1-4 0zm2-.5a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0V3a.5.5 0 0 0-.5-.5m3.378-.628c.482 0 .872-.39.872-.872h1.5v4.25H10v1.5H6v-1.5h1.25V3.206c-.27.107-.564.166-.872.166H6v-1.5zm5 0c.482 0 .872-.39.872-.872h1.5v4.25H15v1.5h-4v-1.5h1.25V3.206c-.27.107-.564.166-.872.166H11v-1.5zM6 11a2 2 0 1 1 4 0v2a2 2 0 1 1-4 0zm2-.5a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0v-2a.5.5 0 0 0-.5-.5m-6.622-.378c.482 0 .872-.39.872-.872h1.5v4.25H5V15H1v-1.5h1.25v-2.044c-.27.107-.564.166-.872.166H1v-1.5zm10 0c.482 0 .872-.39.872-.872h1.5v4.25H15V15h-4v-1.5h1.25v-2.044c-.27.107-.564.166-.872.166H11v-1.5z",
      clipRule: "evenodd"
    })
  });
}
const BinaryIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBinaryIcon
  });
});
BinaryIcon.displayName = 'BinaryIcon';
var BinaryIcon$1 = BinaryIcon;

function SvgBlockQuoteIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.5 2H0v12h1.5zM13 2H4v1.5h9zm3 3.5H4V7h12zM13 9H4v1.5h9zm3 3.5H4V14h12z",
      clipRule: "evenodd"
    })
  });
}
const BlockQuoteIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBlockQuoteIcon
  });
});
BlockQuoteIcon.displayName = 'BlockQuoteIcon';
var BlockQuoteIcon$1 = BlockQuoteIcon;

function SvgBoldIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M4.75 3a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h4.375a2.875 2.875 0 0 0 1.496-5.33A2.875 2.875 0 0 0 8.375 3zm.75 5.75v2.75h3.625a1.375 1.375 0 0 0 0-2.75zm2.877-1.5a1.375 1.375 0 0 0-.002-2.75H5.5v2.75z",
      clipRule: "evenodd"
    })
  });
}
const BoldIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBoldIcon
  });
});
BoldIcon.displayName = 'BoldIcon';
var BoldIcon$1 = BoldIcon;

function SvgBookIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.75 1a.75.75 0 0 0-.75.75v13.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zM7.5 2.5h-4v6.055l1.495-1.36a.75.75 0 0 1 1.01 0L7.5 8.555zm-4 8.082 2-1.818 2.246 2.041A.75.75 0 0 0 9 10.25V2.5h3.5v12h-9z",
      clipRule: "evenodd"
    })
  });
}
const BookIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBookIcon
  });
});
BookIcon.displayName = 'BookIcon';
var BookIcon$1 = BookIcon;

function SvgBookmarkFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M2.75 0A.75.75 0 0 0 2 .75v14.5a.75.75 0 0 0 1.28.53L8 11.06l4.72 4.72a.75.75 0 0 0 1.28-.53V.75a.75.75 0 0 0-.75-.75z"
    })
  });
}
const BookmarkFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBookmarkFillIcon
  });
});
BookmarkFillIcon.displayName = 'BookmarkFillIcon';
var BookmarkFillIcon$1 = BookmarkFillIcon;

function SvgBookmarkIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2 .75A.75.75 0 0 1 2.75 0h10.5a.75.75 0 0 1 .75.75v14.5a.75.75 0 0 1-1.28.53L8 11.06l-4.72 4.72A.75.75 0 0 1 2 15.25zm1.5.75v11.94l3.97-3.97a.75.75 0 0 1 1.06 0l3.97 3.97V1.5z",
      clipRule: "evenodd"
    })
  });
}
const BookmarkIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBookmarkIcon
  });
});
BookmarkIcon.displayName = 'BookmarkIcon';
var BookmarkIcon$1 = BookmarkIcon;

function SvgBooksIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 17 17",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.5 4.5v10h1v-10zM1 3a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1zM6.5 1.5v13h2v-13zM6 0a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m11.63 7.74 1.773 6.773.967-.254-1.773-6.771zm-.864-1.324a1 1 0 0 0-.714 1.221l2.026 7.74a1 1 0 0 0 1.22.713l1.936-.506a1 1 0 0 0 .714-1.22l-2.026-7.74a1 1 0 0 0-1.22-.714z",
      clipRule: "evenodd"
    })]
  });
}
const BooksIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBooksIcon
  });
});
BooksIcon.displayName = 'BooksIcon';
var BooksIcon$1 = BooksIcon;

function SvgBracketsCurlyIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M5.5 2a2.75 2.75 0 0 0-2.75 2.75v1C2.75 6.44 2.19 7 1.5 7H1v2h.5c.69 0 1.25.56 1.25 1.25v1A2.75 2.75 0 0 0 5.5 14H6v-1.5h-.5c-.69 0-1.25-.56-1.25-1.25v-1c0-.93-.462-1.752-1.168-2.25A2.75 2.75 0 0 0 4.25 5.75v-1c0-.69.56-1.25 1.25-1.25H6V2zM13.25 4.75A2.75 2.75 0 0 0 10.5 2H10v1.5h.5c.69 0 1.25.56 1.25 1.25v1c0 .93.462 1.752 1.168 2.25a2.75 2.75 0 0 0-1.168 2.25v1c0 .69-.56 1.25-1.25 1.25H10V14h.5a2.75 2.75 0 0 0 2.75-2.75v-1c0-.69.56-1.25 1.25-1.25h.5V7h-.5c-.69 0-1.25-.56-1.25-1.25z"
    })
  });
}
const BracketsCurlyIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBracketsCurlyIcon
  });
});
BracketsCurlyIcon.displayName = 'BracketsCurlyIcon';
var BracketsCurlyIcon$1 = BracketsCurlyIcon;

function SvgBracketsSquareIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1H5v1.5H2.5v11H5V15H1.75a.75.75 0 0 1-.75-.75zm12.5.75H11V1h3.25a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H11v-1.5h2.5z",
      clipRule: "evenodd"
    })
  });
}
const BracketsSquareIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBracketsSquareIcon
  });
});
BracketsSquareIcon.displayName = 'BracketsSquareIcon';
var BracketsSquareIcon$1 = BracketsSquareIcon;

function SvgBracketsXIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#BracketsXIcon_svg__a)",
      children: [jsx("path", {
        d: "M1.75 4.75A2.75 2.75 0 0 1 4.5 2H5v1.5h-.5c-.69 0-1.25.56-1.25 1.25v1c0 .93-.462 1.752-1.168 2.25a2.75 2.75 0 0 1 1.168 2.25v1c0 .69.56 1.25 1.25 1.25H5V14h-.5a2.75 2.75 0 0 1-2.75-2.75v-1C1.75 9.56 1.19 9 .5 9H0V7h.5c.69 0 1.25-.56 1.25-1.25zM11.5 2a2.75 2.75 0 0 1 2.75 2.75v1c0 .69.56 1.25 1.25 1.25h.5v2h-.5c-.69 0-1.25.56-1.25 1.25v1A2.75 2.75 0 0 1 11.5 14H11v-1.5h.5c.69 0 1.25-.56 1.25-1.25v-1c0-.93.462-1.752 1.168-2.25a2.75 2.75 0 0 1-1.168-2.25v-1c0-.69-.56-1.25-1.25-1.25H11V2z"
      }), jsx("path", {
        d: "M4.97 6.03 6.94 8 4.97 9.97l1.06 1.06L8 9.06l1.97 1.97 1.06-1.06L9.06 8l1.97-1.97-1.06-1.06L8 6.94 6.03 4.97z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const BracketsXIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBracketsXIcon
  });
});
BracketsXIcon.displayName = 'BracketsXIcon';
var BracketsXIcon$1 = BracketsXIcon;

function SvgBranchIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 4a3 3 0 1 1 5.186 2.055 3.23 3.23 0 0 0 2 1.155 3.001 3.001 0 1 1-.152 1.494A4.73 4.73 0 0 1 4.911 6.86a3 3 0 0 1-.161.046v2.19a3.001 3.001 0 1 1-1.5 0v-2.19A3 3 0 0 1 1 4m3-1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3M2.5 12a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0m7-3.75a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0",
      clipRule: "evenodd"
    })
  });
}
const BranchIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBranchIcon
  });
});
BranchIcon.displayName = 'BranchIcon';
var BranchIcon$1 = BranchIcon;

function SvgBriefcaseFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M5 4V2.75C5 1.784 5.784 1 6.75 1h2.5c.966 0 1.75.784 1.75 1.75V4h3.25a.75.75 0 0 1 .75.75v9.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75v-9.5A.75.75 0 0 1 1.75 4zm1.5-1.25a.25.25 0 0 1 .25-.25h2.5a.25.25 0 0 1 .25.25V4h-3zm-4 5.423V6.195A7.72 7.72 0 0 0 8 8.485c2.15 0 4.095-.875 5.5-2.29v1.978A9.2 9.2 0 0 1 8 9.985a9.2 9.2 0 0 1-5.5-1.812",
      clipRule: "evenodd"
    })
  });
}
const BriefcaseFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBriefcaseFillIcon
  });
});
BriefcaseFillIcon.displayName = 'BriefcaseFillIcon';
var BriefcaseFillIcon$1 = BriefcaseFillIcon;

function SvgBriefcaseIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 4H5V2.75C5 1.784 5.784 1 6.75 1h2.5c.966 0 1.75.784 1.75 1.75V4h3.25a.75.75 0 0 1 .75.75v9.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75v-9.5A.75.75 0 0 1 1.75 4m5-1.5a.25.25 0 0 0-.25.25V4h3V2.75a.25.25 0 0 0-.25-.25zM2.5 8.173V13.5h11V8.173A9.2 9.2 0 0 1 8 9.985a9.2 9.2 0 0 1-5.5-1.812m0-1.978A7.72 7.72 0 0 0 8 8.485c2.15 0 4.095-.875 5.5-2.29V5.5h-11z",
      clipRule: "evenodd"
    })
  });
}
const BriefcaseIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBriefcaseIcon
  });
});
BriefcaseIcon.displayName = 'BriefcaseIcon';
var BriefcaseIcon$1 = BriefcaseIcon;

function SvgBugIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M5.5 5a2.5 2.5 0 1 1 4.792 1H5.708A2.5 2.5 0 0 1 5.5 5M4.13 6.017a4 4 0 1 1 7.74 0l.047.065L14 4l1.06 1.06-2.41 2.412q.178.493.268 1.028H16V10h-3.02a6 6 0 0 1-.33 1.528l2.41 2.412L14 15l-2.082-2.082C11.002 14.187 9.588 15 8 15c-1.587 0-3.002-.813-3.918-2.082L2 15 .94 13.94l2.41-2.412A6 6 0 0 1 3.02 10H0V8.5h3.082q.09-.535.269-1.028L.939 5.061 2 4l2.082 2.081zm.812 1.538A4.4 4.4 0 0 0 4.5 9.5c0 2.347 1.698 4 3.5 4s3.5-1.653 3.5-4c0-.713-.163-1.375-.442-1.945z",
      clipRule: "evenodd"
    })
  });
}
const BugIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgBugIcon
  });
});
BugIcon.displayName = 'BugIcon';
var BugIcon$1 = BugIcon;

function SvgCalendarClockIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M4.5 0v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75H6v-1.5H2.5V7H15V2.75a.75.75 0 0 0-.75-.75H11.5V0H10v2H6V0zm9 5.5v-2h-11v2z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M10.25 10.5V12c0 .199.079.39.22.53l1 1 1.06-1.06-.78-.78V10.5z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7 12a4 4 0 1 1 8 0 4 4 0 0 1-8 0m4-2.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5",
      clipRule: "evenodd"
    })]
  });
}
const CalendarClockIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCalendarClockIcon
  });
});
CalendarClockIcon.displayName = 'CalendarClockIcon';
var CalendarClockIcon$1 = CalendarClockIcon;

function SvgCalendarEventIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M8.5 10.25a1.75 1.75 0 1 1 3.5 0 1.75 1.75 0 0 1-3.5 0"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M10 2H6V0H4.5v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H11.5V0H10zM2.5 3.5v2h11v-2zm0 10V7h11v6.5z",
      clipRule: "evenodd"
    })]
  });
}
const CalendarEventIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCalendarEventIcon
  });
});
CalendarEventIcon.displayName = 'CalendarEventIcon';
var CalendarEventIcon$1 = CalendarEventIcon;

function SvgCalendarIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M4.5 0v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H11.5V0H10v2H6V0zm9 3.5v2h-11v-2zM2.5 7v6.5h11V7z",
      clipRule: "evenodd"
    })
  });
}
const CalendarIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCalendarIcon
  });
});
CalendarIcon.displayName = 'CalendarIcon';
var CalendarIcon$1 = CalendarIcon;

function SvgCaretDownSquareIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M8 10a.75.75 0 0 1-.59-.286l-2.164-2.75a.75.75 0 0 1 .589-1.214h4.33a.75.75 0 0 1 .59 1.214l-2.166 2.75A.75.75 0 0 1 8 10"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z",
      clipRule: "evenodd"
    })]
  });
}
const CaretDownSquareIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCaretDownSquareIcon
  });
});
CaretDownSquareIcon.displayName = 'CaretDownSquareIcon';
var CaretDownSquareIcon$1 = CaretDownSquareIcon;

function SvgCaretUpSquareIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M8 5.75a.75.75 0 0 1 .59.286l2.164 2.75A.75.75 0 0 1 10.165 10h-4.33a.75.75 0 0 1-.59-1.214l2.166-2.75A.75.75 0 0 1 8 5.75"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z",
      clipRule: "evenodd"
    })]
  });
}
const CaretUpSquareIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCaretUpSquareIcon
  });
});
CaretUpSquareIcon.displayName = 'CaretUpSquareIcon';
var CaretUpSquareIcon$1 = CaretUpSquareIcon;

function SvgCatalogCloudIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.5 13.25V4.792c.306.134.644.208 1 .208h8v1H13V.75a.75.75 0 0 0-.75-.75H3.5A2.5 2.5 0 0 0 1 2.5v10.75A2.75 2.75 0 0 0 3.75 16H4v-1.5h-.25c-.69 0-1.25-.56-1.25-1.25m9-9.75h-8a1 1 0 0 1 0-2h8z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M10.179 7a3.61 3.61 0 0 0-3.464 2.595 3.251 3.251 0 0 0 .443 6.387.8.8 0 0 0 .163.018h5.821C14.758 16 16 14.688 16 13.107c0-1.368-.931-2.535-2.229-2.824A3.61 3.61 0 0 0 10.18 7m-2.805 7.496q.023 0 .044.004h5.555a1 1 0 0 1 .1-.002l.07.002c.753 0 1.357-.607 1.357-1.393s-.604-1.393-1.357-1.393h-.107a.75.75 0 0 1-.75-.75v-.357a2.107 2.107 0 0 0-4.199-.26.75.75 0 0 1-.698.656 1.75 1.75 0 0 0-.015 3.493",
      clipRule: "evenodd"
    })]
  });
}
const CatalogCloudIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCatalogCloudIcon
  });
});
CatalogCloudIcon.displayName = 'CatalogCloudIcon';
var CatalogCloudIcon$1 = CatalogCloudIcon;

function SvgCatalogGearIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      fillRule: "evenodd",
      clipPath: "url(#CatalogGearIcon_svg__a)",
      clipRule: "evenodd",
      children: [jsx("path", {
        d: "M12.5 7V5h-8c-.356 0-.694-.074-1-.208v8.458c0 .69.56 1.25 1.25 1.25H7.5V16H4.75A2.75 2.75 0 0 1 2 13.25V2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75V7zm-8-3.5a1 1 0 0 1 0-2h8v2z"
      }), jsx("path", {
        d: "M12.75 9.31c.328.09.632.24.9.434l.959-.764.935 1.172-.958.765c.129.301.207.63.222.974l1.195.273-.333 1.463-1.195-.273c-.16.296-.372.56-.624.781l.532 1.105-1.35.651-.533-1.104a2.8 2.8 0 0 1-1 0l-.531 1.104-1.352-.65.532-1.105a2.8 2.8 0 0 1-.624-.782l-1.195.273-.334-1.462 1.196-.273c.015-.345.093-.674.222-.975l-.958-.764.935-1.173.958.765a2.8 2.8 0 0 1 .901-.435V8.085h1.5zm.374 2.71a1.124 1.124 0 1 1-2.248 0 1.124 1.124 0 0 1 2.248 0"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const CatalogGearIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCatalogGearIcon
  });
});
CatalogGearIcon.displayName = 'CatalogGearIcon';
var CatalogGearIcon$1 = CatalogGearIcon;

function SvgCatalogHIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3.5 13.25V4.792c.306.134.644.208 1 .208h8v2H14V.75a.75.75 0 0 0-.75-.75H4.5A2.5 2.5 0 0 0 2 2.5v10.75A2.75 2.75 0 0 0 4.75 16H8v-1.5H4.75c-.69 0-1.25-.56-1.25-1.25m9-9.75h-8a1 1 0 0 1 0-2h8z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M9 8v8h1.5v-3.25h3V16H15V8h-1.5v3.25h-3V8z"
    })]
  });
}
const CatalogHIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCatalogHIcon
  });
});
CatalogHIcon.displayName = 'CatalogHIcon';
var CatalogHIcon$1 = CatalogHIcon;

function SvgCatalogHomeIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M14 .75v7.143l-.285-.247A2.64 2.64 0 0 0 12.5 7.05V5h-8c-.356 0-.694-.074-1-.208v8.458c0 .69.56 1.25 1.25 1.25H7V16H4.75A2.75 2.75 0 0 1 2 13.25V2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75M3.5 2.5a1 1 0 0 0 1 1h8v-2h-8a1 1 0 0 0-1 1",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M12.475 8.67a.75.75 0 0 0-.95 0l-2.75 2.25a.75.75 0 0 0-.275.58v3.75c0 .414.336.75.75.75h5.5a.75.75 0 0 0 .75-.75V11.5a.75.75 0 0 0-.275-.58zM10 14.5v-2.645l2-1.636 2 1.636V14.5h-1.25v-2h-1.5v2z",
      clipRule: "evenodd"
    })]
  });
}
const CatalogHomeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCatalogHomeIcon
  });
});
CatalogHomeIcon.displayName = 'CatalogHomeIcon';
var CatalogHomeIcon$1 = CatalogHomeIcon;

function SvgCatalogIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M14 .75a.75.75 0 0 0-.75-.75H4.5A2.5 2.5 0 0 0 2 2.5v10.75A2.75 2.75 0 0 0 4.75 16h8.5a.75.75 0 0 0 .75-.75zM3.5 4.792v8.458c0 .69.56 1.25 1.25 1.25h7.75V5h-8c-.356 0-.694-.074-1-.208m9-1.292v-2h-8a1 1 0 0 0 0 2z",
      clipRule: "evenodd"
    })
  });
}
const CatalogIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCatalogIcon
  });
});
CatalogIcon.displayName = 'CatalogIcon';
var CatalogIcon$1 = CatalogIcon;

function SvgCatalogOffIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M14.03.75v10.69l-1.5-1.5V1.5H4.78c-.2 0-.39.047-.558.131L3.136.545A2.74 2.74 0 0 1 4.78 0h8.5a.75.75 0 0 1 .75.75"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.03 3.56 1 2.53l1.06-1.06 13 13L14 15.53l-.017-.017a.75.75 0 0 1-.703.487H4.53a2.5 2.5 0 0 1-2.5-2.5zm8.94 8.94 1.56 1.56v.44h-8a1 1 0 1 1 0-2zM9.47 11H4.53c-.355 0-.693.074-1 .208V5.061z",
      clipRule: "evenodd"
    })]
  });
}
const CatalogOffIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCatalogOffIcon
  });
});
CatalogOffIcon.displayName = 'CatalogOffIcon';
var CatalogOffIcon$1 = CatalogOffIcon;

function SvgCatalogSharedIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M14 6.76a3 3 0 0 0-1.5.553V5h-8c-.356 0-.694-.074-1-.208v8.458c0 .69.56 1.25 1.25 1.25H8V16H4.75A2.75 2.75 0 0 1 2 13.25V2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75zM3.5 2.5a1 1 0 0 0 1 1h8v-2h-8a1 1 0 0 0-1 1",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M14.25 11.5c-.407 0-.782-.139-1.079-.372l-1.671.836v.072l1.671.836a1.75 1.75 0 1 1-.67 1.342l-1.672-.836a1.75 1.75 0 1 1 0-2.756l1.671-.836V9.75a1.75 1.75 0 1 1 1.75 1.75"
    })]
  });
}
const CatalogSharedIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCatalogSharedIcon
  });
});
CatalogSharedIcon.displayName = 'CatalogSharedIcon';
var CatalogSharedIcon$1 = CatalogSharedIcon;

function SvgChainIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 24 24",
    ...props,
    children: [jsxs("g", {
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      strokeWidth: 2.5,
      clipPath: "url(#ChainIcon_svg__a)",
      children: [jsx("path", {
        d: "M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"
      }), jsx("path", {
        d: "M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h24v24H0z"
        })
      })
    })]
  });
}
const ChainIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgChainIcon
  });
});
ChainIcon.displayName = 'ChainIcon';
var ChainIcon$1 = ChainIcon;

function SvgChartLineIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M1 1v13.25c0 .414.336.75.75.75H15v-1.5H2.5V1z"
    }), jsx("path", {
      fill: "currentColor",
      d: "m15.03 5.03-1.06-1.06L9.5 8.44 7 5.94 3.47 9.47l1.06 1.06L7 8.06l2.5 2.5z"
    })]
  });
}
const ChartLineIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgChartLineIcon
  });
});
ChartLineIcon.displayName = 'ChartLineIcon';
var ChartLineIcon$1 = ChartLineIcon;

function SvgCheckCircleBadgeIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "m10.47 5.47 1.06 1.06L7 11.06 4.47 8.53l1.06-1.06L7 8.94zM16 12.5a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0"
    }), jsx("path", {
      fill: "currentColor",
      d: "M1.5 8a6.5 6.5 0 0 1 13-.084c.54.236 1.031.565 1.452.967Q16 8.448 16 8a8 8 0 1 0-7.117 7.952 5 5 0 0 1-.967-1.453A6.5 6.5 0 0 1 1.5 8"
    })]
  });
}
const CheckCircleBadgeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCheckCircleBadgeIcon
  });
});
CheckCircleBadgeIcon.displayName = 'CheckCircleBadgeIcon';
var CheckCircleBadgeIcon$1 = CheckCircleBadgeIcon;

function SvgCheckCircleFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m11.53-1.47-1.06-1.06L7 8.94 5.53 7.47 4.47 8.53l2 2 .53.53.53-.53z",
      clipRule: "evenodd"
    })
  });
}
const CheckCircleFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCheckCircleFillIcon
  });
});
CheckCircleFillIcon.displayName = 'CheckCircleFillIcon';
var CheckCircleFillIcon$1 = CheckCircleFillIcon;

function SvgCheckCircleIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M11.53 6.53 7 11.06 4.47 8.53l1.06-1.06L7 8.94l3.47-3.47z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13",
      clipRule: "evenodd"
    })]
  });
}
const CheckCircleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCheckCircleIcon
  });
});
CheckCircleIcon.displayName = 'CheckCircleIcon';
var CheckCircleIcon$1 = CheckCircleIcon;

function SvgCheckLineIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M15.06 2.06 14 1 5.53 9.47 2.06 6 1 7.06l4.53 4.531zM1.03 15.03h14v-1.5h-14z",
      clipRule: "evenodd"
    })
  });
}
const CheckLineIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCheckLineIcon
  });
});
CheckLineIcon.displayName = 'CheckLineIcon';
var CheckLineIcon$1 = CheckLineIcon;

function SvgCheckSmallIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M12.03 6.03 7 11.06 3.97 8.03l1.06-1.06L7 8.94l3.97-3.97z",
      clipRule: "evenodd"
    })
  });
}
const CheckSmallIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCheckSmallIcon
  });
});
CheckSmallIcon.displayName = 'CheckSmallIcon';
var CheckSmallIcon$1 = CheckSmallIcon;

function SvgCheckboxIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M1.75 2a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h11.5a.75.75 0 0 0 .75-.75V9h-1.5v4.5h-10v-10H10V2z"
    }), jsx("path", {
      fill: "currentColor",
      d: "m15.03 4.03-1.06-1.06L7.5 9.44 5.53 7.47 4.47 8.53l3.03 3.03z"
    })]
  });
}
const CheckboxIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCheckboxIcon
  });
});
CheckboxIcon.displayName = 'CheckboxIcon';
var CheckboxIcon$1 = CheckboxIcon;

function SvgChecklistIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "m5.5 2 1.06 1.06-3.53 3.531L1 4.561 2.06 3.5l.97.97zM15.03 4.53h-7v-1.5h7zM1.03 14.53v-1.5h14v1.5zM8.03 9.53h7v-1.5h-7zM6.56 8.06 5.5 7 3.03 9.47l-.97-.97L1 9.56l2.03 2.031z"
    })
  });
}
const ChecklistIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgChecklistIcon
  });
});
ChecklistIcon.displayName = 'ChecklistIcon';
var ChecklistIcon$1 = ChecklistIcon;

function SvgChevronDoubleDownIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M10.947 7.954 8 10.891 5.056 7.954 3.997 9.016l4.004 3.993 4.005-3.993z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M10.947 3.994 8 6.931 5.056 3.994 3.997 5.056 8.001 9.05l4.005-3.993z"
    })]
  });
}
const ChevronDoubleDownIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgChevronDoubleDownIcon
  });
});
ChevronDoubleDownIcon.displayName = 'ChevronDoubleDownIcon';
var ChevronDoubleDownIcon$1 = ChevronDoubleDownIcon;

function SvgChevronDoubleLeftIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M8.047 10.944 5.11 8l2.937-2.944-1.062-1.06L2.991 8l3.994 4.003z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M12.008 10.944 9.07 8l2.938-2.944-1.062-1.06L6.952 8l3.994 4.003z"
    })]
  });
}
const ChevronDoubleLeftIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgChevronDoubleLeftIcon
  });
});
ChevronDoubleLeftIcon.displayName = 'ChevronDoubleLeftIcon';
var ChevronDoubleLeftIcon$1 = ChevronDoubleLeftIcon;

function SvgChevronDoubleRightIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "m7.954 5.056 2.937 2.946-2.937 2.945 1.062 1.059 3.993-4.004-3.993-4.005z"
    }), jsx("path", {
      fill: "currentColor",
      d: "m3.994 5.056 2.937 2.946-2.937 2.945 1.062 1.059L9.05 8.002 5.056 3.997z"
    })]
  });
}
const ChevronDoubleRightIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgChevronDoubleRightIcon
  });
});
ChevronDoubleRightIcon.displayName = 'ChevronDoubleRightIcon';
var ChevronDoubleRightIcon$1 = ChevronDoubleRightIcon;

function SvgChevronDoubleUpIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M5.056 8.047 8 5.11l2.944 2.937 1.06-1.062L8 2.991 3.997 6.985z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M5.056 12.008 8 9.07l2.944 2.937 1.06-1.062L8 6.952l-4.003 3.994z"
    })]
  });
}
const ChevronDoubleUpIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgChevronDoubleUpIcon
  });
});
ChevronDoubleUpIcon.displayName = 'ChevronDoubleUpIcon';
var ChevronDoubleUpIcon$1 = ChevronDoubleUpIcon;

function SvgChevronDownIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 8.917 10.947 6 12 7.042 8 11 4 7.042 5.053 6z",
      clipRule: "evenodd"
    })
  });
}
const ChevronDownIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgChevronDownIcon
  });
});
ChevronDownIcon.displayName = 'ChevronDownIcon';
var ChevronDownIcon$1 = ChevronDownIcon;

function SvgChevronUpIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 7.083 5.053 10 4 8.958 8 5l4 3.958L10.947 10z",
      clipRule: "evenodd"
    })
  });
}
const ChevronUpIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgChevronUpIcon
  });
});
ChevronUpIcon.displayName = 'ChevronUpIcon';
var ChevronUpIcon$1 = ChevronUpIcon;

function SvgCircleIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M12.5 8a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0"
    })
  });
}
const CircleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCircleIcon
  });
});
CircleIcon.displayName = 'CircleIcon';
var CircleIcon$1 = CircleIcon;

function SvgCircleOfflineIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m11.667 5.392 1.362-1.363-1.06-1.06-9 9 1.06 1.06 1.363-1.362a4.5 4.5 0 0 0 6.276-6.276m-1.083 1.083-4.11 4.109a3 3 0 0 0 4.11-4.11M8 3.5q.606.002 1.164.152L7.811 5.006A3 3 0 0 0 5.006 7.81L3.652 9.164A4.5 4.5 0 0 1 8 3.5",
      clipRule: "evenodd"
    })
  });
}
const CircleOfflineIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCircleOfflineIcon
  });
});
CircleOfflineIcon.displayName = 'CircleOfflineIcon';
var CircleOfflineIcon$1 = CircleOfflineIcon;

function SvgCircleOutlineIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M5 8a3 3 0 1 0 6 0 3 3 0 0 0-6 0m3-4.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9",
      clipRule: "evenodd"
    })
  });
}
const CircleOutlineIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCircleOutlineIcon
  });
});
CircleOutlineIcon.displayName = 'CircleOutlineIcon';
var CircleOutlineIcon$1 = CircleOutlineIcon;

function SvgClipboardIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M5.5 0a.75.75 0 0 0-.75.75V1h-2a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75h-2V.75A.75.75 0 0 0 10.5 0zm5.75 2.5v.75a.75.75 0 0 1-.75.75h-5a.75.75 0 0 1-.75-.75V2.5H3.5v11h9v-11zm-5 0v-1h3.5v1z",
      clipRule: "evenodd"
    })
  });
}
const ClipboardIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgClipboardIcon
  });
});
ClipboardIcon.displayName = 'ClipboardIcon';
var ClipboardIcon$1 = ClipboardIcon;

function SvgClockKeyIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M8 1.5a6.5 6.5 0 0 0-5.07 10.57l-1.065 1.065A8 8 0 1 1 15.418 11h-1.65A6.5 6.5 0 0 0 8 1.5"
    }), jsx("path", {
      fill: "currentColor",
      d: "M7.25 8V4h1.5v3.25H11v1.5H8A.75.75 0 0 1 7.25 8"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M4 13a3 3 0 0 1 5.959-.5h4.291a.75.75 0 0 1 .75.75V16h-1.5v-2h-1v2H11v-2H9.83A3.001 3.001 0 0 1 4 13m3-1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3",
      clipRule: "evenodd"
    })]
  });
}
const ClockKeyIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgClockKeyIcon
  });
});
ClockKeyIcon.displayName = 'ClockKeyIcon';
var ClockKeyIcon$1 = ClockKeyIcon;

function SvgCloudDatabaseIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M3.394 4.586a4.752 4.752 0 0 1 9.351.946A3.75 3.75 0 0 1 15.787 8H14.12a2.25 2.25 0 0 0-1.871-1H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.475-.402.75.75 0 0 1-.698.657A2.75 2.75 0 0 0 4 11.49V13a.8.8 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6.25 10.5c0-.851.67-1.42 1.293-1.731C8.211 8.435 9.08 8.25 10 8.25s1.79.185 2.457.519c.622.31 1.293.88 1.293 1.731v2.277c-.014.836-.677 1.397-1.293 1.705-.668.333-1.537.518-2.457.518s-1.79-.185-2.457-.518c-.616-.308-1.279-.869-1.293-1.705V10.5m1.964 2.64c.418.209 1.049.36 1.786.36s1.368-.151 1.786-.36c.209-.105.337-.21.406-.29a.3.3 0 0 0 .057-.096l.001-.004v-.423c-.636.273-1.423.423-2.25.423s-1.614-.15-2.25-.423v.427l.005.014a.3.3 0 0 0 .053.082c.069.08.197.185.406.29M7.75 10.5v-.004l.005-.014a.3.3 0 0 1 .053-.082c.069-.08.197-.185.406-.29.418-.209 1.049-.36 1.786-.36s1.368.151 1.786.36c.209.105.337.21.406.29a.3.3 0 0 1 .057.096l.001.004v.004l-.005.014a.3.3 0 0 1-.053.082c-.069.08-.197.185-.406.29-.418.209-1.049.36-1.786.36s-1.368-.151-1.786-.36a1.3 1.3 0 0 1-.406-.29.3.3 0 0 1-.058-.096z",
      clipRule: "evenodd"
    })]
  });
}
const CloudDatabaseIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCloudDatabaseIcon
  });
});
CloudDatabaseIcon.displayName = 'CloudDatabaseIcon';
var CloudDatabaseIcon$1 = CloudDatabaseIcon;

function SvgCloudDownloadIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M8 2a4.75 4.75 0 0 0-4.606 3.586 4.251 4.251 0 0 0 .427 8.393A.8.8 0 0 0 4 14v-1.511a2.75 2.75 0 0 1 .077-5.484.75.75 0 0 0 .697-.657 3.25 3.25 0 0 1 6.476.402v.5c0 .414.336.75.75.75h.25a2.25 2.25 0 1 1-.188 4.492L12 12.49V14l.077-.004q.086.004.173.004a3.75 3.75 0 0 0 .495-7.468A4.75 4.75 0 0 0 8 2"
    }), jsx("path", {
      fill: "currentColor",
      d: "M7.25 11.19 5.03 8.97l-1.06 1.06L8 14.06l4.03-4.03-1.06-1.06-2.22 2.22V6h-1.5z"
    })]
  });
}
const CloudDownloadIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCloudDownloadIcon
  });
});
CloudDownloadIcon.displayName = 'CloudDownloadIcon';
var CloudDownloadIcon$1 = CloudDownloadIcon;

function SvgCloudIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3.394 5.586a4.752 4.752 0 0 1 9.351.946 3.75 3.75 0 0 1-.668 7.464L12 14H4a.8.8 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393m.72 6.914h7.762a.8.8 0 0 1 .186-.008q.092.008.188.008a2.25 2.25 0 0 0 0-4.5H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.475-.402.75.75 0 0 1-.698.657 2.75 2.75 0 0 0-.024 5.488z",
      clipRule: "evenodd"
    })
  });
}
const CloudIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCloudIcon
  });
});
CloudIcon.displayName = 'CloudIcon';
var CloudIcon$1 = CloudIcon;

function SvgCloudKeyIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3.394 5.586a4.752 4.752 0 0 1 9.351.946A3.75 3.75 0 0 1 15.787 9H14.12a2.25 2.25 0 0 0-1.871-1H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.475-.402.75.75 0 0 1-.698.657A2.75 2.75 0 0 0 4 12.49V14a.8.8 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393M15.25 10.5h-4.291a3 3 0 1 0-.13 1.5H12v2h1.5v-2h1v2H16v-2.75a.75.75 0 0 0-.75-.75M8 9.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3",
      clipRule: "evenodd"
    })
  });
}
const CloudKeyIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCloudKeyIcon
  });
});
CloudKeyIcon.displayName = 'CloudKeyIcon';
var CloudKeyIcon$1 = CloudKeyIcon;

function SvgCloudModelIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M3.394 5.586a4.752 4.752 0 0 1 9.351.946A3.75 3.75 0 0 1 15.787 9H14.12a2.25 2.25 0 0 0-1.871-1H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.475-.402.75.75 0 0 1-.698.657A2.75 2.75 0 0 0 4 12.49V14a.8.8 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 7a2.25 2.25 0 0 1 2.03 3.22l.5.5a2.25 2.25 0 1 1-1.06 1.06l-.5-.5A2.25 2.25 0 1 1 8 7m.75 2.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0m3.5 3.5a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0",
      clipRule: "evenodd"
    })]
  });
}
const CloudModelIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCloudModelIcon
  });
});
CloudModelIcon.displayName = 'CloudModelIcon';
var CloudModelIcon$1 = CloudModelIcon;

function SvgCloudOffIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M13.97 14.53 2.47 3.03l-1 1 1.628 1.628a4.252 4.252 0 0 0 .723 8.32A.8.8 0 0 0 4 14h7.44l1.53 1.53zM4.077 7.005a.75.75 0 0 0 .29-.078L9.939 12.5H4.115l-.062-.007a2.75 2.75 0 0 1 .024-5.488",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M4.8 3.24a4.75 4.75 0 0 1 7.945 3.293 3.75 3.75 0 0 1 1.928 6.58l-1.067-1.067A2.25 2.25 0 0 0 12.25 8H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-5.388-2.448z"
    })]
  });
}
const CloudOffIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCloudOffIcon
  });
});
CloudOffIcon.displayName = 'CloudOffIcon';
var CloudOffIcon$1 = CloudOffIcon;

function SvgCloudUploadIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M8 2a4.75 4.75 0 0 0-4.606 3.586 4.251 4.251 0 0 0 .427 8.393A.8.8 0 0 0 4 14v-1.511a2.75 2.75 0 0 1 .077-5.484.75.75 0 0 0 .697-.657 3.25 3.25 0 0 1 6.476.402v.5c0 .414.336.75.75.75h.25a2.25 2.25 0 1 1-.188 4.492L12 12.49V14l.077-.004q.086.004.173.004a3.75 3.75 0 0 0 .495-7.468A4.75 4.75 0 0 0 8 2"
    }), jsx("path", {
      fill: "currentColor",
      d: "m8.75 8.81 2.22 2.22 1.06-1.06L8 5.94 3.97 9.97l1.06 1.06 2.22-2.22V14h1.5z"
    })]
  });
}
const CloudUploadIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCloudUploadIcon
  });
});
CloudUploadIcon.displayName = 'CloudUploadIcon';
var CloudUploadIcon$1 = CloudUploadIcon;

function SvgCodeIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 17 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M4.03 12.06 5.091 11l-2.97-2.97 2.97-2.97L4.031 4 0 8.03zM12.091 4l4.03 4.03-4.03 4.03-1.06-1.06L14 8.03l-2.97-2.97z"
    })
  });
}
const CodeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCodeIcon
  });
});
CodeIcon.displayName = 'CodeIcon';
var CodeIcon$1 = CodeIcon;

function SvgColorFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7.5 1v1.59l4.88 4.88a.75.75 0 0 1 0 1.06l-4.242 4.243a2.75 2.75 0 0 1-3.89 0l-2.421-2.422a2.75 2.75 0 0 1 0-3.889L6 2.29V1zM6 8V4.41L2.888 7.524a1.25 1.25 0 0 0 0 1.768l2.421 2.421a1.25 1.25 0 0 0 1.768 0L10.789 8 7.5 4.71V8zm7.27 1.51a.76.76 0 0 0-1.092.001 8.5 8.5 0 0 0-1.216 1.636c-.236.428-.46.953-.51 1.501-.054.576.083 1.197.587 1.701a2.385 2.385 0 0 0 3.372 0c.505-.504.644-1.126.59-1.703-.05-.55-.274-1.075-.511-1.503a8.5 8.5 0 0 0-1.22-1.633m-.995 2.363c.138-.25.3-.487.451-.689.152.201.313.437.452.687.19.342.306.657.33.913.02.228-.03.377-.158.505a.885.885 0 0 1-1.25 0c-.125-.125-.176-.272-.155-.501.024-.256.14-.572.33-.915",
      clipRule: "evenodd"
    })
  });
}
const ColorFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgColorFillIcon
  });
});
ColorFillIcon.displayName = 'ColorFillIcon';
var ColorFillIcon$1 = ColorFillIcon;

function SvgColumnIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6.5 9V6h3v3zm3 1.5v3h-3v-3zm1.5-.75v-9a.75.75 0 0 0-.75-.75h-4.5A.75.75 0 0 0 5 .75v13.5c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75zM6.5 4.5v-3h3v3z",
      clipRule: "evenodd"
    })
  });
}
const ColumnIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgColumnIcon
  });
});
ColumnIcon.displayName = 'ColumnIcon';
var ColumnIcon$1 = ColumnIcon;

function SvgColumnsIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zM2.5 13.5v-11H5v11zm4 0h3v-11h-3zm4.5-11v11h2.5v-11z",
      clipRule: "evenodd"
    })
  });
}
const ColumnsIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgColumnsIcon
  });
});
ColumnsIcon.displayName = 'ColumnsIcon';
var ColumnsIcon$1 = ColumnsIcon;

function SvgCommandPaletteIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      stroke: "currentColor",
      strokeWidth: 0.25,
      d: "M4.316 6.737H5.38v2.526H4.316c-1.346 0-2.441 1.057-2.441 2.415a2.441 2.441 0 1 0 4.882 0V10.62h2.486v1.057a2.441 2.441 0 1 0 4.882 0c0-1.358-1.094-2.415-2.44-2.415h-1.058V6.737h1.057c1.347 0 2.441-1.057 2.441-2.415a2.441 2.441 0 1 0-4.882 0V5.38H6.757V4.322a2.441 2.441 0 1 0-4.882 0c0 1.358 1.095 2.415 2.44 2.415ZM5.38 4.322v1.073H4.316A1.08 1.08 0 0 1 3.25 4.322c0-.585.485-1.064 1.065-1.064s1.065.479 1.065 1.064Zm7.368 0a1.08 1.08 0 0 1-1.065 1.073h-1.057V4.322c0-.587.479-1.064 1.057-1.064.58 0 1.066.479 1.066 1.064Zm-3.506 2.4v2.557H6.757V6.72zM3.251 11.67a1.08 1.08 0 0 1 1.065-1.073H5.38v1.073c0 .585-.486 1.064-1.065 1.064-.58 0-1.065-.479-1.065-1.064Zm7.376 0v-1.073h1.057a1.08 1.08 0 0 1 1.066 1.073c0 .585-.486 1.064-1.066 1.064a1.065 1.065 0 0 1-1.057-1.064Z"
    })
  });
}
const CommandPaletteIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCommandPaletteIcon
  });
});
CommandPaletteIcon.displayName = 'CommandPaletteIcon';
var CommandPaletteIcon$1 = CommandPaletteIcon;

function SvgConnectIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7.78 3.97 5.03 1.22a.75.75 0 0 0-1.06 0L1.22 3.97a.75.75 0 0 0 0 1.06l2.75 2.75a.75.75 0 0 0 1.06 0l2.75-2.75a.75.75 0 0 0 0-1.06m-1.59.53L4.5 6.19 2.81 4.5 4.5 2.81zM15 11.75a3.25 3.25 0 1 0-6.5 0 3.25 3.25 0 0 0 6.5 0M11.75 10a1.75 1.75 0 1 1 0 3.5 1.75 1.75 0 0 1 0-3.5",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M14.25 1H9v1.5h4.5V7H15V1.75a.75.75 0 0 0-.75-.75M1 9v5.25c0 .414.336.75.75.75H7v-1.5H2.5V9z"
    })]
  });
}
const ConnectIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgConnectIcon
  });
});
ConnectIcon.displayName = 'ConnectIcon';
var ConnectIcon$1 = ConnectIcon;

function SvgCopyIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75H5v3.25c0 .414.336.75.75.75h8.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75H11V1.75a.75.75 0 0 0-.75-.75zM9.5 5V2.5h-7v7H5V5.75A.75.75 0 0 1 5.75 5zm-3 8.5v-7h7v7z",
      clipRule: "evenodd"
    })
  });
}
const CopyIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCopyIcon
  });
});
CopyIcon.displayName = 'CopyIcon';
var CopyIcon$1 = CopyIcon;

function SvgCursorIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#CursorIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M1.22 1.22a.75.75 0 0 1 .802-.169l13.5 5.25a.75.75 0 0 1-.043 1.413L9.597 9.597l-1.883 5.882a.75.75 0 0 1-1.413.043l-5.25-13.5a.75.75 0 0 1 .169-.802m1.847 1.847 3.864 9.937 1.355-4.233a.75.75 0 0 1 .485-.485l4.233-1.355z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M16 0H0v16h16z"
        })
      })
    })]
  });
}
const CursorIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCursorIcon
  });
});
CursorIcon.displayName = 'CursorIcon';
var CursorIcon$1 = CursorIcon;

function SvgCursorTypeIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M8 3.75h1c.69 0 1.25.56 1.25 1.25v6c0 .69-.56 1.25-1.25 1.25H8v1.5h1c.788 0 1.499-.331 2-.863a2.74 2.74 0 0 0 2 .863h1v-1.5h-1c-.69 0-1.25-.56-1.25-1.25V5c0-.69.56-1.25 1.25-1.25h1v-1.5h-1c-.788 0-1.499.331-2 .863a2.74 2.74 0 0 0-2-.863H8z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M5.936 8.003 3 5.058 4.062 4l3.993 4.004-3.993 4.005L3 10.948z"
    })]
  });
}
const CursorTypeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgCursorTypeIcon
  });
});
CursorTypeIcon.displayName = 'CursorTypeIcon';
var CursorTypeIcon$1 = CursorTypeIcon;

function SvgDagIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 1.75A.75.75 0 0 1 8.75 1h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5A.75.75 0 0 1 8 5.25v-1H5.5c-.69 0-1.25.56-1.25 1.25h2a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-2c0 .69.56 1.25 1.25 1.25H8v-1a.75.75 0 0 1 .75-.75h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5a.75.75 0 0 1-.75-.75v-1H5.5a2.75 2.75 0 0 1-2.75-2.75h-2A.75.75 0 0 1 0 9.75v-3.5a.75.75 0 0 1 .75-.75h2A2.75 2.75 0 0 1 5.5 2.75H8zm1.5.75v2h4v-2zM1.5 9V7h4v2zm8 4.5v-2h4v2z",
      clipRule: "evenodd"
    })
  });
}
const DagIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgDagIcon
  });
});
DagIcon.displayName = 'DagIcon';
var DagIcon$1 = DagIcon;

function SvgDIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M5.75 4.5a.75.75 0 0 0-.75.75v5.5c0 .414.336.75.75.75h2a3.5 3.5 0 1 0 0-7zM6.5 10V6h1.25a2 2 0 1 1 0 4z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z",
      clipRule: "evenodd"
    })]
  });
}
const DIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgDIcon
  });
});
DIcon.displayName = 'DIcon';
var DIcon$1 = DIcon;

function SvgDangerFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m15.78 11.533-4.242 4.243a.75.75 0 0 1-.53.22H4.996a.75.75 0 0 1-.53-.22L.224 11.533a.75.75 0 0 1-.22-.53v-6.01a.75.75 0 0 1 .22-.53L4.467.22a.75.75 0 0 1 .53-.22h6.01a.75.75 0 0 1 .53.22l4.243 4.242c.141.141.22.332.22.53v6.011a.75.75 0 0 1-.22.53m-8.528-.785a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0m1.5-5.75v4h-1.5v-4z",
      clipRule: "evenodd"
    })
  });
}
const DangerFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgDangerFillIcon
  });
});
DangerFillIcon.displayName = 'DangerFillIcon';
var DangerFillIcon$1 = DangerFillIcon;

function SvgDashIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M15 8.75H1v-1.5h14z",
      clipRule: "evenodd"
    })
  });
}
const DashIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgDashIcon
  });
});
DashIcon.displayName = 'DashIcon';
var DashIcon$1 = DashIcon;

function SvgDashboardIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5 8.75v3h4.75v-3zm0-1.5h4.75V2.5H2.5zm6.25-6.5v3h4.75v-3zm0 11V7h4.75v6.5z",
      clipRule: "evenodd"
    })
  });
}
const DashboardIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgDashboardIcon
  });
});
DashboardIcon.displayName = 'DashboardIcon';
var DashboardIcon$1 = DashboardIcon;

function SvgDataIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8.646.368a.75.75 0 0 0-1.292 0l-3.25 5.5A.75.75 0 0 0 4.75 7h6.5a.75.75 0 0 0 .646-1.132zM8 2.224 9.936 5.5H6.064zM8.5 9.25a.75.75 0 0 1 .75-.75h5a.75.75 0 0 1 .75.75v5a.75.75 0 0 1-.75.75h-5a.75.75 0 0 1-.75-.75zM10 10v3.5h3.5V10zM1 11.75a3.25 3.25 0 1 1 6.5 0 3.25 3.25 0 0 1-6.5 0M4.25 10a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5",
      clipRule: "evenodd"
    })
  });
}
const DataIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgDataIcon
  });
});
DataIcon.displayName = 'DataIcon';
var DataIcon$1 = DataIcon;

function SvgDatabaseIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.727 3.695c-.225.192-.227.298-.227.305s.002.113.227.305c.223.19.59.394 1.108.58C4.865 5.256 6.337 5.5 8 5.5s3.135-.244 4.165-.615c.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305s-.002-.113-.227-.305c-.223-.19-.59-.394-1.108-.58C11.135 2.744 9.663 2.5 8 2.5s-3.135.244-4.165.615c-.519.186-.885.39-1.108.58M13.5 5.94a7 7 0 0 1-.826.358C11.442 6.74 9.789 7 8 7s-3.442-.26-4.673-.703a7 7 0 0 1-.827-.358V8c0 .007.002.113.227.305.223.19.59.394 1.108.58C4.865 9.256 6.337 9.5 8 9.5s3.135-.244 4.165-.615c.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305zM15 8V4c0-.615-.348-1.1-.755-1.447-.41-.349-.959-.63-1.571-.85C11.442 1.26 9.789 1 8 1s-3.442.26-4.673.703c-.613.22-1.162.501-1.572.85C1.348 2.9 1 3.385 1 4v8c0 .615.348 1.1.755 1.447.41.349.959.63 1.572.85C4.558 14.74 6.21 15 8 15s3.441-.26 4.674-.703c.612-.22 1.161-.501 1.571-.85.407-.346.755-.832.755-1.447zm-1.5 1.939a7 7 0 0 1-.826.358C11.442 10.74 9.789 11 8 11s-3.442-.26-4.673-.703a7 7 0 0 1-.827-.358V12c0 .007.002.113.227.305.223.19.59.394 1.108.58 1.03.371 2.502.615 4.165.615s3.135-.244 4.165-.615c.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305z",
      clipRule: "evenodd"
    })
  });
}
const DatabaseIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgDatabaseIcon
  });
});
DatabaseIcon.displayName = 'DatabaseIcon';
var DatabaseIcon$1 = DatabaseIcon;

function SvgDecimalIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3 10a3 3 0 1 0 6 0V6a3 3 0 0 0-6 0zm3 1.5A1.5 1.5 0 0 1 4.5 10V6a1.5 1.5 0 1 1 3 0v4A1.5 1.5 0 0 1 6 11.5M10 10a3 3 0 1 0 6 0V6a3 3 0 1 0-6 0zm3 1.5a1.5 1.5 0 0 1-1.5-1.5V6a1.5 1.5 0 0 1 3 0v4a1.5 1.5 0 0 1-1.5 1.5",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M1 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2"
    })]
  });
}
const DecimalIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgDecimalIcon
  });
});
DecimalIcon.displayName = 'DecimalIcon';
var DecimalIcon$1 = DecimalIcon;

function SvgDotsCircleIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#DotsCircleIcon_svg__a)",
      children: [jsx("path", {
        d: "M6 8a.75.75 0 1 1-1.5 0A.75.75 0 0 1 6 8M8 8.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5M10.75 8.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0",
        clipRule: "evenodd"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const DotsCircleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgDotsCircleIcon
  });
});
DotsCircleIcon.displayName = 'DotsCircleIcon';
var DotsCircleIcon$1 = DotsCircleIcon;

function SvgDownloadIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M1 13.5h14V15H1zM12.53 6.53l-1.06-1.06-2.72 2.72V1h-1.5v7.19L4.53 5.47 3.47 6.53 8 11.06z"
    })
  });
}
const DownloadIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgDownloadIcon
  });
});
DownloadIcon.displayName = 'DownloadIcon';
var DownloadIcon$1 = DownloadIcon;

function SvgDragIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M5.25 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M10.75 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M5.25 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M10.75 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M5.25 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M10.75 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5"
    })
  });
}
const DragIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgDragIcon
  });
});
DragIcon.displayName = 'DragIcon';
var DragIcon$1 = DragIcon;

function SvgErdIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M10 1.75a.75.75 0 0 1 .75-.75h3.5a.75.75 0 0 1 .75.75v3a.75.75 0 0 1-.75.75h-3.062l-.692.922.004.078v3l-.004.078.691.922h3.063a.75.75 0 0 1 .75.75v3a.75.75 0 0 1-.75.75h-3.5a.75.75 0 0 1-.75-.75v-2.833l-.875-1.167h-2.25L6 11.417v2.833a.75.75 0 0 1-.75.75h-3.5a.75.75 0 0 1-.75-.75v-3a.75.75 0 0 1 .75-.75h3.063l.691-.922A1 1 0 0 1 5.5 9.5v-3q0-.039.004-.078L4.813 5.5H1.75A.75.75 0 0 1 1 4.75v-3A.75.75 0 0 1 1.75 1h3.5a.75.75 0 0 1 .75.75v2.833l.875 1.167h2.25L10 4.583zm1.5.75V4h2V2.5zm0 11V12h2v1.5zM2.5 4V2.5h2V4zm0 8v1.5h2V12zM7 8.75v-1.5h2v1.5z",
      clipRule: "evenodd"
    })
  });
}
const ErdIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgErdIcon
  });
});
ErdIcon.displayName = 'ErdIcon';
var ErdIcon$1 = ErdIcon;

function SvgExpandLessIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 17",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M12.06 1.06 11 0 8.03 2.97 5.06 0 4 1.06l4.03 4.031zM4 15l4.03-4.03L12.06 15 11 16.06l-2.97-2.969-2.97 2.97z"
    })
  });
}
const ExpandLessIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgExpandLessIcon
  });
});
ExpandLessIcon.displayName = 'ExpandLessIcon';
var ExpandLessIcon$1 = ExpandLessIcon;

function SvgExpandMoreIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 17",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "m4 4.03 1.06 1.061 2.97-2.97L11 5.091l1.06-1.06L8.03 0zM12.06 12.091l-4.03 4.03L4 12.091l1.06-1.06L8.03 14 11 11.03z"
    })
  });
}
const ExpandMoreIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgExpandMoreIcon
  });
});
ExpandMoreIcon.displayName = 'ExpandMoreIcon';
var ExpandMoreIcon$1 = ExpandMoreIcon;

function SvgFaceFrownIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#FaceFrownIcon_svg__a)",
      children: [jsx("path", {
        d: "M6 5.25a.75.75 0 0 0 0 1.5h.007a.75.75 0 0 0 0-1.5zM9.25 6a.75.75 0 0 1 .75-.75h.007a.75.75 0 0 1 0 1.5H10A.75.75 0 0 1 9.25 6M10.07 11.12a.75.75 0 0 0 1.197-.903l-.001-.001v-.001l-.003-.003-.005-.006-.015-.02a3 3 0 0 0-.217-.246 4.7 4.7 0 0 0-.626-.546C9.858 9 9.04 8.584 8 8.584s-1.858.416-2.4.81a4.7 4.7 0 0 0-.795.733l-.048.06-.015.019-.005.006-.002.003-.031.044.03-.042a.75.75 0 1 0 1.22.875q.032-.039.103-.115c.096-.1.24-.235.426-.37.375-.273.89-.523 1.517-.523s1.142.25 1.517.523a3.2 3.2 0 0 1 .529.485l.021.025z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M8 .583a7.417 7.417 0 1 0 0 14.834A7.417 7.417 0 0 0 8 .583M2.083 8a5.917 5.917 0 1 1 11.834 0A5.917 5.917 0 0 1 2.083 8",
        clipRule: "evenodd"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const FaceFrownIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFaceFrownIcon
  });
});
FaceFrownIcon.displayName = 'FaceFrownIcon';
var FaceFrownIcon$1 = FaceFrownIcon;

function SvgFaceNeutralIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      fillRule: "evenodd",
      clipPath: "url(#FaceNeutralIcon_svg__a)",
      clipRule: "evenodd",
      children: [jsx("path", {
        d: "M8 2.084a5.917 5.917 0 1 0 0 11.833A5.917 5.917 0 0 0 8 2.084M.583 8a7.417 7.417 0 1 1 14.834 0A7.417 7.417 0 0 1 .583 8"
      }), jsx("path", {
        d: "M4.583 10a.75.75 0 0 1 .75-.75h5.334a.75.75 0 1 1 0 1.5H5.333a.75.75 0 0 1-.75-.75M5.25 6A.75.75 0 0 1 6 5.25h.007a.75.75 0 0 1 0 1.5H6A.75.75 0 0 1 5.25 6M9.25 6a.75.75 0 0 1 .75-.75h.007a.75.75 0 1 1 0 1.5H10A.75.75 0 0 1 9.25 6"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const FaceNeutralIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFaceNeutralIcon
  });
});
FaceNeutralIcon.displayName = 'FaceNeutralIcon';
var FaceNeutralIcon$1 = FaceNeutralIcon;

function SvgFaceSmileIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#FaceSmileIcon_svg__a)",
      children: [jsx("path", {
        fillRule: "evenodd",
        d: "M8 2.084a5.917 5.917 0 1 0 0 11.833A5.917 5.917 0 0 0 8 2.084M.583 8a7.417 7.417 0 1 1 14.834 0A7.417 7.417 0 0 1 .583 8",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M4.883 8.734a.75.75 0 0 1 1.048.146l.002.003.021.026q.032.038.103.114c.096.1.24.235.426.37.375.274.89.524 1.517.524s1.142-.25 1.517-.523a3.2 3.2 0 0 0 .55-.511.75.75 0 0 1 1.2.9l.029-.042-.03.043-.001.002-.002.002-.005.007-.015.019-.048.059q-.06.073-.17.188c-.143.15-.354.348-.626.546-.54.393-1.359.81-2.399.81s-1.858-.417-2.4-.81a4.7 4.7 0 0 1-.795-.734l-.048-.059-.015-.02-.005-.006-.002-.002v-.002h-.002a.75.75 0 0 1 .15-1.05"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M5.25 6A.75.75 0 0 1 6 5.25h.007a.75.75 0 0 1 0 1.5H6A.75.75 0 0 1 5.25 6M9.25 6a.75.75 0 0 1 .75-.75h.007a.75.75 0 1 1 0 1.5H10A.75.75 0 0 1 9.25 6",
        clipRule: "evenodd"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const FaceSmileIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFaceSmileIcon
  });
});
FaceSmileIcon.displayName = 'FaceSmileIcon';
var FaceSmileIcon$1 = FaceSmileIcon;

function SvgFileCodeIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 17",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16H2zm7.5 1.81 1.94 1.94H9.5z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M7.47 9.97 4.44 13l3.03 3.03 1.06-1.06L6.56 13l1.97-1.97zM11.03 9.97l-1.06 1.06L11.94 13l-1.97 1.97 1.06 1.06L14.06 13z"
    })]
  });
}
const FileCodeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFileCodeIcon
  });
});
FileCodeIcon.displayName = 'FileCodeIcon';
var FileCodeIcon$1 = FileCodeIcon;

function SvgFileDocumentIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16H2zm7.5 1.81 1.94 1.94H9.5z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M5 11.5V13h9v-1.5zM14 16H5v-1.5h9z"
    })]
  });
}
const FileDocumentIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFileDocumentIcon
  });
});
FileDocumentIcon.displayName = 'FileDocumentIcon';
var FileDocumentIcon$1 = FileDocumentIcon;

function SvgFileIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53v9a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75zm1.5.75v12h9V7H8.75A.75.75 0 0 1 8 6.25V2.5zm6 1.06 1.94 1.94H9.5z",
      clipRule: "evenodd"
    })
  });
}
const FileIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFileIcon
  });
});
FileIcon.displayName = 'FileIcon';
var FileIcon$1 = FileIcon;

function SvgFileImageIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16H2zm7.5 1.81 1.94 1.94H9.5z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M10.466 10a.75.75 0 0 0-.542.27l-3.75 4.5A.75.75 0 0 0 6.75 16h6.5a.75.75 0 0 0 .75-.75V13.5a.75.75 0 0 0-.22-.53l-2.75-2.75a.75.75 0 0 0-.564-.22m2.034 3.81v.69H8.351l2.2-2.639zM6.5 7.25a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5M5.75 9.5a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0",
      clipRule: "evenodd"
    })]
  });
}
const FileImageIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFileImageIcon
  });
});
FileImageIcon.displayName = 'FileImageIcon';
var FileImageIcon$1 = FileImageIcon;

function SvgFileModelIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.75 1a.75.75 0 0 0-.75.75V16h1.5V2.5H8v3.75c0 .414.336.75.75.75h3.75v3H14V6.25a.75.75 0 0 0-.22-.53l-4.5-4.5A.75.75 0 0 0 8.75 1zm8.69 4.5L9.5 3.56V5.5z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M11.75 11.5a2.25 2.25 0 1 1-2.03 1.28l-.5-.5a2.25 2.25 0 1 1 1.06-1.06l.5.5c.294-.141.623-.22.97-.22m.75 2.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0M8.25 9.5a.75.75 0 1 1 0 1.5.75.75 0 0 1 0-1.5",
      clipRule: "evenodd"
    })]
  });
}
const FileModelIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFileModelIcon
  });
});
FileModelIcon.displayName = 'FileModelIcon';
var FileModelIcon$1 = FileModelIcon;

function SvgFilterIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75V4a.75.75 0 0 1-.22.53L10 9.31v4.94a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1-.75-.75V9.31L1.22 4.53A.75.75 0 0 1 1 4zm1.5.75v1.19l4.78 4.78c.141.14.22.331.22.53v4.5h1V9a.75.75 0 0 1 .22-.53l4.78-4.78V2.5z",
      clipRule: "evenodd"
    })
  });
}
const FilterIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFilterIcon
  });
});
FilterIcon.displayName = 'FilterIcon';
var FilterIcon$1 = FilterIcon;

function SvgFlagPointerIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3 2.5h5.439a.5.5 0 0 1 .39.188l4 5a.5.5 0 0 1 0 .624l-4 5a.5.5 0 0 1-.39.188H3a.5.5 0 0 1-.5-.5V3a.5.5 0 0 1 .5-.5M1 3a2 2 0 0 1 2-2h5.439A2 2 0 0 1 10 1.75l4 5a2 2 0 0 1 0 2.5l-4 5a2 2 0 0 1-1.562.75H3a2 2 0 0 1-2-2zm6 7a2 2 0 1 0 0-4 2 2 0 0 0 0 4",
      clipRule: "evenodd"
    })
  });
}
const FlagPointerIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFlagPointerIcon
  });
});
FlagPointerIcon.displayName = 'FlagPointerIcon';
var FlagPointerIcon$1 = FlagPointerIcon;

function SvgFloatIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M0 5.25A2.25 2.25 0 0 0 2.25 3h1.5v8.5H6V13H0v-1.5h2.25V6c-.627.471-1.406.75-2.25.75zM10 5.75A2.75 2.75 0 0 1 12.75 3h.39a2.86 2.86 0 0 1 1.57 5.252l-2.195 1.44a2.25 2.25 0 0 0-1.014 1.808H16V13h-6v-1.426a3.75 3.75 0 0 1 1.692-3.135l2.194-1.44A1.36 1.36 0 0 0 13.14 4.5h-.389c-.69 0-1.25.56-1.25 1.25V6H10zM8 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2"
    })
  });
}
const FloatIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFloatIcon
  });
});
FloatIcon.displayName = 'FloatIcon';
var FloatIcon$1 = FloatIcon;

function SvgFolderBranchFillIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75H5.5c0-.98.403-1.866 1.05-2.5a3.5 3.5 0 1 1 5.945-2.661 3.5 3.5 0 0 1 1.505-.339c.744 0 1.433.232 2 .627V4.75a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M9.75 12.145a2 2 0 1 1-1.5 0v-1.29a2 2 0 1 1 2.538-.957c.3.585.812 1.017 1.416 1.221a2 2 0 1 1-.096 1.53 4 4 0 0 1-2.358-1.577zM8.5 14a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0m5.5-2.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1M8.5 9a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0",
      clipRule: "evenodd"
    })]
  });
}
const FolderBranchFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFolderBranchFillIcon
  });
});
FolderBranchFillIcon.displayName = 'FolderBranchFillIcon';
var FolderBranchFillIcon$1 = FolderBranchFillIcon;

function SvgFolderBranchIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 0 1 .75.75V8h-1.5V5.5h-7a.75.75 0 0 1-.53-.22L5.555 3.866a1.25 1.25 0 0 0-.883-.366H1.5v9H5V14H.75a.75.75 0 0 1-.75-.75zM9 8.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1M7 9a2 2 0 1 1 3.778.917c.376.58.888 1.031 1.414 1.227a2 2 0 1 1-.072 1.54c-.977-.207-1.795-.872-2.37-1.626v1.087a2 2 0 1 1-1.5 0v-1.29A2 2 0 0 1 7 9m7 2.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1m-5 2a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1",
      clipRule: "evenodd"
    })
  });
}
const FolderBranchIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFolderBranchIcon
  });
});
FolderBranchIcon.displayName = 'FolderBranchIcon';
var FolderBranchIcon$1 = FolderBranchIcon;

function SvgFolderCloudFilledIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#FolderCloudFilledIcon_svg__a)",
      children: [jsx("path", {
        d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 0 1 .75.75v5.02a4.4 4.4 0 0 0-.921-.607 5.11 5.11 0 0 0-9.512-.753A4.75 4.75 0 0 0 2.917 14H.75a.75.75 0 0 1-.75-.75z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M6.715 9.595a3.608 3.608 0 0 1 7.056.688C15.07 10.572 16 11.739 16 13.107 16 14.688 14.757 16 13.143 16H7.32a.8.8 0 0 1-.163-.018 3.25 3.25 0 0 1-.443-6.387m.703 4.905-.044-.004a1.75 1.75 0 0 1 .015-3.493.75.75 0 0 0 .698-.657 2.108 2.108 0 0 1 4.199.261v.357c0 .415.335.75.75.75h.107c.753 0 1.357.607 1.357 1.393s-.604 1.393-1.357 1.393q-.036 0-.07-.002a1 1 0 0 0-.1.002z",
        clipRule: "evenodd"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const FolderCloudFilledIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFolderCloudFilledIcon
  });
});
FolderCloudFilledIcon.displayName = 'FolderCloudFilledIcon';
var FolderCloudFilledIcon$1 = FolderCloudFilledIcon;

function SvgFolderCloudIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#FolderCloudIcon_svg__a)",
      children: [jsx("path", {
        d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75H3v-1.5H1.5v-9h3.172c.331 0 .649.132.883.366L6.97 5.28c.14.141.331.22.53.22h7V8H16V4.75a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M10.179 7a3.61 3.61 0 0 0-3.464 2.595 3.251 3.251 0 0 0 .443 6.387.8.8 0 0 0 .163.018h5.821C14.758 16 16 14.688 16 13.107c0-1.368-.931-2.535-2.229-2.824A3.61 3.61 0 0 0 10.18 7m-2.805 7.496q.023 0 .044.004h5.555a1 1 0 0 1 .1-.002l.07.002c.753 0 1.357-.607 1.357-1.393s-.604-1.393-1.357-1.393h-.107a.75.75 0 0 1-.75-.75v-.357a2.107 2.107 0 0 0-4.199-.26.75.75 0 0 1-.698.656 1.75 1.75 0 0 0-.015 3.493",
        clipRule: "evenodd"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const FolderCloudIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFolderCloudIcon
  });
});
FolderCloudIcon.displayName = 'FolderCloudIcon';
var FolderCloudIcon$1 = FolderCloudIcon;

function SvgFolderFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2z"
    })
  });
}
const FolderFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFolderFillIcon
  });
});
FolderFillIcon.displayName = 'FolderFillIcon';
var FolderFillIcon$1 = FolderFillIcon;

function SvgFolderIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 0 1 .75.75v8.5a.75.75 0 0 1-.75.75H.75a.75.75 0 0 1-.75-.75zm1.5.75v9h13v-7h-7a.75.75 0 0 1-.53-.22L5.555 3.866a1.25 1.25 0 0 0-.883-.366z",
      clipRule: "evenodd"
    })
  });
}
const FolderIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFolderIcon
  });
});
FolderIcon.displayName = 'FolderIcon';
var FolderIcon$1 = FolderIcon;

function SvgFontIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#FontIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M5.197 3.473a.75.75 0 0 0-1.393-.001L-.006 13H1.61l.6-1.5h4.562l.596 1.5h1.614zM6.176 10 4.498 5.776 2.809 10zm4.07-2.385c.593-.205 1.173-.365 1.754-.365a1.5 1.5 0 0 1 1.42 1.014A3.8 3.8 0 0 0 12 8c-.741 0-1.47.191-2.035.607A2.3 2.3 0 0 0 9 10.5c0 .81.381 1.464.965 1.893.565.416 1.294.607 2.035.607.524 0 1.042-.096 1.5-.298V13H15V8.75a3 3 0 0 0-3-3c-.84 0-1.614.23-2.245.448zM13.5 10.5a.8.8 0 0 0-.353-.685C12.897 9.631 12.5 9.5 12 9.5c-.5 0-.897.131-1.146.315a.8.8 0 0 0-.354.685c0 .295.123.515.354.685.25.184.645.315 1.146.315.502 0 .897-.131 1.147-.315.23-.17.353-.39.353-.685",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const FontIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFontIcon
  });
});
FontIcon.displayName = 'FontIcon';
var FontIcon$1 = FontIcon;

function SvgForkIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2 2.75a2.75 2.75 0 1 1 3.5 2.646V6.75h3.75A2.75 2.75 0 0 1 12 9.5v.104a2.751 2.751 0 1 1-1.5 0V9.5c0-.69-.56-1.25-1.25-1.25H5.5v1.354a2.751 2.751 0 1 1-1.5 0V5.396A2.75 2.75 0 0 1 2 2.75M4.75 1.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5M3.5 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0m6.5 0a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0",
      clipRule: "evenodd"
    })
  });
}
const ForkIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgForkIcon
  });
});
ForkIcon.displayName = 'ForkIcon';
var ForkIcon$1 = ForkIcon;

function SvgFullscreenExitIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M6 1v4.25a.75.75 0 0 1-.75.75H1V4.5h3.5V1zM10 15v-4.25a.75.75 0 0 1 .75-.75H15v1.5h-3.5V15zM10.75 6H15V4.5h-3.5V1H10v4.25c0 .414.336.75.75.75M1 10h4.25a.75.75 0 0 1 .75.75V15H4.5v-3.5H1z"
    })
  });
}
const FullscreenExitIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFullscreenExitIcon
  });
});
FullscreenExitIcon.displayName = 'FullscreenExitIcon';
var FullscreenExitIcon$1 = FullscreenExitIcon;

function SvgFullscreenIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M6 1H1.75a.75.75 0 0 0-.75.75V6h1.5V2.5H6zM10 2.5V1h4.25a.75.75 0 0 1 .75.75V6h-1.5V2.5zM10 13.5h3.5V10H15v4.25a.75.75 0 0 1-.75.75H10zM2.5 10v3.5H6V15H1.75a.75.75 0 0 1-.75-.75V10z"
    })
  });
}
const FullscreenIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFullscreenIcon
  });
});
FullscreenIcon.displayName = 'FullscreenIcon';
var FullscreenIcon$1 = FullscreenIcon;

function SvgFunctionIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#FunctionIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M9.93 2.988c-.774-.904-2.252-.492-2.448.682L7.094 6h2.005a2.75 2.75 0 0 1 2.585 1.81l.073.202 2.234-2.063 1.018 1.102-2.696 2.489.413 1.137c.18.494.65.823 1.175.823H15V13h-1.1a2.75 2.75 0 0 1-2.585-1.81l-.198-.547-2.61 2.408-1.017-1.102 3.07-2.834-.287-.792A1.25 1.25 0 0 0 9.099 7.5H6.844l-.846 5.076c-.405 2.43-3.464 3.283-5.067 1.412l1.139-.976c.774.904 2.252.492 2.448-.682l.805-4.83H3V6h2.573l.43-2.576C6.407.994 9.465.14 11.07 2.012z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M16 0H0v16h16z"
        })
      })
    })]
  });
}
const FunctionIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgFunctionIcon
  });
});
FunctionIcon.displayName = 'FunctionIcon';
var FunctionIcon$1 = FunctionIcon;

function SvgGearFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7.966 0q-.51 0-1.005.063a.75.75 0 0 0-.62.51l-.639 1.946q-.315.13-.61.294L3.172 2.1a.75.75 0 0 0-.784.165c-.481.468-.903.996-1.255 1.572a.75.75 0 0 0 .013.802l1.123 1.713a6 6 0 0 0-.15.66L.363 8.07a.75.75 0 0 0-.36.716c.067.682.22 1.34.447 1.962a.75.75 0 0 0 .635.489l2.042.19q.195.276.422.529l-.27 2.032a.75.75 0 0 0 .336.728 8 8 0 0 0 1.812.874.75.75 0 0 0 .778-.192l1.422-1.478a6 6 0 0 0 .677 0l1.422 1.478a.75.75 0 0 0 .778.192 8 8 0 0 0 1.812-.874.75.75 0 0 0 .335-.728l-.269-2.032a6 6 0 0 0 .422-.529l2.043-.19a.75.75 0 0 0 .634-.49c.228-.621.38-1.279.447-1.961a.75.75 0 0 0-.36-.716l-1.756-1.056a6 6 0 0 0-.15-.661l1.123-1.713a.75.75 0 0 0 .013-.802 8 8 0 0 0-1.255-1.572.75.75 0 0 0-.784-.165l-1.92.713q-.295-.163-.61-.294L9.589.573a.75.75 0 0 0-.619-.51A8 8 0 0 0 7.965 0m.018 10.25a2.25 2.25 0 1 0 0-4.5 2.25 2.25 0 0 0 0 4.5",
      clipRule: "evenodd"
    })
  });
}
const GearFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgGearFillIcon
  });
});
GearFillIcon.displayName = 'GearFillIcon';
var GearFillIcon$1 = GearFillIcon;

function SvgGearIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      fillRule: "evenodd",
      clipPath: "url(#GearIcon_svg__a)",
      clipRule: "evenodd",
      children: [jsx("path", {
        d: "M7.984 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6m-1.5 3a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0"
      }), jsx("path", {
        d: "M7.966 0q-.51 0-1.005.063a.75.75 0 0 0-.62.51l-.639 1.946q-.315.13-.61.294L3.172 2.1a.75.75 0 0 0-.784.165c-.481.468-.903.996-1.255 1.572a.75.75 0 0 0 .013.802l1.123 1.713a6 6 0 0 0-.15.66L.363 8.07a.75.75 0 0 0-.36.716c.067.682.22 1.34.447 1.962a.75.75 0 0 0 .635.489l2.042.19q.195.276.422.529l-.27 2.032a.75.75 0 0 0 .336.728 8 8 0 0 0 1.812.874.75.75 0 0 0 .778-.192l1.422-1.478a6 6 0 0 0 .677 0l1.422 1.478a.75.75 0 0 0 .778.192 8 8 0 0 0 1.812-.874.75.75 0 0 0 .335-.728l-.269-2.032a6 6 0 0 0 .422-.529l2.043-.19a.75.75 0 0 0 .634-.49c.228-.621.38-1.279.447-1.961a.75.75 0 0 0-.36-.716l-1.756-1.056a6 6 0 0 0-.15-.661l1.123-1.713a.75.75 0 0 0 .013-.802 8 8 0 0 0-1.255-1.572.75.75 0 0 0-.784-.165l-1.92.713q-.295-.163-.61-.294L9.589.573a.75.75 0 0 0-.619-.51A8 8 0 0 0 7.965 0m-.95 3.328.597-1.819a7 7 0 0 1 .705 0l.597 1.819a.75.75 0 0 0 .472.476q.519.177.97.468a.75.75 0 0 0 .668.073l1.795-.668q.234.264.44.552l-1.05 1.6a.75.75 0 0 0-.078.667q.181.501.24 1.05a.75.75 0 0 0 .359.567l1.642.988q-.06.351-.156.687l-1.909.178a.75.75 0 0 0-.569.353q-.287.463-.672.843a.75.75 0 0 0-.219.633l.252 1.901a7 7 0 0 1-.635.306l-1.33-1.381a.75.75 0 0 0-.63-.225 4.5 4.5 0 0 1-1.08 0 .75.75 0 0 0-.63.225l-1.33 1.381a7 7 0 0 1-.634-.306l.252-1.9a.75.75 0 0 0-.219-.634 4.5 4.5 0 0 1-.672-.843.75.75 0 0 0-.569-.353l-1.909-.178a7 7 0 0 1-.156-.687L3.2 8.113a.75.75 0 0 0 .36-.567q.056-.549.239-1.05a.75.75 0 0 0-.078-.666L2.67 4.229q.206-.288.44-.552l1.795.668a.75.75 0 0 0 .667-.073c.3-.193.626-.351.97-.468a.75.75 0 0 0 .472-.476"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const GearIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgGearIcon
  });
});
GearIcon.displayName = 'GearIcon';
var GearIcon$1 = GearIcon;

function SvgGiftIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3 3.25A2.25 2.25 0 0 1 5.25 1C6.365 1 7.36 1.522 8 2.335A3.5 3.5 0 0 1 10.75 1a2.25 2.25 0 0 1 2.122 3h1.378a.75.75 0 0 1 .75.75v3a.75.75 0 0 1-.75.75H14v5.75a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75V8.5h-.25A.75.75 0 0 1 1 7.75v-3A.75.75 0 0 1 1.75 4h1.378A2.3 2.3 0 0 1 3 3.25M5.25 4h1.937A2 2 0 0 0 5.25 2.5a.75.75 0 0 0 0 1.5m2 1.5H2.5V7h4.75zm0 3H3.5v5h3.75zm1.5 5v-5h3.75v5zm0-6.5V5.5h4.75V7zm.063-3h1.937a.75.75 0 0 0 0-1.5A2 2 0 0 0 8.813 4",
      clipRule: "evenodd"
    })
  });
}
const GiftIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgGiftIcon
  });
});
GiftIcon.displayName = 'GiftIcon';
var GiftIcon$1 = GiftIcon;

function SvgGitCommitIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 5.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5M4.07 7.25a4.001 4.001 0 0 1 7.86 0H16v1.5h-4.07a4.001 4.001 0 0 1-7.86 0H0v-1.5z",
      clipRule: "evenodd"
    })
  });
}
const GitCommitIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgGitCommitIcon
  });
});
GitCommitIcon.displayName = 'GitCommitIcon';
var GitCommitIcon$1 = GitCommitIcon;

function SvgGlobeIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#GlobeIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m5.354-5.393q.132-.347.287-.666A6.51 6.51 0 0 0 1.543 7.25h2.971c.067-1.777.368-3.399.84-4.643m.661 4.643c.066-1.627.344-3.062.742-4.11.23-.607.485-1.046.73-1.32.247-.274.421-.32.513-.32s.266.046.512.32.501.713.731 1.32c.398 1.048.676 2.483.742 4.11zm3.97 1.5h-3.97c.066 1.627.344 3.062.742 4.11.23.607.485 1.046.73 1.32.247.274.421.32.513.32s.266-.046.512-.32.501-.713.731-1.32c.398-1.048.676-2.483.742-4.11m1.501-1.5c-.067-1.777-.368-3.399-.84-4.643a8 8 0 0 0-.287-.666 6.51 6.51 0 0 1 4.098 5.309zm2.971 1.5h-2.971c-.067 1.777-.368 3.399-.84 4.643a8 8 0 0 1-.287.666 6.51 6.51 0 0 0 4.098-5.309m-9.943 0H1.543a6.51 6.51 0 0 0 4.098 5.309 8 8 0 0 1-.287-.666c-.472-1.244-.773-2.866-.84-4.643",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 16h16V0H0z"
        })
      })
    })]
  });
}
const GlobeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgGlobeIcon
  });
});
GlobeIcon.displayName = 'GlobeIcon';
var GlobeIcon$1 = GlobeIcon;

function SvgGridDashIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M1 1.75V4h1.5V2.5H4V1H1.75a.75.75 0 0 0-.75.75M15 14.25V12h-1.5v1.5H12V15h2.25a.75.75 0 0 0 .75-.75M12 1h2.25a.75.75 0 0 1 .75.75V4h-1.5V2.5H12zM1.75 15H4v-1.5H2.5V12H1v2.25a.75.75 0 0 0 .75.75M10 2.5H6V1h4zM6 15h4v-1.5H6zM13.5 10V6H15v4zM1 6v4h1.5V6z"
    })
  });
}
const GridDashIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgGridDashIcon
  });
});
GridDashIcon.displayName = 'GridDashIcon';
var GridDashIcon$1 = GridDashIcon;

function SvgGridIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v4.5c0 .414.336.75.75.75h4.5A.75.75 0 0 0 7 6.25v-4.5A.75.75 0 0 0 6.25 1zm.75 4.5v-3h3v3zM1.75 9a.75.75 0 0 0-.75.75v4.5c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-4.5A.75.75 0 0 0 6.25 9zm.75 4.5v-3h3v3zM9 1.75A.75.75 0 0 1 9.75 1h4.5a.75.75 0 0 1 .75.75v4.49a.75.75 0 0 1-.75.75h-4.5A.75.75 0 0 1 9 6.24zm1.5.75v2.99h3V2.5zM9.75 9a.75.75 0 0 0-.75.75v4.5c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-4.5a.75.75 0 0 0-.75-.75zm.75 4.5v-3h3v3z",
      clipRule: "evenodd"
    })
  });
}
const GridIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgGridIcon
  });
});
GridIcon.displayName = 'GridIcon';
var GridIcon$1 = GridIcon;

function SvgH1Icon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M1 3v10h1.5V8.75H6V13h1.5V3H6v4.25H2.5V3zM11.25 3A2.25 2.25 0 0 1 9 5.25v1.5c.844 0 1.623-.279 2.25-.75v5.5H9V13h6v-1.5h-2.25V3z"
    })
  });
}
const H1Icon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgH1Icon
  });
});
H1Icon.displayName = 'H1Icon';
var H1Icon$1 = H1Icon;

function SvgH2Icon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M1 3v10h1.5V8.75H6V13h1.5V3H6v4.25H2.5V3zM11.75 3A2.75 2.75 0 0 0 9 5.75V6h1.5v-.25c0-.69.56-1.25 1.25-1.25h.39a1.36 1.36 0 0 1 .746 2.498L10.692 8.44A3.75 3.75 0 0 0 9 11.574V13h6v-1.5h-4.499a2.25 2.25 0 0 1 1.014-1.807l2.194-1.44A2.86 2.86 0 0 0 12.14 3z"
    })
  });
}
const H2Icon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgH2Icon
  });
});
H2Icon.displayName = 'H2Icon';
var H2Icon$1 = H2Icon;

function SvgH3Icon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M1 3h1.5v4.25H6V3h1.5v10H6V8.75H2.5V13H1zM9 5.75A2.75 2.75 0 0 1 11.75 3h.375a2.875 2.875 0 0 1 1.937 5 2.875 2.875 0 0 1-1.937 5h-.375A2.75 2.75 0 0 1 9 10.25V10h1.5v.25c0 .69.56 1.25 1.25 1.25h.375a1.375 1.375 0 1 0 0-2.75H11v-1.5h1.125a1.375 1.375 0 1 0 0-2.75h-.375c-.69 0-1.25.56-1.25 1.25V6H9z"
    })
  });
}
const H3Icon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgH3Icon
  });
});
H3Icon.displayName = 'H3Icon';
var H3Icon$1 = H3Icon;

function SvgHistoryIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#HistoryIcon_svg__a)",
      children: [jsx("path", {
        d: "m3.507 7.73.963-.962 1.06 1.06-2.732 2.732L-.03 7.732l1.06-1.06.979.978a7 7 0 1 1 2.041 5.3l1.061-1.06a5.5 5.5 0 1 0-1.604-4.158"
      }), jsx("path", {
        d: "M8.25 8V4h1.5v3.69l1.78 1.78-1.06 1.06-2-2A.75.75 0 0 1 8.25 8"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const HistoryIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgHistoryIcon
  });
});
HistoryIcon.displayName = 'HistoryIcon';
var HistoryIcon$1 = HistoryIcon;

function SvgHomeIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7.625 1.1a.75.75 0 0 1 .75 0l6.25 3.61a.75.75 0 0 1 .375.65v8.89a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75V10H7v4.25a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75V5.355a.75.75 0 0 1 .375-.65zM2.5 5.79V13.5h3V9.25a.75.75 0 0 1 .75-.75h3.5a.75.75 0 0 1 .75.75v4.25h3V5.792L8 2.616z",
      clipRule: "evenodd"
    })
  });
}
const HomeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgHomeIcon
  });
});
HomeIcon.displayName = 'HomeIcon';
var HomeIcon$1 = HomeIcon;

function SvgImageIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6.25 3.998a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5m-.75 2.25a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.492a.75.75 0 0 1-.75.75H5.038l-.009.009-.008-.009H1.75a.75.75 0 0 1-.75-.75zm12.5 11.742H6.544l4.455-4.436 2.47 2.469.031-.03zm0-10.992v6.934l-1.97-1.968a.75.75 0 0 0-1.06-.001l-6.052 6.027H2.5V2.5z",
      clipRule: "evenodd"
    })]
  });
}
const ImageIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgImageIcon
  });
});
ImageIcon.displayName = 'ImageIcon';
var ImageIcon$1 = ImageIcon;

function SvgIndentDecreaseIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M16 2H0v1.5h16zM16 5.5H7V7h9zM16 9H7v1.5h9zM16 12.5H0V14h16zM3.97 11.03.94 8l3.03-3.03 1.06 1.06L3.06 8l1.97 1.97z"
    })
  });
}
const IndentDecreaseIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgIndentDecreaseIcon
  });
});
IndentDecreaseIcon.displayName = 'IndentDecreaseIcon';
var IndentDecreaseIcon$1 = IndentDecreaseIcon;

function SvgIndentIncreaseIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M16 2H0v1.5h16zM16 5.5H7V7h9zM16 9H7v1.5h9zM16 12.5H0V14h16zM2.03 4.97 5.06 8l-3.03 3.03L.97 9.97 2.94 8 .97 6.03z"
    })
  });
}
const IndentIncreaseIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgIndentIncreaseIcon
  });
});
IndentIncreaseIcon.displayName = 'IndentIncreaseIcon';
var IndentIncreaseIcon$1 = IndentIncreaseIcon;

function SvgInfinityIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m8 6.94 1.59-1.592a3.75 3.75 0 1 1 0 5.304L8 9.06l-1.591 1.59a3.75 3.75 0 1 1 0-5.303zm2.652-.531a2.25 2.25 0 1 1 0 3.182L9.06 8zM6.939 8 5.35 6.409a2.25 2.25 0 1 0 0 3.182l1.588-1.589z",
      clipRule: "evenodd"
    })
  });
}
const InfinityIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgInfinityIcon
  });
});
InfinityIcon.displayName = 'InfinityIcon';
var InfinityIcon$1 = InfinityIcon;

function SvgInfoBookIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M11.75 4.5a.75.75 0 1 1 0 1.5.75.75 0 0 1 0-1.5M12.5 7.75a.75.75 0 0 0-.75-.75h-1.5v1.5H11V11h1.5z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 2.75A.75.75 0 0 1 .75 2h4.396A3.75 3.75 0 0 1 8 3.317 3.75 3.75 0 0 1 10.854 2h4.396a.75.75 0 0 1 .75.75v10.5a.75.75 0 0 1-.75.75h-4.396a2.25 2.25 0 0 0-2.012 1.244l-.171.341a.75.75 0 0 1-1.342 0l-.17-.341A2.25 2.25 0 0 0 5.145 14H.75a.75.75 0 0 1-.75-.75zm1.5.75v9h3.646c.765 0 1.494.233 2.104.646V4.927l-.092-.183A2.25 2.25 0 0 0 5.146 3.5zm7.25 1.427v8.219a3.75 3.75 0 0 1 2.104-.646H14.5v-9h-3.646a2.25 2.25 0 0 0-2.012 1.244z",
      clipRule: "evenodd"
    })]
  });
}
const InfoBookIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgInfoBookIcon
  });
});
InfoBookIcon.displayName = 'InfoBookIcon';
var InfoBookIcon$1 = InfoBookIcon;

function SvgInfoFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0m-8.75 3V7h1.5v4zM8 4.5A.75.75 0 1 1 8 6a.75.75 0 0 1 0-1.5",
      clipRule: "evenodd"
    })
  });
}
const InfoFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgInfoFillIcon
  });
});
InfoFillIcon.displayName = 'InfoFillIcon';
var InfoFillIcon$1 = InfoFillIcon;

function SvgIngestionIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M15 2.5a.75.75 0 0 0-.75-.75h-3a.75.75 0 0 0-.75.75V6H12V3.25h1.5v9.5H12V10h-1.5v3.5c0 .414.336.75.75.75h3a.75.75 0 0 0 .75-.75z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3.75 0c1.26 0 2.322.848 2.648 2.004A2.75 2.75 0 0 1 9 4.75v2.5h3v1.5H9v2.5a2.75 2.75 0 0 1-2.602 2.746 2.751 2.751 0 1 1-3.47-3.371 2.751 2.751 0 0 1 0-5.25A2.751 2.751 0 0 1 3.75 0M5 2.75a1.25 1.25 0 1 0-2.5 0 1.25 1.25 0 0 0 2.5 0m-.428 2.625a2.76 2.76 0 0 0 1.822-1.867A1.25 1.25 0 0 1 7.5 4.75v2.5H6.396a2.76 2.76 0 0 0-1.824-1.875M6.396 8.75H7.5v2.5a1.25 1.25 0 0 1-1.106 1.242 2.76 2.76 0 0 0-1.822-1.867A2.76 2.76 0 0 0 6.396 8.75M3.75 12a1.25 1.25 0 1 1 0 2.5 1.25 1.25 0 0 1 0-2.5m0-5.25a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5",
      clipRule: "evenodd"
    })]
  });
}
const IngestionIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgIngestionIcon
  });
});
IngestionIcon.displayName = 'IngestionIcon';
var IngestionIcon$1 = IngestionIcon;

function SvgItalicIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M9.648 4.5H12V3H6v1.5h2.102l-1.75 7H4V13h6v-1.5H7.898z",
      clipRule: "evenodd"
    })
  });
}
const ItalicIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgItalicIcon
  });
});
ItalicIcon.displayName = 'ItalicIcon';
var ItalicIcon$1 = ItalicIcon;

function SvgKeyIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a4 4 0 0 1 7.93-.75h7.32A.75.75 0 0 1 16 8v3h-1.5V8.75H13V11h-1.5V8.75H7.93A4.001 4.001 0 0 1 0 8m4-2.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5",
      clipRule: "evenodd"
    })
  });
}
const KeyIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgKeyIcon
  });
});
KeyIcon.displayName = 'KeyIcon';
var KeyIcon$1 = KeyIcon;

function SvgKeyboardIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75zm.75 10.5v-9h13v9zm2.75-8h-1.5V6h1.5zm1.5 0V6h1.5V4.5zm3 0V6h1.5V4.5zm3 0V6h1.5V4.5zm-1.5 2.75h-1.5v1.5h1.5zm1.5 1.5v-1.5h1.5v1.5zm-4.5 0v-1.5h-1.5v1.5zm-3 0v-1.5h-1.5v1.5zM11 10H5v1.5h6z",
      clipRule: "evenodd"
    })
  });
}
const KeyboardIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgKeyboardIcon
  });
});
KeyboardIcon.displayName = 'KeyboardIcon';
var KeyboardIcon$1 = KeyboardIcon;

function SvgLayerGraphIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 3.75a2.75 2.75 0 1 1 3.5 2.646v3.208c.916.259 1.637.98 1.896 1.896h3.208a2.751 2.751 0 1 1 0 1.5H6.396A2.751 2.751 0 1 1 3 9.604V6.396A2.75 2.75 0 0 1 1 3.75m11.25 9.75a1.25 1.25 0 1 1 0-2.5 1.25 1.25 0 0 1 0 2.5m-8.5-11a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m0 8.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M10 1.5h3.75a.75.75 0 0 1 .75.75V6H13V3h-3z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7.75 4a.75.75 0 0 0-.75.75v3.5c0 .414.336.75.75.75h3.5a.75.75 0 0 0 .75-.75v-3.5a.75.75 0 0 0-.75-.75zm.75 3.5v-2h2v2z",
      clipRule: "evenodd"
    })]
  });
}
const LayerGraphIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgLayerGraphIcon
  });
});
LayerGraphIcon.displayName = 'LayerGraphIcon';
var LayerGraphIcon$1 = LayerGraphIcon;

function SvgLayerIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M13.5 2.5H7V1h7.25a.75.75 0 0 1 .75.75V9h-1.5z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 7.75A.75.75 0 0 1 1.75 7h6.5a.75.75 0 0 1 .75.75v6.5a.75.75 0 0 1-.75.75h-6.5a.75.75 0 0 1-.75-.75zm1.5.75v5h5v-5z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M4 5.32h6.5V12H12V4.57a.75.75 0 0 0-.75-.75H4z"
    })]
  });
}
const LayerIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgLayerIcon
  });
});
LayerIcon.displayName = 'LayerIcon';
var LayerIcon$1 = LayerIcon;

function SvgLeafIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M5 6a2.75 2.75 0 0 0 2.75 2.75h2.395a2 2 0 1 0 0-1.5H7.75C7.06 7.25 6.5 6.69 6.5 6V2H5zm6.5 2a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0",
      clipRule: "evenodd"
    })
  });
}
const LeafIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgLeafIcon
  });
});
LeafIcon.displayName = 'LeafIcon';
var LeafIcon$1 = LeafIcon;

function SvgLettersIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6.25 1h2.174a2.126 2.126 0 0 1 1.81 3.243 2.126 2.126 0 0 1-1.36 3.761H6.25a.75.75 0 0 1-.75-.75V1.75A.75.75 0 0 1 6.25 1M7 6.504V5.252h1.874a.626.626 0 1 1 0 1.252zm2.05-3.378c0 .345-.28.625-.625.626H7.001L7 2.5h1.424c.346 0 .626.28.626.626M3.307 6a.75.75 0 0 1 .697.473L6.596 13H4.982l-.238-.6H1.855l-.24.6H0l2.61-6.528A.75.75 0 0 1 3.307 6m-.003 2.776.844 2.124H2.455z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M12.5 15a2.5 2.5 0 0 0 2.5-2.5h-1.5a1 1 0 1 1-2 0v-1.947c0-.582.472-1.053 1.053-1.053.523 0 .947.424.947.947v.053H15v-.053A2.447 2.447 0 0 0 12.553 8 2.553 2.553 0 0 0 10 10.553V12.5a2.5 2.5 0 0 0 2.5 2.5"
    })]
  });
}
const LettersIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgLettersIcon
  });
});
LettersIcon.displayName = 'LettersIcon';
var LettersIcon$1 = LettersIcon;

function SvgLibrariesIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "m8.301 1.522 5.25 13.5 1.398-.544-5.25-13.5zM1 15V1h1.5v14zM5 15V1h1.5v14z"
    })
  });
}
const LibrariesIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgLibrariesIcon
  });
});
LibrariesIcon.displayName = 'LibrariesIcon';
var LibrariesIcon$1 = LibrariesIcon;

function SvgLightningIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M9.49.04a.75.75 0 0 1 .51.71V6h3.25a.75.75 0 0 1 .596 1.206l-6.5 8.5A.75.75 0 0 1 6 15.25V10H2.75a.75.75 0 0 1-.596-1.206l6.5-8.5A.75.75 0 0 1 9.491.04M4.269 8.5H6.75a.75.75 0 0 1 .75.75v3.785L11.732 7.5H9.25a.75.75 0 0 1-.75-.75V2.965z",
      clipRule: "evenodd"
    })
  });
}
const LightningIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgLightningIcon
  });
});
LightningIcon.displayName = 'LightningIcon';
var LightningIcon$1 = LightningIcon;

function SvgLinkIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M4 4h3v1.5H4a2.5 2.5 0 0 0 0 5h3V12H4a4 4 0 0 1 0-8M12 10.5H9V12h3a4 4 0 0 0 0-8H9v1.5h3a2.5 2.5 0 0 1 0 5"
    }), jsx("path", {
      fill: "currentColor",
      d: "M4 8.75h8v-1.5H4z"
    })]
  });
}
const LinkIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgLinkIcon
  });
});
LinkIcon.displayName = 'LinkIcon';
var LinkIcon$1 = LinkIcon;

function SvgLinkOffIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M14.035 11.444A4 4 0 0 0 12 4H9v1.5h3a2.5 2.5 0 0 1 .917 4.826zM14 13.53 2.47 2l-1 1 1.22 1.22A4.002 4.002 0 0 0 4 12h3v-1.5H4a2.5 2.5 0 0 1-.03-5l1.75 1.75H4v1.5h3.22L13 14.53z"
    }), jsx("path", {
      fill: "currentColor",
      d: "m9.841 7.25 1.5 1.5H12v-1.5z"
    })]
  });
}
const LinkOffIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgLinkOffIcon
  });
});
LinkOffIcon.displayName = 'LinkOffIcon';
var LinkOffIcon$1 = LinkOffIcon;

function SvgListBorderIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M12 8.75H7v-1.5h5zM7 5.5h5V4H7zM12 12H7v-1.5h5zM4.75 5.5a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5M5.5 8A.75.75 0 1 1 4 8a.75.75 0 0 1 1.5 0M4.75 12a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75v11h11v-11z",
      clipRule: "evenodd"
    })]
  });
}
const ListBorderIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgListBorderIcon
  });
});
ListBorderIcon.displayName = 'ListBorderIcon';
var ListBorderIcon$1 = ListBorderIcon;

function SvgListClearIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      fill: "currentColor",
      clipPath: "url(#ListClearIcon_svg__a)",
      children: jsx("path", {
        d: "M15.03 13.97 13.06 12l1.97-1.97-1.06-1.06L12 10.94l-1.97-1.97-1.06 1.06L10.94 12l-1.97 1.97 1.06 1.06L12 13.06l1.97 1.97zM5 11.5H1V10h4zM11 3.5H1V2h10zM7 7.5H1V6h6z"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 16h16V0H0z"
        })
      })
    })]
  });
}
const ListClearIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgListClearIcon
  });
});
ListClearIcon.displayName = 'ListClearIcon';
var ListClearIcon$1 = ListClearIcon;

function SvgLockFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M12 6V4a4 4 0 0 0-8 0v2H2.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75zM5.5 6h5V4a2.5 2.5 0 0 0-5 0zm1.75 7V9h1.5v4z",
      clipRule: "evenodd"
    })
  });
}
const LockFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgLockFillIcon
  });
});
LockFillIcon.displayName = 'LockFillIcon';
var LockFillIcon$1 = LockFillIcon;

function SvgLockIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.25 9v4h1.5V9z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M12 6V4a4 4 0 0 0-8 0v2H2.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75zm.5 1.5v7h-9v-7zM5.5 4v2h5V4a2.5 2.5 0 0 0-5 0",
      clipRule: "evenodd"
    })]
  });
}
const LockIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgLockIcon
  });
});
LockIcon.displayName = 'LockIcon';
var LockIcon$1 = LockIcon;

function SvgLockUnlockedIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M10 11.75v-1.5H6v1.5z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M13.25 6H5.5V4a2.5 2.5 0 0 1 5 0v.5H12V4a4 4 0 0 0-8 0v2H2.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75M3.5 7.5h9v7h-9z",
      clipRule: "evenodd"
    })]
  });
}
const LockUnlockedIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgLockUnlockedIcon
  });
});
LockUnlockedIcon.displayName = 'LockUnlockedIcon';
var LockUnlockedIcon$1 = LockUnlockedIcon;

function SvgMIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M6.42 5.415A.75.75 0 0 0 5 5.75V11h1.5V8.927l.83 1.658a.75.75 0 0 0 1.34 0l.83-1.658V11H11V5.75a.75.75 0 0 0-1.42-.335L8 8.573z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z",
      clipRule: "evenodd"
    })]
  });
}
const MIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgMIcon
  });
});
MIcon.displayName = 'MIcon';
var MIcon$1 = MIcon;

function SvgMailIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75zm.75 2.347V12.5h13V4.347L9.081 8.604a1.75 1.75 0 0 1-2.162 0zM13.15 3.5H2.85l4.996 3.925a.25.25 0 0 0 .308 0z",
      clipRule: "evenodd"
    })
  });
}
const MailIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgMailIcon
  });
});
MailIcon.displayName = 'MailIcon';
var MailIcon$1 = MailIcon;

function SvgMenuIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M15 4H1V2.5h14zm0 4.75H1v-1.5h14zm0 4.75H1V12h14z",
      clipRule: "evenodd"
    })
  });
}
const MenuIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgMenuIcon
  });
});
MenuIcon.displayName = 'MenuIcon';
var MenuIcon$1 = MenuIcon;

function SvgMinusBoxIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M11.5 8.75h-7v-1.5h7z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z",
      clipRule: "evenodd"
    })]
  });
}
const MinusBoxIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgMinusBoxIcon
  });
});
MinusBoxIcon.displayName = 'MinusBoxIcon';
var MinusBoxIcon$1 = MinusBoxIcon;

function SvgMinusCircleFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16m3.5-7.25h-7v-1.5h7z",
      clipRule: "evenodd"
    })
  });
}
const MinusCircleFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgMinusCircleFillIcon
  });
});
MinusCircleFillIcon.displayName = 'MinusCircleFillIcon';
var MinusCircleFillIcon$1 = MinusCircleFillIcon;

function SvgMinusCircleIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M4.5 8.75v-1.5h7v1.5z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13",
      clipRule: "evenodd"
    })]
  });
}
const MinusCircleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgMinusCircleIcon
  });
});
MinusCircleIcon.displayName = 'MinusCircleIcon';
var MinusCircleIcon$1 = MinusCircleIcon;

function SvgModelsIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#ModelsIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M0 4.75a2.75 2.75 0 0 1 5.145-1.353l4.372-.95a2.75 2.75 0 1 1 3.835 2.823l.282 2.257a2.75 2.75 0 1 1-2.517 4.46l-2.62 1.145.003.118a2.75 2.75 0 1 1-4.415-2.19L3.013 7.489A2.75 2.75 0 0 1 0 4.75M2.75 3.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m2.715 1.688q.027-.164.033-.333l4.266-.928a2.75 2.75 0 0 0 2.102 1.546l.282 2.257c-.377.165-.71.412-.976.719zM4.828 6.55a2.8 2.8 0 0 1-.413.388l1.072 3.573q.13-.012.263-.012c.945 0 1.778.476 2.273 1.202l2.5-1.093a2.8 2.8 0 0 1 .012-.797zM12 10.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0M5.75 12a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5M11 2.75a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const ModelsIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgModelsIcon
  });
});
ModelsIcon.displayName = 'ModelsIcon';
var ModelsIcon$1 = ModelsIcon;

function SvgNoIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.5 8a6.5 6.5 0 0 1 10.535-5.096l-9.131 9.131A6.47 6.47 0 0 1 1.5 8m2.465 5.096a6.5 6.5 0 0 0 9.131-9.131z",
      clipRule: "evenodd"
    })
  });
}
const NoIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgNoIcon
  });
});
NoIcon.displayName = 'NoIcon';
var NoIcon$1 = NoIcon;

function SvgNotebookIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3 1.75A.75.75 0 0 1 3.75 1h10.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H3.75a.75.75 0 0 1-.75-.75V12.5H1V11h2V8.75H1v-1.5h2V5H1V3.5h2zm1.5.75v11H6v-11zm3 0v11h6v-11z",
      clipRule: "evenodd"
    })
  });
}
const NotebookIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgNotebookIcon
  });
});
NotebookIcon.displayName = 'NotebookIcon';
var NotebookIcon$1 = NotebookIcon;

function SvgNotificationIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 1a5 5 0 0 0-5 5v1.99c0 .674-.2 1.332-.573 1.892l-1.301 1.952A.75.75 0 0 0 1.75 13h3.5v.25a2.75 2.75 0 1 0 5.5 0V13h3.5a.75.75 0 0 0 .624-1.166l-1.301-1.952A3.4 3.4 0 0 1 13 7.99V6a5 5 0 0 0-5-5m1.25 12h-2.5v.25a1.25 1.25 0 1 0 2.5 0zM4.5 6a3.5 3.5 0 1 1 7 0v1.99c0 .97.287 1.918.825 2.724l.524.786H3.15l.524-.786A4.9 4.9 0 0 0 4.5 7.99z",
      clipRule: "evenodd"
    })
  });
}
const NotificationIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgNotificationIcon
  });
});
NotificationIcon.displayName = 'NotificationIcon';
var NotificationIcon$1 = NotificationIcon;

function SvgNotificationOffIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m14.47 13.53-12-12-1 1L3.28 4.342A5 5 0 0 0 3 6v1.99c0 .674-.2 1.332-.573 1.892l-1.301 1.952A.75.75 0 0 0 1.75 13h3.5v.25a2.75 2.75 0 1 0 5.5 0V13h1.19l1.53 1.53zM13.038 8.5A3.4 3.4 0 0 1 13 7.99V6a5 5 0 0 0-7.965-4.026l1.078 1.078A3.5 3.5 0 0 1 11.5 6v1.99q0 .238.023.472l.038.038zM4.5 6q0-.21.024-.415L10.44 11.5H3.151l.524-.786A4.9 4.9 0 0 0 4.5 7.99zm2.25 7.25V13h2.5v.25a1.25 1.25 0 1 1-2.5 0",
      clipRule: "evenodd"
    })
  });
}
const NotificationOffIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgNotificationOffIcon
  });
});
NotificationOffIcon.displayName = 'NotificationOffIcon';
var NotificationOffIcon$1 = NotificationOffIcon;

function SvgNumbersIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M7.889 1A2.39 2.39 0 0 0 5.5 3.389H7c0-.491.398-.889.889-.889h.371a.74.74 0 0 1 .292 1.42l-1.43.613A2.68 2.68 0 0 0 5.5 6.992V8h5V6.5H7.108c.12-.26.331-.472.604-.588l1.43-.613A2.24 2.24 0 0 0 8.26 1zM2.75 6a1.5 1.5 0 0 1-1.5 1.5H1V9h.25c.546 0 1.059-.146 1.5-.401V11.5H1V13h5v-1.5H4.25V6zM10 12.85A2.15 2.15 0 0 0 12.15 15h.725a2.125 2.125 0 0 0 1.617-3.504 2.138 2.138 0 0 0-1.656-3.521l-.713.008A2.15 2.15 0 0 0 10 10.133v.284h1.5v-.284a.65.65 0 0 1 .642-.65l.712-.009a.638.638 0 1 1 .008 1.276H12v1.5h.875a.625.625 0 1 1 0 1.25h-.725a.65.65 0 0 1-.65-.65v-.267H10z"
    })
  });
}
const NumbersIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgNumbersIcon
  });
});
NumbersIcon.displayName = 'NumbersIcon';
var NumbersIcon$1 = NumbersIcon;

function SvgOfficeIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M4 8.75h8v-1.5H4zM7 5.75H4v-1.5h3zM4 11.75h8v-1.5H4z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V5a.75.75 0 0 0-.75-.75H10v-2.5A.75.75 0 0 0 9.25 1zm.75 1.5h6V5c0 .414.336.75.75.75h4.25v7.75h-11z",
      clipRule: "evenodd"
    })]
  });
}
const OfficeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgOfficeIcon
  });
});
OfficeIcon.displayName = 'OfficeIcon';
var OfficeIcon$1 = OfficeIcon;

function SvgOverflowIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M8 1a1.75 1.75 0 1 0 0 3.5A1.75 1.75 0 0 0 8 1M8 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M8 11.5A1.75 1.75 0 1 0 8 15a1.75 1.75 0 0 0 0-3.5"
    })
  });
}
const OverflowIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgOverflowIcon
  });
});
OverflowIcon.displayName = 'OverflowIcon';
var OverflowIcon$1 = OverflowIcon;

function SvgPageBottomIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 3.06 2.06 2l5.97 5.97L14 2l1.06 1.06-7.03 7.031zm14.03 10.47v1.5h-14v-1.5z",
      clipRule: "evenodd"
    })
  });
}
const PageBottomIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPageBottomIcon
  });
});
PageBottomIcon.displayName = 'PageBottomIcon';
var PageBottomIcon$1 = PageBottomIcon;

function SvgPageFirstIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m12.97 1 1.06 1.06-5.97 5.97L14.03 14l-1.06 1.06-7.03-7.03zM2.5 15.03H1v-14h1.5z",
      clipRule: "evenodd"
    })
  });
}
const PageFirstIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPageFirstIcon
  });
});
PageFirstIcon.displayName = 'PageFirstIcon';
var PageFirstIcon$1 = PageFirstIcon;

function SvgPageLastIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3.06 1 2 2.06l5.97 5.97L2 14l1.06 1.06 7.031-7.03zm10.47 14.03h1.5v-14h-1.5z",
      clipRule: "evenodd"
    })
  });
}
const PageLastIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPageLastIcon
  });
});
PageLastIcon.displayName = 'PageLastIcon';
var PageLastIcon$1 = PageLastIcon;

function SvgPageTopIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m1 12.97 1.06 1.06 5.97-5.97L14 14.03l1.06-1.06-7.03-7.03zM15.03 2.5V1h-14v1.5z",
      clipRule: "evenodd"
    })
  });
}
const PageTopIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPageTopIcon
  });
});
PageTopIcon.displayName = 'PageTopIcon';
var PageTopIcon$1 = PageTopIcon;

function SvgPauseIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M10 12V4h1.5v8zm-5.5 0V4H6v8z",
      clipRule: "evenodd"
    })
  });
}
const PauseIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPauseIcon
  });
});
PauseIcon.displayName = 'PauseIcon';
var PauseIcon$1 = PauseIcon;

function SvgPencilIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M13.487 1.513a1.75 1.75 0 0 0-2.474 0L1.22 11.306a.75.75 0 0 0-.22.53v2.5c0 .414.336.75.75.75h2.5a.75.75 0 0 0 .53-.22l9.793-9.793a1.75 1.75 0 0 0 0-2.475zm-1.414 1.06a.25.25 0 0 1 .354 0l1.086 1.086a.25.25 0 0 1 0 .354L12 5.525l-1.44-1.44zM9.5 5.146l-7 7v1.44h1.44l7-7z",
      clipRule: "evenodd"
    })
  });
}
const PencilIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPencilIcon
  });
});
PencilIcon.displayName = 'PencilIcon';
var PencilIcon$1 = PencilIcon;

function SvgPieChartIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M9.25 0a.75.75 0 0 0-.75.75v6c0 .414.336.75.75.75h6a.75.75 0 0 0 .75-.75A6.75 6.75 0 0 0 9.25 0M10 1.553A5.25 5.25 0 0 1 14.447 6H10z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6.75 2.5a6.75 6.75 0 1 0 6.75 6.75.75.75 0 0 0-.75-.75H7.5V3.25a.75.75 0 0 0-.75-.75M1.5 9.25A5.25 5.25 0 0 1 6 4.053V9.25c0 .414.336.75.75.75h5.197A5.251 5.251 0 0 1 1.5 9.25",
      clipRule: "evenodd"
    })]
  });
}
const PieChartIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPieChartIcon
  });
});
PieChartIcon.displayName = 'PieChartIcon';
var PieChartIcon$1 = PieChartIcon;

function SvgPinCancelIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M5.75 0A.75.75 0 0 0 5 .75v1.19l9 9V9a.75.75 0 0 0-.22-.53l-2.12-2.122a2.25 2.25 0 0 1-.66-1.59V.75a.75.75 0 0 0-.75-.75zM10.94 12l2.53 2.53 1.06-1.06-11.5-11.5-1.06 1.06 2.772 2.773q-.157.301-.4.545L2.22 8.47A.75.75 0 0 0 2 9v2.25c0 .414.336.75.75.75h4.5v4h1.5v-4z"
    })
  });
}
const PinCancelIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPinCancelIcon
  });
});
PinCancelIcon.displayName = 'PinCancelIcon';
var PinCancelIcon$1 = PinCancelIcon;

function SvgPinFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M5 .75A.75.75 0 0 1 5.75 0h4.5a.75.75 0 0 1 .75.75v4.007c0 .597.237 1.17.659 1.591L13.78 8.47c.141.14.22.331.22.53v2.25a.75.75 0 0 1-.75.75h-4.5v4h-1.5v-4h-4.5a.75.75 0 0 1-.75-.75V9a.75.75 0 0 1 .22-.53L4.34 6.348A2.25 2.25 0 0 0 5 4.758z"
    })
  });
}
const PinFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPinFillIcon
  });
});
PinFillIcon.displayName = 'PinFillIcon';
var PinFillIcon$1 = PinFillIcon;

function SvgPinIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M5.75 0A.75.75 0 0 0 5 .75v4.007a2.25 2.25 0 0 1-.659 1.591L2.22 8.47A.75.75 0 0 0 2 9v2.25c0 .414.336.75.75.75h4.5v4h1.5v-4h4.5a.75.75 0 0 0 .75-.75V9a.75.75 0 0 0-.22-.53L11.66 6.348A2.25 2.25 0 0 1 11 4.758V.75a.75.75 0 0 0-.75-.75zm.75 4.757V1.5h3v3.257a3.75 3.75 0 0 0 1.098 2.652L12.5 9.311V10.5h-9V9.31L5.402 7.41A3.75 3.75 0 0 0 6.5 4.757",
      clipRule: "evenodd"
    })
  });
}
const PinIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPinIcon
  });
});
PinIcon.displayName = 'PinIcon';
var PinIcon$1 = PinIcon;

function SvgPipelineIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M10.75 6.75A5.75 5.75 0 0 0 5 1H1.75a.75.75 0 0 0-.75.75V6c0 .414.336.75.75.75H5a.25.25 0 0 1 .25.25v2.25A5.75 5.75 0 0 0 11 15h3.25a.75.75 0 0 0 .75-.75V10a.75.75 0 0 0-.75-.75H11a.25.25 0 0 1-.25-.25zM5.5 2.53a4.25 4.25 0 0 1 3.75 4.22V9a1.75 1.75 0 0 0 1.25 1.678v2.793A4.25 4.25 0 0 1 6.75 9.25V7A1.75 1.75 0 0 0 5.5 5.322zM4 2.5v2.75H2.5V2.5zm9.5 8.25H12v2.75h1.5z",
      clipRule: "evenodd"
    })
  });
}
const PipelineIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPipelineIcon
  });
});
PipelineIcon.displayName = 'PipelineIcon';
var PipelineIcon$1 = PipelineIcon;

function SvgPlayCircleFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m7.125-2.815A.75.75 0 0 0 6 5.835v4.33a.75.75 0 0 0 1.125.65l3.75-2.166a.75.75 0 0 0 0-1.299z",
      clipRule: "evenodd"
    })
  });
}
const PlayCircleFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPlayCircleFillIcon
  });
});
PlayCircleFillIcon.displayName = 'PlayCircleFillIcon';
var PlayCircleFillIcon$1 = PlayCircleFillIcon;

function SvgPlayCircleIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M11.25 8a.75.75 0 0 1-.375.65l-3.75 2.165A.75.75 0 0 1 6 10.165v-4.33a.75.75 0 0 1 1.125-.65l3.75 2.165a.75.75 0 0 1 .375.65"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0",
      clipRule: "evenodd"
    })]
  });
}
const PlayCircleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPlayCircleIcon
  });
});
PlayCircleIcon.displayName = 'PlayCircleIcon';
var PlayCircleIcon$1 = PlayCircleIcon;

function SvgPlayIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M12.125 8.864a.75.75 0 0 0 0-1.3l-6-3.464A.75.75 0 0 0 5 4.75v6.928a.75.75 0 0 0 1.125.65z"
    })
  });
}
const PlayIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPlayIcon
  });
});
PlayIcon.displayName = 'PlayIcon';
var PlayIcon$1 = PlayIcon;

function SvgPlugIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m14.168 2.953.893-.892L14 1l-.893.893a4 4 0 0 0-5.077.48l-.884.884a.75.75 0 0 0 0 1.061l4.597 4.596a.75.75 0 0 0 1.06 0l.884-.884a4 4 0 0 0 .48-5.077M12.627 6.97l-.354.353-3.536-3.535.354-.354a2.5 2.5 0 1 1 3.536 3.536M7.323 10.152 5.91 8.737l1.414-1.414-1.06-1.06-1.415 1.414-.53-.53a.75.75 0 0 0-1.06 0l-.885.883a4 4 0 0 0-.48 5.077L1 14l1.06 1.06.893-.892a4 4 0 0 0 5.077-.48l.884-.885a.75.75 0 0 0 0-1.06l-.53-.53 1.414-1.415-1.06-1.06zm-3.889 2.475a2.5 2.5 0 0 0 3.536 0l.353-.354-3.535-3.536-.354.354a2.5 2.5 0 0 0 0 3.536",
      clipRule: "evenodd"
    })
  });
}
const PlugIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPlugIcon
  });
});
PlugIcon.displayName = 'PlugIcon';
var PlugIcon$1 = PlugIcon;

function SvgPlusCircleFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16m-.75-4.5V8.75H4.5v-1.5h2.75V4.5h1.5v2.75h2.75v1.5H8.75v2.75z",
      clipRule: "evenodd"
    })
  });
}
const PlusCircleFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPlusCircleFillIcon
  });
});
PlusCircleFillIcon.displayName = 'PlusCircleFillIcon';
var PlusCircleFillIcon$1 = PlusCircleFillIcon;

function SvgPlusCircleIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.25 11.5V8.75H4.5v-1.5h2.75V4.5h1.5v2.75h2.75v1.5H8.75v2.75z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0",
      clipRule: "evenodd"
    })]
  });
}
const PlusCircleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPlusCircleIcon
  });
});
PlusCircleIcon.displayName = 'PlusCircleIcon';
var PlusCircleIcon$1 = PlusCircleIcon;

function SvgPlusSquareIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.25 7.25V4.5h1.5v2.75h2.75v1.5H8.75v2.75h-1.5V8.75H4.5v-1.5z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75v11h11v-11z",
      clipRule: "evenodd"
    })]
  });
}
const PlusSquareIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgPlusSquareIcon
  });
});
PlusSquareIcon.displayName = 'PlusSquareIcon';
var PlusSquareIcon$1 = PlusSquareIcon;

function SvgQueryEditorIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M12 12H8v-1.5h4zM5.53 11.53 7.56 9.5 5.53 7.47 4.47 8.53l.97.97-.97.97z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 3V2.5h11V4zm0 1.5v8h11v-8z",
      clipRule: "evenodd"
    })]
  });
}
const QueryEditorIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgQueryEditorIcon
  });
});
QueryEditorIcon.displayName = 'QueryEditorIcon';
var QueryEditorIcon$1 = QueryEditorIcon;

function SvgQueryIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#QueryIcon_svg__a)",
      children: [jsx("path", {
        fillRule: "evenodd",
        d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16h-.75a.75.75 0 0 1-.75-.75zm7.5 1.81 1.94 1.94H9.5z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M5.53 9.97 8.56 13l-3.03 3.03-1.06-1.06L6.44 13l-1.97-1.97zM14 14.5H9V16h5z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const QueryIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgQueryIcon
  });
});
QueryIcon.displayName = 'QueryIcon';
var QueryIcon$1 = QueryIcon;

function SvgQuestionMarkFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16m2.207-10.189a2.25 2.25 0 0 1-1.457 2.56V9h-1.5V7.75A.75.75 0 0 1 8 7a.75.75 0 1 0-.75-.75h-1.5a2.25 2.25 0 0 1 4.457-.439M7.25 10.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0",
      clipRule: "evenodd"
    })
  });
}
const QuestionMarkFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgQuestionMarkFillIcon
  });
});
QuestionMarkFillIcon.displayName = 'QuestionMarkFillIcon';
var QuestionMarkFillIcon$1 = QuestionMarkFillIcon;

function SvgQuestionMarkIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.25 10.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0M10.079 7.111A2.25 2.25 0 1 0 5.75 6.25h1.5A.75.75 0 1 1 8 7a.75.75 0 0 0-.75.75V9h1.5v-.629a2.25 2.25 0 0 0 1.329-1.26"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13",
      clipRule: "evenodd"
    })]
  });
}
const QuestionMarkIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgQuestionMarkIcon
  });
});
QuestionMarkIcon.displayName = 'QuestionMarkIcon';
var QuestionMarkIcon$1 = QuestionMarkIcon;

function SvgReaderModeIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M13 4.5h-3V6h3zM13 7.25h-3v1.5h3zM13 10h-3v1.5h3z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75zm.75 10.5v-9h5.75v9zm7.25 0h5.75v-9H8.75z",
      clipRule: "evenodd"
    })]
  });
}
const ReaderModeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgReaderModeIcon
  });
});
ReaderModeIcon.displayName = 'ReaderModeIcon';
var ReaderModeIcon$1 = ReaderModeIcon;

function SvgRedoIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#RedoIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "m13.19 5-2.72-2.72 1.06-1.06 4.53 4.53-4.53 4.53-1.06-1.06 2.72-2.72H4.5a3 3 0 1 0 0 6H9V14H4.5a4.5 4.5 0 0 1 0-9z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 16h16V0H0z"
        })
      })
    })]
  });
}
const RedoIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgRedoIcon
  });
});
RedoIcon.displayName = 'RedoIcon';
var RedoIcon$1 = RedoIcon;

function SvgRefreshIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 8a7 7 0 0 1 11.85-5.047l.65.594V2H15v4h-4V4.5h1.32l-.496-.453-.007-.007a5.5 5.5 0 1 0 .083 7.839l1.063 1.058A7 7 0 0 1 1 8",
      clipRule: "evenodd"
    })
  });
}
const RefreshIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgRefreshIcon
  });
});
RefreshIcon.displayName = 'RefreshIcon';
var RefreshIcon$1 = RefreshIcon;

function SvgReplyIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("mask", {
      id: "ReplyIcon_svg__a",
      width: 16,
      height: 16,
      x: 0,
      y: 0,
      maskUnits: "userSpaceOnUse",
      style: {
        maskType: 'alpha'
      },
      children: jsx("path", {
        fill: "currentColor",
        d: "M0 0h16v16H0z"
      })
    }), jsx("g", {
      mask: "url(#ReplyIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        d: "M3.333 3.333V6q0 .834.584 1.417Q4.5 8 5.333 8h6.117l-2.4-2.4.95-.933 4 4-4 4-.95-.934 2.4-2.4H5.333a3.21 3.21 0 0 1-2.358-.975A3.21 3.21 0 0 1 2 6V3.333z"
      })
    })]
  });
}
const ReplyIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgReplyIcon
  });
});
ReplyIcon.displayName = 'ReplyIcon';
var ReplyIcon$1 = ReplyIcon;

function SvgRobot2Icon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 24 24",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#Robot2Icon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        d: "M2.4 24v-6q0-.99.705-1.695A2.3 2.3 0 0 1 4.8 15.6h14.4q.99 0 1.695.705T21.6 18v6zm6-9.6q-2.49 0-4.245-1.755T2.4 8.4t1.755-4.245T8.4 2.4h7.2q2.49 0 4.245 1.755T21.6 8.4t-1.755 4.245T15.6 14.4zm-3.6 7.2h14.4V18H4.8zM8.4 12h7.2q1.5 0 2.55-1.05T19.2 8.4t-1.05-2.55T15.6 4.8H8.4q-1.5 0-2.55 1.05T4.8 8.4t1.05 2.55T8.4 12m0-2.4q.51 0 .855-.345T9.6 8.4t-.345-.855A1.16 1.16 0 0 0 8.4 7.2q-.51 0-.855.345T7.2 8.4t.345.855.855.345m7.2 0q.51 0 .855-.345T16.8 8.4t-.345-.855A1.16 1.16 0 0 0 15.6 7.2q-.51 0-.855.345T14.4 8.4t.345.855.855.345"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h24v24H0z"
        })
      })
    })]
  });
}
const Robot2Icon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgRobot2Icon
  });
});
Robot2Icon.displayName = 'Robot2Icon';
var Robot2Icon$1 = Robot2Icon;

function SvgRobotIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 0a.75.75 0 0 1 .75.75V3h5.5a.75.75 0 0 1 .75.75V6h.25a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-.75.75H15v2.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75V12H.75a.75.75 0 0 1-.75-.75v-4.5A.75.75 0 0 1 .75 6H1V3.75A.75.75 0 0 1 1.75 3h5.5V.75A.75.75 0 0 1 8 0M2.5 4.5v9h11v-9zM5 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2m7-1a1 1 0 1 1-2 0 1 1 0 0 1 2 0m-6.25 2.25a.75.75 0 0 0 0 1.5h4.5a.75.75 0 0 0 0-1.5z",
      clipRule: "evenodd"
    })
  });
}
const RobotIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgRobotIcon
  });
});
RobotIcon.displayName = 'RobotIcon';
var RobotIcon$1 = RobotIcon;

function SvgRocketIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "m2.873 6.374 1.557.658q.28-.558.578-1.077.3-.517.659-1.036L4.549 4.7zm2.834 1.654L7.98 10.28q.839-.319 1.796-.976a12.8 12.8 0 0 0 1.796-1.495q1.397-1.395 2.185-3.1.788-1.702.688-3.138-1.437-.1-3.152.687A10.7 10.7 0 0 0 8.18 4.441a12.8 12.8 0 0 0-1.497 1.793q-.659.957-.977 1.794m3.551-1.295A1.53 1.53 0 0 1 8.8 5.607q0-.667.46-1.126a1.55 1.55 0 0 1 1.137-.459q.678 0 1.137.459.459.457.459 1.126 0 .667-.459 1.126a1.55 1.55 0 0 1-1.137.458q-.68 0-1.138-.458m.38 6.397 1.675-1.674-.219-1.116q-.52.36-1.037.648-.52.288-1.078.568zM15.882.116a9.17 9.17 0 0 1-.47 4.693q-.847 2.282-2.922 4.355l.399 1.973q.08.399-.04.777-.12.38-.4.658L9.1 15.92l-1.676-3.926L4.01 8.586.08 6.912l3.332-3.348q.28-.279.668-.399.39-.12.789-.04l1.975.399q2.074-2.072 4.35-2.93a9.1 9.1 0 0 1 4.689-.478M1.496 11.117a2.36 2.36 0 0 1 1.706-.707 2.29 2.29 0 0 1 1.707.687q.698.697.688 1.704a2.35 2.35 0 0 1-.708 1.704q-.499.498-1.667.857Q2.055 15.722 0 16q.28-2.052.638-3.219.36-1.166.858-1.664m1.138 1.116q-.2.2-.4.728a5 5 0 0 0-.279 1.066q.54-.08 1.068-.27.528-.188.728-.388a.86.86 0 0 0 .26-.578.72.72 0 0 0-.22-.578.76.76 0 0 0-.579-.229.83.83 0 0 0-.578.25"
    })
  });
}
const RocketIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgRocketIcon
  });
});
RocketIcon.displayName = 'RocketIcon';
var RocketIcon$1 = RocketIcon;

function SvgRunningIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#RunningIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M8 1.5A6.5 6.5 0 0 0 1.5 8H0a8 8 0 0 1 8-8zm0 13A6.5 6.5 0 0 0 14.5 8H16a8 8 0 0 1-8 8z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const RunningIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgRunningIcon
  });
});
RunningIcon.displayName = 'RunningIcon';
var RunningIcon$1 = RunningIcon;

function SvgSaveIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M10 9.25H6v1.5h4z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1H11a.75.75 0 0 1 .53.22l3.25 3.25c.141.14.22.331.22.53v9.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75H5v3.75c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75V2.81l2.5 2.5v8.19h-11zm4 0h3v3h-3z",
      clipRule: "evenodd"
    })]
  });
}
const SaveIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSaveIcon
  });
});
SaveIcon.displayName = 'SaveIcon';
var SaveIcon$1 = SaveIcon;

function SvgSchemaIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.75 0A.75.75 0 0 0 2 .75v3a.75.75 0 0 0 .75.75h1v7a2.75 2.75 0 0 0 2.75 2.75H7v1c0 .414.336.75.75.75h5.5a.75.75 0 0 0 .75-.75v-3a.75.75 0 0 0-.75-.75h-5.5a.75.75 0 0 0-.75.75v.5h-.5c-.69 0-1.25-.56-1.25-1.25V8.45c.375.192.8.3 1.25.3H7v.75c0 .414.336.75.75.75h5.5A.75.75 0 0 0 14 9.5v-3a.75.75 0 0 0-.75-.75h-5.5A.75.75 0 0 0 7 6.5v.75h-.5c-.69 0-1.25-.56-1.25-1.25V4.5h8a.75.75 0 0 0 .75-.75v-3a.75.75 0 0 0-.75-.75zm.75 3V1.5h9V3zm5 10v1.5h4V13zm0-4.25v-1.5h4v1.5z",
      clipRule: "evenodd"
    })
  });
}
const SchemaIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSchemaIcon
  });
});
SchemaIcon.displayName = 'SchemaIcon';
var SchemaIcon$1 = SchemaIcon;

function SvgSchoolIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M16 7a.75.75 0 0 0-.37-.647l-7.25-4.25a.75.75 0 0 0-.76 0L.37 6.353a.75.75 0 0 0 0 1.294L3 9.188V12a.75.75 0 0 0 .4.663l4.25 2.25a.75.75 0 0 0 .7 0l4.25-2.25A.75.75 0 0 0 13 12V9.188l1.5-.879V12H16zm-7.62 4.897 3.12-1.83v1.481L8 13.401l-3.5-1.853v-1.48l3.12 1.829a.75.75 0 0 0 .76 0M8 3.619 2.233 7 8 10.38 13.767 7z",
      clipRule: "evenodd"
    })
  });
}
const SchoolIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSchoolIcon
  });
});
SchoolIcon.displayName = 'SchoolIcon';
var SchoolIcon$1 = SchoolIcon;

function SvgSearchDataIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.651 3.128a.75.75 0 0 0-1.302 0l-1 1.75A.75.75 0 0 0 6 6h2a.75.75 0 0 0 .651-1.122zM4.75 6.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M7.5 7.25a.75.75 0 0 1 .75-.75h2a.75.75 0 0 1 .75.75v2a.75.75 0 0 1-.75.75h-2a.75.75 0 0 1-.75-.75z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 7a7 7 0 1 1 12.45 4.392l2.55 2.55-1.06 1.061-2.55-2.55A7 7 0 0 1 0 7m7-5.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11",
      clipRule: "evenodd"
    })]
  });
}
const SearchDataIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSearchDataIcon
  });
});
SearchDataIcon.displayName = 'SearchDataIcon';
var SearchDataIcon$1 = SearchDataIcon;

function SvgSearchIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#SearchIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M8 1a7 7 0 1 0 4.39 12.453l2.55 2.55 1.06-1.06-2.55-2.55A7 7 0 0 0 8 1M2.5 8a5.5 5.5 0 1 1 11 0 5.5 5.5 0 0 1-11 0",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const SearchIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSearchIcon
  });
});
SearchIcon.displayName = 'SearchIcon';
var SearchIcon$1 = SearchIcon;

function SvgSecurityIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2 1.75A.75.75 0 0 1 2.75 1h10.5a.75.75 0 0 1 .75.75v7.465a5.75 5.75 0 0 1-2.723 4.889l-2.882 1.784a.75.75 0 0 1-.79 0l-2.882-1.784A5.75 5.75 0 0 1 2 9.214zm1.5.75V7h3.75V2.5zm5.25 0V7h3.75V2.5zm3.75 6H8.75v5.404l1.737-1.076A4.25 4.25 0 0 0 12.5 9.215zm-5.25 5.404V8.5H3.5v.715a4.25 4.25 0 0 0 2.013 3.613z",
      clipRule: "evenodd"
    })
  });
}
const SecurityIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSecurityIcon
  });
});
SecurityIcon.displayName = 'SecurityIcon';
var SecurityIcon$1 = SecurityIcon;

function SvgSendIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M16 8a.75.75 0 0 1-.435.68l-13.5 6.25a.75.75 0 0 1-1.02-.934L3.202 8 1.044 2.004a.75.75 0 0 1 1.021-.935l13.5 6.25A.75.75 0 0 1 16 8m-11.473.75-1.463 4.065L13.464 8l-10.4-4.815L4.527 7.25H8v1.5z",
      clipRule: "evenodd"
    })
  });
}
const SendIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSendIcon
  });
});
SendIcon.displayName = 'SendIcon';
var SendIcon$1 = SendIcon;

function SvgShareIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M3.97 5.03 8 1l4.03 4.03-1.06 1.061-2.22-2.22v7.19h-1.5V3.87l-2.22 2.22z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M2.5 13.56v-6.5H1v7.25c0 .415.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V7.06h-1.5v6.5z"
    })]
  });
}
const ShareIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgShareIcon
  });
});
ShareIcon.displayName = 'ShareIcon';
var ShareIcon$1 = ShareIcon;

function SvgShieldIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2 1.75A.75.75 0 0 1 2.75 1h10.5a.75.75 0 0 1 .75.75v7.465a5.75 5.75 0 0 1-2.723 4.889l-2.882 1.784a.75.75 0 0 1-.79 0l-2.882-1.784A5.75 5.75 0 0 1 2 9.214zm1.5.75V7h3.75V2.5zm5.25 0V7h3.75V2.5zm3.75 6H8.75v5.404l1.737-1.076A4.25 4.25 0 0 0 12.5 9.215zm-5.25 5.404V8.5H3.5v.715a4.25 4.25 0 0 0 2.013 3.613z",
      clipRule: "evenodd"
    })
  });
}
const ShieldIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgShieldIcon
  });
});
ShieldIcon.displayName = 'ShieldIcon';
var ShieldIcon$1 = ShieldIcon;

function SvgShortcutIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M14.25 14H9v-1.5h4.5v-10h-10V6H2V1.75A.75.75 0 0 1 2.75 1h11.5a.75.75 0 0 1 .75.75v11.5a.75.75 0 0 1-.75.75"
    }), jsx("path", {
      fill: "currentColor",
      d: "M2 8h5v5H5.5v-2.872a2.251 2.251 0 0 0 .75 4.372V16A3.75 3.75 0 0 1 3.7 9.5H2z"
    })]
  });
}
const ShortcutIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgShortcutIcon
  });
});
ShortcutIcon.displayName = 'ShortcutIcon';
var ShortcutIcon$1 = ShortcutIcon;

function SvgSidebarAutoIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 17 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1zM4 2.5H2.5v11H4z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "m9.06 8 1.97 1.97-1.06 1.06L6.94 8l3.03-3.03 1.06 1.06zM11.97 6.03 13.94 8l-1.97 1.97 1.06 1.06L16.06 8l-3.03-3.03z"
    })]
  });
}
const SidebarAutoIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSidebarAutoIcon
  });
});
SidebarAutoIcon.displayName = 'SidebarAutoIcon';
var SidebarAutoIcon$1 = SidebarAutoIcon;

function SvgSidebarCollapseIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1zM4 2.5H2.5v11H4z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "m9.81 8.75 1.22 1.22-1.06 1.06L6.94 8l3.03-3.03 1.06 1.06-1.22 1.22H14v1.5z"
    })]
  });
}
const SidebarCollapseIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSidebarCollapseIcon
  });
});
SidebarCollapseIcon.displayName = 'SidebarCollapseIcon';
var SidebarCollapseIcon$1 = SidebarCollapseIcon;

function SvgSidebarExpandIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1zM4 2.5H2.5v11H4z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M11.19 8.75 9.97 9.97l1.06 1.06L14.06 8l-3.03-3.03-1.06 1.06 1.22 1.22H7v1.5z"
    })]
  });
}
const SidebarExpandIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSidebarExpandIcon
  });
});
SidebarExpandIcon.displayName = 'SidebarExpandIcon';
var SidebarExpandIcon$1 = SidebarExpandIcon;

function SvgSidebarIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11H4v11zm3 0h8v-11h-8z",
      clipRule: "evenodd"
    })
  });
}
const SidebarIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSidebarIcon
  });
});
SidebarIcon.displayName = 'SidebarIcon';
var SidebarIcon$1 = SidebarIcon;

function SvgSlidersIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2 3.104V2h1.5v1.104a2.751 2.751 0 0 1 0 5.292V14H2V8.396a2.751 2.751 0 0 1 0-5.292M1.5 5.75a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0M12.5 2v1.104a2.751 2.751 0 0 0 0 5.292V14H14V8.396a2.751 2.751 0 0 0 0-5.292V2zm.75 2.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5M7.25 14v-1.104a2.751 2.751 0 0 1 0-5.292V2h1.5v5.604a2.751 2.751 0 0 1 0 5.292V14zM8 11.5A1.25 1.25 0 1 1 8 9a1.25 1.25 0 0 1 0 2.5",
      clipRule: "evenodd"
    })
  });
}
const SlidersIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSlidersIcon
  });
});
SlidersIcon.displayName = 'SlidersIcon';
var SlidersIcon$1 = SlidersIcon;

function SvgSortAlphabeticalAscendingIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#SortAlphabeticalAscendingIcon_svg__a)",
      children: [jsx("path", {
        fillRule: "evenodd",
        d: "M4.307 0a.75.75 0 0 1 .697.473L7.596 7H5.982l-.238-.6H2.855l-.24.6H1L3.61.472A.75.75 0 0 1 4.307 0m-.852 4.9h1.693l-.844-2.124z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M4.777 9.5H1.5V8h4.75a.75.75 0 0 1 .607 1.191L3.723 13.5H7V15H2.25a.75.75 0 0 1-.607-1.191zM12 .94l4.03 4.03-1.06 1.06-2.22-2.22V10h-1.5V3.81L9.03 6.03 7.97 4.97z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const SortAlphabeticalAscendingIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSortAlphabeticalAscendingIcon
  });
});
SortAlphabeticalAscendingIcon.displayName = 'SortAlphabeticalAscendingIcon';
var SortAlphabeticalAscendingIcon$1 = SortAlphabeticalAscendingIcon;

function SvgSortAlphabeticalDescendingIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#SortAlphabeticalDescendingIcon_svg__a)",
      children: [jsx("path", {
        fillRule: "evenodd",
        d: "M4.307 0a.75.75 0 0 1 .697.473L7.596 7H5.982l-.238-.6H2.855l-.24.6H1L3.61.472A.75.75 0 0 1 4.307 0m-.852 4.9h1.693l-.844-2.124z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M4.777 9.5H1.5V8h4.75a.75.75 0 0 1 .607 1.191L3.723 13.5H7V15H2.25a.75.75 0 0 1-.607-1.191zM12 15.06l-4.03-4.03 1.06-1.06 2.22 2.22V6h1.5v6.19l2.22-2.22 1.06 1.06z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const SortAlphabeticalDescendingIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSortAlphabeticalDescendingIcon
  });
});
SortAlphabeticalDescendingIcon.displayName = 'SortAlphabeticalDescendingIcon';
var SortAlphabeticalDescendingIcon$1 = SortAlphabeticalDescendingIcon;

function SvgSortAlphabeticalLeftIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#SortAlphabeticalLeftIcon_svg__a)",
      children: [jsx("path", {
        d: "M.94 4 4.97-.03l1.06 1.06-2.22 2.22H10v1.5H3.81l2.22 2.22-1.06 1.06z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M4.307 9a.75.75 0 0 1 .697.473L7.596 16H5.982l-.238-.6H2.855l-.24.6H1l2.61-6.528A.75.75 0 0 1 4.307 9m-.852 4.9h1.693l-.844-2.124z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M11.777 10.5H8.5V9h4.75a.75.75 0 0 1 .607 1.191L10.723 14.5H14V16H9.25a.75.75 0 0 1-.607-1.191z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const SortAlphabeticalLeftIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSortAlphabeticalLeftIcon
  });
});
SortAlphabeticalLeftIcon.displayName = 'SortAlphabeticalLeftIcon';
var SortAlphabeticalLeftIcon$1 = SortAlphabeticalLeftIcon;

function SvgSortAlphabeticalRightIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#SortAlphabeticalRightIcon_svg__a)",
      children: [jsx("path", {
        d: "m14.06 4-4.03 4.03-1.06-1.06 2.22-2.22H5v-1.5h6.19L8.97 1.03l1.06-1.06z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M4.307 9a.75.75 0 0 1 .697.473L7.596 16H5.982l-.238-.6H2.855l-.24.6H1l2.61-6.528A.75.75 0 0 1 4.307 9m-.852 4.9h1.693l-.844-2.124z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M11.777 10.5H8.5V9h4.75a.75.75 0 0 1 .607 1.191l-3.134 4.31H14V16H9.25a.75.75 0 0 1-.607-1.192z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const SortAlphabeticalRightIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSortAlphabeticalRightIcon
  });
});
SortAlphabeticalRightIcon.displayName = 'SortAlphabeticalRightIcon';
var SortAlphabeticalRightIcon$1 = SortAlphabeticalRightIcon;

function SvgSortAscendingIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "m11.5.94 4.03 4.03-1.06 1.06-2.22-2.22V10h-1.5V3.81L8.53 6.03 7.47 4.97zM1 4.5h4V6H1zM1 12.5h10V14H1zM8 8.5H1V10h7z"
    })
  });
}
const SortAscendingIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSortAscendingIcon
  });
});
SortAscendingIcon.displayName = 'SortAscendingIcon';
var SortAscendingIcon$1 = SortAscendingIcon;

function SvgSortDescendingIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 3.5h10V2H1zm0 8h4V10H1zm7-4H1V6h7zm3.5 7.56 4.03-4.03-1.06-1.06-2.22 2.22V6h-1.5v6.19L8.53 9.97l-1.06 1.06z",
      clipRule: "evenodd"
    })
  });
}
const SortDescendingIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSortDescendingIcon
  });
});
SortDescendingIcon.displayName = 'SortDescendingIcon';
var SortDescendingIcon$1 = SortDescendingIcon;

function SvgSortUnsortedIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M11.5.94 7.47 4.97l1.06 1.06 2.22-2.22v8.38L8.53 9.97l-1.06 1.06 4.03 4.03 4.03-4.03-1.06-1.06-2.22 2.22V3.81l2.22 2.22 1.06-1.06zM6 3.5H1V5h5zM6 11.5H1V13h5zM1 7.5h5V9H1z"
    })
  });
}
const SortUnsortedIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSortUnsortedIcon
  });
});
SortUnsortedIcon.displayName = 'SortUnsortedIcon';
var SortUnsortedIcon$1 = SortUnsortedIcon;

function SvgSparkleDoubleFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M4.739 9.622a.75.75 0 0 0-1.478 0l-.152.876a.75.75 0 0 1-.61.61l-.878.153a.75.75 0 0 0 0 1.478l.877.152a.75.75 0 0 1 .61.61l.153.878a.75.75 0 0 0 1.478 0l.152-.877a.75.75 0 0 1 .61-.61l.878-.153a.75.75 0 0 0 0-1.478l-.877-.152a.75.75 0 0 1-.61-.61zM10.737.611a.75.75 0 0 0-1.474 0l-.264 1.398A3.75 3.75 0 0 1 6.01 5l-1.398.264a.75.75 0 0 0 0 1.474l1.398.264A3.75 3.75 0 0 1 9 9.99l.264 1.398a.75.75 0 0 0 1.474 0l.264-1.398A3.75 3.75 0 0 1 13.99 7l1.398-.264a.75.75 0 0 0 0-1.474l-1.398-.264A3.75 3.75 0 0 1 11 2.01z",
      clipRule: "evenodd"
    })
  });
}
const SparkleDoubleFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSparkleDoubleFillIcon
  });
});
SparkleDoubleFillIcon.displayName = 'SparkleDoubleFillIcon';
var SparkleDoubleFillIcon$1 = SparkleDoubleFillIcon;

function SvgSparkleDoubleIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M3.848 10.627 4 9.75l.152.877a1.5 1.5 0 0 0 1.221 1.22L6.25 12l-.877.152a1.5 1.5 0 0 0-1.22 1.221L4 14.25l-.152-.877a1.5 1.5 0 0 0-1.221-1.22L1.75 12l.877-.152a1.5 1.5 0 0 0 1.22-1.221"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M4 9a.75.75 0 0 1 .739.621l.152.877a.75.75 0 0 0 .61.61l.878.153a.75.75 0 0 1 0 1.478l-.877.152a.75.75 0 0 0-.61.61l-.153.878a.75.75 0 0 1-1.478 0l-.152-.877a.75.75 0 0 0-.61-.61l-.878-.153a.75.75 0 0 1 0-1.478l.877-.152a.75.75 0 0 0 .61-.61l.153-.878A.75.75 0 0 1 4 9m0 2.92-.08.08q.042.039.08.08.038-.042.08-.08zM10 0c.36 0 .67.257.737.611l.264 1.398A3.75 3.75 0 0 0 13.99 5l1.398.264a.75.75 0 0 1 0 1.474l-1.398.264A3.75 3.75 0 0 0 11 9.99l-.264 1.398a.75.75 0 0 1-1.474 0l-.264-1.398A3.75 3.75 0 0 0 6.01 7l-1.398-.264a.75.75 0 0 1 0-1.474l1.398-.264A3.75 3.75 0 0 0 9 2.01L9.263.611A.75.75 0 0 1 10 0m0 3.682A5.26 5.26 0 0 1 7.682 6 5.26 5.26 0 0 1 10 8.318 5.26 5.26 0 0 1 12.318 6 5.26 5.26 0 0 1 10 3.682",
      clipRule: "evenodd"
    })]
  });
}
const SparkleDoubleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSparkleDoubleIcon
  });
});
SparkleDoubleIcon.displayName = 'SparkleDoubleIcon';
var SparkleDoubleIcon$1 = SparkleDoubleIcon;

function SvgSparkleFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M13.667 6.575c1.376.452 1.376 2.398 0 2.85l-2.472.813a1.5 1.5 0 0 0-.957.957l-.813 2.472c-.452 1.376-2.398 1.376-2.85 0l-.813-2.472a1.5 1.5 0 0 0-.956-.957l-2.473-.813c-1.376-.452-1.376-2.398 0-2.85l2.473-.813a1.5 1.5 0 0 0 .956-.956l.813-2.473c.452-1.376 2.398-1.376 2.85 0l.813 2.473a1.5 1.5 0 0 0 .957.956z",
      clipRule: "evenodd"
    })
  });
}
const SparkleFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSparkleFillIcon
  });
});
SparkleFillIcon.displayName = 'SparkleFillIcon';
var SparkleFillIcon$1 = SparkleFillIcon;

function SvgSparkleIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M10.726 8.813 13.199 8l-2.473-.813a3 3 0 0 1-1.913-1.913L8 2.801l-.813 2.473a3 3 0 0 1-1.913 1.913L2.801 8l2.473.813a3 3 0 0 1 1.913 1.913L8 13.199l.813-2.473a3 3 0 0 1 1.913-1.913m2.941.612c1.376-.452 1.376-2.398 0-2.85l-2.472-.813a1.5 1.5 0 0 1-.957-.956l-.813-2.473c-.452-1.376-2.398-1.376-2.85 0l-.813 2.473a1.5 1.5 0 0 1-.956.956l-2.473.813c-1.376.452-1.376 2.398 0 2.85l2.473.813a1.5 1.5 0 0 1 .956.957l.813 2.472c.452 1.376 2.398 1.376 2.85 0l.813-2.472a1.5 1.5 0 0 1 .957-.957z",
      clipRule: "evenodd"
    })
  });
}
const SparkleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSparkleIcon
  });
});
SparkleIcon.displayName = 'SparkleIcon';
var SparkleIcon$1 = SparkleIcon;

function SvgSparkleRectangleIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 2.75A.75.75 0 0 1 .75 2H8v1.5H1.5v9h13V10H16v3.25a.75.75 0 0 1-.75.75H.75a.75.75 0 0 1-.75-.75zm12.987-.14a.75.75 0 0 0-1.474 0l-.137.728a1.93 1.93 0 0 1-1.538 1.538l-.727.137a.75.75 0 0 0 0 1.474l.727.137c.78.147 1.39.758 1.538 1.538l.137.727a.75.75 0 0 0 1.474 0l.137-.727c.147-.78.758-1.39 1.538-1.538l.727-.137a.75.75 0 0 0 0-1.474l-.727-.137a1.93 1.93 0 0 1-1.538-1.538z",
      clipRule: "evenodd"
    })
  });
}
const SparkleRectangleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSparkleRectangleIcon
  });
});
SparkleRectangleIcon.displayName = 'SparkleRectangleIcon';
var SparkleRectangleIcon$1 = SparkleRectangleIcon;

function SvgSpeechBubbleIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M6 7a.75.75 0 1 1-1.5 0A.75.75 0 0 1 6 7M8 7.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5M10.75 7.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6 1a6 6 0 1 0 0 12v2.25a.75.75 0 0 0 1.28.53L10.061 13A6 6 0 0 0 10 1zM1.5 7A4.5 4.5 0 0 1 6 2.5h4a4.5 4.5 0 1 1 0 9h-.25a.75.75 0 0 0-.53.22L7.5 13.44v-1.19a.75.75 0 0 0-.75-.75H6A4.5 4.5 0 0 1 1.5 7",
      clipRule: "evenodd"
    })]
  });
}
const SpeechBubbleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSpeechBubbleIcon
  });
});
SpeechBubbleIcon.displayName = 'SpeechBubbleIcon';
var SpeechBubbleIcon$1 = SpeechBubbleIcon;

function SvgSpeechBubblePlusIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.25 9.5V7.75H5.5v-1.5h1.75V4.5h1.5v1.75h1.75v1.5H8.75V9.5z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6 1a6 6 0 0 0-6 6v.25a5.75 5.75 0 0 0 5 5.701v2.299a.75.75 0 0 0 1.28.53L9.06 13H10a6 6 0 0 0 0-12zM1.5 7A4.5 4.5 0 0 1 6 2.5h4a4.5 4.5 0 1 1 0 9H8.75a.75.75 0 0 0-.53.22L6.5 13.44v-1.19a.75.75 0 0 0-.75-.75A4.25 4.25 0 0 1 1.5 7.25z",
      clipRule: "evenodd"
    })]
  });
}
const SpeechBubblePlusIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSpeechBubblePlusIcon
  });
});
SpeechBubblePlusIcon.displayName = 'SpeechBubblePlusIcon';
var SpeechBubblePlusIcon$1 = SpeechBubblePlusIcon;

function SvgSpeechBubbleQuestionMarkIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6 1a6 6 0 0 0-6 6v.25a5.75 5.75 0 0 0 5 5.701v2.299a.75.75 0 0 0 1.28.53L9.06 13H10a6 6 0 0 0 0-12zM1.5 7A4.5 4.5 0 0 1 6 2.5h4a4.5 4.5 0 1 1 0 9H8.75a.75.75 0 0 0-.53.22L6.5 13.44v-1.19a.75.75 0 0 0-.75-.75A4.25 4.25 0 0 1 1.5 7.25zm8.707-1.689A2.25 2.25 0 0 1 8 8h-.75V6.5H8a.75.75 0 1 0-.75-.75h-1.5a2.25 2.25 0 0 1 4.457-.439M7.25 9.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0",
      clipRule: "evenodd"
    })
  });
}
const SpeechBubbleQuestionMarkIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSpeechBubbleQuestionMarkIcon
  });
});
SpeechBubbleQuestionMarkIcon.displayName = 'SpeechBubbleQuestionMarkIcon';
var SpeechBubbleQuestionMarkIcon$1 = SpeechBubbleQuestionMarkIcon;

function SvgSpeechBubbleStarIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M8 3.5a.5.5 0 0 1 .476.345l.56 1.728h1.817a.5.5 0 0 1 .294.904l-1.47 1.068.562 1.728a.5.5 0 0 1-.77.559L8 8.764 6.53 9.832a.5.5 0 0 1-.769-.56l.561-1.727-1.47-1.068a.5.5 0 0 1 .295-.904h1.816l.561-1.728A.5.5 0 0 1 8 3.5"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6 1a6 6 0 1 0 0 12v2.25a.75.75 0 0 0 1.28.53L10.061 13A6 6 0 0 0 10 1zM1.5 7A4.5 4.5 0 0 1 6 2.5h4a4.5 4.5 0 1 1 0 9h-.25a.75.75 0 0 0-.53.22L7.5 13.44v-1.19a.75.75 0 0 0-.75-.75H6A4.5 4.5 0 0 1 1.5 7",
      clipRule: "evenodd"
    })]
  });
}
const SpeechBubbleStarIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSpeechBubbleStarIcon
  });
});
SpeechBubbleStarIcon.displayName = 'SpeechBubbleStarIcon';
var SpeechBubbleStarIcon$1 = SpeechBubbleStarIcon;

function SvgStarFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M7.995 0a.75.75 0 0 1 .714.518l1.459 4.492h4.723a.75.75 0 0 1 .44 1.356l-3.82 2.776 1.459 4.492a.75.75 0 0 1-1.154.838l-3.82-2.776-3.821 2.776a.75.75 0 0 1-1.154-.838L4.48 9.142.66 6.366A.75.75 0 0 1 1.1 5.01h4.723L7.282.518A.75.75 0 0 1 7.995 0"
    })
  });
}
const StarFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgStarFillIcon
  });
});
StarFillIcon.displayName = 'StarFillIcon';
var StarFillIcon$1 = StarFillIcon;

function SvgStarIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7.995 0a.75.75 0 0 1 .714.518l1.459 4.492h4.723a.75.75 0 0 1 .44 1.356l-3.82 2.776 1.459 4.492a.75.75 0 0 1-1.154.838l-3.82-2.776-3.821 2.776a.75.75 0 0 1-1.154-.838L4.48 9.142.66 6.366A.75.75 0 0 1 1.1 5.01h4.723L7.282.518A.75.75 0 0 1 7.995 0m0 3.177-.914 2.814a.75.75 0 0 1-.713.519h-2.96l2.394 1.739a.75.75 0 0 1 .273.839l-.915 2.814 2.394-1.74a.75.75 0 0 1 .882 0l2.394 1.74-.914-2.814a.75.75 0 0 1 .272-.839l2.394-1.74H9.623a.75.75 0 0 1-.713-.518z",
      clipRule: "evenodd"
    })
  });
}
const StarIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgStarIcon
  });
});
StarIcon.displayName = 'StarIcon';
var StarIcon$1 = StarIcon;

function SvgStopCircleFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16M6.125 5.5a.625.625 0 0 0-.625.625v3.75c0 .345.28.625.625.625h3.75c.345 0 .625-.28.625-.625v-3.75a.625.625 0 0 0-.625-.625z",
      clipRule: "evenodd"
    })
  });
}
const StopCircleFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgStopCircleFillIcon
  });
});
StopCircleFillIcon.displayName = 'StopCircleFillIcon';
var StopCircleFillIcon$1 = StopCircleFillIcon;

function SvgStopCircleIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M5.5 6a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5H6a.5.5 0 0 1-.5-.5z",
      clipRule: "evenodd"
    })
  });
}
const StopCircleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgStopCircleIcon
  });
});
StopCircleIcon.displayName = 'StopCircleIcon';
var StopCircleIcon$1 = StopCircleIcon;

function SvgStopIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M4.5 4a.5.5 0 0 0-.5.5v7a.5.5 0 0 0 .5.5h7a.5.5 0 0 0 .5-.5v-7a.5.5 0 0 0-.5-.5z"
    })
  });
}
const StopIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgStopIcon
  });
});
StopIcon.displayName = 'StopIcon';
var StopIcon$1 = StopIcon;

function SvgStorefrontIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3.52 2.3a.75.75 0 0 1 .6-.3h7.76a.75.75 0 0 1 .6.3l2.37 3.158a.75.75 0 0 1 .15.45v.842q0 .059-.009.115A2.31 2.31 0 0 1 14 8.567v5.683a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75V8.567A2.31 2.31 0 0 1 1 6.75v-.841a.75.75 0 0 1 .15-.45zm7.605 6.068c.368.337.847.557 1.375.6V13.5h-9V8.968a2.3 2.3 0 0 0 1.375-.6c.411.377.96.607 1.563.607.602 0 1.15-.23 1.562-.607.411.377.96.607 1.563.607.602 0 1.15-.23 1.562-.607m2.375-2.21v.532l-.001.019a.813.813 0 0 1-1.623 0l-.008-.076a1 1 0 0 0 .012-.133V4zm-3.113.445a1 1 0 0 0-.013.106.813.813 0 0 1-1.624-.019V3.5h1.63v3q0 .053.007.103M7.25 3.5v3.19l-.001.019a.813.813 0 0 1-1.623 0l-.006-.064V3.5zM4.12 4 2.5 6.16v.531l.001.019a.813.813 0 0 0 1.619.045z",
      clipRule: "evenodd"
    })
  });
}
const StorefrontIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgStorefrontIcon
  });
});
StorefrontIcon.displayName = 'StorefrontIcon';
var StorefrontIcon$1 = StorefrontIcon;

function SvgStreamIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.52 7.48a6.5 6.5 0 0 1 12.722-1.298l-.09-.091a3.75 3.75 0 0 0-5.304 0L6.091 8.848a2.25 2.25 0 0 1-3.182 0L1.53 7.47zm.238 2.338A6.5 6.5 0 0 0 14.48 8.52l-.01.01-1.379-1.378a2.25 2.25 0 0 0-3.182 0L7.152 9.909a3.75 3.75 0 0 1-5.304 0z",
      clipRule: "evenodd"
    })
  });
}
const StreamIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgStreamIcon
  });
});
StreamIcon.displayName = 'StreamIcon';
var StreamIcon$1 = StreamIcon;

function SvgSyncIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M8 2.5a5.48 5.48 0 0 1 3.817 1.54l.009.009.5.451H11V6h4V2h-1.5v1.539l-.651-.588A7 7 0 0 0 1 8h1.5A5.5 5.5 0 0 1 8 2.5M1 10h4v1.5H3.674l.5.451.01.01A5.5 5.5 0 0 0 13.5 8h1.499a7 7 0 0 1-11.849 5.048L2.5 12.46V14H1z"
    })
  });
}
const SyncIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgSyncIcon
  });
});
SyncIcon.displayName = 'SyncIcon';
var SyncIcon$1 = SyncIcon;

function SvgTableGlassesIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H4v-1.5H2.5V7H5v2h1.5V7h3v2H11V7h2.5v2H15V1.75a.75.75 0 0 0-.75-.75zM13.5 5.5v-3h-11v3z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M11.75 10a.75.75 0 0 0-.707.5H9.957a.75.75 0 0 0-.708-.5H5.75a.75.75 0 0 0-.75.75v1.75a2.5 2.5 0 0 0 5 0V12h1v.5a2.5 2.5 0 0 0 5 0v-1.75a.75.75 0 0 0-.75-.75zm.75 2.5v-1h2v1a1 1 0 1 1-2 0m-6-1v1a1 1 0 1 0 2 0v-1z",
      clipRule: "evenodd"
    })]
  });
}
const TableGlassesIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTableGlassesIcon
  });
});
TableGlassesIcon.displayName = 'TableGlassesIcon';
var TableGlassesIcon$1 = TableGlassesIcon;

function SvgTableGlobeIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H7v-1.5h-.5V7H15V1.75a.75.75 0 0 0-.75-.75zM5 7v6.5H2.5V7zm8.5-1.5v-3h-11v3z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M11.625 7.25a4.375 4.375 0 1 0 0 8.75 4.375 4.375 0 0 0 0-8.75M9.952 9.287a10.5 10.5 0 0 0-.185 1.588H8.85a2.88 2.88 0 0 1 1.103-1.588m1.547-.02c-.116.41-.196.963-.23 1.608h.712c-.034-.646-.114-1.198-.23-1.608a2.5 2.5 0 0 0-.126-.353q-.06.13-.126.353m0 4.716c-.116-.41-.196-.963-.23-1.608h.712c-.034.646-.114 1.198-.23 1.608-.043.15-.086.265-.126.353a2.5 2.5 0 0 1-.126-.353m1.799-4.696c.098.475.158 1.016.185 1.588h.918a2.88 2.88 0 0 0-1.103-1.588m.185 3.088h.918a2.88 2.88 0 0 1-1.103 1.588c.098-.475.158-1.016.185-1.588m-4.634 0h.918c.027.572.087 1.113.185 1.588a2.88 2.88 0 0 1-1.103-1.588",
      clipRule: "evenodd"
    })]
  });
}
const TableGlobeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTableGlobeIcon
  });
});
TableGlobeIcon.displayName = 'TableGlobeIcon';
var TableGlobeIcon$1 = TableGlobeIcon;

function SvgTableIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75v3h11v-3zm0 11V7H5v6.5zm4 0h3V7h-3zM11 7v6.5h2.5V7z",
      clipRule: "evenodd"
    })
  });
}
const TableIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTableIcon
  });
});
TableIcon.displayName = 'TableIcon';
var TableIcon$1 = TableIcon;

function SvgTableLightningIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H8v-1.5H6.5V7h7v2H15V1.75a.75.75 0 0 0-.75-.75zM5 7H2.5v6.5H5zm8.5-1.5v-3h-11v3z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "m8.43 11.512 3-3.5 1.14.976-1.94 2.262H14a.75.75 0 0 1 .57 1.238l-3 3.5-1.14-.976 1.94-2.262H9a.75.75 0 0 1-.57-1.238"
    })]
  });
}
const TableLightningIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTableLightningIcon
  });
});
TableLightningIcon.displayName = 'TableLightningIcon';
var TableLightningIcon$1 = TableLightningIcon;

function SvgTableModelIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H6.5V7H15V1.75a.75.75 0 0 0-.75-.75zM5 7v6.5H2.5V7zm8.5-1.5v-3h-11v3z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7.25 8.5a1.25 1.25 0 1 1 2.488.177l1.48 1.481a2 2 0 0 1 1.563 0l.731-.731a1.25 1.25 0 1 1 1.06 1.06l-.73.732a2 2 0 0 1 0 1.562l.731.731a1.25 1.25 0 1 1-1.06 1.06l-.732-.73a2 2 0 0 1-2.636-1.092H9.5a1.25 1.25 0 1 1 0-1.5h.645l.013-.031-1.481-1.481A1.25 1.25 0 0 1 7.25 8.5M11.5 12a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0",
      clipRule: "evenodd"
    })]
  });
}
const TableModelIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTableModelIcon
  });
});
TableModelIcon.displayName = 'TableModelIcon';
var TableModelIcon$1 = TableModelIcon;

function SvgTableOnlineViewIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 15",
    ...props,
    children: [jsx("path", {
      stroke: "currentColor",
      strokeWidth: 1.5,
      d: "M.75 1A.25.25 0 0 1 1 .75h12a.25.25 0 0 1 .25.25v12a.25.25 0 0 1-.25.25H1A.25.25 0 0 1 .75 13zM1 5.25h12M5.25 13V6"
    }), jsx("path", {
      fill: "#fff",
      d: "M7 7h9v7.5H7z"
    }), jsx("path", {
      stroke: "currentColor",
      d: "M10.5 12H15m0 0-1.5-1.5M15 12l-1.5 1.5M12.5 9.5H8m0 0L9.5 11M8 9.5 9.5 8"
    })]
  });
}
const TableOnlineViewIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTableOnlineViewIcon
  });
});
TableOnlineViewIcon.displayName = 'TableOnlineViewIcon';
var TableOnlineViewIcon$1 = TableOnlineViewIcon;

function SvgTableStreamIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H6.5V7h3v1H11V7h2.5v1H15V1.75a.75.75 0 0 0-.75-.75zM5 7v6.5H2.5V7zm8.5-1.5v-3h-11v3z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M9.024 10.92a1.187 1.187 0 0 1 1.876.03 2.687 2.687 0 0 0 4.247.066l.439-.548-1.172-.937-.438.548a1.187 1.187 0 0 1-1.876-.03 2.687 2.687 0 0 0-4.247-.066l-.439.548 1.172.937z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M9.024 13.92a1.187 1.187 0 0 1 1.876.03 2.687 2.687 0 0 0 4.247.066l.439-.548-1.172-.937-.438.548a1.187 1.187 0 0 1-1.876-.03 2.687 2.687 0 0 0-4.247-.066l-.439.548 1.172.937z"
    })]
  });
}
const TableStreamIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTableStreamIcon
  });
});
TableStreamIcon.displayName = 'TableStreamIcon';
var TableStreamIcon$1 = TableStreamIcon;

function SvgTableVectorIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H8v-1.5H6.5V7h7v2H15V1.75a.75.75 0 0 0-.75-.75zM5 7H2.5v6.5H5zm8.5-1.5v-3h-11v3z",
      clipRule: "evenodd"
    }), jsx("circle", {
      cx: 12,
      cy: 12,
      r: 1,
      fill: "currentColor"
    }), jsx("circle", {
      cx: 12.5,
      cy: 9.5,
      r: 0.5,
      fill: "currentColor"
    }), jsx("circle", {
      cx: 9.5,
      cy: 12.5,
      r: 0.5,
      fill: "currentColor"
    }), jsx("circle", {
      cx: 13.75,
      cy: 13.75,
      r: 0.75,
      fill: "currentColor"
    }), jsx("circle", {
      cx: 9.75,
      cy: 9.75,
      r: 0.75,
      fill: "currentColor"
    }), jsx("path", {
      stroke: "currentColor",
      strokeWidth: 0.3,
      d: "M13.5 13.5 12 12m0 0 .5-2.5M12 12l-2.5.5M10 10l2 2"
    })]
  });
}
const TableVectorIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTableVectorIcon
  });
});
TableVectorIcon.displayName = 'TableVectorIcon';
var TableVectorIcon$1 = TableVectorIcon;

function SvgTableViewIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      fillRule: "evenodd",
      clipPath: "url(#TableViewIcon_svg__a)",
      clipRule: "evenodd",
      children: [jsx("path", {
        d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H4v-1.5H2.5V7H5v2h1.5V7h3v2H11V7h2.5v2H15V1.75a.75.75 0 0 0-.75-.75zM13.5 5.5v-3h-11v3z"
      }), jsx("path", {
        d: "M11.75 10a.75.75 0 0 0-.707.5H9.957a.75.75 0 0 0-.707-.5h-3.5a.75.75 0 0 0-.75.75v1.75a2.5 2.5 0 0 0 5 0V12h1v.5a2.5 2.5 0 0 0 5 0v-1.75a.75.75 0 0 0-.75-.75zm.75 2.5v-1h2v1a1 1 0 1 1-2 0m-6-1v1a1 1 0 1 0 2 0v-1z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const TableViewIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTableViewIcon
  });
});
TableViewIcon.displayName = 'TableViewIcon';
var TableViewIcon$1 = TableViewIcon;

function SvgTagIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M5 6a1 1 0 1 0 0-2 1 1 0 0 0 0 2"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.768 1.018a.75.75 0 0 0-.75.75v6.1c0 .199.079.39.22.53l6.884 6.885a.75.75 0 0 0 1.06 0l6.101-6.1a.75.75 0 0 0 0-1.061L8.4 1.237a.75.75 0 0 0-.53-.22zm6.884 12.674L2.518 7.557v-5.04h5.04l6.134 6.135z",
      clipRule: "evenodd"
    })]
  });
}
const TagIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTagIcon
  });
});
TagIcon.displayName = 'TagIcon';
var TagIcon$1 = TagIcon;

function SvgTargetIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M1.75 1H6v1.5H2.5V6H1V1.75A.75.75 0 0 1 1.75 1M14.25 1H10v1.5h3.5V6H15V1.75a.75.75 0 0 0-.75-.75M10 13.5h3.5V10H15v4.25a.75.75 0 0 1-.75.75H10zM2.5 13.5V10H1v4.25c0 .414.336.75.75.75H6v-1.5z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6M6.5 8a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0",
      clipRule: "evenodd"
    })]
  });
}
const TargetIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTargetIcon
  });
});
TargetIcon.displayName = 'TargetIcon';
var TargetIcon$1 = TargetIcon;

function SvgTerminalIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.5 13.5v-11h11v11zM1 1.7a.7.7 0 0 1 .7-.7h12.6a.7.7 0 0 1 .7.7v12.6a.7.7 0 0 1-.7.7H1.7a.7.7 0 0 1-.7-.7zm7 8.83h4v-1.5H8zm-.97-1.97-1.5 1.5L4.47 9l.97-.97-.97-.97L5.53 6l1.5 1.5.53.53z",
      clipRule: "evenodd"
    })
  });
}
const TerminalIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTerminalIcon
  });
});
TerminalIcon.displayName = 'TerminalIcon';
var TerminalIcon$1 = TerminalIcon;

function SvgTextBoxIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11zM5 6h2.25v5.5h1.5V6H11V4.5H5z",
      clipRule: "evenodd"
    })
  });
}
const TextBoxIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTextBoxIcon
  });
});
TextBoxIcon.displayName = 'TextBoxIcon';
var TextBoxIcon$1 = TextBoxIcon;

function SvgTextUnderlineIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M9.83 2.004v2.309c.437-.18 1.005-.318 1.5-.318.74 0 1.668.284 2.03.647.363.363.97 1.474.97 2.858s-.382 2.46-.966 2.889c-.564.415-1.293.606-2.035.606a3.7 3.7 0 0 1-1.5-.298v.298h-1.5V2.004zm0 5.496c0-.988.211-1.444.483-1.69.271-.246.763-.315 1.016-.315s.69.017 1.046.279.543.606.543 1.726-.23 1.413-.543 1.684-.723.311-1.046.311-.725-.106-1.016-.363c-.215-.189-.484-.644-.484-1.632m-8.327 5H0v1.503h16V12.5H1.503M4.25 5.245c-.582 0-1.16.16-1.755.365l-.49-1.418c.63-.218 1.404-.447 2.245-.447a3 3 0 0 1 3 3v4.25h-1.5v-.299a3.7 3.7 0 0 1-1.5.299c-.741 0-1.47-.192-2.035-.607a2.3 2.3 0 0 1-.965-1.893c0-.81.381-1.465.965-1.894.564-.415 1.294-.606 2.035-.606.494 0 .982.084 1.42.263a1.5 1.5 0 0 0-1.42-1.013M5.397 7.81c.23.17.353.39.353.685a.8.8 0 0 1-.353.685c-.25.183-.646.315-1.147.315s-.897-.132-1.147-.315a.8.8 0 0 1-.353-.685c0-.295.123-.516.353-.685.25-.184.646-.315 1.147-.315s.897.13 1.147.315",
      clipRule: "evenodd"
    })
  });
}
const TextUnderlineIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTextUnderlineIcon
  });
});
TextUnderlineIcon.displayName = 'TextUnderlineIcon';
var TextUnderlineIcon$1 = TextUnderlineIcon;

function SvgThumbsDownIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#ThumbsDownIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M13.655 2.274a.8.8 0 0 0-.528-.19h-1.044v5.833h1.044a.79.79 0 0 0 .79-.643V2.725a.8.8 0 0 0-.262-.451m-3.072 6.233V2.083H3.805a.58.58 0 0 0-.583.496v.001l-.92 6a.585.585 0 0 0 .583.67h3.782a.75.75 0 0 1 .75.75v2.667a1.25 1.25 0 0 0 .8 1.166zm1.238.91L9.352 14.97a.75.75 0 0 1-.685.446 2.75 2.75 0 0 1-2.75-2.75V10.75h-3.02A2.082 2.082 0 0 1 .82 8.354l.92-6A2.085 2.085 0 0 1 3.816.584h9.29a2.29 2.29 0 0 1 2.303 1.982 1 1 0 0 1 .007.1v4.667a1 1 0 0 1-.007.1 2.29 2.29 0 0 1-2.303 1.984z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const ThumbsDownIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgThumbsDownIcon
  });
});
ThumbsDownIcon.displayName = 'ThumbsDownIcon';
var ThumbsDownIcon$1 = ThumbsDownIcon;

function SvgThumbsUpIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#ThumbsUpIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M6.648 1.029a.75.75 0 0 1 .685-.446 2.75 2.75 0 0 1 2.75 2.75V5.25h3.02a2.083 2.083 0 0 1 2.079 2.396l-.92 6a2.085 2.085 0 0 1-2.08 1.77H2.668a2.083 2.083 0 0 1-2.084-2.082V8.667a2.083 2.083 0 0 1 2.084-2.083h1.512zM3.917 8.084h-1.25a.583.583 0 0 0-.584.583v4.667a.583.583 0 0 0 .584.583h1.25zm1.5 5.833h6.778a.58.58 0 0 0 .583-.496l.92-6a.584.584 0 0 0-.583-.67H9.333a.75.75 0 0 1-.75-.75V3.332a1.25 1.25 0 0 0-.8-1.166L5.417 7.493z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const ThumbsUpIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgThumbsUpIcon
  });
});
ThumbsUpIcon.displayName = 'ThumbsUpIcon';
var ThumbsUpIcon$1 = ThumbsUpIcon;

function SvgTrashIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M6 0a.75.75 0 0 0-.712.513L4.46 3H1v1.5h1.077l1.177 10.831A.75.75 0 0 0 4 16h8a.75.75 0 0 0 .746-.669L13.923 4.5H15V3h-3.46L10.713.513A.75.75 0 0 0 10 0zm3.96 3-.5-1.5H6.54L6.04 3zM3.585 4.5l1.087 10h6.654l1.087-10z",
      clipRule: "evenodd"
    })
  });
}
const TrashIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTrashIcon
  });
});
TrashIcon.displayName = 'TrashIcon';
var TrashIcon$1 = TrashIcon;

function SvgTreeIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.004 9.602a2.751 2.751 0 1 0 3.371 3.47 2.751 2.751 0 0 0 5.25 0 2.751 2.751 0 1 0 3.371-3.47A2.75 2.75 0 0 0 11.25 7h-2.5v-.604a2.751 2.751 0 1 0-1.5 0V7h-2.5a2.75 2.75 0 0 0-2.746 2.602M2.75 11a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m4.5-2.5h-2.5a1.25 1.25 0 0 0-1.242 1.106 2.76 2.76 0 0 1 1.867 1.822A2.76 2.76 0 0 1 7.25 9.604zm1.5 0v1.104c.892.252 1.6.942 1.875 1.824a2.76 2.76 0 0 1 1.867-1.822A1.25 1.25 0 0 0 11.25 8.5zM12 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0m-5.25 0a1.25 1.25 0 1 0 2.5 0 1.25 1.25 0 0 0-2.5 0M8 5a1.25 1.25 0 1 1 0-2.5A1.25 1.25 0 0 1 8 5",
      clipRule: "evenodd"
    })
  });
}
const TreeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgTreeIcon
  });
});
TreeIcon.displayName = 'TreeIcon';
var TreeIcon$1 = TreeIcon;

function SvgUnderlineIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M4.544 6.466 4.6 2.988l1.5.024-.056 3.478A1.978 1.978 0 1 0 10 6.522V3h1.5v3.522a3.478 3.478 0 1 1-6.956-.056M12 13H4v-1.5h8z",
      clipRule: "evenodd"
    })
  });
}
const UnderlineIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgUnderlineIcon
  });
});
UnderlineIcon.displayName = 'UnderlineIcon';
var UnderlineIcon$1 = UnderlineIcon;

function SvgUndoIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#UndoIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        d: "M2.81 6.5h8.69a3 3 0 0 1 0 6H7V14h4.5a4.5 4.5 0 0 0 0-9H2.81l2.72-2.72-1.06-1.06-4.53 4.53 4.53 4.53 1.06-1.06z"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M16 16H0V0h16z"
        })
      })
    })]
  });
}
const UndoIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgUndoIcon
  });
});
UndoIcon.displayName = 'UndoIcon';
var UndoIcon$1 = UndoIcon;

function SvgUploadIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M1 13.56h14v1.5H1zM12.53 5.53l-1.06 1.061-2.72-2.72v7.19h-1.5V3.87l-2.72 2.72-1.06-1.06L8 1z"
    })
  });
}
const UploadIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgUploadIcon
  });
});
UploadIcon.displayName = 'UploadIcon';
var UploadIcon$1 = UploadIcon;

function SvgUsbIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M8 0a.75.75 0 0 1 .65.375l1.299 2.25a.75.75 0 0 1-.65 1.125H8.75V9.5h2.75V8h-.25a.75.75 0 0 1-.75-.75v-2a.75.75 0 0 1 .75-.75h2a.75.75 0 0 1 .75.75v2a.75.75 0 0 1-.75.75H13v2.25a.75.75 0 0 1-.75.75h-3.5v1.668a1.75 1.75 0 1 1-1.5 0V11h-3.5a.75.75 0 0 1-.75-.75V7.832a1.75 1.75 0 1 1 1.5 0V9.5h2.75V3.75h-.549a.75.75 0 0 1-.65-1.125l1.3-2.25A.75.75 0 0 1 8 0"
    })
  });
}
const UsbIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgUsbIcon
  });
});
UsbIcon.displayName = 'UsbIcon';
var UsbIcon$1 = UsbIcon;

function SvgUserBadgeIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 5.25a2.75 2.75 0 1 0 0 5.5 2.75 2.75 0 0 0 0-5.5M6.75 8a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m4.401 2.5.386-.867A2.75 2.75 0 0 1 7.3 0h1.4a2.75 2.75 0 0 1 2.513 1.633l.386.867h1.651a.75.75 0 0 1 .75.75v12a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75v-12a.75.75 0 0 1 .75-.75zm1.756-.258A1.25 1.25 0 0 1 7.3 1.5h1.4c.494 0 .942.29 1.143.742l.114.258H6.043zM8 12a8.7 8.7 0 0 0-4.5 1.244V4h9v9.244A8.7 8.7 0 0 0 8 12m0 1.5c1.342 0 2.599.364 3.677 1H4.323A7.2 7.2 0 0 1 8 13.5",
      clipRule: "evenodd"
    })]
  });
}
const UserBadgeIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgUserBadgeIcon
  });
});
UserBadgeIcon.displayName = 'UserBadgeIcon';
var UserBadgeIcon$1 = UserBadgeIcon;

function SvgUserCircleIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M5.25 6.75a2.75 2.75 0 1 1 5.5 0 2.75 2.75 0 0 1-5.5 0M8 5.5A1.25 1.25 0 1 0 8 8a1.25 1.25 0 0 0 0-2.5",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 0 0-4.773 10.912A8.73 8.73 0 0 1 8 11c1.76 0 3.4.52 4.773 1.412A6.5 6.5 0 0 0 8 1.5m3.568 11.934A7.23 7.23 0 0 0 8 12.5a7.23 7.23 0 0 0-3.568.934A6.47 6.47 0 0 0 8 14.5a6.47 6.47 0 0 0 3.568-1.066",
      clipRule: "evenodd"
    })]
  });
}
const UserCircleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgUserCircleIcon
  });
});
UserCircleIcon.displayName = 'UserCircleIcon';
var UserCircleIcon$1 = UserCircleIcon;

function SvgUserGroupIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.25 3.75a2.75 2.75 0 1 1 5.5 0 2.75 2.75 0 0 1-5.5 0M5 2.5A1.25 1.25 0 1 0 5 5a1.25 1.25 0 0 0 0-2.5M9.502 14H.75a.75.75 0 0 1-.75-.75V11a.75.75 0 0 1 .164-.469C1.298 9.115 3.077 8 5.125 8c1.76 0 3.32.822 4.443 1.952A5.55 5.55 0 0 1 11.75 9.5c1.642 0 3.094.745 4.041 1.73a.75.75 0 0 1 .209.52v1.5a.75.75 0 0 1-.75.75zM1.5 12.5v-1.228C2.414 10.228 3.72 9.5 5.125 9.5c1.406 0 2.71.728 3.625 1.772V12.5zm8.75 0h4.25v-.432A4.17 4.17 0 0 0 11.75 11c-.53 0-1.037.108-1.5.293zM11.75 3.5a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5M11 5.75a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0",
      clipRule: "evenodd"
    })
  });
}
const UserGroupIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgUserGroupIcon
  });
});
UserGroupIcon.displayName = 'UserGroupIcon';
var UserGroupIcon$1 = UserGroupIcon;

function SvgUserIcon$1(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 1a3.25 3.25 0 1 0 0 6.5A3.25 3.25 0 0 0 8 1M6.25 4.25a1.75 1.75 0 1 1 3.5 0 1.75 1.75 0 0 1-3.5 0M8 9a8.74 8.74 0 0 0-6.836 3.287.75.75 0 0 0-.164.469v1.494c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75v-1.494a.75.75 0 0 0-.164-.469A8.74 8.74 0 0 0 8 9m-5.5 4.5v-.474A7.23 7.23 0 0 1 8 10.5c2.2 0 4.17.978 5.5 2.526v.474z",
      clipRule: "evenodd"
    })
  });
}
const UserIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgUserIcon$1
  });
});
UserIcon.displayName = 'UserIcon';
var UserIcon$1 = UserIcon;

function SvgVisibleIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      fillRule: "evenodd",
      clipPath: "url(#VisibleIcon_svg__a)",
      clipRule: "evenodd",
      children: [jsx("path", {
        d: "M8 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6M6.5 8a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0"
      }), jsx("path", {
        d: "M8 2A8.39 8.39 0 0 0 .028 7.777a.75.75 0 0 0 0 .466 8.389 8.389 0 0 0 15.944 0 .75.75 0 0 0 0-.466A8.39 8.39 0 0 0 8 2m0 10.52a6.89 6.89 0 0 1-6.465-4.51 6.888 6.888 0 0 1 12.93 0A6.89 6.89 0 0 1 8 12.52"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const VisibleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgVisibleIcon
  });
});
VisibleIcon.displayName = 'VisibleIcon';
var VisibleIcon$1 = VisibleIcon;

function SvgVisibleOffIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsxs("g", {
      fill: "currentColor",
      clipPath: "url(#VisibleOffIcon_svg__a)",
      children: [jsx("path", {
        fillRule: "evenodd",
        d: "m11.634 13.195 1.335 1.335 1.061-1.06-11.5-11.5-1.06 1.06 1.027 1.028a8.4 8.4 0 0 0-2.469 3.72.75.75 0 0 0 0 .465 8.39 8.39 0 0 0 11.606 4.951m-1.14-1.14-1.301-1.301a3 3 0 0 1-3.946-3.946L3.56 5.121A6.9 6.9 0 0 0 1.535 8.01a6.89 6.89 0 0 0 8.96 4.045",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M15.972 8.243a8.4 8.4 0 0 1-1.946 3.223l-1.06-1.06a6.9 6.9 0 0 0 1.499-2.396 6.89 6.89 0 0 0-8.187-4.293L5.082 2.522a8.389 8.389 0 0 1 10.89 5.256.75.75 0 0 1 0 .465"
      }), jsx("path", {
        d: "M11 8q0 .21-.028.411L7.589 5.028q.201-.027.41-.028a3 3 0 0 1 3 3"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
const VisibleOffIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgVisibleOffIcon
  });
});
VisibleOffIcon.displayName = 'VisibleOffIcon';
var VisibleOffIcon$1 = VisibleOffIcon;

function SvgWarningFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8.649 1.374a.75.75 0 0 0-1.298 0l-7.25 12.5A.75.75 0 0 0 .75 15h14.5a.75.75 0 0 0 .649-1.126zM7.25 10V6.5h1.5V10zm1.5 1.75a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0",
      clipRule: "evenodd"
    })
  });
}
const WarningFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgWarningFillIcon
  });
});
WarningFillIcon.displayName = 'WarningFillIcon';
var WarningFillIcon$1 = WarningFillIcon;

function SvgWorkflowsIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3.75 4a1.25 1.25 0 1 0 0-2.5 1.25 1.25 0 0 0 0 2.5m2.646-.5a2.751 2.751 0 1 1 0-1.5h5.229a3.375 3.375 0 0 1 .118 6.748L8.436 11.11a.75.75 0 0 1-.872 0l-3.3-2.357a1.875 1.875 0 0 0 .111 3.747h5.229a2.751 2.751 0 1 1 0 1.5H4.375a3.375 3.375 0 0 1-.118-6.748L7.564 4.89a.75.75 0 0 1 .872 0l3.3 2.357a1.875 1.875 0 0 0-.111-3.747zm7.104 9.75a1.25 1.25 0 1 1-2.5 0 1.25 1.25 0 0 1 2.5 0M8 6.422 5.79 8 8 9.578 10.21 8z",
      clipRule: "evenodd"
    })
  });
}
const WorkflowsIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgWorkflowsIcon
  });
});
WorkflowsIcon.displayName = 'WorkflowsIcon';
var WorkflowsIcon$1 = WorkflowsIcon;

function SvgWorkspacesIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M2.5 1a.75.75 0 0 0-.75.75v3c0 .414.336.75.75.75H6V4H3.25V2.5h9.5V4H10v1.5h3.5a.75.75 0 0 0 .75-.75v-3A.75.75 0 0 0 13.5 1z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 12.25c0-1.26.848-2.322 2.004-2.648A2.75 2.75 0 0 1 4.75 7h2.5V4h1.5v3h2.5a2.75 2.75 0 0 1 2.746 2.602 2.751 2.751 0 1 1-3.371 3.47 2.751 2.751 0 0 1-5.25 0A2.751 2.751 0 0 1 0 12.25M2.75 11a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m2.625.428a2.76 2.76 0 0 0-1.867-1.822A1.25 1.25 0 0 1 4.75 8.5h2.5v1.104c-.892.252-1.6.942-1.875 1.824M8.75 9.604V8.5h2.5c.642 0 1.17.483 1.242 1.106a2.76 2.76 0 0 0-1.867 1.822A2.76 2.76 0 0 0 8.75 9.604M12 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0m-5.25 0a1.25 1.25 0 1 0 2.5 0 1.25 1.25 0 0 0-2.5 0",
      clipRule: "evenodd"
    })]
  });
}
const WorkspacesIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgWorkspacesIcon
  });
});
WorkspacesIcon.displayName = 'WorkspacesIcon';
var WorkspacesIcon$1 = WorkspacesIcon;

function SvgWrenchIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 24 24",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#WrenchIcon_svg__a)",
      children: jsx("path", {
        stroke: "currentColor",
        strokeLinecap: "round",
        strokeLinejoin: "round",
        strokeWidth: 2,
        d: "M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.121 2.121 0 1 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h24v24H0z"
        })
      })
    })]
  });
}
const WrenchIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgWrenchIcon
  });
});
WrenchIcon.displayName = 'WrenchIcon';
var WrenchIcon$1 = WrenchIcon;

function SvgXCircleFillIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16m1.97-4.97L8 9.06l-1.97 1.97-1.06-1.06L6.94 8 4.97 6.03l1.06-1.06L8 6.94l1.97-1.97 1.06 1.06L9.06 8l1.97 1.97z",
      clipRule: "evenodd"
    })
  });
}
const XCircleFillIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgXCircleFillIcon
  });
});
XCircleFillIcon.displayName = 'XCircleFillIcon';
var XCircleFillIcon$1 = XCircleFillIcon;

function SvgXCircleIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M6.94 8 4.97 6.03l1.06-1.06L8 6.94l1.97-1.97 1.06 1.06L9.06 8l1.97 1.97-1.06 1.06L8 9.06l-1.97 1.97-1.06-1.06z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13",
      clipRule: "evenodd"
    })]
  });
}
const XCircleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgXCircleIcon
  });
});
XCircleIcon.displayName = 'XCircleIcon';
var XCircleIcon$1 = XCircleIcon;

function SvgZoomInIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 17",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M8.75 7.25H11v1.5H8.75V11h-1.5V8.75H5v-1.5h2.25V5h1.5z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 1a7 7 0 1 0 4.39 12.453l2.55 2.55 1.06-1.06-2.55-2.55A7 7 0 0 0 8 1M2.5 8a5.5 5.5 0 1 1 11 0 5.5 5.5 0 0 1-11 0",
      clipRule: "evenodd"
    })]
  });
}
const ZoomInIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgZoomInIcon
  });
});
ZoomInIcon.displayName = 'ZoomInIcon';
var ZoomInIcon$1 = ZoomInIcon;

function SvgZoomMarqueeSelection(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M1 1.75V4h1.5V2.5H4V1H1.75a.75.75 0 0 0-.75.75M14.25 1H12v1.5h1.5V4H15V1.75a.75.75 0 0 0-.75-.75M4 15H1.75a.75.75 0 0 1-.75-.75V12h1.5v1.5H4zM6 2.5h4V1H6zM1 10V6h1.5v4z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M4.053 9.27a5.217 5.217 0 1 1 9.397 3.122l1.69 1.69-1.062 1.06-1.688-1.69A5.217 5.217 0 0 1 4.053 9.27M9.27 5.555a3.717 3.717 0 1 0 0 7.434 3.717 3.717 0 0 0 0-7.434",
      clipRule: "evenodd"
    })]
  });
}
const ZoomMarqueeSelection = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgZoomMarqueeSelection
  });
});
ZoomMarqueeSelection.displayName = 'ZoomMarqueeSelection';
var ZoomMarqueeSelection$1 = ZoomMarqueeSelection;

function SvgZoomOutIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 17",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M11 7.25H5v1.5h6z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 8a7 7 0 1 1 12.45 4.392l2.55 2.55-1.06 1.061-2.55-2.55A7 7 0 0 1 1 8m7-5.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11",
      clipRule: "evenodd"
    })]
  });
}
const ZoomOutIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgZoomOutIcon
  });
});
ZoomOutIcon.displayName = 'ZoomOutIcon';
var ZoomOutIcon$1 = ZoomOutIcon;

function getAccordionEmotionStyles(clsPrefix, theme) {
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
        outlineColor: `${theme.colors.primary} !important`,
        outlineStyle: 'auto !important'
      }
    },
    [`& > ${classItem} > ${classHeader} > ${classArrow}`]: {
      fontSize: theme.general.iconSize,
      right: 12
    },
    [classArrow]: {
      color: theme.colors.textSecondary
    },
    [`& > ${classItemActive} > ${classHeader} > ${classArrow}`]: {
      transform: 'translateY(-50%) rotate(180deg)'
    },
    [classContent]: {
      border: '0 none',
      backgroundColor: theme.colors.backgroundPrimary
    },
    [classContentBox]: {
      padding: '8px 16px 16px'
    },
    [`& > ${classItem} > ${classHeader}`]: {
      padding: '6px 44px 6px 0',
      lineHeight: theme.typography.lineHeightBase
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getAccordionEmotionStyles;");
}
const AccordionPanel = _ref => {
  let {
    dangerouslySetAntdProps,
    dangerouslyAppendEmotionCSS,
    children,
    ...props
  } = _ref;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Collapse.Panel, {
      ...props,
      ...dangerouslySetAntdProps,
      css: dangerouslyAppendEmotionCSS,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
};
const Accordion = /* #__PURE__ */(() => {
  const Accordion = _ref2 => {
    let {
      dangerouslySetAntdProps,
      dangerouslyAppendEmotionCSS,
      displayMode = 'multiple',
      analyticsEvents,
      componentId,
      valueHasNoPii,
      onChange,
      ...props
    } = _ref2;
    const {
      theme,
      getPrefixedClassName
    } = useDesignSystemTheme();
    // While this component is called `Accordion` for correctness, in AntD it is called `Collapse`.
    const clsPrefix = getPrefixedClassName('collapse');
    const eventContext = useDesignSystemEventComponentCallbacks({
      componentType: DesignSystemEventProviderComponentTypes.Accordion,
      componentId,
      analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
      valueHasNoPii
    });
    const onChangeWrapper = useCallback(newValue => {
      if (Array.isArray(newValue)) {
        eventContext.onValueChange(JSON.stringify(newValue));
      } else {
        eventContext.onValueChange(newValue);
      }
      onChange === null || onChange === void 0 || onChange(newValue);
    }, [eventContext, onChange]);
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Collapse
      // eslint-disable-next-line @databricks/no-unstable-nested-components -- go/no-nested-components
      , {
        expandIcon: () => jsx(ChevronDownIcon$1, {}),
        expandIconPosition: "right",
        accordion: displayMode === 'single',
        ...props,
        ...dangerouslySetAntdProps,
        css: [getAccordionEmotionStyles(clsPrefix, theme), dangerouslyAppendEmotionCSS, addDebugOutlineStylesIfEnabled(theme), process.env.NODE_ENV === "production" ? "" : ";label:Accordion;"],
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
  error: DangerFillIcon$1,
  warning: WarningFillIcon$1,
  success: CheckCircleFillIcon$1,
  info: InfoFillIcon$1
};
const SeverityIcon = /*#__PURE__*/forwardRef(function (props, ref) {
  const FilledIcon = filledIconsMap[props.severity];
  return jsx(FilledIcon, {
    ref: ref,
    ...props
  });
});

const Alert = _ref => {
  let {
    componentId,
    analyticsEvents,
    dangerouslySetAntdProps,
    closable = true,
    closeIconLabel = 'Close alert',
    onClose,
    ...props
  } = _ref;
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Alert,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : []
  });
  const closeButtonEventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId: componentId ? `${componentId}.close` : 'codegen_design_system_src_design_system_alert_alert.tsx_50',
    analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick]
  });
  const {
    elementRef: closeIconRef
  } = useNotifyOnFirstView({
    onView: eventContext.onView
  });
  const clsPrefix = getPrefixedClassName('alert');
  const mergedProps = {
    ...props,
    type: props.type || 'error',
    showIcon: true,
    closable
  };
  useEffect(() => {
    if (closeIconRef.current) {
      var _closeIconRef$current;
      closeIconRef.current.removeAttribute('aria-label');
      (_closeIconRef$current = closeIconRef.current.closest('button')) === null || _closeIconRef$current === void 0 || _closeIconRef$current.setAttribute('aria-label', closeIconLabel);
    }
  }, [mergedProps.closable, closeIconLabel, closeIconRef]);
  const onCloseWrapper = e => {
    closeButtonEventContext.onClick();
    onClose === null || onClose === void 0 || onClose(e);
  };
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Alert$1, {
      ...addDebugOutlineIfEnabled(),
      ...mergedProps,
      onClose: onCloseWrapper,
      className: classnames(mergedProps.className),
      css: getAlertEmotionStyles(clsPrefix, theme, mergedProps),
      icon: jsx(SeverityIcon, {
        severity: mergedProps.type
      })
      // Antd calls this prop `closeText` but we can use it to set any React element to replace the close icon.
      ,
      closeText: mergedProps.closable && jsx(CloseIcon, {
        ref: closeIconRef,
        "aria-label": closeIconLabel,
        css: /*#__PURE__*/css({
          fontSize: theme.general.iconSize
        }, process.env.NODE_ENV === "production" ? "" : ";label:Alert;")
      })
      // Always set a description for consistent styling (e.g. icon size)
      ,
      description: props.description || ' ',
      ...dangerouslySetAntdProps
    })
  });
};
const getAlertEmotionStyles = (clsPrefix, theme, props) => {
  const classCloseIcon = `.${clsPrefix}-close-icon`;
  const classCloseButton = `.${clsPrefix}-close-button`;
  const classCloseText = `.${clsPrefix}-close-text`;
  const classDescription = `.${clsPrefix}-description`;
  const classMessage = `.${clsPrefix}-message`;
  const classWithDescription = `.${clsPrefix}-with-description`;
  const classWithIcon = `.${clsPrefix}-icon`;
  const ALERT_ICON_HEIGHT = 16;
  const ALERT_ICON_FONT_SIZE = 16;
  const styles = {
    // General
    padding: theme.spacing.sm,
    [`${classMessage}, &${classWithDescription} ${classMessage}`]: {
      // TODO(giles): These three rules are all the same as the H3 styles. We can refactor them out into a shared object.
      fontSize: theme.typography.fontSizeBase,
      fontWeight: theme.typography.typographyBoldFontWeight,
      lineHeight: theme.typography.lineHeightBase
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
        outlineColor: theme.colors.primary
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
      marginTop: 2
    },
    [`${classCloseIcon}, ${classCloseButton}, ${classCloseText} > span`]: {
      lineHeight: theme.typography.lineHeightBase,
      height: ALERT_ICON_HEIGHT,
      width: ALERT_ICON_HEIGHT,
      fontSize: ALERT_ICON_FONT_SIZE,
      marginTop: 2,
      '& > span': {
        lineHeight: theme.typography.lineHeightBase
      }
    },
    // No description
    ...(!props.description && {
      display: 'flex',
      alignItems: 'center',
      [classWithIcon]: {
        fontSize: ALERT_ICON_FONT_SIZE,
        marginTop: 0
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
    }),
    // Warning
    ...(props.type === 'warning' && {
      color: theme.colors.textValidationWarning,
      borderColor: theme.colors.yellow300
    }),
    // Error
    ...(props.type === 'error' && {
      color: theme.colors.textValidationDanger,
      borderColor: theme.colors.red300
    }),
    // Banner
    ...(props.banner && {
      borderStyle: 'solid',
      borderWidth: `${theme.general.borderWidth}px 0`
    }),
    // After closed
    '&[data-show="false"]': {
      borderWidth: 0,
      padding: 0,
      width: 0,
      height: 0
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getAlertEmotionStyles;");
};

const getGlobalStyles = theme => {
  return /*#__PURE__*/css({
    'body, .mfe-root': {
      backgroundColor: theme.colors.backgroundPrimary,
      color: theme.colors.textPrimary,
      '--dubois-global-background-color': theme.colors.backgroundPrimary,
      '--dubois-global-color': theme.colors.textPrimary
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getGlobalStyles;");
};
const ApplyGlobalStyles = () => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Global, {
    styles: getGlobalStyles(theme)
  });
};

/**
 * @deprecated Use `TypeaheadCombobox` instead.
 */
const AutoComplete = /* #__PURE__ */(() => {
  const AutoComplete = _ref => {
    let {
      dangerouslySetAntdProps,
      ...props
    } = _ref;
    const {
      theme
    } = useDesignSystemTheme();
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(AutoComplete$1, {
        ...addDebugOutlineIfEnabled(),
        dropdownStyle: {
          boxShadow: theme.general.shadowLow,
          ...getDarkModePortalStyles(theme)
        },
        ...props,
        ...dangerouslySetAntdProps,
        css: /*#__PURE__*/css(getAnimationCss(theme.options.enableAnimation), process.env.NODE_ENV === "production" ? "" : ";label:AutoComplete;")
      })
    });
  };
  AutoComplete.Option = AutoComplete$1.Option;
  return AutoComplete;
})();

function SvgDatabricksIcon(props) {
  return jsxs("svg", {
    viewBox: "0 0 24 24",
    width: "100%",
    height: "100%",
    xmlns: "http://www.w3.org/2000/svg",
    ...props,
    children: [jsx("rect", {
      width: "24",
      height: "24",
      fill: "#FF3621"
    }), jsx("path", {
      d: "m18.8 10.515-6.8228 3.945-7.3059-4.215-0.35138 0.195v3.06l7.6573 4.41 6.8228-3.93v1.62l-6.8228 3.945-7.3059-4.215-0.35138 0.195v0.525l7.6573 4.41 7.6427-4.41v-3.06l-0.3514-0.195-7.2913 4.2-6.8374-3.93v-1.62l6.8374 3.93 7.6427-4.41v-3.015l-0.3807-0.225-7.262 4.185-6.486-3.72 6.486-3.735 5.3294 3.075 0.4685-0.27v-0.375l-5.7979-3.345-7.6573 4.41v0.48l7.6573 4.41 6.8228-3.945v1.62z",
      fill: "#fff"
    })]
  });
}

function SvgUserIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "100%",
    height: "auto",
    fill: "none",
    viewBox: "0 0 400 400",
    ...props,
    children: jsx("path", {
      d: "M200 69.333c-44.873 0-81.25 36.377-81.25 81.25s36.377 81.25 81.25 81.25 81.25-36.377 81.25-81.25-36.377-81.25-81.25-81.25Zm0 200c-69.162 0-130.835 32.119-170.89 82.181A18.748 18.748 0 0 0 25 363.228v37.355c0 10.356 8.395 18.75 18.75 18.75h312.5c10.355 0 18.75-8.394 18.75-18.75v-37.355c0-4.258-1.449-8.39-4.11-11.714-40.055-50.062-101.728-82.181-170.89-82.181Z",
      fill: "#5F7281"
    })
  });
}

/**
 * `LegacyTooltip` is deprecated in favor of the new `Tooltip` component
 * @deprecated
 */
const LegacyTooltip = _ref => {
  let {
    children,
    title,
    placement = 'top',
    dataTestId,
    dangerouslySetAntdProps,
    silenceScreenReader = false,
    useAsLabel = false,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const tooltipRef = useRef(null);
  const duboisId = useUniqueId('dubois-tooltip-component-');
  const id = dangerouslySetAntdProps !== null && dangerouslySetAntdProps !== void 0 && dangerouslySetAntdProps.id ? dangerouslySetAntdProps === null || dangerouslySetAntdProps === void 0 ? void 0 : dangerouslySetAntdProps.id : duboisId;
  if (!title) {
    return jsx(React__default.Fragment, {
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
  const liveTitle = title && /*#__PURE__*/React__default.isValidElement(title) ? /*#__PURE__*/React__default.cloneElement(title, titleProps) : jsx("span", {
    ...titleProps,
    children: title
  });
  const ariaProps = {
    'aria-hidden': false
  };
  const addAriaProps = e => {
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
  const removeAriaProps = e => {
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
    onMouseEnter: e => {
      addAriaProps(e);
    },
    onMouseLeave: e => {
      removeAriaProps(e);
    },
    onFocus: e => {
      addAriaProps(e);
    },
    onBlur: e => {
      removeAriaProps(e);
    }
  };
  const childWithProps = /*#__PURE__*/React__default.isValidElement(children) ? ( /*#__PURE__*/React__default.cloneElement(children, {
    ...ariaProps,
    ...interactionProps,
    ...children.props
  })) : _isNil(children) ? children : jsx("span", {
    ...ariaProps,
    ...interactionProps,
    children: children
  });
  const {
    overlayInnerStyle,
    overlayStyle,
    ...delegatedDangerouslySetAntdProps
  } = dangerouslySetAntdProps || {};
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Tooltip, {
      id: id,
      ref: tooltipRef,
      title: liveTitle,
      placement: placement
      // Always trigger on hover and focus
      ,
      trigger: ['hover', 'focus'],
      overlayInnerStyle: {
        backgroundColor: '#2F3941',
        lineHeight: '22px',
        padding: '4px 8px',
        boxShadow: theme.general.shadowLow,
        ...overlayInnerStyle,
        ...getDarkModePortalStyles(theme)
      },
      overlayStyle: {
        zIndex: theme.options.zIndexBase + 70,
        ...overlayStyle
      },
      css: /*#__PURE__*/css({
        ...getAnimationCss(theme.options.enableAnimation)
      }, process.env.NODE_ENV === "production" ? "" : ";label:LegacyTooltip;"),
      ...delegatedDangerouslySetAntdProps,
      ...props,
      children: childWithProps
    })
  });
};

/**
 * `LegacyInfoTooltip` is deprecated in favor of the new `InfoTooltip` component
 * @deprecated
 */
const LegacyInfoTooltip = _ref => {
  let {
    title,
    tooltipProps,
    iconTitle,
    isKeyboardFocusable = true,
    ...iconProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(LegacyTooltip, {
    useAsLabel: true,
    title: title,
    ...tooltipProps,
    children: jsx("span", {
      ...addDebugOutlineIfEnabled(),
      style: {
        display: 'inline-flex'
      },
      children: jsx(InfoCircleOutlined, {
        tabIndex: isKeyboardFocusable ? 0 : -1,
        "aria-hidden": "false",
        "aria-label": iconTitle,
        alt: iconTitle,
        css: /*#__PURE__*/css({
          fontSize: theme.typography.fontSizeSm,
          color: theme.colors.textSecondary
        }, process.env.NODE_ENV === "production" ? "" : ";label:LegacyInfoTooltip;"),
        ...iconProps
      })
    })
  });
};

const InfoPopover = _ref => {
  let {
    children,
    popoverProps,
    iconTitle,
    iconProps,
    isKeyboardFocusable = true,
    ariaLabel = 'More details'
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const [open, setOpen] = useState(false);
  const handleKeyDown = event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      setOpen(!open);
    }
  };
  return jsxs(Root$5, {
    open: open,
    onOpenChange: setOpen,
    children: [jsx(Trigger$3, {
      asChild: true,
      children: jsx("span", {
        style: {
          display: 'inline-flex',
          cursor: 'pointer'
        }
        // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
        ,
        tabIndex: isKeyboardFocusable ? 0 : -1,
        onKeyDown: handleKeyDown,
        "aria-label": iconTitle ? undefined : ariaLabel,
        children: jsx(InfoCircleOutlined, {
          "aria-hidden": iconTitle ? false : true,
          title: iconTitle,
          "aria-label": iconTitle,
          css: /*#__PURE__*/css({
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary
          }, process.env.NODE_ENV === "production" ? "" : ";label:InfoPopover;"),
          ...iconProps
        })
      })
    }), jsxs(Content$4, {
      align: "start",
      ...popoverProps,
      children: [children, jsx(Arrow$2, {})]
    })]
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$s() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$h = process.env.NODE_ENV === "production" ? {
  name: "1amee4m",
  styles: "line-height:0"
} : {
  name: "c77cjr-trigger",
  styles: "line-height:0;label:trigger;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$s
};
const OverflowPopover = _ref => {
  let {
    items,
    renderLabel,
    tooltipText,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const [showTooltip, setShowTooltip] = useState(true);
  const label = `+${items.length}`;
  let trigger = jsx("span", {
    css: _ref2$h,
    ...addDebugOutlineIfEnabled(),
    children: jsx(Trigger$3, {
      asChild: true,
      children: jsx(Button, {
        componentId: "something",
        type: "link",
        children: renderLabel ? renderLabel(label) : label
      })
    })
  });
  if (showTooltip) {
    trigger = jsx(LegacyTooltip, {
      title: tooltipText || 'See more items',
      children: trigger
    });
  }
  return jsxs(Root$5, {
    onOpenChange: open => setShowTooltip(!open),
    children: [trigger, jsx(Content$4, {
      align: "start",
      ...props,
      ...addDebugOutlineIfEnabled(),
      children: jsx("div", {
        css: /*#__PURE__*/css({
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs
        }, process.env.NODE_ENV === "production" ? "" : ";label:OverflowPopover;"),
        children: items.map((item, index) => jsx("div", {
          children: item
        }, `overflow-${index}`))
      })
    })]
  });
};

const SIZE = new Map([['xl', {
  avatarSize: 48,
  fontSize: 18,
  groupShift: 12,
  iconSize: 24
}], ['lg', {
  avatarSize: 40,
  fontSize: 16,
  groupShift: 8,
  iconSize: 20
}], ['md', {
  avatarSize: 32,
  fontSize: 14,
  groupShift: 4,
  iconSize: 16
}], ['sm', {
  avatarSize: 24,
  fontSize: 12,
  groupShift: 4,
  iconSize: 14
}], ['xs', {
  avatarSize: 20,
  fontSize: 12,
  groupShift: 2,
  iconSize: 12
}]]);
const DEFAULT_SIZE = 'sm';
function getAvatarEmotionStyles(_ref) {
  let {
    backgroundColor,
    size = DEFAULT_SIZE,
    theme
  } = _ref;
  const {
    avatarSize,
    fontSize,
    iconSize
  } = SIZE.get(size);
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
      borderRadius: theme.borders.borderRadiusMd,
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

/** Generate random number from a string between 0 - (maxRange - 1) */
function getRandomNumberFromString(_ref2) {
  let {
    value,
    maxRange
  } = _ref2;
  let hash = 0;
  let char = 0;
  if (value.length === 0) return hash;
  for (let i = 0; i < value.length; i++) {
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
  switch (randomNumber) {
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
  const {
    theme
  } = useDesignSystemTheme();
  const styles = getAvatarEmotionStyles({
    size: props.size,
    theme,
    backgroundColor: 'backgroundColor' in props ? props.backgroundColor : undefined
  });
  switch (props.type) {
    case 'entity':
      if ('src' in props && props.src) {
        return jsx("img", {
          css: [styles.default, styles.img, styles.system, process.env.NODE_ENV === "production" ? "" : ";label:Avatar;"],
          src: props.src,
          alt: props.label
        });
      }
      if ('icon' in props && props.icon) {
        return jsx("div", {
          css: [styles.default, styles.system, styles.icon, process.env.NODE_ENV === "production" ? "" : ";label:Avatar;"],
          role: "img",
          "aria-label": props.label,
          children: props.icon
        });
      }

      // display first initial of name when no image / icon is provided
      return jsx("div", {
        css: [styles.default, styles.system, styles.icon, {
          backgroundColor: getAvatarBackgroundColor(props.label, theme)
        }, process.env.NODE_ENV === "production" ? "" : ";label:Avatar;"],
        children: jsx("abbr", {
          css: styles.abbr,
          title: props.label,
          children: props.label.substring(0, 1)
        })
      });
    case 'user':
      if ('label' in props && props.label.trim()) {
        if (props.src) {
          return jsx("img", {
            css: [styles.default, styles.img, styles.user, process.env.NODE_ENV === "production" ? "" : ";label:Avatar;"],
            src: props.src,
            alt: props.label
          });
        } else if (props.icon) {
          return jsx("div", {
            css: [styles.default, styles.user, styles.icon, process.env.NODE_ENV === "production" ? "" : ";label:Avatar;"],
            role: "img",
            "aria-label": props.label,
            children: props.icon
          });
        }
        // display first initial of name when no image / icon is provided
        return jsx("div", {
          css: [styles.default, styles.user, styles.icon, {
            backgroundColor: getAvatarBackgroundColor(props.label, theme)
          }, process.env.NODE_ENV === "production" ? "" : ";label:Avatar;"],
          children: jsx("abbr", {
            css: styles.abbr,
            title: props.label,
            children: props.label.substring(0, 1)
          })
        });
      }

      // default to user icon when no user info is provided
      return jsx("div", {
        css: [styles.default, styles.user, styles.icon, styles.userIcon, process.env.NODE_ENV === "production" ? "" : ";label:Avatar;"],
        role: "img",
        "aria-label": "user",
        children: jsx(SvgUserIcon, {})
      });
  }
}
function DBAssistantAvatar(_ref3) {
  let {
    size
  } = _ref3;
  return jsx(Avatar, {
    size: size,
    type: "entity",
    label: "Assistant",
    icon: jsx(SvgDatabricksIcon, {})
  });
}
function AssistantAvatar(_ref4) {
  let {
    backgroundColor,
    size
  } = _ref4;
  return jsx(Avatar, {
    backgroundColor: backgroundColor,
    size: size,
    type: "entity",
    label: "Assistant",
    icon: jsx(SparkleDoubleIcon$1, {})
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
function AvatarGroup(_ref5) {
  let {
    size = DEFAULT_SIZE,
    users
  } = _ref5;
  const {
    theme
  } = useDesignSystemTheme();
  const styles = getAvatarGroupEmotionStyles(theme);
  const displayedUsers = useMemo(() => users.slice(0, MAX_AVATAR_GROUP_USERS), [users]);
  const extraUsers = useMemo(() => users.slice(MAX_AVATAR_GROUP_USERS), [users]);
  const {
    avatarSize,
    groupShift: avatarShift,
    fontSize
  } = SIZE.get(size);
  return jsxs("div", {
    css: styles.container,
    children: [jsx("div", {
      css: /*#__PURE__*/css({
        ...styles.avatarsContainer,
        width: (avatarSize + 2 - avatarShift) * displayedUsers.length + avatarShift
      }, process.env.NODE_ENV === "production" ? "" : ";label:AvatarGroup;"),
      children: displayedUsers.map((user, idx) => jsx("div", {
        css: /*#__PURE__*/css({
          ...styles.avatar,
          left: -avatarShift * idx
        }, process.env.NODE_ENV === "production" ? "" : ";label:AvatarGroup;"),
        children: jsx(Avatar, {
          size: size,
          type: "user",
          ...user
        })
      }, `${user.label}-idx`))
    }), extraUsers.length > 0 && jsx(OverflowPopover, {
      items: extraUsers.map(user => user.label),
      tooltipText: "Show more users",
      renderLabel: label => jsx("span", {
        css: /*#__PURE__*/css({
          fontSize: `${fontSize}px !important`
        }, process.env.NODE_ENV === "production" ? "" : ";label:AvatarGroup;"),
        children: label
      })
    })]
  });
}

const Breadcrumb = /* #__PURE__ */(() => {
  const Breadcrumb = _ref => {
    let {
      dangerouslySetAntdProps,
      includeTrailingCaret = true,
      ...props
    } = _ref;
    const {
      theme,
      classNamePrefix
    } = useDesignSystemTheme();
    const separatorClass = `.${classNamePrefix}-breadcrumb-separator`;
    const styles = /*#__PURE__*/css({
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
          outlineColor: `${theme.colors.primary} !important`,
          outlineStyle: 'auto !important'
        }
      },
      [separatorClass]: {
        fontSize: theme.general.iconFontSize
      },
      '& > span': {
        display: 'inline-flex',
        alignItems: 'center'
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:styles;");
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsxs(Breadcrumb$1, {
        ...addDebugOutlineIfEnabled(),
        separator: jsx(ChevronRightIcon, {}),
        ...props,
        ...dangerouslySetAntdProps,
        css: /*#__PURE__*/css(getAnimationCss(theme.options.enableAnimation), styles, process.env.NODE_ENV === "production" ? "" : ";label:Breadcrumb;"),
        children: [props.children, includeTrailingCaret && props.children && jsx(Breadcrumb.Item, {
          children: " "
        })]
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
  const seedValue = seed.split('').map(char => char.charCodeAt(0)).reduce((prev, curr) => prev + curr, 0);

  // This is a simple sine wave function that will always return a number between 0 and 1.
  // This produces a value akin to `Math.random()`, but has deterministic output.
  // Of course, sine curves are not a perfectly random distribution between 0 and 1, but
  // it's close enough for our purposes.
  return Math.sin(seedValue) / 2 + 0.5;
}

// This is a simple Fisher-Yates shuffler using the above PRNG.
function shuffleArray(arr, seed) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(pseudoRandomNumberGeneratorFromSeed(seed + String(i)) * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// Finally, we shuffle a list off offsets to apply to the widths of the cells.
// This ensures that the cells are not all the same width, but that they are
// random to produce a more realistic looking skeleton.
function getOffsets(seed) {
  return shuffleArray([48, 24, 0], seed);
}
const skeletonLoading = keyframes({
  '0%': {
    backgroundPosition: '100% 50%'
  },
  '100%': {
    backgroundPosition: '0 50%'
  }
});
const genSkeletonAnimatedColor = function (theme) {
  let frameRate = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 60;
  // TODO: Pull this from the themes; it's not currently available.
  const color = theme.isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(31, 38, 45, 0.1)';
  // Light mode value copied from Ant's Skeleton animation
  const colorGradientEnd = theme.isDarkMode ? 'rgba(99, 99, 99, 0.24)' : 'rgba(129, 129, 129, 0.24)';
  return /*#__PURE__*/css({
    animationDuration: '1.4s',
    background: `linear-gradient(90deg, ${color} 25%, ${colorGradientEnd} 37%, ${color} 63%)`,
    backgroundSize: '400% 100%',
    animationName: skeletonLoading,
    animationTimingFunction: `steps(${frameRate}, end)`,
    // Based on data from perf dashboard, p95 loading time goes up to about 20s, so about 14 iterations is needed.
    animationIterationCount: 14,
    '@media only percy': {
      animation: 'none'
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:genSkeletonAnimatedColor;");
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$r() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const GenericContainerStyles = process.env.NODE_ENV === "production" ? {
  name: "12h7em6",
  styles: "cursor:progress;border-radius:var(--border-radius)"
} : {
  name: "19fx6jo-GenericContainerStyles",
  styles: "cursor:progress;border-radius:var(--border-radius);label:GenericContainerStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$r
};
const GenericSkeleton = _ref => {
  let {
    label,
    frameRate = 60,
    style,
    loading = true,
    loadingDescription = 'GenericSkeleton',
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs("div", {
    ...addDebugOutlineIfEnabled(),
    css: [GenericContainerStyles, genSkeletonAnimatedColor(theme, frameRate), process.env.NODE_ENV === "production" ? "" : ";label:GenericSkeleton;"],
    style: {
      ...style,
      ['--border-radius']: `${theme.general.borderRadiusBase}px`
    },
    ...restProps,
    children: [loading && jsx(LoadingState, {
      description: loadingDescription
    }), jsx("span", {
      css: visuallyHidden,
      children: label
    })]
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$q() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const paragraphContainerStyles = process.env.NODE_ENV === "production" ? {
  name: "sj05g9",
  styles: "cursor:progress;width:100%;height:20px;display:flex;justify-content:flex-start;align-items:center"
} : {
  name: "u3a3v7-paragraphContainerStyles",
  styles: "cursor:progress;width:100%;height:20px;display:flex;justify-content:flex-start;align-items:center;label:paragraphContainerStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$q
};
const paragraphFillStyles = process.env.NODE_ENV === "production" ? {
  name: "10nptxl",
  styles: "border-radius:var(--border-radius);height:8px"
} : {
  name: "h6xifd-paragraphFillStyles",
  styles: "border-radius:var(--border-radius);height:8px;label:paragraphFillStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$q
};
const ParagraphSkeleton = _ref => {
  let {
    label,
    seed = '',
    frameRate = 60,
    style,
    loading = true,
    loadingDescription = 'ParagraphSkeleton',
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const offsetWidth = getOffsets(seed)[0];
  return jsxs("div", {
    ...addDebugOutlineIfEnabled(),
    css: paragraphContainerStyles,
    style: {
      ...style,
      ['--border-radius']: `${theme.general.borderRadiusBase}px`
    },
    ...restProps,
    children: [loading && jsx(LoadingState, {
      description: loadingDescription
    }), jsx("span", {
      css: visuallyHidden,
      children: label
    }), jsx("div", {
      "aria-hidden": true,
      css: [paragraphFillStyles, genSkeletonAnimatedColor(theme, frameRate), {
        width: `calc(100% - ${offsetWidth}px)`
      }, process.env.NODE_ENV === "production" ? "" : ";label:ParagraphSkeleton;"]
    })]
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$p() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const titleContainerStyles = process.env.NODE_ENV === "production" ? {
  name: "116rc6i",
  styles: "cursor:progress;width:100%;height:28px;display:flex;justify-content:flex-start;align-items:center"
} : {
  name: "1dar8xl-titleContainerStyles",
  styles: "cursor:progress;width:100%;height:28px;display:flex;justify-content:flex-start;align-items:center;label:titleContainerStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$p
};
const titleFillStyles = process.env.NODE_ENV === "production" ? {
  name: "9fmdbb",
  styles: "border-radius:var(--border-radius);height:12px;width:100%"
} : {
  name: "1vyd6dg-titleFillStyles",
  styles: "border-radius:var(--border-radius);height:12px;width:100%;label:titleFillStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$p
};
const TitleSkeleton = _ref => {
  let {
    label,
    frameRate = 60,
    style,
    loading = true,
    loadingDescription = 'TitleSkeleton',
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs("div", {
    ...addDebugOutlineIfEnabled(),
    css: titleContainerStyles,
    style: {
      ...style,
      ['--border-radius']: `${theme.general.borderRadiusBase}px`
    },
    ...restProps,
    children: [loading && jsx(LoadingState, {
      description: loadingDescription
    }), jsx("span", {
      css: visuallyHidden,
      children: label
    }), jsx("div", {
      "aria-hidden": true,
      css: [titleFillStyles, genSkeletonAnimatedColor(theme, frameRate), process.env.NODE_ENV === "production" ? "" : ";label:TitleSkeleton;"]
    })]
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$o() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
// Class names that can be used to reference children within
// Should not be used outside of design system
// TODO: PE-239 Maybe we could add "dangerous" into the names or make them completely random.
function randomString() {
  return _times(20, () => _random(35).toString(36)).join('');
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
  cell: process.env.NODE_ENV === "production" ? {
    name: "fdi8dv",
    styles: "display:inline-grid;position:relative;flex:1;box-sizing:border-box;padding-left:var(--table-spacing-sm);padding-right:var(--table-spacing-sm);word-break:break-word;overflow:hidden;& .anticon{vertical-align:text-bottom;}"
  } : {
    name: "10gwa9a-cell",
    styles: "display:inline-grid;position:relative;flex:1;box-sizing:border-box;padding-left:var(--table-spacing-sm);padding-right:var(--table-spacing-sm);word-break:break-word;overflow:hidden;& .anticon{vertical-align:text-bottom;};label:cell;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
  },
  header: process.env.NODE_ENV === "production" ? {
    name: "ik7qgz",
    styles: "font-weight:bold;align-items:flex-end;display:flex;overflow:hidden;&[aria-sort]{cursor:pointer;user-select:none;}.table-header-text{color:var(--table-header-text-color);}.table-header-icon-container{color:var(--table-header-sort-icon-color);display:none;}&[aria-sort]:hover{.table-header-icon-container, .table-header-text{color:var(--table-header-focus-color);}}&[aria-sort]:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}}&:hover, &[aria-sort=\"ascending\"], &[aria-sort=\"descending\"]{.table-header-icon-container{display:inline;}}"
  } : {
    name: "1xg6jn4-header",
    styles: "font-weight:bold;align-items:flex-end;display:flex;overflow:hidden;&[aria-sort]{cursor:pointer;user-select:none;}.table-header-text{color:var(--table-header-text-color);}.table-header-icon-container{color:var(--table-header-sort-icon-color);display:none;}&[aria-sort]:hover{.table-header-icon-container, .table-header-text{color:var(--table-header-focus-color);}}&[aria-sort]:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}}&:hover, &[aria-sort=\"ascending\"], &[aria-sort=\"descending\"]{.table-header-icon-container{display:inline;}};label:header;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
  },
  row: process.env.NODE_ENV === "production" ? {
    name: "ndcf6g",
    styles: "display:flex;&.table-isHeader{> *{background-color:var(--table-header-background-color);}.table-isScrollable &{position:sticky;top:0;z-index:1;}}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:var(--row-checkbox-opacity, 0);}&:not(.table-row-isGrid)&:hover{&:not(.table-isHeader){background-color:var(--table-row-hover);}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:1;}}.table-row-select-cell input[type=\"checkbox\"]:focus ~ *{opacity:1;}> *{padding-top:var(--table-row-vertical-padding);padding-bottom:var(--table-row-vertical-padding);border-bottom:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid > *{border-right:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid > :first-of-type{border-left:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid.table-isHeader:first-of-type > *{border-top:1px solid;border-color:var(--table-separator-color);}"
  } : {
    name: "8dtpje-row",
    styles: "display:flex;&.table-isHeader{> *{background-color:var(--table-header-background-color);}.table-isScrollable &{position:sticky;top:0;z-index:1;}}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:var(--row-checkbox-opacity, 0);}&:not(.table-row-isGrid)&:hover{&:not(.table-isHeader){background-color:var(--table-row-hover);}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:1;}}.table-row-select-cell input[type=\"checkbox\"]:focus ~ *{opacity:1;}> *{padding-top:var(--table-row-vertical-padding);padding-bottom:var(--table-row-vertical-padding);border-bottom:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid > *{border-right:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid > :first-of-type{border-left:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid.table-isHeader:first-of-type > *{border-top:1px solid;border-color:var(--table-separator-color);};label:row;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
  }
};

// For performance, these styles are defined outside of the component so they are not redefined on every render.
// We're also using CSS Variables rather than any dynamic styles so that the style object remains static.
const tableStyles = {
  tableWrapper: /*#__PURE__*/css({
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
  }, process.env.NODE_ENV === "production" ? "" : ";label:tableWrapper;"),
  table: process.env.NODE_ENV === "production" ? {
    name: "oxmfz7",
    styles: ".table-isScrollable &{flex:1;overflow:auto;}"
  } : {
    name: "1csnd2v-table",
    styles: ".table-isScrollable &{flex:1;overflow:auto;};label:table;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
  },
  headerButtonTarget: process.env.NODE_ENV === "production" ? {
    name: "sezlox",
    styles: "align-items:flex-end;display:flex;overflow:hidden;width:100%;justify-content:inherit;&:focus{.table-header-text{color:var(--table-header-focus-color);}.table-header-icon-container{color:var(--table-header-focus-color);display:inline;}}&:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}}"
  } : {
    name: "1iv4trp-headerButtonTarget",
    styles: "align-items:flex-end;display:flex;overflow:hidden;width:100%;justify-content:inherit;&:focus{.table-header-text{color:var(--table-header-focus-color);}.table-header-icon-container{color:var(--table-header-focus-color);display:inline;}}&:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}};label:headerButtonTarget;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
  },
  sortHeaderIconOnRight: process.env.NODE_ENV === "production" ? {
    name: "1hdiaor",
    styles: "margin-left:var(--table-spacing-xs)"
  } : {
    name: "evh3p3-sortHeaderIconOnRight",
    styles: "margin-left:var(--table-spacing-xs);label:sortHeaderIconOnRight;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
  },
  sortHeaderIconOnLeft: process.env.NODE_ENV === "production" ? {
    name: "d4plmt",
    styles: "margin-right:var(--table-spacing-xs)"
  } : {
    name: "1gr7edl-sortHeaderIconOnLeft",
    styles: "margin-right:var(--table-spacing-xs);label:sortHeaderIconOnLeft;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
  },
  checkboxCell: process.env.NODE_ENV === "production" ? {
    name: "4cdr0s",
    styles: "display:flex;align-items:center;flex:0;padding-left:var(--table-spacing-sm);padding-top:0;padding-bottom:0;min-width:var(--table-spacing-md);max-width:var(--table-spacing-md);box-sizing:content-box"
  } : {
    name: "17au6u2-checkboxCell",
    styles: "display:flex;align-items:center;flex:0;padding-left:var(--table-spacing-sm);padding-top:0;padding-bottom:0;min-width:var(--table-spacing-md);max-width:var(--table-spacing-md);box-sizing:content-box;label:checkboxCell;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
  },
  resizeHandleContainer: /*#__PURE__*/css({
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
  }, process.env.NODE_ENV === "production" ? "" : ";label:resizeHandleContainer;"),
  resizeHandle: process.env.NODE_ENV === "production" ? {
    name: "55zery",
    styles: "width:1px;background:var(--table-resize-handle-color)"
  } : {
    name: "1ot7jju-resizeHandle",
    styles: "width:1px;background:var(--table-resize-handle-color);label:resizeHandle;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
  },
  paginationContainer: process.env.NODE_ENV === "production" ? {
    name: "ehlmid",
    styles: "display:flex;justify-content:flex-end;padding-top:var(--table-spacing-sm);padding-bottom:var(--table-spacing-sm)"
  } : {
    name: "p324df-paginationContainer",
    styles: "display:flex;justify-content:flex-end;padding-top:var(--table-spacing-sm);padding-bottom:var(--table-spacing-sm);label:paginationContainer;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
  }
};
var tableStyles$1 = tableStyles;

const TableContext = /*#__PURE__*/createContext({
  size: 'default',
  grid: false
});
const Table = /*#__PURE__*/forwardRef(function Table(_ref, ref) {
  let {
    children,
    size = 'default',
    someRowsSelected,
    style,
    pagination,
    empty,
    className,
    scrollable = false,
    grid = false,
    ...rest
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const [shouldBeFocusable, setShouldBeFocusable] = useState(false);
  const tableContentRef = useRef(null);
  useImperativeHandle(ref, () => tableContentRef.current);
  useEffect(() => {
    const ref = tableContentRef.current;
    if (ref) {
      if (ref.scrollHeight > ref.clientHeight) {
        setShouldBeFocusable(true);
      } else {
        setShouldBeFocusable(false);
      }
    }
  }, []);
  return jsx(DesignSystemEventSuppressInteractionProviderContext.Provider, {
    value: DesignSystemEventSuppressInteractionTrueContextValue,
    children: jsx(TableContext.Provider, {
      value: useMemo(() => {
        return {
          size,
          someRowsSelected,
          grid
        };
      }, [size, someRowsSelected, grid]),
      children: jsxs("div", {
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
          ['--table-separator-color']: theme.colors.borderDecorative,
          ['--table-resize-handle-color']: theme.colors.borderDecorative,
          ['--table-spacing-md']: `${theme.spacing.md}px`,
          ['--table-spacing-sm']: `${theme.spacing.sm}px`,
          ['--table-spacing-xs']: `${theme.spacing.xs}px`
        },
        css: [tableStyles$1.tableWrapper, /*#__PURE__*/css({
          minHeight: !empty && pagination ? 150 : 100
        }, process.env.NODE_ENV === "production" ? "" : ";label:Table;"), process.env.NODE_ENV === "production" ? "" : ";label:Table;"],
        className: classnames({
          'table-isScrollable': scrollable,
          'table-isGrid': grid
        }, className),
        children: [jsxs("div", {
          role: "table",
          ref: tableContentRef,
          css: tableStyles$1.table
          // Needed to make panel body content focusable when scrollable for keyboard-only users to be able to focus & scroll
          // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
          ,
          tabIndex: shouldBeFocusable ? 0 : -1,
          children: [children, empty && jsx("div", {
            css: /*#__PURE__*/css({
              padding: theme.spacing.lg
            }, process.env.NODE_ENV === "production" ? "" : ";label:Table;"),
            children: empty
          })]
        }), !empty && pagination && jsx("div", {
          css: tableStyles$1.paginationContainer,
          children: pagination
        })]
      })
    })
  });
});

const TableCell = /*#__PURE__*/forwardRef(function (_ref, ref) {
  let {
    children,
    className,
    ellipsis = false,
    multiline = false,
    align = 'left',
    style,
    wrapContent = true,
    ...rest
  } = _ref;
  const {
    size,
    grid
  } = useContext(TableContext);
  let typographySize = 'md';
  if (size === 'small') {
    typographySize = 'sm';
  }
  const content = wrapContent === true ? jsx(Typography.Text, {
    ellipsis: !multiline,
    size: typographySize,
    title: !multiline && typeof children === 'string' && children || undefined,
    children: children
  }) : children;
  return jsx("div", {
    ...rest,
    role: "cell",
    style: {
      textAlign: align,
      ...style
    },
    ref: ref
    // PE-259 Use more performance className for grid but keep css= for compatibility.
    ,
    css: !grid ? repeatingElementsStyles.cell : undefined,
    className: classnames(grid && tableClassNames.cell, className),
    children: content
  });
});

const TableRowContext = /*#__PURE__*/createContext({
  isHeader: false
});
const TableRow = /*#__PURE__*/forwardRef(function TableRow(_ref, ref) {
  let {
    children,
    className,
    style,
    isHeader = false,
    verticalAlignment,
    ...rest
  } = _ref;
  const {
    size,
    grid
  } = useContext(TableContext);
  const {
    theme
  } = useDesignSystemTheme();

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
  return jsx(TableRowContext.Provider, {
    value: useMemo(() => {
      return {
        isHeader
      };
    }, [isHeader]),
    children: jsx("div", {
      ...rest,
      ref: ref,
      role: "row",
      style: {
        ...style,
        ['--table-row-vertical-padding']: `${rowPadding}px`
      }
      // PE-259 Use more performance className for grid but keep css= for consistency.
      ,
      css: !grid ? repeatingElementsStyles.row : undefined,
      className: classnames(className, grid && tableClassNames.row, {
        'table-isHeader': isHeader,
        'table-row-isGrid': grid
      }),
      children: children
    })
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$n() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const TableRowActionStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "gk361n",
    styles: "width:32px;padding-top:var(--vertical-padding);padding-bottom:var(--vertical-padding);display:flex;align-items:start;justify-content:center"
  } : {
    name: "q9pljs-container",
    styles: "width:32px;padding-top:var(--vertical-padding);padding-bottom:var(--vertical-padding);display:flex;align-items:start;justify-content:center;label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$n
  }
};
const TableRowAction = /*#__PURE__*/forwardRef(function TableRowAction(_ref, ref) {
  let {
    children,
    style,
    className,
    ...rest
  } = _ref;
  const {
    size
  } = useContext(TableContext);
  const {
    isHeader
  } = useContext(TableRowContext);
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    ...rest,
    ref: ref,
    role: isHeader ? 'columnheader' : 'cell',
    style: {
      ...style,
      ['--vertical-padding']: size === 'default' ? `${theme.spacing.xs}px` : 0
    },
    css: TableRowActionStyles.container,
    className: className,
    children: children
  });
});

/** @deprecated Use `TableRowAction` instead */
const TableRowMenuContainer = TableRowAction;

function _EMOTION_STRINGIFIED_CSS_ERROR__$m() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const TableSkeletonStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "6kz1wu",
    styles: "display:flex;flex-direction:column;align-items:flex-start"
  } : {
    name: "1we0er9-container",
    styles: "display:flex;flex-direction:column;align-items:flex-start;label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$m
  },
  cell: process.env.NODE_ENV === "production" ? {
    name: "1t820zr",
    styles: "width:100%;height:8px;border-radius:4px;background:var(--table-skeleton-color);margin-top:var(--table-skeleton-row-vertical-margin);margin-bottom:var(--table-skeleton-row-vertical-margin)"
  } : {
    name: "1m8dl5b-cell",
    styles: "width:100%;height:8px;border-radius:4px;background:var(--table-skeleton-color);margin-top:var(--table-skeleton-row-vertical-margin);margin-bottom:var(--table-skeleton-row-vertical-margin);label:cell;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$m
  }
};
const TableSkeleton = _ref => {
  let {
    lines = 1,
    seed = '',
    frameRate = 60,
    style
    // TODO: Re-enable this after Clusters fixes tests: https://databricks.slack.com/archives/C04LYE3F8HX/p1679597678339659
    /** children, ...rest */
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    size
  } = useContext(TableContext);
  const widths = getOffsets(seed);
  return jsx("div", {
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
    children: [...Array(lines)].map((_, idx) => jsx("div", {
      css: [TableSkeletonStyles.cell, genSkeletonAnimatedColor(theme, frameRate), {
        width: `calc(100% - ${widths[idx % widths.length]}px)`
      }, process.env.NODE_ENV === "production" ? "" : ";label:TableSkeleton;"]
    }, idx))
  });
};
const TableSkeletonRows = _ref2 => {
  let {
    table,
    actionColumnIds = [],
    numRows = 3,
    loading = true,
    loadingDescription = 'Table skeleton rows'
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs(Fragment, {
    children: [loading && jsx(LoadingState, {
      description: loadingDescription
    }), [...Array(numRows).keys()].map(i => jsx(TableRow, {
      children: table.getFlatHeaders().map(header => {
        var _meta$styles, _meta$numSkeletonLine;
        const meta = header.column.columnDef.meta;
        return actionColumnIds.includes(header.id) ? jsx(TableRowAction, {
          children: jsx(TableSkeleton, {
            style: {
              width: theme.general.iconSize
            }
          })
        }, `cell-${header.id}-${i}`) : jsx(TableCell, {
          style: (_meta$styles = meta === null || meta === void 0 ? void 0 : meta.styles) !== null && _meta$styles !== void 0 ? _meta$styles : (meta === null || meta === void 0 ? void 0 : meta.width) !== undefined ? {
            maxWidth: meta.width
          } : {},
          children: jsx(TableSkeleton, {
            seed: `skeleton-${header.id}-${i}`,
            lines: (_meta$numSkeletonLine = meta === null || meta === void 0 ? void 0 : meta.numSkeletonLines) !== null && _meta$numSkeletonLine !== void 0 ? _meta$numSkeletonLine : undefined
          })
        }, `cell-${header.id}-${i}`);
      })
    }, i))]
  });
};

// Loading state requires a width since it'll have no content
const LOADING_STATE_DEFAULT_WIDTH = 300;
function getStyles$1(args) {
  const {
    theme,
    loading,
    width,
    disableHover,
    hasTopBar,
    hasBottomBar,
    hasHref
  } = args;
  const hoverOrFocusStyle = {
    boxShadow: disableHover || loading ? '' : theme.general.shadowLow,
    ...(hasHref && {
      border: `1px solid ${theme.colors.actionDefaultBorderHover}`
    })
  };
  return /*#__PURE__*/css({
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
    width: width !== null && width !== void 0 ? width : 'fit-content',
    borderRadius: theme.borders.borderRadiusMd,
    borderColor: theme.colors.border,
    borderWidth: '1px',
    borderStyle: 'solid',
    '&:hover': hoverOrFocusStyle,
    '&:focus': hoverOrFocusStyle,
    cursor: disableHover || loading ? 'default' : 'pointer',
    transition: `box-shadow 0.2s ease-in-out`,
    textDecoration: 'none !important',
    ...getAnimationCss(theme.options.enableAnimation)
  }, process.env.NODE_ENV === "production" ? "" : ";label:getStyles;");
}
function getBottomBarStyles(theme) {
  return /*#__PURE__*/css({
    marginTop: theme.spacing.sm,
    borderBottomRightRadius: theme.borders.borderRadiusMd,
    borderBottomLeftRadius: theme.borders.borderRadiusMd,
    overflow: 'hidden'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getBottomBarStyles;");
}
function getTopBarStyles(theme) {
  return /*#__PURE__*/css({
    marginBottom: theme.spacing.sm,
    borderTopRightRadius: theme.borders.borderRadiusMd,
    borderTopLeftRadius: theme.borders.borderRadiusMd,
    overflow: 'hidden'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getTopBarStyles;");
}
const Card = _ref => {
  let {
    children,
    customLoadingContent,
    dangerouslyAppendEmotionCSS,
    loading,
    width,
    bottomBarContent,
    topBarContent,
    disableHover,
    onClick,
    href,
    navigateFn,
    anchorProps,
    componentId,
    analyticsEvents,
    ...dataAndAttributes
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const hasTopBar = !_isUndefined(topBarContent);
  const hasBottomBar = !_isUndefined(bottomBarContent);
  const cardStyle = /*#__PURE__*/css(getStyles$1({
    theme,
    loading,
    width,
    disableHover,
    hasBottomBar,
    hasTopBar,
    hasHref: Boolean(href)
  }), process.env.NODE_ENV === "production" ? "" : ";label:cardStyle;");
  const ref = React__default.useRef(null);
  const bottomBar = bottomBarContent ? jsx("div", {
    css: /*#__PURE__*/css(getBottomBarStyles(theme), process.env.NODE_ENV === "production" ? "" : ";label:bottomBar;"),
    children: bottomBarContent
  }) : null;
  const topBar = topBarContent ? jsx("div", {
    css: /*#__PURE__*/css(getTopBarStyles(theme), process.env.NODE_ENV === "production" ? "" : ";label:topBar;"),
    children: topBarContent
  }) : null;
  const contentPadding = hasTopBar || hasBottomBar ? theme.spacing.lg : 0;
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Card,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnClick]
  });
  const navigate = useCallback(() => {
    if (navigateFn) {
      navigateFn();
    }
  }, [navigateFn]);
  const handleClick = useCallback(e => {
    var _ref$current;
    eventContext.onClick();
    navigate();
    onClick === null || onClick === void 0 || onClick(e);
    (_ref$current = ref.current) === null || _ref$current === void 0 || _ref$current.blur();
  }, [navigate, eventContext, onClick]);
  const handleSelection = useCallback(e => {
    eventContext.onClick();
    e.preventDefault();
    navigate();
    onClick === null || onClick === void 0 || onClick(e);
  }, [navigate, eventContext, onClick]);
  const content = jsx("div", {
    ref: ref
    // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
    ,
    tabIndex: 0,
    ...addDebugOutlineIfEnabled(),
    css: href ? [] : [cardStyle, dangerouslyAppendEmotionCSS],
    onClick: loading || href ? undefined : handleClick,
    ...dataAndAttributes,
    ...(href && {
      role: 'link',
      onKeyDown: e => {
        var _dataAndAttributes$on;
        if (e.key === 'Enter' || e.key === ' ') {
          handleSelection(e);
        }
        (_dataAndAttributes$on = dataAndAttributes.onKeyDown) === null || _dataAndAttributes$on === void 0 || _dataAndAttributes$on.call(dataAndAttributes, e);
      }
    }),
    children: loading ? jsx(DefaultCardLoadingContent, {
      width: width,
      customLoadingContent: customLoadingContent
    }) : jsxs(Fragment, {
      children: [topBar, jsx("div", {
        css: /*#__PURE__*/css({
          padding: `0px ${contentPadding}px`,
          flexGrow: 1
        }, process.env.NODE_ENV === "production" ? "" : ";label:content;"),
        children: children
      }), bottomBar]
    })
  });
  return href ? jsx("a", {
    css: [cardStyle, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Card;"],
    href: href,
    ...anchorProps,
    children: content
  }) : content;
};
function DefaultCardLoadingContent(_ref2) {
  let {
    customLoadingContent,
    width
  } = _ref2;
  if (customLoadingContent) {
    return jsx(Fragment, {
      children: customLoadingContent
    });
  }
  return jsxs("div", {
    css: /*#__PURE__*/css({
      width: width !== null && width !== void 0 ? width : LOADING_STATE_DEFAULT_WIDTH
    }, process.env.NODE_ENV === "production" ? "" : ";label:DefaultCardLoadingContent;"),
    children: [jsx(TitleSkeleton, {
      label: "Loading...",
      style: {
        width: '50%'
      }
    }), [...Array(3).keys()].map(i => jsx(ParagraphSkeleton, {
      label: "Loading..."
    }, i))]
  });
}

function getCheckboxEmotionStyles(clsPrefix, theme) {
  let isHorizontal = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
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
  const styles = {
    [`.${clsPrefix}`]: {
      top: 'unset',
      lineHeight: theme.typography.lineHeightBase
    },
    [`&${classWrapper}, ${classWrapper}`]: {
      alignItems: 'center',
      lineHeight: theme.typography.lineHeightBase
    },
    // Top level styles are for the unchecked state
    [classInner]: {
      borderColor: theme.colors.actionDefaultBorderDefault
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
      columnGap: 0
    },
    ...(isHorizontal && {
      [`&${classContainer}`]: {
        display: 'flex',
        flexDirection: 'row',
        columnGap: theme.spacing.sm,
        rowGap: 0,
        [`& > ${classContainer}-item`]: {
          marginRight: 0
        }
      }
    }),
    // Keyboard focus
    [`${classInput}:focus-visible + ${classInner}`]: {
      outlineWidth: '2px',
      outlineColor: theme.colors.primary,
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
const getWrapperStyle = _ref => {
  let {
    clsPrefix,
    theme,
    wrapperStyle = {}
  } = _ref;
  const styles = {
    height: theme.typography.lineHeightBase,
    lineHeight: theme.typography.lineHeightBase,
    [`&& + .${clsPrefix}-hint, && + .${clsPrefix}-form-message`]: {
      paddingLeft: theme.spacing.lg,
      marginTop: 0
    },
    ...wrapperStyle
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getWrapperStyle;");
};
const DuboisCheckbox = /*#__PURE__*/forwardRef(function Checkbox(_ref2, ref) {
  let {
    isChecked,
    onChange,
    children,
    isDisabled = false,
    style,
    wrapperStyle,
    dangerouslySetAntdProps,
    className,
    componentId,
    analyticsEvents,
    ...restProps
  } = _ref2;
  const {
    theme,
    classNamePrefix,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('checkbox');
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Checkbox,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii: true
  });
  const onChangeHandler = event => {
    eventContext.onValueChange(event.target.checked);
    onChange === null || onChange === void 0 || onChange(event.target.checked, event);
  };
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx("div", {
      ...addDebugOutlineIfEnabled(),
      className: classnames(className, `${clsPrefix}-container`),
      css: getWrapperStyle({
        clsPrefix: classNamePrefix,
        theme,
        wrapperStyle
      }),
      children: jsx(Checkbox$1, {
        checked: isChecked === null ? undefined : isChecked,
        ref: ref,
        onChange: onChangeHandler,
        disabled: isDisabled,
        indeterminate: isChecked === null
        // Individual checkboxes don't depend on isHorizontal flag, orientation and spacing is handled by end users
        ,
        css: /*#__PURE__*/css(importantify(getCheckboxEmotionStyles(clsPrefix, theme, false)), process.env.NODE_ENV === "production" ? "" : ";label:Checkbox;"),
        style: style,
        "aria-checked": isChecked === null ? 'mixed' : isChecked,
        ...restProps,
        ...dangerouslySetAntdProps,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    })
  });
});
const CheckboxGroup = /*#__PURE__*/forwardRef(function CheckboxGroup(_ref3, ref) {
  let {
    children,
    layout = 'vertical',
    ...props
  } = _ref3;
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('checkbox');
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Checkbox$1.Group, {
      ...addDebugOutlineIfEnabled(),
      ref: ref,
      ...props,
      css: getCheckboxEmotionStyles(clsPrefix, theme, layout === 'horizontal'),
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
});
const CheckboxNamespace = /* #__PURE__ */Object.assign(DuboisCheckbox, {
  Group: CheckboxGroup
});
const Checkbox = CheckboxNamespace;

// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
const __INTERNAL_DO_NOT_USE__Group = CheckboxGroup;

const infoIconStyles = theme => ({
  display: 'inline-flex',
  paddingLeft: theme.spacing.xs,
  color: theme.colors.textSecondary,
  pointerEvents: 'all'
});
const getNewChildren = (children, props, disabledReason, ref) => {
  const childCount = Children.count(children);
  const tooltip = jsx(LegacyTooltip, {
    title: disabledReason,
    placement: "right",
    dangerouslySetAntdProps: {
      getPopupContainer: () => ref.current || document.body
    },
    children: jsx("span", {
      css: theme => infoIconStyles(theme),
      onClick: e => {
        if (props.disabled) {
          e.stopPropagation();
        }
      },
      children: jsx(InfoIcon, {
        role: "presentation",
        alt: "Disabled state reason",
        "aria-hidden": "false"
      })
    })
  });
  if (childCount === 1) {
    return getChild(children, Boolean(props['disabled']), disabledReason, tooltip, 0, childCount);
  }
  return Children.map(children, (child, idx) => {
    return getChild(child, Boolean(props['disabled']), disabledReason, tooltip, idx, childCount);
  });
};
const getChild = (child, isDisabled, disabledReason, tooltip, index, siblingCount) => {
  const HintColumnType = jsx(HintColumn, {}).type;
  const isHintColumnType = Boolean(child && typeof child !== 'string' && typeof child !== 'number' && typeof child !== 'boolean' && 'type' in child && (child === null || child === void 0 ? void 0 : child.type) === HintColumnType);
  if (isDisabled && disabledReason && child && isHintColumnType) {
    return jsxs(Fragment, {
      children: [tooltip, child]
    });
  } else if (index === siblingCount - 1 && isDisabled && disabledReason) {
    return jsxs(Fragment, {
      children: [child, tooltip]
    });
  }
  return child;
};

const DropdownContext = /*#__PURE__*/createContext({
  isOpen: false,
  setIsOpen: isOpen => {}
});
const useDropdownContext = () => React__default.useContext(DropdownContext);
const Root$4 = _ref => {
  let {
    children,
    ...props
  } = _ref;
  const [isOpen, setIsOpen] = React__default.useState(Boolean(props.defaultOpen || props.open));
  const useExternalState = useRef(props.open !== undefined || props.onOpenChange !== undefined).current;
  useEffect(() => {
    if (useExternalState) {
      setIsOpen(Boolean(props.open));
    }
  }, [useExternalState, props.open]);
  const handleOpenChange = isOpen => {
    if (!useExternalState) {
      setIsOpen(isOpen);
    }

    // In case the consumer doesn't manage open state but wants to listen to the callback
    if (props.onOpenChange) {
      props.onOpenChange(isOpen);
    }
  };
  return jsx(DropdownMenu$1.Root, {
    ...props,
    ...(!useExternalState && {
      open: isOpen,
      onOpenChange: handleOpenChange
    }),
    children: jsx(DropdownContext.Provider, {
      value: {
        isOpen: useExternalState ? props.open : isOpen,
        setIsOpen: useExternalState ? props.onOpenChange : handleOpenChange
      },
      children: children
    })
  });
};
const Content$3 = /*#__PURE__*/forwardRef(function Content(_ref2, ref) {
  let {
    children,
    minWidth = 220,
    forceCloseOnEscape,
    onEscapeKeyDown,
    onKeyDown,
    ...props
  } = _ref2;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const {
    setIsOpen
  } = useDropdownContext();
  const {
    isInsideModal
  } = useModalContext();
  return jsx(DropdownMenu$1.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(DropdownMenu$1.Content, {
      ...addDebugOutlineIfEnabled(),
      ref: ref,
      loop: true,
      css: [contentStyles, {
        minWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      sideOffset: 4,
      align: "start",
      onKeyDown: e => {
        // This is a workaround for Radix's DropdownMenu.Content not receiving Escape key events
        // when nested inside a modal. We need to stop propagation of the event so that the modal
        // doesn't close when the DropdownMenu should.
        if (e.key === 'Escape') {
          if (isInsideModal || forceCloseOnEscape) {
            e.stopPropagation();
            setIsOpen === null || setIsOpen === void 0 || setIsOpen(false);
          }
          onEscapeKeyDown === null || onEscapeKeyDown === void 0 || onEscapeKeyDown(e.nativeEvent);
        }
        onKeyDown === null || onKeyDown === void 0 || onKeyDown(e);
      },
      ...props,
      onWheel: e => {
        var _props$onWheel;
        e.stopPropagation();
        props === null || props === void 0 || (_props$onWheel = props.onWheel) === null || _props$onWheel === void 0 || _props$onWheel.call(props, e);
      },
      onTouchMove: e => {
        var _props$onTouchMove;
        e.stopPropagation();
        props === null || props === void 0 || (_props$onTouchMove = props.onTouchMove) === null || _props$onTouchMove === void 0 || _props$onTouchMove.call(props, e);
      },
      children: children
    })
  });
});
const SubContent$1 = /*#__PURE__*/forwardRef(function Content(_ref3, ref) {
  let {
    children,
    minWidth = 220,
    ...props
  } = _ref3;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const [contentFitsInViewport, setContentFitsInViewport] = React__default.useState(true);
  const [dataSide, setDataSide] = React__default.useState(null);
  const {
    isOpen
  } = useSubContext();
  const useResponsiveDropdownPosition = safex('databricks.fe.designsystem.useResponsiveDropdownPosition', false);
  const elemRef = useRef(null);
  useImperativeHandle(ref, () => elemRef.current);
  const checkAvailableWidth = useCallback(() => {
    if (useResponsiveDropdownPosition) {
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
    }
  }, [useResponsiveDropdownPosition]);
  useEffect(() => {
    if (useResponsiveDropdownPosition) {
      window.addEventListener('resize', checkAvailableWidth);
      checkAvailableWidth();
      return () => {
        window.removeEventListener('resize', checkAvailableWidth);
      };
    }
    return;
  }, [useResponsiveDropdownPosition, checkAvailableWidth]);
  useEffect(() => {
    if (useResponsiveDropdownPosition) {
      if (isOpen) {
        setTimeout(() => {
          checkAvailableWidth();
        }, 25);
      }
    }
  }, [isOpen, useResponsiveDropdownPosition, checkAvailableWidth]);
  let transformCalc = `calc(var(--radix-dropdown-menu-content-available-width) + var(--radix-dropdown-menu-trigger-width) * -1)`;
  if (dataSide === 'left') {
    transformCalc = `calc(var(--radix-dropdown-menu-trigger-width) - var(--radix-dropdown-menu-content-available-width))`;
  }
  const responsiveCss = `
    transform-origin: var(--radix-dropdown-menu-content-transform-origin) !important;
    transform: translateX(${transformCalc}) !important;
`;
  return jsx(DropdownMenu$1.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(DropdownMenu$1.SubContent, {
      ...addDebugOutlineIfEnabled(),
      ref: elemRef,
      loop: true,
      css: [contentStyles, {
        minWidth
      }, useResponsiveDropdownPosition ? contentFitsInViewport ? '' : responsiveCss : '', process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      sideOffset: -2,
      alignOffset: -5,
      ...props,
      children: children
    })
  });
});
const Trigger$2 = /*#__PURE__*/forwardRef(function Trigger(_ref4, ref) {
  let {
    children,
    ...props
  } = _ref4;
  return jsx(DropdownMenu$1.Trigger, {
    ...addDebugOutlineIfEnabled(),
    ref: ref,
    ...props,
    children: children
  });
});
const Item$1 = /*#__PURE__*/forwardRef(function Item(_ref5, ref) {
  let {
    children,
    disabledReason,
    danger,
    onClick,
    componentId,
    analyticsEvents,
    ...props
  } = _ref5;
  const itemRef = useRef(null);
  useImperativeHandle(ref, () => itemRef.current);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.DropdownMenuItem,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnClick]
  });
  return jsx(DropdownMenu$1.Item, {
    css: theme => [dropdownItemStyles, danger && dangerItemStyles(theme)],
    ref: itemRef,
    onClick: e => {
      if (props.disabled) {
        e.preventDefault();
      } else {
        if (!props.asChild) {
          eventContext.onClick();
        }
        onClick === null || onClick === void 0 || onClick(e);
      }
    },
    ...props,
    children: getNewChildren(children, props, disabledReason, itemRef)
  });
});
const Label$2 = /*#__PURE__*/forwardRef(function Label(_ref6, ref) {
  let {
    children,
    ...props
  } = _ref6;
  return jsx(DropdownMenu$1.Label, {
    ref: ref,
    css: [dropdownItemStyles, theme => ({
      color: theme.colors.textSecondary,
      '&:hover': {
        cursor: 'default'
      }
    }), process.env.NODE_ENV === "production" ? "" : ";label:Label;"],
    ...props,
    children: children
  });
});
const Separator$2 = /*#__PURE__*/forwardRef(function Separator(_ref7, ref) {
  let {
    children,
    ...props
  } = _ref7;
  return jsx(DropdownMenu$1.Separator, {
    ref: ref,
    css: dropdownSeparatorStyles,
    ...props,
    children: children
  });
});
const SubTrigger$1 = /*#__PURE__*/forwardRef(function TriggerItem(_ref8, ref) {
  let {
    children,
    disabledReason,
    ...props
  } = _ref8;
  const subTriggerRef = useRef(null);
  useImperativeHandle(ref, () => subTriggerRef.current);
  return jsxs(DropdownMenu$1.SubTrigger, {
    ref: subTriggerRef,
    css: [dropdownItemStyles, theme => ({
      '&[data-state="open"]': {
        backgroundColor: theme.colors.actionTertiaryBackgroundHover
      }
    }), process.env.NODE_ENV === "production" ? "" : ";label:TriggerItem;"],
    ...props,
    children: [getNewChildren(children, props, disabledReason, subTriggerRef), jsx(HintColumn, {
      css: theme => ({
        margin: CONSTANTS$1.subMenuIconMargin(theme),
        display: 'flex',
        alignSelf: 'stretch',
        alignItems: 'center'
      }),
      children: jsx(ChevronRightIcon, {
        css: theme => ({
          fontSize: CONSTANTS$1.subMenuIconSize(theme)
        })
      })
    })]
  });
});

/**
 * Deprecated. Use `SubTrigger` instead.
 * @deprecated
 */
const TriggerItem = SubTrigger$1;
const CheckboxItem$1 = /*#__PURE__*/forwardRef(function CheckboxItem(_ref9, ref) {
  let {
    children,
    disabledReason,
    componentId,
    analyticsEvents,
    onCheckedChange,
    ...props
  } = _ref9;
  const checkboxItemRef = useRef(null);
  useImperativeHandle(ref, () => checkboxItemRef.current);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.DropdownMenuCheckboxItem,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii: true
  });
  const onCheckedChangeWrapper = useCallback(checked => {
    eventContext.onValueChange(checked);
    onCheckedChange === null || onCheckedChange === void 0 || onCheckedChange(checked);
  }, [eventContext, onCheckedChange]);
  return jsx(DropdownMenu$1.CheckboxItem, {
    ref: checkboxItemRef,
    css: theme => [dropdownItemStyles, checkboxItemStyles(theme)],
    onCheckedChange: onCheckedChangeWrapper,
    ...props,
    children: getNewChildren(children, props, disabledReason, checkboxItemRef)
  });
});
const RadioGroup$1 = /*#__PURE__*/forwardRef(function RadioGroup(_ref10, ref) {
  let {
    children,
    componentId,
    analyticsEvents,
    onValueChange,
    valueHasNoPii,
    ...props
  } = _ref10;
  const radioGroupItemRef = useRef(null);
  useImperativeHandle(ref, () => radioGroupItemRef.current);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.DropdownMenuRadioGroup,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii
  });
  const onValueChangeWrapper = useCallback(value => {
    eventContext.onValueChange(value);
    onValueChange === null || onValueChange === void 0 || onValueChange(value);
  }, [eventContext, onValueChange]);
  return jsx(DropdownMenu$1.RadioGroup, {
    ref: radioGroupItemRef,
    onValueChange: onValueChangeWrapper,
    ...props,
    children: children
  });
});
const ItemIndicator$1 = /*#__PURE__*/forwardRef(function ItemIndicator(_ref11, ref) {
  let {
    children,
    ...props
  } = _ref11;
  return jsx(DropdownMenu$1.ItemIndicator, {
    ref: ref,
    css: theme => ({
      marginLeft: -(CONSTANTS$1.checkboxIconWidth(theme) + CONSTANTS$1.checkboxPaddingRight(theme)),
      position: 'absolute',
      fontSize: theme.general.iconFontSize
    }),
    ...props,
    children: children !== null && children !== void 0 ? children : jsx(CheckIcon, {
      css: theme => ({
        color: theme.colors.textSecondary
      })
    })
  });
});
const Arrow$1 = /*#__PURE__*/forwardRef(function Arrow(_ref12, ref) {
  let {
    children,
    ...props
  } = _ref12;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DropdownMenu$1.Arrow, {
    css: /*#__PURE__*/css({
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
    }, process.env.NODE_ENV === "production" ? "" : ";label:Arrow;"),
    ref: ref,
    width: 12,
    height: 6,
    ...props,
    children: children
  });
});
const RadioItem$1 = /*#__PURE__*/forwardRef(function RadioItem(_ref13, ref) {
  let {
    children,
    disabledReason,
    ...props
  } = _ref13;
  const radioItemRef = useRef(null);
  useImperativeHandle(ref, () => radioItemRef.current);
  return jsx(DropdownMenu$1.RadioItem, {
    ref: radioItemRef,
    css: theme => [dropdownItemStyles, checkboxItemStyles(theme)],
    ...props,
    children: getNewChildren(children, props, disabledReason, radioItemRef)
  });
});
const SubContext = /*#__PURE__*/createContext({
  isOpen: false
});
const useSubContext = () => React__default.useContext(SubContext);
const Sub$1 = _ref14 => {
  var _props$defaultOpen;
  let {
    children,
    onOpenChange,
    ...props
  } = _ref14;
  const [isOpen, setIsOpen] = React__default.useState((_props$defaultOpen = props.defaultOpen) !== null && _props$defaultOpen !== void 0 ? _props$defaultOpen : false);
  const handleOpenChange = isOpen => {
    onOpenChange === null || onOpenChange === void 0 || onOpenChange(isOpen);
    setIsOpen(isOpen);
  };
  return jsx(DropdownMenu$1.Sub, {
    onOpenChange: handleOpenChange,
    ...props,
    children: jsx(SubContext.Provider, {
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
const HintColumn = /*#__PURE__*/forwardRef(function HintColumn(_ref15, ref) {
  let {
    children,
    ...props
  } = _ref15;
  return jsx("div", {
    ref: ref,
    css: [metaTextStyles, "margin-left:auto;" + (process.env.NODE_ENV === "production" ? "" : ";label:HintColumn;")],
    ...props,
    children: children
  });
});
const HintRow$1 = /*#__PURE__*/forwardRef(function HintRow(_ref16, ref) {
  let {
    children,
    ...props
  } = _ref16;
  return jsx("div", {
    ref: ref,
    css: [metaTextStyles, "min-width:100%;" + (process.env.NODE_ENV === "production" ? "" : ";label:HintRow;")],
    ...props,
    children: children
  });
});
const IconWrapper = /*#__PURE__*/forwardRef(function IconWrapper(_ref17, ref) {
  let {
    children,
    ...props
  } = _ref17;
  return jsx("div", {
    ref: ref,
    css: theme => ({
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
  itemPaddingVertical(theme) {
    // The number from the mocks is the midpoint between constants
    return 0.5 * theme.spacing.xs + 0.5 * theme.spacing.sm;
  },
  itemPaddingHorizontal(theme) {
    return theme.spacing.sm;
  },
  checkboxIconWidth(theme) {
    return theme.general.iconFontSize;
  },
  checkboxPaddingLeft(theme) {
    return theme.spacing.sm + theme.spacing.xs;
  },
  checkboxPaddingRight(theme) {
    return theme.spacing.sm;
  },
  subMenuIconMargin(theme) {
    // Negative margin so the icons can be larger without increasing the overall item height
    const iconMarginVertical = this.itemPaddingVertical(theme) / 2;
    const iconMarginRight = -this.itemPaddingVertical(theme) + theme.spacing.sm * 1.5;
    return `${-iconMarginVertical}px ${-iconMarginRight}px ${-iconMarginVertical}px auto`;
  },
  subMenuIconSize(theme) {
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
  }
};
const dropdownContentStyles = theme => ({
  backgroundColor: theme.colors.backgroundPrimary,
  color: theme.colors.textPrimary,
  lineHeight: theme.typography.lineHeightBase,
  border: `1px solid ${theme.colors.borderDecorative}`,
  borderRadius: theme.borders.borderRadiusMd,
  padding: `${theme.spacing.xs}px 0`,
  boxShadow: theme.general.shadowLow,
  userSelect: 'none',
  ...getDarkModePortalStyles(theme),
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
      textDecoration: 'none'
    }
  })
});
const contentStyles = theme => ({
  ...dropdownContentStyles(theme)
});
const dropdownItemStyles = theme => ({
  padding: `${CONSTANTS$1.itemPaddingVertical(theme)}px ${CONSTANTS$1.itemPaddingHorizontal(theme)}px`,
  display: 'flex',
  flexWrap: 'wrap',
  alignItems: 'center',
  outline: 'unset',
  '&:hover': {
    cursor: 'pointer'
  },
  '&:focus': {
    backgroundColor: theme.colors.actionTertiaryBackgroundHover
  },
  '&[data-disabled]': {
    pointerEvents: 'none',
    color: theme.colors.actionDisabledText
  }
});
const dangerItemStyles = theme => ({
  color: theme.colors.textValidationDanger,
  '&:hover, &:focus': {
    backgroundColor: theme.colors.actionDangerDefaultBackgroundHover
  }
});
const checkboxItemStyles = theme => ({
  position: 'relative',
  paddingLeft: CONSTANTS$1.checkboxIconWidth(theme) + CONSTANTS$1.checkboxPaddingLeft(theme) + CONSTANTS$1.checkboxPaddingRight(theme)
});
const metaTextStyles = theme => ({
  color: theme.colors.textSecondary,
  fontSize: theme.typography.fontSizeSm,
  '[data-disabled] &': {
    color: theme.colors.actionDisabledText
  }
});
const dropdownSeparatorStyles = theme => ({
  height: 1,
  margin: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
  backgroundColor: theme.colors.borderDecorative
});

var DropdownMenu = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Arrow: Arrow$1,
  CheckboxItem: CheckboxItem$1,
  Content: Content$3,
  Group: Group$2,
  HintColumn: HintColumn,
  HintRow: HintRow$1,
  IconWrapper: IconWrapper,
  Item: Item$1,
  ItemIndicator: ItemIndicator$1,
  Label: Label$2,
  RadioGroup: RadioGroup$1,
  RadioItem: RadioItem$1,
  Root: Root$4,
  Separator: Separator$2,
  Sub: Sub$1,
  SubContent: SubContent$1,
  SubTrigger: SubTrigger$1,
  Trigger: Trigger$2,
  TriggerItem: TriggerItem,
  dropdownContentStyles: dropdownContentStyles,
  dropdownItemStyles: dropdownItemStyles,
  dropdownSeparatorStyles: dropdownSeparatorStyles
});

const Trigger$1 = ContextMenuTrigger;
const ItemIndicator = ContextMenuItemIndicator;
const Group$1 = ContextMenuGroup;
const Arrow = ContextMenuArrow;
const Sub = ContextMenuSub;
const ContextMenuProps = /*#__PURE__*/createContext({
  isOpen: false,
  setIsOpen: isOpen => {}
});
const useContextMenuProps = () => React__default.useContext(ContextMenuProps);
const Root$3 = _ref => {
  let {
    children,
    onOpenChange,
    ...props
  } = _ref;
  const [isOpen, setIsOpen] = React__default.useState(false);
  const handleChange = isOpen => {
    setIsOpen(isOpen);
    onOpenChange === null || onOpenChange === void 0 || onOpenChange(isOpen);
  };
  return jsx(ContextMenu$2, {
    onOpenChange: handleChange,
    ...props,
    children: jsx(ContextMenuProps.Provider, {
      value: {
        isOpen,
        setIsOpen
      },
      children: children
    })
  });
};
const SubTrigger = _ref2 => {
  let {
    children,
    disabledReason,
    withChevron,
    ...props
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  const ref = useRef(null);
  return jsxs(ContextMenuSubTrigger, {
    ...props,
    css: dropdownItemStyles(theme),
    ref: ref,
    children: [getNewChildren(children, props, disabledReason, ref), withChevron && jsx(ContextMenu.Hint, {
      children: jsx(ChevronRightIcon, {})
    })]
  });
};
const Content$2 = _ref3 => {
  let {
    children,
    minWidth,
    forceCloseOnEscape,
    onEscapeKeyDown,
    onKeyDown,
    ...childrenProps
  } = _ref3;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const {
    theme
  } = useDesignSystemTheme();
  const {
    isInsideModal
  } = useModalContext();
  const {
    isOpen,
    setIsOpen
  } = useContextMenuProps();
  return jsx(ContextMenuPortal, {
    container: getPopupContainer && getPopupContainer(),
    children: isOpen && jsx(ContextMenuContent, {
      ...addDebugOutlineIfEnabled(),
      onKeyDown: e => {
        // This is a workaround for Radix's ContextMenu.Content not receiving Escape key events
        // when nested inside a modal. We need to stop propagation of the event so that the modal
        // doesn't close when the DropdownMenu should.
        if (e.key === 'Escape') {
          if (isInsideModal || forceCloseOnEscape) {
            e.stopPropagation();
            setIsOpen(false);
          }
          onEscapeKeyDown === null || onEscapeKeyDown === void 0 || onEscapeKeyDown(e.nativeEvent);
        }
        onKeyDown === null || onKeyDown === void 0 || onKeyDown(e);
      },
      ...childrenProps,
      css: [dropdownContentStyles(theme), {
        minWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      children: children
    })
  });
};
const SubContent = _ref4 => {
  let {
    children,
    minWidth,
    ...childrenProps
  } = _ref4;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(ContextMenuPortal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(ContextMenuSubContent, {
      ...addDebugOutlineIfEnabled(),
      ...childrenProps,
      css: [dropdownContentStyles(theme), {
        minWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:SubContent;"],
      children: children
    })
  });
};
const Item = _ref5 => {
  let {
    children,
    disabledReason,
    onClick,
    componentId,
    analyticsEvents,
    asChild,
    ...props
  } = _ref5;
  const {
    theme
  } = useDesignSystemTheme();
  const ref = useRef(null);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.ContextMenuItem,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnClick]
  });
  const onClickWrapper = useCallback(e => {
    if (!asChild) {
      eventContext.onClick();
    }
    onClick === null || onClick === void 0 || onClick(e);
  }, [asChild, eventContext, onClick]);
  return jsx(ContextMenuItem, {
    ...props,
    asChild: asChild,
    onClick: onClickWrapper,
    css: dropdownItemStyles(theme),
    ref: ref,
    children: getNewChildren(children, props, disabledReason, ref)
  });
};
const CheckboxItem = _ref6 => {
  let {
    children,
    disabledReason,
    onCheckedChange,
    componentId,
    analyticsEvents,
    ...props
  } = _ref6;
  const {
    theme
  } = useDesignSystemTheme();
  const ref = useRef(null);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.ContextMenuCheckboxItem,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii: true
  });
  const onCheckedChangeWrapper = useCallback(checked => {
    eventContext.onValueChange(checked);
    onCheckedChange === null || onCheckedChange === void 0 || onCheckedChange(checked);
  }, [eventContext, onCheckedChange]);
  return jsxs(ContextMenuCheckboxItem, {
    ...props,
    onCheckedChange: onCheckedChangeWrapper,
    css: dropdownItemStyles(theme),
    ref: ref,
    children: [jsx(ContextMenuItemIndicator, {
      css: itemIndicatorStyles(theme),
      children: jsx(CheckIcon, {})
    }), !props.checked && jsx("div", {
      style: {
        width: theme.general.iconFontSize + theme.spacing.xs
      }
    }), getNewChildren(children, props, disabledReason, ref)]
  });
};
const RadioGroup = _ref7 => {
  let {
    onValueChange,
    componentId,
    analyticsEvents,
    valueHasNoPii,
    ...props
  } = _ref7;
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.ContextMenuRadioGroup,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii
  });
  const onValueChangeWrapper = useCallback(value => {
    eventContext.onValueChange(value);
    onValueChange === null || onValueChange === void 0 || onValueChange(value);
  }, [eventContext, onValueChange]);
  return jsx(ContextMenuRadioGroup, {
    ...props,
    onValueChange: onValueChangeWrapper
  });
};
const RadioItem = _ref8 => {
  let {
    children,
    disabledReason,
    ...props
  } = _ref8;
  const {
    theme
  } = useDesignSystemTheme();
  const ref = useRef(null);
  return jsxs(ContextMenuRadioItem, {
    ...props,
    css: [dropdownItemStyles(theme), {
      '&[data-state="unchecked"]': {
        paddingLeft: theme.general.iconFontSize + theme.spacing.xs + theme.spacing.sm
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:RadioItem;"],
    ref: ref,
    children: [jsx(ContextMenuItemIndicator, {
      css: itemIndicatorStyles(theme),
      children: jsx(CheckIcon, {})
    }), getNewChildren(children, props, disabledReason, ref)]
  });
};
const Label$1 = _ref9 => {
  let {
    children,
    ...props
  } = _ref9;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(ContextMenuLabel, {
    ...props,
    css: /*#__PURE__*/css({
      color: theme.colors.textSecondary,
      padding: `${theme.spacing.sm - 2}px ${theme.spacing.sm}px`
    }, process.env.NODE_ENV === "production" ? "" : ";label:Label;"),
    children: children
  });
};
const Hint$1 = _ref10 => {
  let {
    children
  } = _ref10;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("span", {
    css: /*#__PURE__*/css({
      display: 'inline-flex',
      marginLeft: 'auto',
      paddingLeft: theme.spacing.sm
    }, process.env.NODE_ENV === "production" ? "" : ";label:Hint;"),
    children: children
  });
};
const Separator$1 = () => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(ContextMenuSeparator, {
    css: dropdownSeparatorStyles(theme)
  });
};
const itemIndicatorStyles = theme => /*#__PURE__*/css({
  display: 'inline-flex',
  paddingRight: theme.spacing.xs
}, process.env.NODE_ENV === "production" ? "" : ";label:itemIndicatorStyles;");
const ContextMenu = {
  Root: Root$3,
  Trigger: Trigger$1,
  Label: Label$1,
  Item,
  Group: Group$1,
  RadioGroup,
  CheckboxItem,
  RadioItem,
  Arrow,
  Separator: Separator$1,
  Sub,
  SubTrigger,
  SubContent,
  Content: Content$2,
  Hint: Hint$1
};

var ContextMenu$1 = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Arrow: Arrow,
  CheckboxItem: CheckboxItem,
  Content: Content$2,
  ContextMenu: ContextMenu,
  Group: Group$1,
  Hint: Hint$1,
  Item: Item,
  ItemIndicator: ItemIndicator,
  Label: Label$1,
  RadioGroup: RadioGroup,
  RadioItem: RadioItem,
  Root: Root$3,
  Separator: Separator$1,
  Sub: Sub,
  SubContent: SubContent,
  SubTrigger: SubTrigger,
  Trigger: Trigger$1,
  itemIndicatorStyles: itemIndicatorStyles
});

function getEmotionStyles(clsPrefix, theme) {
  const classFocused = `.${clsPrefix}-focused`;
  const classActiveBar = `.${clsPrefix}-active-bar`;
  const classSeparator = `.${clsPrefix}-separator`;
  const classSuffix = `.${clsPrefix}-suffix`;
  const styles = {
    height: 32,
    borderRadius: theme.borders.borderRadiusMd,
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
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getEmotionStyles;");
}
const getDropdownStyles$1 = theme => {
  return {
    zIndex: theme.options.zIndexBase + 50
  };
};
function useDatePickerStyles() {
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('picker');
  return getEmotionStyles(clsPrefix, theme);
}
const AccessibilityWrapper = _ref => {
  let {
    children,
    ariaLive = 'assertive',
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) {
      const inputs = theme.isDarkMode ? ref.current.querySelectorAll('.du-bois-dark-picker-input > input') : ref.current.querySelectorAll('.du-bois-light-picker-input > input');
      inputs.forEach(input => input.setAttribute('aria-live', ariaLive));
    }
  }, [ref, ariaLive, theme.isDarkMode]);
  return jsx("div", {
    ...restProps,
    ref: ref,
    children: children
  });
};
const DuboisDatePicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  const {
    ariaLive,
    wrapperDivProps,
    ...restProps
  } = props;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AccessibilityWrapper, {
      ...addDebugOutlineIfEnabled(),
      ...wrapperDivProps,
      ariaLive: ariaLive,
      children: jsx(DatePicker, {
        css: styles,
        ref: ref,
        ...restProps,
        popupStyle: {
          ...getDropdownStyles$1(theme),
          ...(props.popupStyle || {})
        }
      })
    })
  });
});
const RangePicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  const {
    ariaLive,
    wrapperDivProps,
    ...restProps
  } = props;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AccessibilityWrapper, {
      ...wrapperDivProps,
      ariaLive: ariaLive,
      children: jsx(DatePicker.RangePicker, {
        ...addDebugOutlineIfEnabled(),
        css: styles,
        ...restProps,
        ref: ref,
        popupStyle: {
          ...getDropdownStyles$1(theme),
          ...(props.popupStyle || {})
        }
      })
    })
  });
});
const TimePicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  const {
    ariaLive,
    wrapperDivProps,
    ...restProps
  } = props;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AccessibilityWrapper, {
      ...addDebugOutlineIfEnabled(),
      ...wrapperDivProps,
      ariaLive: ariaLive,
      children: jsx(DatePicker.TimePicker, {
        css: styles,
        ...restProps,
        ref: ref,
        popupStyle: {
          ...getDropdownStyles$1(theme),
          ...(props.popupStyle || {})
        }
      })
    })
  });
});
const QuarterPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  const {
    ariaLive,
    wrapperDivProps,
    ...restProps
  } = props;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AccessibilityWrapper, {
      ...addDebugOutlineIfEnabled(),
      ...wrapperDivProps,
      ariaLive: ariaLive,
      children: jsx(DatePicker.QuarterPicker, {
        css: styles,
        ...restProps,
        ref: ref,
        popupStyle: {
          ...getDropdownStyles$1(theme),
          ...(props.popupStyle || {})
        }
      })
    })
  });
});
const WeekPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  const {
    ariaLive,
    wrapperDivProps,
    ...restProps
  } = props;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AccessibilityWrapper, {
      ...addDebugOutlineIfEnabled(),
      ...wrapperDivProps,
      ariaLive: ariaLive,
      children: jsx(DatePicker.WeekPicker, {
        css: styles,
        ...restProps,
        ref: ref,
        popupStyle: {
          ...getDropdownStyles$1(theme),
          ...(props.popupStyle || {})
        }
      })
    })
  });
});
const MonthPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  const {
    ariaLive,
    wrapperDivProps,
    ...restProps
  } = props;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AccessibilityWrapper, {
      ...addDebugOutlineIfEnabled(),
      ...wrapperDivProps,
      ariaLive: ariaLive,
      children: jsx(DatePicker.MonthPicker, {
        css: styles,
        ...restProps,
        ref: ref,
        popupStyle: {
          ...getDropdownStyles$1(theme),
          ...(props.popupStyle || {})
        }
      })
    })
  });
});
const YearPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  const {
    ariaLive,
    wrapperDivProps,
    ...restProps
  } = props;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(AccessibilityWrapper, {
      ...addDebugOutlineIfEnabled(),
      ...wrapperDivProps,
      ariaLive: ariaLive,
      children: jsx(DatePicker.YearPicker, {
        css: styles,
        ...restProps,
        ref: ref,
        popupStyle: {
          ...getDropdownStyles$1(theme),
          ...(props.popupStyle || {})
        }
      })
    })
  });
});

/**
 * `LegacyDatePicker` was added as a temporary solution pending an
 * official Du Bois replacement. Use with caution.
 * @deprecated
 */
const LegacyDatePicker = /* #__PURE__ */Object.assign(DuboisDatePicker, {
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  RangePicker,
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  TimePicker,
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  QuarterPicker,
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  WeekPicker,
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  MonthPicker,
  /**
   * See deprecation notice for `LegacyDatePicker`.
   * @deprecated
   */
  YearPicker
});

const dialogComboboxContextDefaults = {
  componentId: 'codegen_design-system_src_design-system_dialogcombobox_providers_dialogcomboboxcontext.tsx_27',
  id: '',
  label: '',
  value: [],
  isInsideDialogCombobox: false,
  multiSelect: false,
  setValue: () => {},
  setIsControlled: () => {},
  stayOpenOnSelection: false,
  isOpen: false,
  setIsOpen: () => {},
  contentWidth: undefined,
  setContentWidth: () => {},
  textOverflowMode: 'multiline',
  setTextOverflowMode: () => {},
  scrollToSelectedElement: true,
  rememberLastScrollPosition: false
};
const DialogComboboxContext = /*#__PURE__*/createContext(dialogComboboxContextDefaults);
const DialogComboboxContextProvider = _ref => {
  let {
    children,
    value
  } = _ref;
  return jsx(DialogComboboxContext.Provider, {
    value: value,
    children: children
  });
};

const useDialogComboboxContext = () => {
  return useContext(DialogComboboxContext);
};

const DialogCombobox = _ref => {
  let {
    children,
    label,
    id,
    value = [],
    open,
    emptyText,
    scrollToSelectedElement = true,
    rememberLastScrollPosition = false,
    componentId,
    analyticsEvents,
    valueHasNoPii,
    ...props
  } = _ref;
  // Used to avoid infinite loop when value is controlled from within the component (DialogComboboxOptionControlledList)
  // We can't remove setValue altogether because uncontrolled component users need to be able to set the value from root for trigger to update
  const [isControlled, setIsControlled] = useState(false);
  const [selectedValue, setSelectedValue] = useState(value);
  const [isOpen, setIsOpen] = useState(Boolean(open));
  const [contentWidth, setContentWidth] = useState();
  const [textOverflowMode, setTextOverflowMode] = useState('multiline');
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.DialogCombobox,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii
  });
  const setSelectedValueWrapper = useCallback(newValue => {
    eventContext.onValueChange(JSON.stringify(newValue));
    setSelectedValue(newValue);
  }, [eventContext]);
  useEffect(() => {
    if ((!Array.isArray(selectedValue) || !Array.isArray(value)) && selectedValue !== value || selectedValue && value && selectedValue.length === value.length && selectedValue.every((v, i) => v === value[i])) {
      return;
    }
    if (!isControlled) {
      setSelectedValueWrapper(value);
    }
  }, [value, isControlled, selectedValue, setSelectedValueWrapper]);
  return jsx(DialogComboboxContextProvider, {
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
      valueHasNoPii
    },
    children: jsx(Root$2, {
      open: open !== undefined ? open : isOpen,
      ...props,
      children: children
    })
  });
};
const Root$2 = props => {
  const {
    children,
    stayOpenOnSelection,
    multiSelect,
    ...restProps
  } = props;
  const {
    value,
    setIsOpen
  } = useDialogComboboxContext();
  const handleOpenChange = open => {
    setIsOpen(open);
  };
  useEffect(() => {
    if (!stayOpenOnSelection && (typeof stayOpenOnSelection === 'boolean' || !multiSelect)) {
      setIsOpen(false);
    }
  }, [value, stayOpenOnSelection, multiSelect, setIsOpen]);
  return jsx(Popover.Root, {
    onOpenChange: handleOpenChange,
    ...restProps,
    children: children
  });
};

const getButtonStyles = theme => {
  return /*#__PURE__*/css({
    color: theme.colors.textPlaceholder,
    fontSize: theme.typography.fontSizeSm,
    marginLeft: theme.spacing.xs,
    ':hover': {
      color: theme.colors.actionTertiaryTextHover
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getButtonStyles;");
};
const ClearSelectionButton = _ref => {
  let {
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(XCircleFillIcon$1, {
    "aria-hidden": "false",
    css: getButtonStyles(theme),
    role: "button",
    "aria-label": "Clear selection",
    ...restProps
  });
};

const EmptyResults = _ref => {
  var _ref2;
  let {
    emptyText
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    emptyText: emptyTextFromContext
  } = useDialogComboboxContext();
  return jsx("div", {
    "aria-live": "assertive",
    css: /*#__PURE__*/css({
      color: theme.colors.textSecondary,
      textAlign: 'center',
      padding: '6px 12px',
      width: '100%',
      boxSizing: 'border-box'
    }, process.env.NODE_ENV === "production" ? "" : ";label:EmptyResults;"),
    children: (_ref2 = emptyTextFromContext !== null && emptyTextFromContext !== void 0 ? emptyTextFromContext : emptyText) !== null && _ref2 !== void 0 ? _ref2 : 'No results found'
  });
};

const HintRow = _ref => {
  let {
    disabled,
    children
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: /*#__PURE__*/css({
      color: theme.colors.textSecondary,
      fontSize: theme.typography.fontSizeSm,
      ...(disabled && {
        color: theme.colors.actionDisabledText
      })
    }, process.env.NODE_ENV === "production" ? "" : ";label:HintRow;"),
    children: children
  });
};

const LoadingSpinner = props => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Spinner, {
    css: /*#__PURE__*/css({
      display: 'flex',
      alignSelf: 'center',
      justifyContent: 'center',
      alignItems: 'center',
      height: theme.general.heightSm,
      width: theme.general.heightSm,
      '> span': {
        fontSize: 20
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:LoadingSpinner;"),
    ...props
  });
};

const SectionHeader = _ref => {
  let {
    children,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    ...props,
    css: /*#__PURE__*/css({
      display: 'flex',
      flexDirection: 'row',
      alignItems: 'flex-start',
      padding: `${theme.spacing.xs}px ${theme.spacing.lg / 2}px`,
      alignSelf: 'stretch',
      fontWeight: 400,
      color: theme.colors.textSecondary
    }, process.env.NODE_ENV === "production" ? "" : ";label:SectionHeader;"),
    children: children
  });
};

const Separator = props => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    ...props,
    css: /*#__PURE__*/css({
      display: 'flex',
      flexDirection: 'row',
      alignItems: 'center',
      margin: `${theme.spacing.xs}px ${theme.spacing.lg / 2}px`,
      border: `1px solid ${theme.colors.borderDecorative}`,
      borderBottom: 0,
      alignSelf: 'stretch'
    }, process.env.NODE_ENV === "production" ? "" : ";label:Separator;")
  });
};

const getDialogComboboxOptionLabelWidth = (theme, width) => {
  const paddingLeft = theme.spacing.xs + theme.spacing.sm;
  const iconWidth = theme.spacing.md;
  const labelMarginLeft = theme.spacing.sm;
  if (typeof width === 'string') {
    return `calc(${width} - ${paddingLeft + iconWidth + labelMarginLeft}px)`;
  }
  return width - paddingLeft + iconWidth + labelMarginLeft;
};
function isOptionDisabled(option) {
  return option.hasAttribute('disabled') && option.getAttribute('disabled') !== 'false';
}
function highlightFirstNonDisabledOption(firstOptionItem) {
  let startAt = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 'start';
  let previousSelection = arguments.length > 2 ? arguments[2] : undefined;
  if (isOptionDisabled(firstOptionItem)) {
    const firstHighlightableOption = findClosestOptionSibling(firstOptionItem, startAt === 'end' ? 'previous' : 'next');
    if (firstHighlightableOption) {
      highlightOption(firstHighlightableOption, previousSelection);
    }
  } else {
    highlightOption(firstOptionItem, previousSelection);
  }
}
function findClosestOptionSibling(element, direction) {
  const nextSibling = direction === 'previous' ? element.previousElementSibling : element.nextElementSibling;
  if ((nextSibling === null || nextSibling === void 0 ? void 0 : nextSibling.getAttribute('role')) === 'option') {
    if (isOptionDisabled(nextSibling)) {
      return findClosestOptionSibling(nextSibling, direction);
    }
    return nextSibling;
  } else if (nextSibling) {
    let nextOptionSibling = nextSibling;
    while (nextOptionSibling && (nextOptionSibling.getAttribute('role') !== 'option' || isOptionDisabled(nextOptionSibling))) {
      nextOptionSibling = direction === 'previous' ? nextOptionSibling.previousElementSibling : nextOptionSibling.nextElementSibling;
    }
    return nextOptionSibling;
  }
  return null;
}
const resetAllHighlightedOptions = currentSelection => {
  const options = getContentOptions(currentSelection);
  options === null || options === void 0 || options.forEach(option => {
    option.setAttribute('tabIndex', '-1');
    option.setAttribute('data-highlighted', 'false');
  });
};
const highlightOption = function (currentSelection, prevSelection) {
  var _currentSelection$scr;
  let focus = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : true;
  if (prevSelection) {
    prevSelection.setAttribute('tabIndex', '-1');
    prevSelection.setAttribute('data-highlighted', 'false');
  }
  if (focus) {
    currentSelection.focus();
  }
  currentSelection.setAttribute('tabIndex', '0');
  currentSelection.setAttribute('data-highlighted', 'true');
  (_currentSelection$scr = currentSelection.scrollIntoView) === null || _currentSelection$scr === void 0 || _currentSelection$scr.call(currentSelection, {
    block: 'center'
  });
};
const findHighlightedOption = options => {
  var _options$find;
  return (_options$find = options.find(option => option.getAttribute('data-highlighted') === 'true')) !== null && _options$find !== void 0 ? _options$find : undefined;
};
const getContentOptions = element => {
  var _element$closest;
  const options = (_element$closest = element.closest('[data-combobox-option-list="true"]')) === null || _element$closest === void 0 ? void 0 : _element$closest.querySelectorAll('[role="option"]');
  return options ? Array.from(options) : undefined;
};
const getKeyboardNavigationFunctions = (handleSelect, _ref) => {
  let {
    onKeyDown,
    onMouseEnter,
    onDefaultKeyDown
  } = _ref;
  return {
    onKeyDown: e => {
      onKeyDown === null || onKeyDown === void 0 || onKeyDown(e);
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          const nextSibling = findClosestOptionSibling(e.currentTarget, 'next');
          if (nextSibling) {
            highlightOption(nextSibling, e.currentTarget);
          } else {
            var _getContentOptions;
            const firstOption = (_getContentOptions = getContentOptions(e.currentTarget)) === null || _getContentOptions === void 0 ? void 0 : _getContentOptions[0];
            if (firstOption) {
              highlightFirstNonDisabledOption(firstOption, 'start', e.currentTarget);
            }
          }
          break;
        case 'ArrowUp':
          e.preventDefault();
          const previousSibling = findClosestOptionSibling(e.currentTarget, 'previous');
          if (previousSibling) {
            highlightOption(previousSibling, e.currentTarget);
          } else {
            var _getContentOptions2;
            const lastOption = (_getContentOptions2 = getContentOptions(e.currentTarget)) === null || _getContentOptions2 === void 0 ? void 0 : _getContentOptions2.slice(-1)[0];
            if (lastOption) {
              highlightFirstNonDisabledOption(lastOption, 'end', e.currentTarget);
            }
          }
          break;
        case 'Enter':
          e.preventDefault();
          handleSelect(e);
          break;
        default:
          onDefaultKeyDown === null || onDefaultKeyDown === void 0 || onDefaultKeyDown(e);
          break;
      }
    },
    onMouseEnter: e => {
      onMouseEnter === null || onMouseEnter === void 0 || onMouseEnter(e);
      resetTabIndexToFocusedElement(e.currentTarget);
    }
  };
};
const resetTabIndexToFocusedElement = elem => {
  resetAllHighlightedOptions(elem);
  elem.setAttribute('tabIndex', '0');
  elem.focus();
};
const dialogComboboxLookAheadKeyDown = (e, setLookAhead, lookAhead) => {
  var _e$currentTarget$pare, _e$currentTarget$pare2;
  if (e.key === 'Escape' || e.key === 'Tab' || e.key === 'Enter') {
    return;
  }
  e.preventDefault();
  const siblings = Array.from((_e$currentTarget$pare = (_e$currentTarget$pare2 = e.currentTarget.parentElement) === null || _e$currentTarget$pare2 === void 0 ? void 0 : _e$currentTarget$pare2.children) !== null && _e$currentTarget$pare !== void 0 ? _e$currentTarget$pare : []);
  // Look for the first sibling that starts with the pressed key + recently pressed keys (lookAhead, cleared after 1.5 seconds of inactivity)
  const nextSiblingIndex = siblings.findIndex(sibling => {
    var _sibling$textContent$, _sibling$textContent;
    const siblingLabel = (_sibling$textContent$ = (_sibling$textContent = sibling.textContent) === null || _sibling$textContent === void 0 ? void 0 : _sibling$textContent.toLowerCase()) !== null && _sibling$textContent$ !== void 0 ? _sibling$textContent$ : '';
    return siblingLabel.startsWith(lookAhead + e.key);
  });
  if (nextSiblingIndex !== -1) {
    const nextSibling = siblings[nextSiblingIndex];
    nextSibling.focus();
    if (setLookAhead) {
      setLookAhead(lookAhead + e.key);
    }
    resetTabIndexToFocusedElement(nextSibling);
  }
};

const getComboboxContentWrapperStyles = (theme, _ref) => {
  let {
    maxHeight = '100vh',
    maxWidth = '100vw',
    minHeight = 0,
    minWidth = 0,
    width
  } = _ref;
  return /*#__PURE__*/css({
    maxHeight,
    maxWidth,
    minHeight,
    minWidth,
    ...(width ? {
      width
    } : {}),
    background: theme.colors.backgroundPrimary,
    color: theme.colors.textPrimary,
    overflow: 'auto',
    // Making sure the content popover overlaps the remove button when opens to the right
    zIndex: theme.options.zIndexBase + 10,
    boxSizing: 'border-box',
    border: `1px solid ${theme.colors.border}`,
    boxShadow: theme.general.shadowLow,
    borderRadius: theme.general.borderRadiusBase,
    colorScheme: theme.isDarkMode ? 'dark' : 'light',
    ...getDarkModePortalStyles(theme)
  }, process.env.NODE_ENV === "production" ? "" : ";label:getComboboxContentWrapperStyles;");
};
const getComboboxOptionItemWrapperStyles = theme => {
  return /*#__PURE__*/css(importantify({
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'flex-start',
    alignSelf: 'stretch',
    padding: '6px 32px 6px 12px',
    lineHeight: theme.typography.lineHeightBase,
    boxSizing: 'content-box',
    cursor: 'pointer',
    userSelect: 'none',
    '&:hover, &[data-highlighted="true"]': {
      background: theme.colors.actionTertiaryBackgroundHover
    },
    '&:focus': {
      background: theme.colors.actionTertiaryBackgroundHover,
      outline: 'none'
    },
    '&[disabled]': {
      pointerEvents: 'none',
      color: theme.colors.actionDisabledText,
      background: theme.colors.backgroundPrimary
    }
  }), process.env.NODE_ENV === "production" ? "" : ";label:getComboboxOptionItemWrapperStyles;");
};
const getComboboxOptionLabelStyles = _ref2 => {
  let {
    theme,
    dangerouslyHideCheck,
    textOverflowMode,
    contentWidth,
    hasHintColumn
  } = _ref2;
  return /*#__PURE__*/css({
    marginLeft: !dangerouslyHideCheck ? theme.spacing.sm : 0,
    fontSize: theme.typography.fontSizeBase,
    fontStyle: 'normal',
    fontWeight: 400,
    cursor: 'pointer',
    overflow: 'hidden',
    wordBreak: 'break-word',
    ...(textOverflowMode === 'ellipsis' && {
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap'
    }),
    ...(contentWidth ? {
      width: getDialogComboboxOptionLabelWidth(theme, contentWidth)
    } : {}),
    ...(hasHintColumn && {
      display: 'flex'
    })
  }, process.env.NODE_ENV === "production" ? "" : ";label:getComboboxOptionLabelStyles;");
};
const getInfoIconStyles = theme => {
  return /*#__PURE__*/css({
    paddingLeft: theme.spacing.xs,
    color: theme.colors.textSecondary,
    pointerEvents: 'all',
    cursor: 'pointer',
    verticalAlign: 'middle'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getInfoIconStyles;");
};
const getCheckboxStyles = (theme, textOverflowMode) => {
  return /*#__PURE__*/css({
    pointerEvents: 'none',
    height: 'unset',
    width: '100%',
    '& > label': {
      display: 'flex',
      width: '100%',
      fontSize: theme.typography.fontSizeBase,
      fontStyle: 'normal',
      fontWeight: 400,
      cursor: 'pointer',
      '& > span:first-of-type': {
        alignSelf: 'flex-start',
        display: 'inline-flex',
        alignItems: 'center',
        paddingTop: theme.spacing.xs / 2
      },
      '& > span:last-of-type': {
        paddingRight: 0,
        width: '100%',
        overflow: 'hidden',
        wordBreak: 'break-word',
        ...(textOverflowMode === 'ellipsis' ? {
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap'
        } : {})
      }
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getCheckboxStyles;");
};
const getFooterStyles = theme => {
  return /*#__PURE__*/css({
    width: '100%',
    background: theme.colors.backgroundPrimary,
    padding: `${theme.spacing.sm}px ${theme.spacing.lg / 2}px ${theme.spacing.sm}px ${theme.spacing.lg / 2}px`,
    position: 'sticky',
    bottom: 0,
    boxSizing: 'border-box',
    '&:has(> .combobox-footer-add-button)': {
      padding: `${theme.spacing.sm}px 0 ${theme.spacing.sm}px 0`,
      '& > :not(.combobox-footer-add-button)': {
        marginLeft: `${theme.spacing.lg / 2}px`,
        marginRight: `${theme.spacing.lg / 2}px`
      },
      '& > .combobox-footer-add-button': {
        justifyContent: 'flex-start !important'
      }
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getFooterStyles;");
};
const getSelectItemWithHintColumnStyles = function () {
  let hintColumnWidthPercent = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 50;
  return /*#__PURE__*/css({
    flexGrow: 1,
    display: 'inline-grid',
    gridTemplateColumns: `${100 - hintColumnWidthPercent}% ${hintColumnWidthPercent}%`
  }, process.env.NODE_ENV === "production" ? "" : ";label:getSelectItemWithHintColumnStyles;");
};
const getHintColumnStyles = (theme, disabled, textOverflowMode) => {
  return /*#__PURE__*/css({
    color: theme.colors.textSecondary,
    fontSize: theme.typography.fontSizeSm,
    textAlign: 'right',
    ...(disabled && {
      color: theme.colors.actionDisabledText
    }),
    ...(textOverflowMode === 'ellipsis' && {
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
      overflow: 'hidden'
    })
  }, process.env.NODE_ENV === "production" ? "" : ";label:getHintColumnStyles;");
};

const DialogComboboxAddButton = _ref => {
  let {
    children,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    isInsideDialogCombobox,
    componentId
  } = useDialogComboboxContext();
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxAddButton` must be used within `DialogCombobox`');
  }
  return jsx(Button, {
    componentId: `${componentId ? componentId : 'design_system.dialogcombobox'}.add_option`,
    ...restProps,
    type: "tertiary",
    className: "combobox-footer-add-button",
    css: /*#__PURE__*/css({
      ...getComboboxOptionItemWrapperStyles(theme),
      ... /*#__PURE__*/css(importantify({
        width: '100%',
        padding: 0,
        display: 'flex',
        alignItems: 'center',
        borderRadius: 0,
        '&:focus': {
          background: theme.colors.actionTertiaryBackgroundHover,
          outline: 'none'
        }
      }), process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxAddButton;")
    }, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxAddButton;"),
    icon: jsx(PlusIcon, {}),
    children: children
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$l() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const defaultMaxHeight = 'var(--radix-popover-content-available-height)';
var _ref2$g = process.env.NODE_ENV === "production" ? {
  name: "1ij1o5n",
  styles: "display:flex;flex-direction:column;align-items:flex-start;justify-content:center"
} : {
  name: "189loa6-DialogComboboxContent",
  styles: "display:flex;flex-direction:column;align-items:flex-start;justify-content:center;label:DialogComboboxContent;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$l
};
const DialogComboboxContent = /*#__PURE__*/forwardRef((_ref, forwardedRef) => {
  let {
    children,
    loading,
    loadingDescription = 'DialogComboboxContent',
    matchTriggerWidth,
    textOverflowMode,
    maxHeight,
    maxWidth,
    minHeight,
    minWidth = 240,
    width,
    align = 'start',
    side = 'bottom',
    sideOffset = 4,
    onEscapeKeyDown,
    onKeyDown,
    forceCloseOnEscape,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    label,
    isInsideDialogCombobox,
    contentWidth,
    setContentWidth,
    textOverflowMode: contextTextOverflowMode,
    setTextOverflowMode,
    multiSelect,
    isOpen,
    rememberLastScrollPosition,
    setIsOpen
  } = useDialogComboboxContext();
  const {
    isInsideModal
  } = useModalContext();
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const [lastScrollPosition, setLastScrollPosition] = useState(0);
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxContent` must be used within `DialogCombobox`');
  }
  const contentRef = useRef(null);
  useImperativeHandle(forwardedRef, () => contentRef.current);
  const realContentWidth = matchTriggerWidth ? 'var(--radix-popover-trigger-width)' : width;
  useEffect(() => {
    if (rememberLastScrollPosition) {
      if (!isOpen && contentRef.current) {
        setLastScrollPosition(contentRef.current.scrollTop);
      } else {
        // Wait for the popover to render and scroll to the last scrolled position
        const interval = setInterval(() => {
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
        return () => clearInterval(interval);
      }
    }
    return;
  }, [isOpen, rememberLastScrollPosition, lastScrollPosition]);
  useEffect(() => {
    if (contentWidth !== realContentWidth) {
      setContentWidth(realContentWidth);
    }
  }, [realContentWidth, contentWidth, setContentWidth]);
  useEffect(() => {
    if (textOverflowMode !== contextTextOverflowMode) {
      setTextOverflowMode(textOverflowMode ? textOverflowMode : 'multiline');
    }
  }, [textOverflowMode, contextTextOverflowMode, setTextOverflowMode]);
  return jsx(Popover.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(Popover.Content, {
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
        width: realContentWidth
      }),
      align: align,
      side: side,
      sideOffset: sideOffset,
      onKeyDown: e => {
        // This is a workaround for Radix's DialogCombobox.Content not receiving Escape key events
        // when nested inside a modal. We need to stop propagation of the event so that the modal
        // doesn't close when the DropdownMenu should.
        if (e.key === 'Escape') {
          if (isInsideModal || forceCloseOnEscape) {
            e.stopPropagation();
            setIsOpen(false);
          }
          onEscapeKeyDown === null || onEscapeKeyDown === void 0 || onEscapeKeyDown(e.nativeEvent);
        }
        onKeyDown === null || onKeyDown === void 0 || onKeyDown(e);
      },
      ...restProps,
      ref: contentRef,
      children: jsx("div", {
        css: _ref2$g,
        children: loading ? jsx(LoadingSpinner, {
          label: "Loading",
          alt: "Loading spinner",
          loadingDescription: loadingDescription
        }) : children ? children : jsx(EmptyResults, {})
      })
    })
  });
});

const getCountBadgeStyles = theme => /*#__PURE__*/css(importantify({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  boxSizing: 'border-box',
  padding: `${theme.spacing.xs / 2}px ${theme.spacing.xs}px`,
  background: theme.colors.tagDefault,
  borderRadius: theme.general.borderRadiusBase,
  fontSize: theme.typography.fontSizeBase,
  height: 20
}), process.env.NODE_ENV === "production" ? "" : ";label:getCountBadgeStyles;");
const DialogComboboxCountBadge = props => {
  const {
    countStartAt,
    ...restOfProps
  } = props;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    value
  } = useDialogComboboxContext();
  return jsx("div", {
    ...restOfProps,
    css: getCountBadgeStyles(theme),
    children: Array.isArray(value) ? countStartAt ? `+${value.length - countStartAt}` : value.length : value ? 1 : 0
  });
};

const DialogComboboxFooter = _ref => {
  let {
    children,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    isInsideDialogCombobox
  } = useDialogComboboxContext();
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxFooter` must be used within `DialogCombobox`');
  }
  return jsx("div", {
    ...restProps,
    css: getFooterStyles(theme),
    children: children
  });
};

const DialogComboboxHintRow = _ref => {
  let {
    children
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: /*#__PURE__*/css({
      minWidth: '100%',
      color: theme.colors.textSecondary,
      fontSize: theme.typography.fontSizeSm,
      '[data-disabled] &': {
        color: theme.colors.actionDisabledText
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxHintRow;"),
    children: children
  });
};

const DialogComboboxOptionListContext = /*#__PURE__*/createContext({
  isInsideDialogComboboxOptionList: false,
  lookAhead: '',
  setLookAhead: () => {}
});
const DialogComboboxOptionListContextProvider = _ref => {
  let {
    children,
    value
  } = _ref;
  return jsx(DialogComboboxOptionListContext.Provider, {
    value: value,
    children: children
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$k() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$f = process.env.NODE_ENV === "production" ? {
  name: "1pgv7dg",
  styles: "display:flex;flex-direction:column;align-items:flex-start;width:100%"
} : {
  name: "1dtf9pj-DialogComboboxOptionList",
  styles: "display:flex;flex-direction:column;align-items:flex-start;width:100%;label:DialogComboboxOptionList;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$k
};
const DialogComboboxOptionList = /*#__PURE__*/forwardRef((_ref, forwardedRef) => {
  let {
    children,
    loading,
    loadingDescription = 'DialogComboboxOptionList',
    withProgressiveLoading,
    ...restProps
  } = _ref;
  const {
    isInsideDialogCombobox
  } = useDialogComboboxContext();
  const ref = useRef(null);
  useImperativeHandle(forwardedRef, () => ref.current);
  const [lookAhead, setLookAhead] = useState('');
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxOptionList` must be used within `DialogCombobox`');
  }
  const lookAheadTimeout = useRef(null);
  useEffect(() => {
    if (lookAheadTimeout.current) {
      clearTimeout(lookAheadTimeout.current);
    }
    lookAheadTimeout.current = setTimeout(() => {
      setLookAhead('');
    }, 1500);
    return () => {
      if (lookAheadTimeout.current) {
        clearTimeout(lookAheadTimeout.current);
      }
    };
  }, [lookAhead]);
  useEffect(() => {
    var _ref$current;
    if (loading && !withProgressiveLoading) {
      return;
    }
    const optionItems = (_ref$current = ref.current) === null || _ref$current === void 0 ? void 0 : _ref$current.querySelectorAll('[role="option"]');
    const hasTabIndexedOption = Array.from(optionItems !== null && optionItems !== void 0 ? optionItems : []).some(optionItem => {
      return optionItem.getAttribute('tabindex') === '0';
    });
    if (!hasTabIndexedOption) {
      const firstOptionItem = optionItems === null || optionItems === void 0 ? void 0 : optionItems[0];
      if (firstOptionItem) {
        highlightFirstNonDisabledOption(firstOptionItem, 'start');
      }
    }
  }, [loading, withProgressiveLoading]);
  const handleOnMouseEnter = event => {
    const target = event.target;
    if (target) {
      var _target$closest;
      const options = target.hasAttribute('data-combobox-option-list') ? target.querySelectorAll('[role="option"]') : target === null || target === void 0 || (_target$closest = target.closest('[data-combobox-option-list="true"]')) === null || _target$closest === void 0 ? void 0 : _target$closest.querySelectorAll('[role="option"]');
      if (options) {
        options.forEach(option => option.removeAttribute('data-highlighted'));
      }
    }
  };
  return jsx("div", {
    ref: ref,
    "aria-busy": loading,
    "data-combobox-option-list": "true",
    css: _ref2$f,
    onMouseEnter: handleOnMouseEnter,
    ...restProps,
    children: jsx(DialogComboboxOptionListContextProvider, {
      value: {
        isInsideDialogComboboxOptionList: true,
        lookAhead,
        setLookAhead
      },
      children: loading ? withProgressiveLoading ? jsxs(Fragment, {
        children: [children, jsx(LoadingSpinner, {
          "aria-label": "Loading",
          alt: "Loading spinner",
          loadingDescription: loadingDescription
        })]
      }) : jsx(LoadingSpinner, {
        "aria-label": "Loading",
        alt: "Loading spinner",
        loadingDescription: loadingDescription
      }) : children && Children.toArray(children).some(child => /*#__PURE__*/React__default.isValidElement(child)) ? children : jsx(EmptyResults, {})
    })
  });
});

const useDialogComboboxOptionListContext = () => {
  return useContext(DialogComboboxOptionListContext);
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$j() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$e = process.env.NODE_ENV === "production" ? {
  name: "zjik7",
  styles: "display:flex"
} : {
  name: "tiu4as-content",
  styles: "display:flex;label:content;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$j
};
var _ref3$6 = process.env.NODE_ENV === "production" ? {
  name: "qhxz92",
  styles: "max-width:100%"
} : {
  name: "i7t2ch-DuboisDialogComboboxOptionListCheckboxItem",
  styles: "max-width:100%;label:DuboisDialogComboboxOptionListCheckboxItem;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$j
};
const DuboisDialogComboboxOptionListCheckboxItem = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    value,
    checked,
    indeterminate,
    onChange,
    children,
    disabledReason,
    _TYPE,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    textOverflowMode,
    contentWidth
  } = useDialogComboboxContext();
  const {
    isInsideDialogComboboxOptionList,
    setLookAhead,
    lookAhead
  } = useDialogComboboxOptionListContext();
  if (!isInsideDialogComboboxOptionList) {
    throw new Error('`DialogComboboxOptionListCheckboxItem` must be used within `DialogComboboxOptionList`');
  }
  const handleSelect = e => {
    if (onChange) {
      onChange(value, e);
    }
  };
  let content = children !== null && children !== void 0 ? children : value;
  if (props.disabled && disabledReason) {
    content = jsxs("div", {
      css: _ref2$e,
      children: [jsx("div", {
        children: content
      }), jsx("div", {
        children: jsx(LegacyTooltip, {
          title: disabledReason,
          placement: "right",
          children: jsx("span", {
            css: getInfoIconStyles(theme),
            children: jsx(InfoIcon, {
              "aria-label": "Disabled status information",
              "aria-hidden": "false"
            })
          })
        })
      })]
    });
  }
  return jsx("div", {
    ref: ref,
    role: "option"
    // Using aria-selected instead of aria-checked because the parent listbox
    ,
    "aria-selected": indeterminate ? false : checked,
    css: [getComboboxOptionItemWrapperStyles(theme), process.env.NODE_ENV === "production" ? "" : ";label:DuboisDialogComboboxOptionListCheckboxItem;"],
    ...props,
    onClick: e => {
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
      onDefaultKeyDown: e => dialogComboboxLookAheadKeyDown(e, setLookAhead, lookAhead)
    }),
    children: jsx(Checkbox, {
      componentId: "codegen_design-system_src_design-system_dialogcombobox_dialogcomboboxoptionlistcheckboxitem.tsx_86",
      disabled: props.disabled,
      isChecked: indeterminate ? null : checked,
      css: [getCheckboxStyles(theme, textOverflowMode), contentWidth ? {
        '& > span:last-of-type': {
          width: getDialogComboboxOptionLabelWidth(theme, contentWidth)
        }
      } : {}, process.env.NODE_ENV === "production" ? "" : ";label:DuboisDialogComboboxOptionListCheckboxItem;"],
      tabIndex: -1
      // Needed because Antd handles keyboard inputs as clicks
      ,
      onClick: e => {
        e.stopPropagation();
        handleSelect(e);
      },
      children: jsx("div", {
        css: _ref3$6,
        children: content
      })
    })
  });
});
DuboisDialogComboboxOptionListCheckboxItem.defaultProps = {
  _TYPE: 'DialogComboboxOptionListCheckboxItem'
};
const DialogComboboxOptionListCheckboxItem = DuboisDialogComboboxOptionListCheckboxItem;

function _EMOTION_STRINGIFIED_CSS_ERROR__$i() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const extractTextContent = node => {
  if (typeof node === 'string' || typeof node === 'number') {
    return node.toString();
  }
  if ( /*#__PURE__*/React__default.isValidElement(node) && node.props.children) {
    return React__default.Children.toArray(node.props.children).map(extractTextContent).join(' ');
  }
  return '';
};
const filterChildren = (children, searchValue) => {
  var _React$Children$map;
  const lowerCaseSearchValue = searchValue.toLowerCase();
  return (_React$Children$map = React__default.Children.map(children, child => {
    if ( /*#__PURE__*/React__default.isValidElement(child)) {
      var _child$props$__EMOTIO, _child$props$__EMOTIO2;
      const childType = (_child$props$__EMOTIO = (_child$props$__EMOTIO2 = child.props['__EMOTION_TYPE_PLEASE_DO_NOT_USE__']) === null || _child$props$__EMOTIO2 === void 0 ? void 0 : _child$props$__EMOTIO2.defaultProps._TYPE) !== null && _child$props$__EMOTIO !== void 0 ? _child$props$__EMOTIO : child.props._TYPE;
      if (childType === 'DialogComboboxOptionListSelectItem' || childType === 'DialogComboboxOptionListCheckboxItem') {
        var _child$props$value$to, _child$props$value;
        const childTextContent = extractTextContent(child).toLowerCase();
        const childValue = (_child$props$value$to = (_child$props$value = child.props.value) === null || _child$props$value === void 0 ? void 0 : _child$props$value.toLowerCase()) !== null && _child$props$value$to !== void 0 ? _child$props$value$to : '';
        return childTextContent.includes(lowerCaseSearchValue) || childValue.includes(lowerCaseSearchValue) ? child : null;
      }
    }
    return child;
  })) === null || _React$Children$map === void 0 ? void 0 : _React$Children$map.filter(child => child);
};
var _ref2$d = process.env.NODE_ENV === "production" ? {
  name: "1d3w5wq",
  styles: "width:100%"
} : {
  name: "csdki6-DialogComboboxOptionListSearch",
  styles: "width:100%;label:DialogComboboxOptionListSearch;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$i
};
const DialogComboboxOptionListSearch = /*#__PURE__*/forwardRef((_ref, forwardedRef) => {
  var _filteredChildren, _filteredChildren2;
  let {
    onChange,
    onSearch,
    virtualized,
    children,
    hasWrapper,
    controlledValue,
    setControlledValue,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    componentId
  } = useDialogComboboxContext();
  const {
    isInsideDialogComboboxOptionList
  } = useDialogComboboxOptionListContext();
  const [searchValue, setSearchValue] = React__default.useState();
  if (!isInsideDialogComboboxOptionList) {
    throw new Error('`DialogComboboxOptionListSearch` must be used within `DialogComboboxOptionList`');
  }
  const handleOnChange = event => {
    if (!virtualized) {
      setSearchValue(event.target.value.toLowerCase());
    }
    setControlledValue === null || setControlledValue === void 0 || setControlledValue(event.target.value);
    onSearch === null || onSearch === void 0 || onSearch(event.target.value);
  };
  let filteredChildren = children;
  if (searchValue && !virtualized && controlledValue === undefined) {
    filteredChildren = filterChildren(hasWrapper ? children.props.children : children, searchValue);
    if (hasWrapper) {
      filteredChildren = /*#__PURE__*/React__default.cloneElement(children, {}, filteredChildren);
    }
  }
  const inputWrapperRef = useRef(null);

  // When the search value changes, highlight the first option
  useEffect(() => {
    if (!inputWrapperRef.current) {
      return;
    }
    const optionItems = getContentOptions(inputWrapperRef.current);
    if (optionItems) {
      // Reset previous highlights
      const highlightedOption = findHighlightedOption(optionItems);
      const firstOptionItem = optionItems === null || optionItems === void 0 ? void 0 : optionItems[0];
      if (firstOptionItem) {
        highlightOption(firstOptionItem, highlightedOption, false);
      }
    }
  }, [searchValue]);
  const handleOnKeyDown = event => {
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
  const childrenIsNotEmpty = Children.toArray(hasWrapper ? children.props.children : children).some(child => /*#__PURE__*/React__default.isValidElement(child));
  return jsxs(Fragment, {
    children: [jsx("div", {
      ref: inputWrapperRef,
      css: /*#__PURE__*/css({
        padding: `${theme.spacing.sm}px ${theme.spacing.lg / 2}px ${theme.spacing.sm}px`,
        width: '100%',
        boxSizing: 'border-box',
        position: 'sticky',
        top: 0,
        background: theme.colors.backgroundPrimary,
        zIndex: theme.options.zIndexBase + 1
      }, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxOptionListSearch;"),
      children: jsx(Input, {
        componentId: componentId ? `${componentId}.search` : 'codegen_design_system_src_design_system_dialogcombobox_dialogcomboboxoptionlistsearch.tsx_173',
        type: "search",
        name: "search",
        ref: forwardedRef,
        prefix: jsx(SearchIcon$1, {}),
        placeholder: "Search",
        onChange: handleOnChange,
        onKeyDown: event => {
          var _restProps$onKeyDown;
          handleOnKeyDown(event);
          (_restProps$onKeyDown = restProps.onKeyDown) === null || _restProps$onKeyDown === void 0 || _restProps$onKeyDown.call(restProps, event);
        },
        value: controlledValue !== null && controlledValue !== void 0 ? controlledValue : searchValue,
        ...restProps
      })
    }), virtualized ? children : (hasWrapper && (_filteredChildren = filteredChildren) !== null && _filteredChildren !== void 0 && (_filteredChildren = _filteredChildren.props.children) !== null && _filteredChildren !== void 0 && _filteredChildren.length || !hasWrapper && (_filteredChildren2 = filteredChildren) !== null && _filteredChildren2 !== void 0 && _filteredChildren2.length) && childrenIsNotEmpty ? jsx("div", {
      "aria-live": "polite",
      css: _ref2$d,
      children: filteredChildren
    }) : jsx(EmptyResults, {})]
  });
});

const selectContextDefaults = {
  isSelect: false
};
const SelectContext = /*#__PURE__*/createContext(selectContextDefaults);
const SelectContextProvider = _ref => {
  let {
    children,
    value
  } = _ref;
  return jsx(SelectContext.Provider, {
    value: value,
    children: children
  });
};

const useSelectContext = () => {
  return useContext(SelectContext);
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$h() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$c = process.env.NODE_ENV === "production" ? {
  name: "zjik7",
  styles: "display:flex"
} : {
  name: "tiu4as-content",
  styles: "display:flex;label:content;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$h
};
var _ref3$5 = process.env.NODE_ENV === "production" ? {
  name: "kjj0ot",
  styles: "padding-top:2px"
} : {
  name: "15osdio-DuboisDialogComboboxOptionListSelectItem",
  styles: "padding-top:2px;label:DuboisDialogComboboxOptionListSelectItem;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$h
};
const DuboisDialogComboboxOptionListSelectItem = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    value,
    checked,
    disabledReason,
    onChange,
    hintColumn,
    hintColumnWidthPercent = 50,
    children,
    _TYPE,
    icon,
    dangerouslyHideCheck,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    stayOpenOnSelection,
    isOpen,
    setIsOpen,
    value: existingValue,
    contentWidth,
    textOverflowMode,
    scrollToSelectedElement
  } = useDialogComboboxContext();
  const {
    isInsideDialogComboboxOptionList,
    lookAhead,
    setLookAhead
  } = useDialogComboboxOptionListContext();
  const {
    isSelect
  } = useSelectContext();
  if (!isInsideDialogComboboxOptionList) {
    throw new Error('`DialogComboboxOptionListSelectItem` must be used within `DialogComboboxOptionList`');
  }
  const itemRef = useRef(null);
  const prevCheckedRef = useRef(checked);
  useImperativeHandle(ref, () => itemRef.current);
  useEffect(() => {
    if (scrollToSelectedElement && isOpen) {
      // Check if checked didn't change since the last update, otherwise the popover is still open and we don't need to scroll
      if (checked && prevCheckedRef.current === checked) {
        // Wait for the popover to render and scroll to the selected element's position
        const interval = setInterval(() => {
          if (itemRef.current) {
            var _itemRef$current, _itemRef$current$scro;
            (_itemRef$current = itemRef.current) === null || _itemRef$current === void 0 || (_itemRef$current$scro = _itemRef$current.scrollIntoView) === null || _itemRef$current$scro === void 0 || _itemRef$current$scro.call(_itemRef$current, {
              behavior: 'smooth',
              block: 'center'
            });
            clearInterval(interval);
          }
        }, 50);
        return () => clearInterval(interval);
      }
      prevCheckedRef.current = checked;
    }
    return;
  }, [isOpen, scrollToSelectedElement, checked]);
  const handleSelect = e => {
    if (onChange) {
      if (isSelect) {
        onChange({
          value,
          label: typeof children === 'string' ? children : value
        }, e);
        if (existingValue !== null && existingValue !== void 0 && existingValue.includes(value)) {
          setIsOpen(false);
        }
        return;
      }
      onChange(value, e);

      // On selecting a previously selected value, manually close the popup, top level logic will not be triggered
      if (!stayOpenOnSelection && existingValue !== null && existingValue !== void 0 && existingValue.includes(value)) {
        setIsOpen(false);
      }
    }
  };
  let content = children !== null && children !== void 0 ? children : value;
  if (props.disabled && disabledReason) {
    content = jsxs("div", {
      css: _ref2$c,
      children: [jsx("div", {
        children: content
      }), jsx(LegacyTooltip, {
        title: disabledReason,
        placement: "right",
        children: jsx("span", {
          css: getInfoIconStyles(theme),
          children: jsx(InfoIcon, {
            "aria-label": "Disabled status information",
            "aria-hidden": "false"
          })
        })
      })]
    });
  }
  return jsxs("div", {
    ref: itemRef,
    css: [getComboboxOptionItemWrapperStyles(theme), {
      '&:focus': {
        background: theme.colors.actionTertiaryBackgroundHover,
        outline: 'none'
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:DuboisDialogComboboxOptionListSelectItem;"],
    ...props,
    onClick: e => {
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
      onDefaultKeyDown: e => dialogComboboxLookAheadKeyDown(e, setLookAhead, lookAhead)
    }),
    role: "option",
    "aria-selected": checked,
    children: [!dangerouslyHideCheck && (checked ? jsx(CheckIcon, {
      css: _ref3$5
    }) : jsx("div", {
      style: {
        width: 16,
        flexShrink: 0
      }
    })), jsxs("label", {
      css: getComboboxOptionLabelStyles({
        theme,
        dangerouslyHideCheck,
        textOverflowMode,
        contentWidth,
        hasHintColumn: Boolean(hintColumn)
      }),
      children: [icon && jsx("span", {
        style: {
          position: 'relative',
          top: 1,
          marginRight: theme.spacing.sm,
          color: theme.colors.textSecondary
        },
        children: icon
      }), hintColumn ? jsxs("span", {
        css: getSelectItemWithHintColumnStyles(hintColumnWidthPercent),
        children: [content, jsx("span", {
          css: getHintColumnStyles(theme, Boolean(props.disabled), textOverflowMode),
          children: hintColumn
        })]
      }) : content]
    })]
  });
});
DuboisDialogComboboxOptionListSelectItem.defaultProps = {
  _TYPE: 'DialogComboboxOptionListSelectItem'
};
const DialogComboboxOptionListSelectItem = DuboisDialogComboboxOptionListSelectItem;

function _EMOTION_STRINGIFIED_CSS_ERROR__$g() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$b = process.env.NODE_ENV === "production" ? {
  name: "1pgv7dg",
  styles: "display:flex;flex-direction:column;align-items:flex-start;width:100%"
} : {
  name: "18t0chz-DialogComboboxOptionControlledList",
  styles: "display:flex;flex-direction:column;align-items:flex-start;width:100%;label:DialogComboboxOptionControlledList;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$g
};
const DialogComboboxOptionControlledList = /*#__PURE__*/forwardRef((_ref, forwardedRef) => {
  let {
    options,
    onChange,
    loading,
    loadingDescription = 'DialogComboboxOptionControlledList',
    withProgressiveLoading,
    withSearch,
    showAllOption,
    allOptionLabel = 'All',
    ...restProps
  } = _ref;
  const {
    isInsideDialogCombobox,
    multiSelect,
    value,
    setValue,
    setIsControlled
  } = useDialogComboboxContext();
  const [lookAhead, setLookAhead] = useState('');
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxOptionControlledList` must be used within `DialogCombobox`');
  }
  const lookAheadTimeout = useRef(null);
  const ref = useRef(null);
  useImperativeHandle(forwardedRef, () => ref.current);
  useEffect(() => {
    if (lookAheadTimeout.current) {
      clearTimeout(lookAheadTimeout.current);
    }
    lookAheadTimeout.current = setTimeout(() => {
      setLookAhead('');
    }, 1500);
    return () => {
      if (lookAheadTimeout.current) {
        clearTimeout(lookAheadTimeout.current);
      }
    };
  }, [lookAhead]);
  useEffect(() => {
    var _ref$current;
    if (loading && !withProgressiveLoading) {
      return;
    }
    const optionItems = (_ref$current = ref.current) === null || _ref$current === void 0 ? void 0 : _ref$current.querySelectorAll('[role="option"]');
    const hasTabIndexedOption = Array.from(optionItems !== null && optionItems !== void 0 ? optionItems : []).some(optionItem => {
      return optionItem.getAttribute('tabindex') === '0';
    });
    if (!hasTabIndexedOption) {
      const firstOptionItem = optionItems === null || optionItems === void 0 ? void 0 : optionItems[0];
      if (firstOptionItem) {
        highlightOption(firstOptionItem, undefined, false);
      }
    }
  }, [loading, withProgressiveLoading]);
  const isOptionChecked = options.reduce((acc, option) => {
    acc[option] = value === null || value === void 0 ? void 0 : value.includes(option);
    return acc;
  }, {});
  const handleUpdate = updatedValue => {
    setIsControlled(true);
    let newValue = [];
    if (multiSelect) {
      if (value.find(item => item === updatedValue)) {
        newValue = value.filter(item => item !== updatedValue);
      } else {
        newValue = [...value, updatedValue];
      }
    } else {
      newValue = [updatedValue];
    }
    setValue(newValue);
    isOptionChecked[updatedValue] = !isOptionChecked[updatedValue];
    if (onChange) {
      onChange(newValue);
    }
  };
  const handleSelectAll = () => {
    setIsControlled(true);
    if (value.length === options.length) {
      setValue([]);
      options.forEach(option => {
        isOptionChecked[option] = false;
      });
      if (onChange) {
        onChange([]);
      }
    } else {
      setValue(options);
      options.forEach(option => {
        isOptionChecked[option] = true;
      });
      if (onChange) {
        onChange(options);
      }
    }
  };
  const renderedOptions = jsxs(Fragment, {
    children: [showAllOption && multiSelect && jsx(DialogComboboxOptionListCheckboxItem, {
      value: "all",
      onChange: handleSelectAll,
      checked: value.length === options.length,
      indeterminate: Boolean(value.length) && value.length !== options.length,
      children: allOptionLabel
    }), options && options.length > 0 ? options.map((option, key) => multiSelect ? jsx(DialogComboboxOptionListCheckboxItem, {
      value: option,
      checked: isOptionChecked[option],
      onChange: handleUpdate,
      children: option
    }, key) : jsx(DialogComboboxOptionListSelectItem, {
      value: option,
      checked: isOptionChecked[option],
      onChange: handleUpdate,
      children: option
    }, key)) : jsx(EmptyResults, {})]
  });
  const optionList = jsx(DialogComboboxOptionList, {
    children: withSearch ? jsx(DialogComboboxOptionListSearch, {
      hasWrapper: true,
      children: renderedOptions
    }) : renderedOptions
  });
  return jsx("div", {
    ref: ref,
    "aria-busy": loading,
    css: _ref2$b,
    ...restProps,
    children: jsx(DialogComboboxOptionListContextProvider, {
      value: {
        isInsideDialogComboboxOptionList: true,
        lookAhead,
        setLookAhead
      },
      children: jsx(Fragment, {
        children: loading ? withProgressiveLoading ? jsxs(Fragment, {
          children: [optionList, jsx(LoadingSpinner, {
            "aria-label": "Loading",
            alt: "Loading spinner",
            loadingDescription: loadingDescription
          })]
        }) : jsx(LoadingSpinner, {
          "aria-label": "Loading",
          alt: "Loading spinner",
          loadingDescription: loadingDescription
        }) : optionList
      })
    })
  });
});

const DialogComboboxSectionHeader = _ref => {
  let {
    children,
    ...props
  } = _ref;
  const {
    isInsideDialogCombobox
  } = useDialogComboboxContext();
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxSectionHeader` must be used within `DialogCombobox`');
  }
  return jsx(SectionHeader, {
    ...props,
    children: children
  });
};

const DialogComboboxSeparator = props => {
  const {
    isInsideDialogCombobox
  } = useDialogComboboxContext();
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxSeparator` must be used within `DialogCombobox`');
  }
  return jsx(Separator, {
    ...props
  });
};

const getTriggerWrapperStyles = (removable, width) => /*#__PURE__*/css(importantify({
  display: 'inline-flex',
  alignItems: 'center',
  ...(width && {
    width: width
  }),
  ...(removable && {
    '& > button:last-of-type': importantify({
      borderBottomLeftRadius: 0,
      borderTopLeftRadius: 0,
      marginLeft: -1
    })
  })
}), process.env.NODE_ENV === "production" ? "" : ";label:getTriggerWrapperStyles;");
const getTriggerStyles = function (theme) {
  let disabled = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;
  let maxWidth = arguments.length > 2 ? arguments[2] : undefined;
  let minWidth = arguments.length > 3 ? arguments[3] : undefined;
  let removable = arguments.length > 4 ? arguments[4] : undefined;
  let width = arguments.length > 5 ? arguments[5] : undefined;
  let validationState = arguments.length > 6 ? arguments[6] : undefined;
  let isBare = arguments.length > 7 ? arguments[7] : undefined;
  let isSelect = arguments.length > 8 ? arguments[8] : undefined;
  const removeButtonInteractionStyles = {
    ...(removable && {
      zIndex: theme.options.zIndexBase + 2,
      '&& + button': {
        marginLeft: -1,
        zIndex: theme.options.zIndexBase + 1
      }
    })
  };
  const validationColor = getValidationStateColor(theme, validationState);
  return /*#__PURE__*/css(importantify({
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
    borderRadius: 4,
    color: theme.colors.textPrimary,
    lineHeight: theme.typography.lineHeightBase,
    fontSize: theme.typography.fontSizeBase,
    cursor: 'pointer',
    ...(width && {
      width: width,
      // Only set flex: 1 to items with width, otherwise in flex containers the trigger will take up all the space and break current usages that depend on content for width
      flex: 1
    }),
    ...(removable && {
      borderBottomRightRadius: 0,
      borderTopRightRadius: 0,
      borderRightColor: 'transparent'
    }),
    '&:hover': {
      background: isBare ? 'transparent' : theme.colors.actionDefaultBackgroundHover,
      borderColor: theme.colors.actionDefaultBorderHover,
      ...removeButtonInteractionStyles
    },
    '&:focus': {
      borderColor: theme.colors.actionDefaultBorderFocus,
      ...removeButtonInteractionStyles
    },
    ...(validationState && {
      borderColor: validationColor,
      '&:hover': {
        borderColor: validationColor
      },
      '&:focus': {
        outlineColor: validationColor,
        outlineOffset: -2
      }
    }),
    ...(isSelect && !disabled && {
      '&&, &&:hover, &&:focus': {
        background: 'transparent'
      },
      '&&:hover': {
        borderColor: theme.colors.actionDefaultBorderHover
      },
      '&&:focus, &[data-state="open"]': {
        outlineColor: theme.colors.actionPrimaryBackgroundDefault,
        outlineWidth: 2,
        outlineOffset: -2,
        outlineStyle: 'solid',
        borderColor: 'transparent',
        boxShadow: 'none'
      }
    }),
    [`&[disabled]`]: {
      background: theme.colors.actionDisabledBackground,
      color: theme.colors.actionDisabledText,
      pointerEvents: 'none',
      userSelect: 'none',
      borderColor: theme.colors.actionDisabledBorder
    }
  }), process.env.NODE_ENV === "production" ? "" : ";label:getTriggerStyles;");
};
const DialogComboboxTrigger = /*#__PURE__*/forwardRef((_ref, forwardedRef) => {
  var _restProps$className;
  let {
    removable = false,
    onRemove,
    children,
    minWidth = 0,
    maxWidth = 9999,
    showTagAfterValueCount = 3,
    allowClear = true,
    controlled,
    onClear,
    wrapperProps,
    width,
    withChevronIcon = true,
    validationState,
    withInlineLabel = true,
    placeholder,
    id: legacyId,
    isBare = false,
    formatDisplayedValue = value => value,
    ...restProps
  } = _ref;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    label,
    id: topLevelId,
    value,
    isInsideDialogCombobox,
    multiSelect,
    setValue
  } = useDialogComboboxContext();
  const {
    isSelect,
    placeholder: selectPlaceholder
  } = useSelectContext();
  const id = topLevelId !== null && topLevelId !== void 0 ? topLevelId : legacyId;
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxTrigger` must be used within `DialogCombobox`');
  }
  const handleRemove = () => {
    if (!onRemove) {
      console.warn('DialogCombobox.Trigger: Attempted remove without providing onRemove handler');
    } else {
      onRemove();
    }
  };
  const handleClear = e => {
    e.stopPropagation();
    if (controlled) {
      setValue([]);
      onClear === null || onClear === void 0 || onClear();
    } else if (!onClear) {
      console.warn('DialogCombobox.Trigger: Attempted clear without providing onClear handler');
    } else {
      onClear();
    }
  };
  const [showTooltip, setShowTooltip] = React__default.useState();
  const triggerContentRef = React__default.useRef(null);
  useEffect(() => {
    if ((value === null || value === void 0 ? void 0 : value.length) > showTagAfterValueCount) {
      setShowTooltip(true);
    } else if (triggerContentRef.current) {
      const {
        clientWidth,
        scrollWidth
      } = triggerContentRef.current;
      setShowTooltip(clientWidth < scrollWidth);
    }
  }, [showTagAfterValueCount, value]);
  const renderFormattedValue = (v, index) => {
    const formattedValue = formatDisplayedValue(v);
    return jsxs(React__default.Fragment, {
      children: [index > 0 && ', ', typeof formattedValue === 'string' ? formattedValue : jsx("span", {
        children: formattedValue
      })]
    }, index);
  };
  const getStringValue = v => {
    const formattedValue = formatDisplayedValue(v);
    return typeof formattedValue === 'string' ? formattedValue : v;
  };
  const numValues = Array.isArray(value) ? value.length : 1;
  const concatenatedValues = Array.isArray(value) ? jsxs(Fragment, {
    children: [value.slice(0, numValues > 10 ? 10 : undefined).map(renderFormattedValue), numValues > 10 && ` + ${numValues - 10}`]
  }) : renderFormattedValue(value, 0);
  const displayedValues = jsx("span", {
    children: concatenatedValues
  });
  const valuesBeforeBadge = Array.isArray(value) ? jsx(Fragment, {
    children: value.slice(0, showTagAfterValueCount).map(renderFormattedValue)
  }) : renderFormattedValue(value, 0);
  let ariaLabel = '';
  if (!isSelect && !id && label) {
    ariaLabel = /*#__PURE__*/React__default.isValidElement(label) ? 'Dialog Combobox' : `${label}`;
    if (value !== null && value !== void 0 && value.length) {
      const stringValues = Array.isArray(value) ? value.map(getStringValue).join(', ') : getStringValue(value);
      ariaLabel += multiSelect ? `, multiselectable, ${value.length} options selected: ${stringValues}` : `, selected option: ${stringValues}`;
    } else {
      ariaLabel += multiSelect ? ', multiselectable, 0 options selected' : ', no option selected';
    }
  } else if (isSelect) {
    var _ref2;
    ariaLabel = (_ref2 = (typeof label === 'string' ? label : '') || restProps['aria-label']) !== null && _ref2 !== void 0 ? _ref2 : '';
  }
  const customSelectContent = isSelect && children ? children : null;
  const dialogComboboxClassname = !isSelect ? `${classNamePrefix}-dialogcombobox` : '';
  const selectV2Classname = isSelect ? `${classNamePrefix}-selectv2` : '';
  const triggerContent = isSelect ? jsxs(Popover.Trigger, {
    ...(ariaLabel && {
      'aria-label': ariaLabel
    }),
    ref: forwardedRef,
    role: "combobox",
    "aria-haspopup": "listbox",
    "aria-invalid": validationState === 'error',
    id: id,
    ...restProps,
    css: getTriggerStyles(theme, restProps.disabled, maxWidth, minWidth, removable, width, validationState, isBare, isSelect),
    children: [jsx("span", {
      css: /*#__PURE__*/css({
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        height: theme.typography.lineHeightBase,
        marginRight: 'auto'
      }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;"),
      ref: triggerContentRef,
      children: value !== null && value !== void 0 && value.length ? customSelectContent !== null && customSelectContent !== void 0 ? customSelectContent : displayedValues : jsx("span", {
        css: /*#__PURE__*/css({
          color: theme.colors.textPlaceholder
        }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;"),
        children: selectPlaceholder
      })
    }), allowClear && value !== null && value !== void 0 && value.length ? jsx(ClearSelectionButton, {
      onClick: handleClear
    }) : null, jsx(ChevronDownIcon$1, {
      css: /*#__PURE__*/css({
        color: theme.colors.textSecondary,
        marginLeft: theme.spacing.xs
      }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;")
    })]
  }) : jsxs(Popover.Trigger, {
    id: id,
    ...(ariaLabel && {
      'aria-label': ariaLabel
    }),
    ref: forwardedRef,
    role: "combobox",
    "aria-haspopup": "listbox",
    "aria-invalid": validationState === 'error',
    ...restProps,
    css: getTriggerStyles(theme, restProps.disabled, maxWidth, minWidth, removable, width, validationState, isBare, isSelect),
    children: [jsxs("span", {
      css: /*#__PURE__*/css({
        display: 'flex',
        alignItems: 'center',
        height: theme.typography.lineHeightBase,
        marginRight: 'auto',
        '&, & > *': {
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis'
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;"),
      ref: triggerContentRef,
      children: [withInlineLabel ? jsxs("span", {
        css: /*#__PURE__*/css({
          height: theme.typography.lineHeightBase,
          marginRight: theme.spacing.xs,
          whiteSpace: 'unset',
          overflow: 'unset',
          textOverflow: 'unset'
        }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;"),
        children: [label, value !== null && value !== void 0 && value.length ? ':' : null]
      }) : !(value !== null && value !== void 0 && value.length) && jsx("span", {
        css: /*#__PURE__*/css({
          color: theme.colors.textPlaceholder
        }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;"),
        children: placeholder
      }), (value === null || value === void 0 ? void 0 : value.length) > showTagAfterValueCount ? jsxs(Fragment, {
        children: [jsx("span", {
          style: {
            marginRight: theme.spacing.xs
          },
          children: valuesBeforeBadge
        }), jsx(DialogComboboxCountBadge, {
          countStartAt: showTagAfterValueCount,
          role: "status",
          "aria-label": "Selected options count"
        })]
      }) : displayedValues]
    }), allowClear && value !== null && value !== void 0 && value.length ? jsx(ClearSelectionButton, {
      onClick: handleClear
    }) : null, withChevronIcon ? jsx(ChevronDownIcon$1, {
      css: /*#__PURE__*/css({
        color: theme.colors.textSecondary,
        justifySelf: 'flex-end',
        marginLeft: theme.spacing.xs
      }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;")
    }) : null]
  });
  return jsxs("div", {
    ...wrapperProps,
    className: `${(_restProps$className = restProps === null || restProps === void 0 ? void 0 : restProps.className) !== null && _restProps$className !== void 0 ? _restProps$className : ''} ${dialogComboboxClassname} ${selectV2Classname}`.trim(),
    css: [getTriggerWrapperStyles(removable, width), wrapperProps === null || wrapperProps === void 0 ? void 0 : wrapperProps.css, process.env.NODE_ENV === "production" ? "" : ";label:DialogComboboxTrigger;"],
    ...addDebugOutlineIfEnabled(),
    children: [showTooltip && value !== null && value !== void 0 && value.length ? jsx(LegacyTooltip, {
      title: customSelectContent !== null && customSelectContent !== void 0 ? customSelectContent : displayedValues,
      children: triggerContent
    }) : triggerContent, removable && jsx(Button, {
      componentId: "codegen_design-system_src_design-system_dialogcombobox_dialogcomboboxtrigger.tsx_355",
      "aria-label": `Remove ${label}`,
      onClick: handleRemove,
      dangerouslySetForceIconStyles: true,
      children: jsx(CloseIcon, {
        "aria-label": `Remove ${label}`,
        "aria-hidden": "false"
      })
    })]
  });
});
/**
 * A custom button trigger that can be wrapped around any button.
 */
const DialogComboboxCustomButtonTriggerWrapper = _ref3 => {
  let {
    children
  } = _ref3;
  return jsx(Popover.Trigger, {
    asChild: true,
    children: children
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$f() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const DEFAULT_WIDTH = 320;
const MIN_WIDTH = 320;
const MAX_WIDTH = '90vw';
const DEFAULT_POSITION = 'right';
const ZINDEX_OVERLAY = 1;
const ZINDEX_CONTENT = ZINDEX_OVERLAY + 1;
var _ref2$a = process.env.NODE_ENV === "production" ? {
  name: "zh83op",
  styles: "flex-grow:1;margin-bottom:0;margin-top:0;white-space:nowrap;overflow:hidden"
} : {
  name: "h5yqvj-Content",
  styles: "flex-grow:1;margin-bottom:0;margin-top:0;white-space:nowrap;overflow:hidden;label:Content;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$f
};
const Content$1 = _ref => {
  let {
    children,
    footer,
    title,
    width,
    position: positionOverride,
    useCustomScrollBehavior,
    expandContentToFullHeight,
    disableOpenAutoFocus,
    onInteractOutside,
    seeThrough,
    hideClose,
    componentId = 'design_system.drawer.content',
    analyticsEvents
  } = _ref;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const {
    theme
  } = useDesignSystemTheme();
  const horizontalContentPadding = theme.spacing.lg;
  const [shouldContentBeFocusable, setShouldContentBeFocusable] = useState(false);
  const contentContainerRef = useRef(null);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Drawer,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : []
  });
  const {
    elementRef: onViewRef
  } = useNotifyOnFirstView({
    onView: eventContext.onView
  });
  const contentRef = useCallback(node => {
    if (!node || !node.clientHeight) return;
    setShouldContentBeFocusable(node.scrollHeight > node.clientHeight);
  }, []);
  const mergedContentRef = useMergeRefs([contentRef, onViewRef]);
  const position = positionOverride !== null && positionOverride !== void 0 ? positionOverride : DEFAULT_POSITION;
  const overlayShow = position === 'right' ? keyframes({
    '0%': {
      transform: 'translate(100%, 0)'
    },
    '100%': {
      transform: 'translate(0, 0)'
    }
  }) : keyframes({
    '0%': {
      transform: 'translate(-100%, 0)'
    },
    '100%': {
      transform: 'translate(0, 0)'
    }
  });
  const dialogPrimitiveContentStyle = /*#__PURE__*/css({
    color: theme.colors.textPrimary,
    backgroundColor: theme.colors.backgroundPrimary,
    boxShadow: 'hsl(206 22% 7% / 35%) 0px 10px 38px -10px, hsl(206 22% 7% / 20%) 0px 10px 20px -15px',
    position: 'fixed',
    top: 0,
    left: position === 'left' ? 0 : undefined,
    right: position === 'right' ? 0 : undefined,
    boxSizing: 'border-box',
    width: width !== null && width !== void 0 ? width : DEFAULT_WIDTH,
    minWidth: MIN_WIDTH,
    maxWidth: MAX_WIDTH,
    zIndex: theme.options.zIndexBase + ZINDEX_CONTENT,
    height: '100vh',
    paddingTop: theme.spacing.md,
    paddingLeft: 0,
    paddingBottom: 0,
    paddingRight: 0,
    overflow: 'hidden',
    '&:focus': {
      outline: 'none'
    },
    '@media (prefers-reduced-motion: no-preference)': {
      animation: `${overlayShow} 350ms cubic-bezier(0.16, 1, 0.3, 1)`
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:dialogPrimitiveContentStyle;");
  return jsxs(DialogPrimitive.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: [jsx(DialogPrimitive.Overlay, {
      css: /*#__PURE__*/css({
        backgroundColor: theme.colors.overlayOverlay,
        position: 'fixed',
        inset: 0,
        // needed so that it covers the PersonaNavSidebar
        zIndex: theme.options.zIndexBase + ZINDEX_OVERLAY,
        opacity: seeThrough ? 0 : 1
      }, process.env.NODE_ENV === "production" ? "" : ";label:Content;")
    }), jsx(DialogPrimitive.DialogContent, {
      ...addDebugOutlineIfEnabled(),
      css: dialogPrimitiveContentStyle,
      style: {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'flex-start',
        opacity: seeThrough ? 0 : 1,
        ...(theme.isDarkMode && {
          borderLeft: `1px solid ${theme.colors.borderDecorative}`,
          boxShadow: 'none'
        })
      },
      "aria-hidden": seeThrough,
      ref: contentContainerRef,
      onOpenAutoFocus: event => {
        if (disableOpenAutoFocus) {
          event.preventDefault();
        }
      },
      onInteractOutside: onInteractOutside,
      children: jsxs(ApplyDesignSystemContextOverrides, {
        getPopupContainer: () => {
          var _contentContainerRef$;
          return (_contentContainerRef$ = contentContainerRef.current) !== null && _contentContainerRef$ !== void 0 ? _contentContainerRef$ : document.body;
        },
        children: [jsxs("div", {
          css: /*#__PURE__*/css({
            flexGrow: 0,
            flexShrink: 1,
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            paddingRight: horizontalContentPadding,
            paddingLeft: horizontalContentPadding,
            marginBottom: theme.spacing.sm
          }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"),
          children: [jsx(DialogPrimitive.Title, {
            title: typeof title === 'string' ? title : undefined,
            asChild: typeof title === 'string',
            css: _ref2$a,
            children: typeof title === 'string' ? jsx(Typography.Title, {
              level: 2,
              withoutMargins: true,
              ellipsis: true,
              children: title
            }) : title
          }), !hideClose && jsx(DialogPrimitive.Close, {
            asChild: true,
            css: /*#__PURE__*/css({
              flexShrink: 1,
              marginLeft: theme.spacing.xs
            }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"),
            children: jsx(Button, {
              componentId: `${componentId}.close`,
              "aria-label": "Close",
              icon: jsx(CloseIcon, {})
            })
          })]
        }), jsxs("div", {
          ref: mergedContentRef
          // Needed to make drawer content focusable when scrollable for keyboard-only users to be able to focus & scroll
          // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
          ,
          tabIndex: shouldContentBeFocusable ? 0 : -1,
          css: /*#__PURE__*/css({
            // in order to have specific content in the drawer scroll with fixed title
            // hide overflow here and remove padding on the right side; content will be responsible for setting right padding
            // so that the scrollbar will appear in the padding right gutter
            paddingRight: useCustomScrollBehavior ? 0 : horizontalContentPadding,
            paddingLeft: horizontalContentPadding,
            overflowY: useCustomScrollBehavior ? 'hidden' : 'auto',
            height: expandContentToFullHeight ? '100%' : undefined,
            ...(!useCustomScrollBehavior ? getShadowScrollStyles(theme) : {})
          }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"),
          children: [children, !footer && jsx(Spacer, {
            size: "lg"
          })]
        }), footer && jsx("div", {
          style: {
            paddingTop: theme.spacing.md,
            paddingRight: horizontalContentPadding,
            paddingLeft: horizontalContentPadding,
            paddingBottom: theme.spacing.lg,
            flexGrow: 0,
            flexShrink: 1
          },
          children: footer
        })]
      })
    })]
  });
};
function Root$1(props) {
  return jsx(DialogPrimitive.Root, {
    ...props
  });
}
function Trigger(props) {
  return jsx(DialogPrimitive.Trigger, {
    asChild: true,
    ...props
  });
}

var Drawer = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Content: Content$1,
  Root: Root$1,
  Trigger: Trigger
});

/**
 * @deprecated Use `DropdownMenu` instead.
 */
const Dropdown = _ref => {
  let {
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Dropdown$1, {
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$e() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const {
  Title: Title$1,
  Paragraph
} = Typography;
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
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getEmptyStyles;");
}
function getEmptyTitleStyles(theme, clsPrefix) {
  const styles = {
    [`&.${clsPrefix}-typography`]: {
      color: theme.colors.textSecondary,
      marginTop: 0,
      marginBottom: 0
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getEmptyTitleStyles;");
}
function getEmptyDescriptionStyles(theme, clsPrefix) {
  const styles = {
    [`&.${clsPrefix}-typography`]: {
      color: theme.colors.textSecondary,
      marginBottom: theme.spacing.md
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getEmptyDescriptionStyles;");
}
var _ref$6 = process.env.NODE_ENV === "production" ? {
  name: "zl1inp",
  styles: "display:flex;justify-content:center"
} : {
  name: "11tid6c-Empty",
  styles: "display:flex;justify-content:center;label:Empty;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$e
};
const Empty = props => {
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    title,
    description,
    image = jsx(ListIcon, {}),
    button,
    dangerouslyAppendEmotionCSS,
    ...dataProps
  } = props;
  return jsx("div", {
    ...dataProps,
    ...addDebugOutlineIfEnabled(),
    css: _ref$6,
    children: jsxs("div", {
      css: [getEmptyStyles(theme), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Empty;"],
      children: [image, title && jsx(Title$1, {
        level: 3,
        css: getEmptyTitleStyles(theme, classNamePrefix),
        children: title
      }), jsx(Paragraph, {
        css: getEmptyDescriptionStyles(theme, classNamePrefix),
        children: description
      }), button]
    })
  });
};

const getFormItemEmotionStyles = _ref => {
  let {
    theme,
    clsPrefix
  } = _ref;
  const clsFormItemLabel = `.${clsPrefix}-form-item-label`;
  const clsFormItemInputControl = `.${clsPrefix}-form-item-control-input`;
  const clsFormItemExplain = `.${clsPrefix}-form-item-explain`;
  const clsHasError = `.${clsPrefix}-form-item-has-error`;
  return /*#__PURE__*/css({
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
  }, process.env.NODE_ENV === "production" ? "" : ";label:getFormItemEmotionStyles;");
};
const FormDubois = /*#__PURE__*/forwardRef(function Form(_ref2, ref) {
  let {
    dangerouslySetAntdProps,
    children,
    ...props
  } = _ref2;
  const mergedProps = {
    ...props,
    layout: props.layout || 'vertical',
    requiredMark: props.requiredMark || false
  };
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Form$1, {
      ...addDebugOutlineIfEnabled(),
      ...mergedProps,
      colon: false,
      ref: ref,
      ...dangerouslySetAntdProps,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
});
const FormItem = _ref3 => {
  let {
    dangerouslySetAntdProps,
    children,
    ...props
  } = _ref3;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Form$1.Item, {
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
const FormNamespace = /* #__PURE__ */Object.assign(FormDubois, {
  Item: FormItem,
  List: Form$1.List,
  useForm: Form$1.useForm
});
const Form = FormNamespace;

// TODO: I'm doing this to support storybook's docgen;
// We should remove this once we have a better storybook integration,
// since these will be exposed in the library's exports.
const __INTERNAL_DO_NOT_USE__FormItem = FormItem;

const getMessageStyles = (clsPrefix, theme) => {
  const errorClass = `.${clsPrefix}-form-error-message`;
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
    [`&${successClass}`]: {
      color: theme.colors.textValidationSuccess
    },
    [`&${warningClass}`]: {
      color: theme.colors.textValidationWarning
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getMessageStyles;");
};
const VALIDATION_STATE_ICONS = {
  error: DangerIcon,
  success: CheckCircleIcon$1,
  warning: WarningIcon
};
function FormMessage(_ref) {
  let {
    id,
    message,
    type = 'error',
    className = '',
    css
  } = _ref;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const stateClass = `${classNamePrefix}-form-${type}-message`;
  const StateIcon = VALIDATION_STATE_ICONS[type];
  const wrapperClass = `${classNamePrefix}-form-message ${className} ${stateClass}`.trim();
  return jsxs("div", {
    ...(id && {
      id
    }),
    className: wrapperClass,
    ...addDebugOutlineIfEnabled(),
    css: [getMessageStyles(classNamePrefix, theme), css, process.env.NODE_ENV === "production" ? "" : ";label:FormMessage;"],
    role: "alert",
    children: [jsx(StateIcon, {}), jsx("div", {
      style: {
        paddingLeft: theme.spacing.xs
      },
      children: message
    })]
  });
}

const getHintStyles = (classNamePrefix, theme) => {
  const styles = {
    display: 'block',
    color: theme.colors.textSecondary,
    lineHeight: theme.typography.lineHeightSm,
    fontSize: theme.typography.fontSizeSm,
    [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-input-affix-wrapper, && + .${classNamePrefix}-select, && + .${classNamePrefix}-selectv2, && + .${classNamePrefix}-dialogcombobox, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group, && + .${classNamePrefix}-typeahead-combobox`]: {
      marginTop: theme.spacing.sm
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getHintStyles;");
};
const Hint = props => {
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    className,
    ...restProps
  } = props;
  return jsx("span", {
    ...addDebugOutlineIfEnabled(),
    className: classnames(`${classNamePrefix}-hint`, className),
    css: getHintStyles(classNamePrefix, theme),
    ...restProps
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$d() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const getLabelStyles$1 = (theme, _ref) => {
  let {
    inline
  } = _ref;
  const styles = {
    '&&': {
      color: theme.colors.textPrimary,
      fontWeight: theme.typography.typographyBoldFontWeight,
      display: inline ? 'inline' : 'block',
      lineHeight: theme.typography.lineHeightBase
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getLabelStyles;");
};
const getLabelWrapperStyles = (classNamePrefix, theme) => {
  const styles = {
    display: 'flex',
    gap: theme.spacing.xs,
    alignItems: 'center',
    [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-input-affix-wrapper, && + .${classNamePrefix}-select, && + .${classNamePrefix}-selectv2, && + .${classNamePrefix}-dialogcombobox, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group, && + .${classNamePrefix}-typeahead-combobox`]: {
      marginTop: theme.spacing.sm
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getLabelWrapperStyles;");
};
var _ref2$9 = process.env.NODE_ENV === "production" ? {
  name: "s5xdrg",
  styles: "display:flex;align-items:center"
} : {
  name: "10kkfun-label",
  styles: "display:flex;align-items:center;label:label;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$d
};
const Label = props => {
  const {
    children,
    className,
    inline,
    required,
    infoPopoverContents,
    infoPopoverProps = {},
    ...restProps
  } = props; // Destructure the new prop
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const label = jsx("label", {
    ...addDebugOutlineIfEnabled(),
    css: [getLabelStyles$1(theme, {
      inline
    }), ...(!infoPopoverContents ? [getLabelWrapperStyles(classNamePrefix, theme)] : [])],
    className: classnames(`${classNamePrefix}-label`, className),
    ...restProps,
    children: jsxs("span", {
      css: _ref2$9,
      children: [children, required && jsx("span", {
        "aria-hidden": "true",
        children: "*"
      })]
    })
  });
  return infoPopoverContents ? jsxs("div", {
    css: getLabelWrapperStyles(classNamePrefix, theme),
    children: [label, jsx(InfoPopover, {
      ...infoPopoverProps,
      children: infoPopoverContents
    })]
  }) : label;
};

function getSelectEmotionStyles(_ref) {
  let {
    clsPrefix,
    theme,
    validationState
  } = _ref;
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
  const validationColor = getValidationStateColor(theme, validationState);
  const styles = {
    ...addDebugOutlineStylesIfEnabled(theme),
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
        outlineColor: theme.colors.actionPrimaryBackgroundDefault,
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
        pointerEvents: 'none'
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
      borderTopRightRadius: theme.borders.borderRadiusMd,
      borderBottomRightRadius: theme.borders.borderRadiusMd,
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
    ...(validationState && {
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
    }),
    ...getAnimationCss(theme.options.enableAnimation)
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSelectEmotionStyles;");
}
function getDropdownStyles(clsPrefix, theme) {
  const classItem = `.${clsPrefix}-item-option`;
  const classItemActive = `.${clsPrefix}-item-option-active`;
  const classItemSelected = `.${clsPrefix}-item-option-selected`;
  const classItemState = `.${clsPrefix}-item-option-state`;
  const styles = {
    borderColor: theme.colors.borderDecorative,
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
    ...getDarkModePortalStyles(theme)
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getDropdownStyles;");
}
function getLoadingIconStyles(theme) {
  return /*#__PURE__*/css({
    fontSize: 20,
    color: theme.colors.textSecondary,
    lineHeight: '20px'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getLoadingIconStyles;");
}
const scrollbarVisibleItemsCount = 8;
const getIconSizeStyle = (theme, newIconDefault) => importantify({
  fontSize: newIconDefault !== null && newIconDefault !== void 0 ? newIconDefault : theme.general.iconFontSize
});
function DuboisSelect(_ref2, ref) {
  let {
    children,
    validationState,
    loading,
    loadingDescription = 'Select',
    mode,
    options,
    notFoundContent,
    optionFilterProp,
    dangerouslySetAntdProps,
    virtual,
    dropdownClassName,
    id,
    onDropdownVisibleChange,
    maxHeight,
    ...restProps
  } = _ref2;
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('select');
  const [isOpen, setIsOpen] = useState(false);
  const [uniqueId, setUniqueId] = useState('');

  // Antd's default is 256, to show half an extra item when scrolling we add 0.5 height extra
  // Reducing to 5.5 as it's default with other components is not an option here because it would break existing usages relying on 8 items being shown by default
  const MAX_HEIGHT = maxHeight !== null && maxHeight !== void 0 ? maxHeight : theme.general.heightSm * 8.5;
  useEffect(() => {
    setUniqueId(id || _uniqueId('dubois-select-'));
  }, [id]);
  useEffect(() => {
    var _document$getElementB;
    // Ant doesn't populate aria-expanded on init (only on user interaction) so we need to do it ourselves
    // in order to pass accessibility tests (Microsoft Accessibility Insights). See: JOBS-11125
    (_document$getElementB = document.getElementById(uniqueId)) === null || _document$getElementB === void 0 || _document$getElementB.setAttribute('aria-expanded', 'false');
  }, [uniqueId]);
  return jsx(ClassNames, {
    children: _ref3 => {
      let {
        css: css$1
      } = _ref3;
      return jsxs(DesignSystemAntDConfigProvider, {
        children: [loading && jsx(LoadingState, {
          description: loadingDescription
        }), jsx(Select$1, {
          onDropdownVisibleChange: visible => {
            onDropdownVisibleChange === null || onDropdownVisibleChange === void 0 || onDropdownVisibleChange(visible);
            setIsOpen(visible);
          }
          // unset aria-owns, aria-controls, and aria-activedescendant if the dropdown is closed;
          // ant always sets these even if the dropdown isn't present in the DOM yet.
          // This was flagged by Microsoft Accessibility Insights. See: JOBS-11125
          ,
          ...(!isOpen ? {
            'aria-owns': undefined,
            'aria-controls': undefined,
            'aria-activedescendant': undefined
          } : {}),
          id: uniqueId,
          css: getSelectEmotionStyles({
            clsPrefix,
            theme,
            validationState
          }),
          removeIcon: jsx(CloseIcon, {
            "aria-hidden": "false",
            css: getIconSizeStyle(theme)
          }),
          clearIcon: jsx(XCircleFillIcon$1, {
            "aria-hidden": "false",
            css: getIconSizeStyle(theme, 12),
            "aria-label": "close-circle"
          }),
          ref: ref,
          suffixIcon: loading && mode === 'tags' ? jsx(LoadingIcon, {
            spin: true,
            "aria-label": "loading",
            "aria-hidden": "false",
            css: getIconSizeStyle(theme, 12)
          }) : jsx(ChevronDownIcon$1, {
            css: getIconSizeStyle(theme)
          }),
          menuItemSelectedIcon: jsx(CheckIcon, {
            css: getIconSizeStyle(theme)
          }),
          showArrow: true,
          dropdownMatchSelectWidth: true,
          notFoundContent: notFoundContent !== null && notFoundContent !== void 0 ? notFoundContent : jsx("div", {
            css: /*#__PURE__*/css({
              color: theme.colors.textSecondary,
              textAlign: 'center'
            }, process.env.NODE_ENV === "production" ? "" : ";label:DuboisSelect;"),
            children: "No results found"
          }),
          dropdownClassName: css$1([getDropdownStyles(clsPrefix, theme), dropdownClassName]),
          listHeight: MAX_HEIGHT,
          maxTagPlaceholder: items => `+ ${items.length} more`,
          mode: mode,
          options: options,
          loading: loading,
          filterOption: true
          // NOTE(FEINF-1102): This is needed to avoid ghost scrollbar that generates error when clicked on exactly 8 elements
          // Because by default AntD uses true for virtual, we want to replicate the same even if there are no children
          ,
          virtual: virtual !== null && virtual !== void 0 ? virtual : children && Array.isArray(children) && children.length !== scrollbarVisibleItemsCount || options && options.length !== scrollbarVisibleItemsCount || !children && !options,
          optionFilterProp: optionFilterProp !== null && optionFilterProp !== void 0 ? optionFilterProp : 'children',
          ...restProps,
          ...dangerouslySetAntdProps,
          children: loading && mode !== 'tags' ? jsxs(Fragment, {
            children: [children, jsx(LegacyOption, {
              disabled: true,
              value: "select-loading-options",
              className: `${clsPrefix}-loading-options`,
              children: jsx(LoadingIcon, {
                "aria-hidden": "false",
                spin: true,
                css: getLoadingIconStyles(theme),
                "aria-label": "loading"
              })
            })]
          }) : children
        })]
      });
    }
  });
}

/** @deprecated Use `LegacySelectOptionProps` */

const LegacySelectOption = /*#__PURE__*/forwardRef(function Option(props, ref) {
  const {
    dangerouslySetAntdProps,
    ...restProps
  } = props;
  return jsx(Select$1.Option, {
    ...restProps,
    ref: ref,
    ...dangerouslySetAntdProps
  });
});

// Needed for rc-select to not throw warning about our component not being Select.Option
LegacySelectOption.isSelectOption = true;

/**
 * @deprecated use LegacySelect.Option instead
 */
const LegacyOption = LegacySelectOption;

/** @deprecated Use `LegacySelectOptGroupProps` */

const LegacySelectOptGroup = /* #__PURE__ */(() => {
  const OptGroup = /*#__PURE__*/forwardRef(function OptGroup(props, ref) {
    return jsx(Select$1.OptGroup, {
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
 */
const LegacyOptGroup = LegacySelectOptGroup;

/**
 * @deprecated Use Select, TypeaheadCombobox, or DialogCombobox depending on your use-case. See http://go/deprecate-ant-select for more information
 */
const LegacySelect = /* #__PURE__ */(() => {
  const DuboisRefForwardedSelect = /*#__PURE__*/forwardRef(DuboisSelect);
  DuboisRefForwardedSelect.Option = LegacySelectOption;
  DuboisRefForwardedSelect.OptGroup = LegacySelectOptGroup;
  return DuboisRefForwardedSelect;
})();

const getRadioInputStyles = _ref => {
  let {
    clsPrefix,
    theme
  } = _ref;
  return {
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
        // This prevents an awkward jitter on the border
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: '50%',
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
        // Ant uses !important
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
          background: theme.colors.actionDefaultBackgroundPress,
          borderColor: theme.colors.actionDefaultBorderFocus,
          boxShadow: `0 0 0 1px ${theme.colors.actionDefaultBackgroundDefault}, 0 0 0 3px ${theme.colors.primary}`
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
  };
};
const getCommonRadioGroupStyles = _ref2 => {
  let {
    theme,
    clsPrefix,
    classNamePrefix
  } = _ref2;
  return /*#__PURE__*/css({
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
  }, process.env.NODE_ENV === "production" ? "" : ";label:getCommonRadioGroupStyles;");
};
const getHorizontalRadioGroupStyles = _ref3 => {
  let {
    theme,
    classNamePrefix
  } = _ref3;
  return /*#__PURE__*/css({
    '&&': {
      display: 'grid',
      gridTemplateRows: '[label] auto [hint] auto',
      gridAutoColumns: 'max-content',
      gridColumnGap: theme.spacing.md
    },
    '& > label': {
      gridRow: 'label / label',
      marginRight: 0
    },
    [`& > label + .${classNamePrefix}-hint`]: {
      display: 'inline-block',
      gridRow: 'hint / hint'
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getHorizontalRadioGroupStyles;");
};
const getVerticalRadioGroupStyles = _ref4 => {
  let {
    theme,
    classNamePrefix
  } = _ref4;
  return /*#__PURE__*/css({
    display: 'flex',
    flexDirection: 'column',
    flexWrap: 'wrap',
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
  }, process.env.NODE_ENV === "production" ? "" : ";label:getVerticalRadioGroupStyles;");
};
const getRadioStyles = _ref5 => {
  let {
    theme,
    clsPrefix
  } = _ref5;
  // Default as bold for standalone radios
  const fontWeight = 'normal';
  const styles = {
    fontWeight
  };
  return /*#__PURE__*/css({
    ...getRadioInputStyles({
      theme,
      clsPrefix
    }),
    ...styles
  }, process.env.NODE_ENV === "production" ? "" : ";label:getRadioStyles;");
};
const DuboisRadio = /*#__PURE__*/forwardRef(function Radio(_ref6, ref) {
  let {
    children,
    dangerouslySetAntdProps,
    ...props
  } = _ref6;
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('radio');
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Radio$1, {
      ...addDebugOutlineIfEnabled(),
      css: getRadioStyles({
        theme,
        clsPrefix
      }),
      ...props,
      ...dangerouslySetAntdProps,
      ref: ref,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
});
const StyledRadioGroup = /*#__PURE__*/forwardRef(function StyledRadioGroup(_ref7, ref) {
  let {
    children,
    dangerouslySetAntdProps,
    componentId,
    analyticsEvents,
    valueHasNoPii,
    onChange,
    ...props
  } = _ref7;
  const {
    theme,
    getPrefixedClassName,
    classNamePrefix
  } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('radio');
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.RadioGroup,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii
  });
  const onChangeWrapper = useCallback(e => {
    var _eventContext$onValue;
    (_eventContext$onValue = eventContext.onValueChange) === null || _eventContext$onValue === void 0 || _eventContext$onValue.call(eventContext, e.target.value);
    onChange === null || onChange === void 0 || onChange(e);
  }, [eventContext, onChange]);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Radio$1.Group, {
      ...addDebugOutlineIfEnabled(),
      ...props,
      css: getCommonRadioGroupStyles({
        theme,
        clsPrefix,
        classNamePrefix
      }),
      onChange: onChangeWrapper,
      ...dangerouslySetAntdProps,
      ref: ref,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
});
const HorizontalGroup = /*#__PURE__*/forwardRef(function HorizontalGroup(_ref8, ref) {
  let {
    dangerouslySetAntdProps,
    ...props
  } = _ref8;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  return jsx(StyledRadioGroup, {
    css: getHorizontalRadioGroupStyles({
      theme,
      classNamePrefix
    }),
    ...props,
    ref: ref,
    ...dangerouslySetAntdProps
  });
});
const Group = /*#__PURE__*/forwardRef(function HorizontalGroup(_ref9, ref) {
  let {
    dangerouslySetAntdProps,
    layout = 'vertical',
    ...props
  } = _ref9;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  return jsx(StyledRadioGroup, {
    css: layout === 'horizontal' ? getHorizontalRadioGroupStyles({
      theme,
      classNamePrefix
    }) : getVerticalRadioGroupStyles({
      theme,
      classNamePrefix
    }),
    ...props,
    ref: ref,
    ...dangerouslySetAntdProps
  });
});

// Note: We are overriding ant's default "Group" with our own.
const RadioNamespace = /* #__PURE__ */Object.assign(DuboisRadio, {
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
 */
const Select = props => {
  const {
    children,
    placeholder,
    value,
    label,
    ...restProps
  } = props;
  return jsx(SelectContextProvider, {
    value: {
      isSelect: true,
      placeholder
    },
    children: jsx(DialogCombobox, {
      label: label,
      value: value ? [value] : [],
      ...restProps,
      children: children
    })
  });
};

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
const SelectContent = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    children,
    minWidth = 150,
    ...restProps
  } = _ref;
  return jsx(DialogComboboxContent, {
    minWidth: minWidth,
    ...restProps,
    ref: ref,
    children: jsx(DialogComboboxOptionList, {
      children: children
    })
  });
});

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
const SelectOption = /*#__PURE__*/forwardRef((props, ref) => {
  const {
    value
  } = useDialogComboboxContext();
  return jsx(DialogComboboxOptionListSelectItem, {
    checked: value && value[0] === props.value,
    ...props,
    ref: ref
  });
});

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
const SelectOptionGroup = props => {
  const {
    name,
    children,
    ...restProps
  } = props;
  return jsxs(Fragment, {
    children: [jsx(DialogComboboxSectionHeader, {
      ...restProps,
      children: name
    }), children]
  });
};

/**
 * Please use `SimpleSelect` unless you have a specific use-case for this primitive.
 * Ask in #dubois if you have questions!
 */
const SelectTrigger = /*#__PURE__*/forwardRef((props, ref) => {
  const {
    children,
    ...restProps
  } = props;
  return jsx(DialogComboboxTrigger, {
    allowClear: false,
    ...restProps,
    ref: ref,
    children: children
  });
});

const SimpleSelectContext = /*#__PURE__*/createContext(undefined);

// This is modelled to match the behavior of native select elements;
// this allows SimpleSelect to be compatible natively with React Hook Form.

const getSelectedOption = (children, value) => {
  const childArray = React__default.Children.toArray(children);
  for (const child of childArray) {
    if ( /*#__PURE__*/React__default.isValidElement(child)) {
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
const getSelectedOptionLabel = (children, value) => {
  const selectedOption = getSelectedOption(children, value);
  if ( /*#__PURE__*/React__default.isValidElement(selectedOption)) {
    return selectedOption.props.children;
  }
  return '';
};

/**
 * This is the future `Select` component which simplifies the API of the current Select primitives.
 * It is temporarily named `SimpleSelect` pending cleanup.
 */
const SimpleSelect = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    defaultValue,
    name,
    placeholder,
    children,
    contentProps,
    onChange,
    onOpenChange,
    id,
    label,
    value,
    validationState,
    forceCloseOnEscape,
    componentId,
    analyticsEvents,
    valueHasNoPii,
    ...rest
  } = _ref;
  const [defaultLabel] = useState(() => {
    if (value) {
      return getSelectedOptionLabel(children, value);
    }
    return '';
  });
  const innerRef = useRef(null);
  useImperativeHandle(ref, () => innerRef.current, []);
  const previousExternalValue = useRef(value);
  const [internalValue, setInternalValue] = useState(value);
  const [selectedLabel, setSelectedLabel] = useState(defaultLabel);
  const isControlled = value !== undefined;
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.SimpleSelect,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii
  });

  // Controlled state setup.
  useEffect(() => {
    if (value !== undefined && value !== previousExternalValue.current) {
      setInternalValue(value);
      previousExternalValue.current = value;
    }
  }, [value]);

  // Uncontrolled state setup.
  useEffect(() => {
    if (isControlled) {
      return;
    }

    // Set initial state.
    const element = innerRef.current;
    const initialValue = defaultValue || (element === null || element === void 0 ? void 0 : element.value) || '';
    setInternalValue(initialValue);
    previousExternalValue.current = initialValue;
  }, [isControlled, defaultValue, value]);

  // Separately update the label when the value changes; this responds
  // to either the controlled or uncontrolled useEffects above.
  useEffect(() => {
    setSelectedLabel(getSelectedOptionLabel(children, internalValue || ''));
  }, [internalValue, children]);

  // Handles controlled state, and propagates changes to the input element.
  const handleChange = newValue => {
    var _innerRef$current;
    eventContext.onValueChange(newValue);
    (_innerRef$current = innerRef.current) === null || _innerRef$current === void 0 || _innerRef$current.setAttribute('value', newValue || '');
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
  return jsx(SimpleSelectContext.Provider, {
    value: {
      value: internalValue,
      onChange: handleChange
    },
    children: jsx(Select
    // SimpleSelect emits its own value change events rather than emitting them from the underlying
    // DialogCombobox due to how SimpleSelect sets its initial state. The Select componentId is explicitly
    // set to undefined to prevent it from emitting events if the componentId prop is required in the future.
    , {
      componentId: undefined,
      value: internalValue,
      placeholder: placeholder,
      label: label !== null && label !== void 0 ? label : rest['aria-label'],
      id: id,
      children: jsxs(SimpleSelectContentWrapper, {
        onOpenChange: onOpenChange,
        children: [jsx(SelectTrigger, {
          ...rest,
          validationState: validationState,
          onClear: () => {
            handleChange('');
          },
          id: id,
          value: internalValue,
          children: selectedLabel || placeholder
        }), jsx("input", {
          type: "hidden",
          ref: innerRef
        }), jsx(SelectContent, {
          forceCloseOnEscape: forceCloseOnEscape,
          ...contentProps,
          children: children
        })]
      })
    })
  });
});
// This component is used to propagate the open state of the DialogCombobox to the SimpleSelect.
// We don't directly pass through `onOpenChange` since it's tied into the actual state; `SimpleSelect` merely
// needs to communicate via the optional prop if the dropdown is open or not and doesn't need to control it.
const SimpleSelectContentWrapper = _ref2 => {
  let {
    children,
    onOpenChange
  } = _ref2;
  const {
    isOpen
  } = useDialogComboboxContext();
  useEffect(() => {
    if (onOpenChange) {
      onOpenChange(Boolean(isOpen));
    }
  }, [isOpen, onOpenChange]);
  return jsx(Fragment, {
    children: children
  });
};
const SimpleSelectOption = /*#__PURE__*/forwardRef((_ref3, ref) => {
  let {
    value,
    children,
    ...rest
  } = _ref3;
  const context = useContext(SimpleSelectContext);
  if (!context) {
    throw new Error('SimpleSelectOption must be used within a SimpleSelect');
  }
  const {
    onChange
  } = context;
  return jsx(SelectOption, {
    ...rest,
    ref: ref,
    value: value,
    onChange: _ref4 => {
      let {
        value
      } = _ref4;
      onChange(value);
    },
    children: children
  });
});
const SimpleSelectOptionGroup = _ref5 => {
  let {
    children,
    label,
    ...props
  } = _ref5;
  const context = useContext(SimpleSelectContext);
  if (!context) {
    throw new Error('SimpleSelectOptionGroup must be used within a SimpleSelect');
  }
  return jsx(SelectOptionGroup, {
    ...props,
    name: label,
    children: children
  });
};

const getSwitchWithLabelStyles = _ref => {
  let {
    clsPrefix,
    theme,
    disabled
  } = _ref;
  // Default value
  const SWITCH_WIDTH = 28;
  const styles = {
    display: 'flex',
    alignItems: 'center',
    ...(disabled && {
      '&&, label': {
        color: theme.colors.actionDisabledText
      }
    }),
    // Switch is Off
    [`&.${clsPrefix}-switch`]: {
      backgroundColor: theme.colors.backgroundPrimary,
      border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
      [`.${clsPrefix}-switch-handle:before`]: {
        boxShadow: `0px 0px 0px 1px ${theme.colors.actionDefaultBorderDefault}`,
        transition: 'none'
      },
      [`&:hover:not(.${clsPrefix}-switch-disabled)`]: {
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`,
        [`.${clsPrefix}-switch-handle:before`]: {
          boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundHover}`
        }
      },
      [`&:active:not(.${clsPrefix}-switch-disabled)`]: {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        border: `1px solid ${theme.colors.actionPrimaryBackgroundPress}`,
        [`.${clsPrefix}-switch-handle:before`]: {
          boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundHover}`
        }
      },
      [`&.${clsPrefix}-switch-disabled`]: {
        backgroundColor: theme.colors.actionDisabledBackground,
        border: `1px solid ${theme.colors.actionDisabledBorder}`,
        [`.${clsPrefix}-switch-handle:before`]: {
          boxShadow: `0px 0px 0px 1px ${theme.colors.actionDisabledBorder}`
        }
      },
      [`&:focus-visible`]: {
        border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
        boxShadow: 'none',
        outlineStyle: 'solid',
        outlineWidth: '1px',
        outlineColor: theme.colors.actionDefaultBorderFocus,
        [`.${clsPrefix}-switch-handle:before`]: {
          boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundDefault}`
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
        border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`
      },
      [`&:active:not(.${clsPrefix}-switch-disabled)`]: {
        backgroundColor: theme.colors.actionPrimaryBackgroundPress
      },
      [`.${clsPrefix}-switch-handle:before`]: {
        boxShadow: `0px 0px 0px 1px ${theme.colors.actionPrimaryBackgroundDefault}`
      },
      [`&.${clsPrefix}-switch-disabled`]: {
        backgroundColor: theme.colors.actionDisabledText,
        border: `1px solid ${theme.colors.actionDisabledText}`,
        [`.${clsPrefix}-switch-handle:before`]: {
          boxShadow: `0px 0px 0px 1px ${theme.colors.actionDisabledText}`
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
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSwitchWithLabelStyles;");
};
const Switch = _ref2 => {
  var _props$id;
  let {
    dangerouslySetAntdProps,
    label,
    labelProps,
    activeLabel,
    inactiveLabel,
    disabledLabel,
    componentId,
    analyticsEvents,
    onChange,
    ...props
  } = _ref2;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const duboisId = useUniqueId('dubois-switch');
  const uniqueId = (_props$id = props.id) !== null && _props$id !== void 0 ? _props$id : duboisId;
  const [isChecked, setIsChecked] = useState(props.checked || props.defaultChecked);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Switch,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii: true
  });
  const handleToggle = (newState, event) => {
    eventContext.onValueChange(newState);
    if (onChange) {
      onChange(newState, event);
    } else {
      setIsChecked(newState);
    }
  };
  const onChangeHandler = (newState, event) => {
    eventContext.onValueChange(newState);
    onChange === null || onChange === void 0 || onChange(newState, event);
  };
  useEffect(() => {
    setIsChecked(props.checked);
  }, [props.checked]);
  const hasNewLabels = activeLabel && inactiveLabel && disabledLabel;
  const stateMessage = isChecked ? activeLabel : inactiveLabel;

  // AntDSwitch's interface does not include `id` even though it passes it through and works as expected
  // We are using this to bypass that check
  const idPropObj = {
    id: uniqueId
  };
  const switchComponent = jsx(Switch$1, {
    ...addDebugOutlineIfEnabled(),
    ...props,
    ...dangerouslySetAntdProps,
    onChange: handleToggle,
    ...idPropObj,
    css: /*#__PURE__*/css({
      ... /*#__PURE__*/css(getAnimationCss(theme.options.enableAnimation), process.env.NODE_ENV === "production" ? "" : ";label:switchComponent;"),
      ...getSwitchWithLabelStyles({
        clsPrefix: classNamePrefix,
        theme
      })
    }, process.env.NODE_ENV === "production" ? "" : ";label:switchComponent;")
  });
  const labelComponent = jsx(Label, {
    inline: true,
    ...labelProps,
    htmlFor: uniqueId,
    style: {
      ...(hasNewLabels && {
        marginRight: theme.spacing.sm
      })
    },
    children: label
  });
  return label ? jsx(DesignSystemAntDConfigProvider, {
    children: jsx("div", {
      ...addDebugOutlineIfEnabled(),
      css: getSwitchWithLabelStyles({
        clsPrefix: classNamePrefix,
        theme,
        disabled: props.disabled
      }),
      children: hasNewLabels ? jsxs(Fragment, {
        children: [labelComponent, jsx("span", {
          style: {
            marginLeft: 'auto',
            marginRight: theme.spacing.sm
          },
          children: `${stateMessage}${props.disabled ? ` (${disabledLabel})` : ''}`
        }), switchComponent]
      }) : jsxs(Fragment, {
        children: [switchComponent, labelComponent]
      })
    })
  }) : jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Switch$1, {
      onChange: onChangeHandler,
      ...addDebugOutlineIfEnabled(),
      ...props,
      ...dangerouslySetAntdProps,
      ...idPropObj,
      css: /*#__PURE__*/css({
        ... /*#__PURE__*/css(getAnimationCss(theme.options.enableAnimation), process.env.NODE_ENV === "production" ? "" : ";label:Switch;"),
        ...getSwitchWithLabelStyles({
          clsPrefix: classNamePrefix,
          theme,
          disabled: props.disabled
        })
      }, process.env.NODE_ENV === "production" ? "" : ";label:Switch;")
    })
  });
};

const typeaheadComboboxContextDefaults = {
  componentId: 'codegen_design-system_src_design-system_typeaheadcombobox_providers_typeaheadcomboboxcontext.tsx_17',
  isInsideTypeaheadCombobox: false,
  multiSelect: false
};
const TypeaheadComboboxContext = /*#__PURE__*/createContext(typeaheadComboboxContextDefaults);
const TypeaheadComboboxContextProvider = _ref => {
  let {
    children,
    value
  } = _ref;
  const [inputWidth, setInputWidth] = useState();
  return jsx(TypeaheadComboboxContext.Provider, {
    value: {
      ...value,
      setInputWidth,
      inputWidth
    },
    children: children
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$c() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$8 = process.env.NODE_ENV === "production" ? {
  name: "18nns55",
  styles: "display:inline-block;width:100%"
} : {
  name: "19m0398-TypeaheadComboboxRoot",
  styles: "display:inline-block;width:100%;label:TypeaheadComboboxRoot;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
};
const TypeaheadComboboxRoot = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    comboboxState,
    multiSelect = false,
    children,
    ...props
  } = _ref;
  const {
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    refs,
    floatingStyles
  } = useFloating({
    whileElementsMounted: autoUpdate,
    middleware: [offset(4), flip(), shift()],
    placement: 'bottom-start'
  });
  return jsx(TypeaheadComboboxContextProvider, {
    value: {
      componentId: props.componentId,
      multiSelect,
      isInsideTypeaheadCombobox: true,
      floatingUiRefs: refs,
      floatingStyles: floatingStyles
    },
    children: jsx("div", {
      ...comboboxState.getComboboxProps({}, {
        suppressRefError: true
      }),
      className: `${classNamePrefix}-typeahead-combobox`,
      css: _ref2$8,
      ...props,
      ref: ref,
      children: children
    })
  });
});

const TypeaheadComboboxStateChangeTypes = useCombobox.stateChangeTypes;
const TypeaheadComboboxMultiSelectStateChangeTypes = useMultipleSelection.stateChangeTypes;
function useComboboxState(_ref) {
  let {
    allItems,
    items,
    itemToString,
    onIsOpenChange,
    allowNewValue = false,
    formValue,
    formOnChange,
    formOnBlur,
    ...props
  } = _ref;
  function getFilteredItems(inputValue) {
    var _inputValue$toLowerCa;
    const lowerCasedInputValue = (_inputValue$toLowerCa = inputValue === null || inputValue === void 0 ? void 0 : inputValue.toLowerCase()) !== null && _inputValue$toLowerCa !== void 0 ? _inputValue$toLowerCa : '';
    // If the input is empty or if there is no matcher supplied, do not filter
    return allItems.filter(item => !inputValue || !props.matcher || props.matcher(item, lowerCasedInputValue));
  }
  const comboboxState = useCombobox({
    onIsOpenChange: onIsOpenChange,
    onInputValueChange: _ref2 => {
      let {
        inputValue
      } = _ref2;
      if (inputValue !== undefined) {
        var _props$setInputValue;
        (_props$setInputValue = props.setInputValue) === null || _props$setInputValue === void 0 || _props$setInputValue.call(props, inputValue);
      }
      if (!props.multiSelect) {
        props.setItems(getFilteredItems(inputValue));
      }
    },
    items: items,
    itemToString(item) {
      return item ? itemToString ? itemToString(item) : item.toString() : '';
    },
    defaultHighlightedIndex: props.multiSelect ? 0 : undefined,
    // after selection for multiselect, highlight the first item.
    scrollIntoView: () => {},
    // disabling scroll because floating-ui repositions the menu
    selectedItem: props.multiSelect ? null : formValue,
    // useMultipleSelection will handle the item selection for multiselect
    stateReducer(state, actionAndChanges) {
      const {
        changes,
        type
      } = actionAndChanges;
      switch (type) {
        case useCombobox.stateChangeTypes.InputBlur:
          if (!props.multiSelect) {
            // If allowNewValue is true, register the input's current value on blur
            if (allowNewValue) {
              const newInputValue = state.inputValue === '' ? null : state.inputValue;
              formOnChange === null || formOnChange === void 0 || formOnChange(newInputValue);
              formOnBlur === null || formOnBlur === void 0 || formOnBlur(newInputValue);
            } else {
              // If allowNewValue is false, clear value on blur
              formOnChange === null || formOnChange === void 0 || formOnChange(null);
              formOnBlur === null || formOnBlur === void 0 || formOnBlur(null);
            }
          } else {
            formOnBlur === null || formOnBlur === void 0 || formOnBlur(state.selectedItem);
          }
          return changes;
        case useCombobox.stateChangeTypes.InputKeyDownEnter:
        case useCombobox.stateChangeTypes.ItemClick:
          formOnChange === null || formOnChange === void 0 || formOnChange(changes.selectedItem);
          return {
            ...changes,
            highlightedIndex: props.multiSelect ? state.highlightedIndex : 0,
            // on multiselect keep the highlighted index unchanged.
            isOpen: props.multiSelect ? true : false // for multiselect, keep the menu open after selection.
          };
        default:
          return changes;
      }
    },
    onStateChange: args => {
      var _props$onStateChange;
      const {
        type,
        selectedItem: newSelectedItem,
        inputValue: newInputValue
      } = args;
      (_props$onStateChange = props.onStateChange) === null || _props$onStateChange === void 0 || _props$onStateChange.call(props, args);
      if (props.multiSelect) {
        switch (type) {
          case useCombobox.stateChangeTypes.InputKeyDownEnter:
          case useCombobox.stateChangeTypes.ItemClick:
          case useCombobox.stateChangeTypes.InputBlur:
            if (newSelectedItem) {
              props.setSelectedItems([...props.selectedItems, newSelectedItem]);
              props.setInputValue('');
              formOnBlur === null || formOnBlur === void 0 || formOnBlur([...props.selectedItems, newSelectedItem]);
            }
            break;
          case useCombobox.stateChangeTypes.InputChange:
            props.setInputValue(newInputValue !== null && newInputValue !== void 0 ? newInputValue : '');
            break;
        }
        // Unselect when clicking selected item
        if (newSelectedItem && props.selectedItems.includes(newSelectedItem)) {
          props.setSelectedItems(props.selectedItems.filter(item => item !== newSelectedItem));
        }
      }
    },
    initialInputValue: props.initialInputValue,
    initialSelectedItem: props.initialSelectedItem
  });
  return comboboxState;
}
function useMultipleSelectionState(selectedItems, setSelectedItems) {
  return useMultipleSelection({
    selectedItems,
    onStateChange(_ref3) {
      let {
        selectedItems: newSelectedItems,
        type
      } = _ref3;
      switch (type) {
        case useMultipleSelection.stateChangeTypes.SelectedItemKeyDownBackspace:
        case useMultipleSelection.stateChangeTypes.SelectedItemKeyDownDelete:
        case useMultipleSelection.stateChangeTypes.DropdownKeyDownBackspace:
        case useMultipleSelection.stateChangeTypes.FunctionRemoveSelectedItem:
          setSelectedItems(newSelectedItems || []);
          break;
      }
    }
  });
}

const useTypeaheadComboboxContext = () => {
  return useContext(TypeaheadComboboxContext);
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$b() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref$5 = process.env.NODE_ENV === "production" ? {
  name: "xv0ss6",
  styles: "padding:0;margin:0;display:flex;flex-direction:column;align-items:flex-start;position:absolute"
} : {
  name: "1sd733r-getTypeaheadComboboxMenuStyles",
  styles: "padding:0;margin:0;display:flex;flex-direction:column;align-items:flex-start;position:absolute;label:getTypeaheadComboboxMenuStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$b
};
const getTypeaheadComboboxMenuStyles = () => {
  return _ref$5;
};
const TypeaheadComboboxMenu = /*#__PURE__*/forwardRef((_ref2, ref) => {
  let {
    comboboxState,
    loading,
    emptyText,
    width,
    minWidth = 240,
    maxWidth,
    minHeight,
    maxHeight,
    listWrapperHeight,
    virtualizerRef,
    children,
    matchTriggerWidth,
    ...restProps
  } = _ref2;
  const {
    getMenuProps,
    isOpen
  } = comboboxState;
  const {
    ref: downshiftRef,
    ...downshiftProps
  } = getMenuProps({}, {
    suppressRefError: true
  });
  const {
    floatingUiRefs,
    floatingStyles,
    isInsideTypeaheadCombobox,
    inputWidth
  } = useTypeaheadComboboxContext();
  if (!isInsideTypeaheadCombobox) {
    throw new Error('`TypeaheadComboboxMenu` must be used within `TypeaheadCombobox`');
  }
  const mergedRef = useMergeRefs([ref, downshiftRef, floatingUiRefs === null || floatingUiRefs === void 0 ? void 0 : floatingUiRefs.setFloating]);
  const {
    theme
  } = useDesignSystemTheme();
  const {
    getPopupContainer
  } = useDesignSystemContext();
  if (!isOpen) return null;
  const hasFragmentWrapper = children && !Array.isArray(children) && children.type === Fragment$1;
  const filterableChildren = hasFragmentWrapper ? children.props.children : children;
  const hasResults = filterableChildren && Children.toArray(filterableChildren).some(child => {
    if ( /*#__PURE__*/React__default.isValidElement(child)) {
      var _child$props$__EMOTIO, _child$props$__EMOTIO2;
      const childType = (_child$props$__EMOTIO = (_child$props$__EMOTIO2 = child.props['__EMOTION_TYPE_PLEASE_DO_NOT_USE__']) === null || _child$props$__EMOTIO2 === void 0 ? void 0 : _child$props$__EMOTIO2.defaultProps._TYPE) !== null && _child$props$__EMOTIO !== void 0 ? _child$props$__EMOTIO : child.props._TYPE;
      return ['TypeaheadComboboxMenuItem', 'TypeaheadComboboxCheckboxItem'].includes(childType);
    }
    return false;
  });
  const footer = Children.toArray(children).filter(child => /*#__PURE__*/React__default.isValidElement(child) && child.props._TYPE === 'TypeaheadComboboxFooter');
  return /*#__PURE__*/createPortal(jsx("ul", {
    ...addDebugOutlineIfEnabled(),
    "aria-busy": loading,
    ...downshiftProps,
    ref: mergedRef,
    css: [getComboboxContentWrapperStyles(theme, {
      maxHeight,
      maxWidth,
      minHeight,
      minWidth,
      width
    }), getTypeaheadComboboxMenuStyles(), matchTriggerWidth && inputWidth && {
      width: inputWidth
    }, process.env.NODE_ENV === "production" ? "" : ";label:TypeaheadComboboxMenu;"],
    style: {
      ...floatingStyles
    },
    ...restProps,
    children: loading ? jsx(LoadingSpinner, {
      "aria-label": "Loading",
      alt: "Loading spinner"
    }) : hasResults ? jsx("div", {
      ref: virtualizerRef,
      style: {
        position: 'relative',
        width: '100%',
        ...(listWrapperHeight && {
          height: listWrapperHeight
        })
      },
      children: children
    }) : jsxs(Fragment, {
      children: [jsx(EmptyResults, {
        emptyText: emptyText
      }), footer]
    })
  }), getPopupContainer ? getPopupContainer() : document.body);
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$a() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const getMenuItemStyles = (theme, isHighlighted, disabled) => {
  return /*#__PURE__*/css({
    ...(disabled && {
      pointerEvents: 'none',
      color: theme.colors.actionDisabledText
    }),
    ...(isHighlighted && {
      background: theme.colors.actionTertiaryBackgroundHover
    })
  }, process.env.NODE_ENV === "production" ? "" : ";label:getMenuItemStyles;");
};
const getLabelStyles = (theme, textOverflowMode) => {
  return /*#__PURE__*/css({
    marginLeft: theme.spacing.sm,
    fontSize: theme.typography.fontSizeBase,
    fontStyle: 'normal',
    fontWeight: 400,
    cursor: 'pointer',
    overflow: 'hidden',
    wordBreak: 'break-word',
    ...(textOverflowMode === 'ellipsis' && {
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap'
    })
  }, process.env.NODE_ENV === "production" ? "" : ";label:getLabelStyles;");
};
var _ref2$7 = process.env.NODE_ENV === "production" ? {
  name: "kjj0ot",
  styles: "padding-top:2px"
} : {
  name: "1uez6s6-TypeaheadComboboxMenuItem",
  styles: "padding-top:2px;label:TypeaheadComboboxMenuItem;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$a
};
var _ref3$4 = process.env.NODE_ENV === "production" ? {
  name: "zjik7",
  styles: "display:flex"
} : {
  name: "1p9uv37-TypeaheadComboboxMenuItem",
  styles: "display:flex;label:TypeaheadComboboxMenuItem;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$a
};
const TypeaheadComboboxMenuItem = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    item,
    index,
    comboboxState,
    textOverflowMode = 'multiline',
    isDisabled,
    disabledReason,
    hintContent,
    onClick: onClickProp,
    children,
    ...restProps
  } = _ref;
  const {
    selectedItem,
    highlightedIndex,
    getItemProps,
    isOpen
  } = comboboxState;
  const isSelected = _isEqual(selectedItem, item);
  const isHighlighted = highlightedIndex === index;
  const {
    theme
  } = useDesignSystemTheme();
  const listItemRef = useRef(null);
  useImperativeHandle(ref, () => listItemRef.current);
  const {
    onClick,
    ...downshiftItemProps
  } = getItemProps({
    item,
    index,
    disabled: isDisabled,
    onMouseUp: e => {
      var _restProps$onMouseUp;
      e.stopPropagation();
      (_restProps$onMouseUp = restProps.onMouseUp) === null || _restProps$onMouseUp === void 0 || _restProps$onMouseUp.call(restProps, e);
    },
    ref: listItemRef
  });
  const handleClick = e => {
    onClickProp === null || onClickProp === void 0 || onClickProp(e);
    onClick === null || onClick === void 0 || onClick(e);
  };

  // Scroll to the highlighted item if it is not in the viewport
  useEffect(() => {
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
        var _listItemRef$current;
        (_listItemRef$current = listItemRef.current) === null || _listItemRef$current === void 0 || _listItemRef$current.scrollIntoView({
          block: 'nearest'
        });
      }
    }
  }, [highlightedIndex, index, isOpen, listItemRef]);
  return jsxs("li", {
    role: "option",
    "aria-selected": isSelected,
    disabled: isDisabled,
    onClick: handleClick,
    css: [getComboboxOptionItemWrapperStyles(theme), getMenuItemStyles(theme, isHighlighted, isDisabled), process.env.NODE_ENV === "production" ? "" : ";label:TypeaheadComboboxMenuItem;"],
    ...downshiftItemProps,
    ...restProps,
    children: [isSelected ? jsx(CheckIcon, {
      css: _ref2$7
    }) : jsx("div", {
      style: {
        width: 16,
        flexShrink: 0
      }
    }), jsxs("label", {
      css: getLabelStyles(theme, textOverflowMode),
      children: [isDisabled && disabledReason ? jsxs("div", {
        css: _ref3$4,
        children: [jsx("div", {
          children: children
        }), jsx("div", {
          css: getInfoIconStyles(theme),
          children: jsx(LegacyInfoTooltip, {
            title: disabledReason
          })
        })]
      }) : children, jsx(HintRow, {
        disabled: isDisabled,
        children: hintContent
      })]
    })]
  });
});
TypeaheadComboboxMenuItem.defaultProps = {
  _TYPE: 'TypeaheadComboboxMenuItem'
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$9() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$6 = process.env.NODE_ENV === "production" ? {
  name: "zjik7",
  styles: "display:flex"
} : {
  name: "1pneh3l-TypeaheadComboboxCheckboxItem",
  styles: "display:flex;label:TypeaheadComboboxCheckboxItem;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$9
};
const TypeaheadComboboxCheckboxItem = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    item,
    index,
    comboboxState,
    selectedItems,
    textOverflowMode = 'multiline',
    isDisabled,
    disabledReason,
    hintContent,
    onClick: onClickProp,
    children,
    ...restProps
  } = _ref;
  const {
    highlightedIndex,
    getItemProps,
    isOpen
  } = comboboxState;
  const isHighlighted = highlightedIndex === index;
  const {
    theme
  } = useDesignSystemTheme();
  const isSelected = selectedItems.includes(item);
  const listItemRef = useRef(null);
  useImperativeHandle(ref, () => listItemRef.current);
  const {
    onClick,
    ...downshiftItemProps
  } = getItemProps({
    item,
    index,
    disabled: isDisabled,
    onMouseUp: e => {
      var _restProps$onMouseUp;
      e.stopPropagation();
      (_restProps$onMouseUp = restProps.onMouseUp) === null || _restProps$onMouseUp === void 0 || _restProps$onMouseUp.call(restProps, e);
    },
    ref: listItemRef
  });
  const handleClick = e => {
    onClickProp === null || onClickProp === void 0 || onClickProp(e);
    onClick(e);
  };

  // Scroll to the highlighted item if it is not in the viewport
  useEffect(() => {
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
        var _listItemRef$current;
        (_listItemRef$current = listItemRef.current) === null || _listItemRef$current === void 0 || _listItemRef$current.scrollIntoView({
          block: 'nearest'
        });
      }
    }
  }, [highlightedIndex, index, isOpen, listItemRef]);
  return jsx("li", {
    role: "option",
    "aria-selected": isSelected,
    disabled: isDisabled,
    onClick: handleClick,
    css: [getComboboxOptionItemWrapperStyles(theme), getMenuItemStyles(theme, isHighlighted, isDisabled), process.env.NODE_ENV === "production" ? "" : ";label:TypeaheadComboboxCheckboxItem;"],
    ...downshiftItemProps,
    ...restProps,
    children: jsx(Checkbox, {
      componentId: "codegen_design-system_src_design-system_typeaheadcombobox_typeaheadcomboboxcheckboxitem.tsx_92",
      disabled: isDisabled,
      isChecked: isSelected,
      css: getCheckboxStyles(theme, textOverflowMode),
      tabIndex: -1
      // Needed because Antd handles keyboard inputs as clicks
      ,
      onClick: e => {
        e.stopPropagation();
      },
      children: jsxs("label", {
        children: [isDisabled && disabledReason ? jsxs("div", {
          css: _ref2$6,
          children: [jsx("div", {
            children: children
          }), jsx("div", {
            css: getInfoIconStyles(theme),
            children: jsx(LegacyInfoTooltip, {
              title: disabledReason
            })
          })]
        }) : children, jsx(HintRow, {
          disabled: isDisabled,
          children: hintContent
        })]
      })
    })
  });
});
TypeaheadComboboxCheckboxItem.defaultProps = {
  _TYPE: 'TypeaheadComboboxCheckboxItem'
};

const getToggleButtonStyles = (theme, disabled) => {
  return /*#__PURE__*/css({
    cursor: 'pointer',
    userSelect: 'none',
    color: theme.colors.textSecondary,
    backgroundColor: 'transparent',
    border: 'none',
    padding: 0,
    marginLeft: theme.spacing.xs,
    height: 16,
    width: 16,
    ...(disabled && {
      pointerEvents: 'none',
      color: theme.colors.actionDisabledText
    })
  }, process.env.NODE_ENV === "production" ? "" : ";label:getToggleButtonStyles;");
};
const TypeaheadComboboxToggleButton = /*#__PURE__*/React__default.forwardRef((_ref, ref) => {
  let {
    disabled,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    onClick
  } = restProps;
  function handleClick(e) {
    e.stopPropagation();
    onClick(e);
  }
  return jsx("button", {
    type: "button",
    "aria-label": "toggle menu",
    ref: ref,
    css: getToggleButtonStyles(theme, disabled),
    ...restProps,
    onClick: handleClick,
    children: jsx(ChevronDownIcon$1, {})
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$8() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$5 = process.env.NODE_ENV === "production" ? {
  name: "l1fpjx",
  styles: "pointer-events:all;vertical-align:text-top"
} : {
  name: "4edanz-TypeaheadComboboxControls",
  styles: "pointer-events:all;vertical-align:text-top;label:TypeaheadComboboxControls;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$8
};
const TypeaheadComboboxControls = _ref => {
  let {
    getDownshiftToggleButtonProps,
    showClearSelectionButton,
    showComboboxToggleButton = true,
    handleClear,
    disabled
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs("div", {
    css: /*#__PURE__*/css({
      position: 'absolute',
      top: theme.spacing.sm,
      right: 7,
      height: 16,
      zIndex: 1
    }, process.env.NODE_ENV === "production" ? "" : ";label:TypeaheadComboboxControls;"),
    children: [showClearSelectionButton && jsx(ClearSelectionButton, {
      onClick: handleClear,
      css: _ref2$5
    }), showComboboxToggleButton && jsx(TypeaheadComboboxToggleButton, {
      ...getDownshiftToggleButtonProps(),
      disabled: disabled
    })]
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$7() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref$4 = process.env.NODE_ENV === "production" ? {
  name: "5ob2ly",
  styles: "display:flex;position:relative"
} : {
  name: "9x5b62-getContainerStyles",
  styles: "display:flex;position:relative;label:getContainerStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$7
};
const getContainerStyles$1 = () => {
  return _ref$4;
};
const getInputStyles$1 = (theme, showComboboxToggleButton) => /*#__PURE__*/css({
  paddingRight: showComboboxToggleButton ? 52 : 26,
  width: '100%',
  minWidth: 72,
  '&:disabled': {
    borderColor: theme.colors.actionDisabledBorder,
    backgroundColor: theme.colors.actionDisabledBackground,
    color: theme.colors.actionDisabledText
  },
  '&:not(:disabled)': {
    backgroundColor: 'transparent'
  }
}, process.env.NODE_ENV === "production" ? "" : ";label:getInputStyles;");
const TypeaheadComboboxInput = /*#__PURE__*/forwardRef((_ref2, ref) => {
  let {
    comboboxState,
    allowClear = true,
    showComboboxToggleButton = true,
    formOnChange,
    onClick,
    clearInputValueOnFocus = false,
    ...restProps
  } = _ref2;
  const {
    isInsideTypeaheadCombobox,
    floatingUiRefs,
    setInputWidth,
    inputWidth,
    componentId
  } = useTypeaheadComboboxContext();
  if (!isInsideTypeaheadCombobox) {
    throw new Error('`TypeaheadComboboxInput` must be used within `TypeaheadCombobox`');
  }
  const {
    getInputProps,
    getToggleButtonProps,
    toggleMenu,
    inputValue,
    setInputValue,
    reset,
    isOpen,
    selectedItem
  } = comboboxState;
  const {
    ref: downshiftRef,
    ...downshiftProps
  } = getInputProps({}, {
    suppressRefError: true
  });
  const mergedRef = useMergeRefs([ref, downshiftRef]);
  const {
    theme
  } = useDesignSystemTheme();
  const handleClick = e => {
    onClick === null || onClick === void 0 || onClick(e);
    toggleMenu();
  };
  const previousInputValue = useRef(null);
  useEffect(() => {
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
  }, [isOpen, inputValue, setInputValue, previousInputValue, clearInputValueOnFocus, selectedItem]);
  const handleClear = () => {
    setInputValue('');
    reset();
    formOnChange === null || formOnChange === void 0 || formOnChange(null);
  };

  // Gets the width of the input and sets it inside the context for rendering the dropdown when `matchTriggerWidth` is true on the menu
  useEffect(() => {
    // Use the DOM reference of the TypeaheadComboboxInput container div to get the width of the input
    if (floatingUiRefs !== null && floatingUiRefs !== void 0 && floatingUiRefs.domReference) {
      var _floatingUiRefs$domRe, _floatingUiRefs$domRe2;
      const width = (_floatingUiRefs$domRe = (_floatingUiRefs$domRe2 = floatingUiRefs.domReference.current) === null || _floatingUiRefs$domRe2 === void 0 ? void 0 : _floatingUiRefs$domRe2.getBoundingClientRect().width) !== null && _floatingUiRefs$domRe !== void 0 ? _floatingUiRefs$domRe : 0;
      // Only update context width when the input width updated
      if (width !== inputWidth) {
        setInputWidth === null || setInputWidth === void 0 || setInputWidth(width);
      }
    }
  }, [floatingUiRefs === null || floatingUiRefs === void 0 ? void 0 : floatingUiRefs.domReference, setInputWidth, inputWidth]);
  return jsxs("div", {
    ref: floatingUiRefs === null || floatingUiRefs === void 0 ? void 0 : floatingUiRefs.setReference,
    css: getContainerStyles$1(),
    className: restProps.className,
    ...addDebugOutlineIfEnabled(),
    children: [jsx(Input, {
      componentId: componentId ? `${componentId}.input` : 'codegen_design-system_src_design-system_typeaheadcombobox_typeaheadcomboboxinput.tsx_126',
      ref: mergedRef,
      ...downshiftProps,
      "aria-controls": comboboxState.isOpen ? downshiftProps['aria-controls'] : undefined,
      onClick: handleClick,
      css: getInputStyles$1(theme, showComboboxToggleButton),
      ...restProps
    }), jsx(TypeaheadComboboxControls, {
      getDownshiftToggleButtonProps: getToggleButtonProps,
      showClearSelectionButton: allowClear && Boolean(inputValue) && !restProps.disabled,
      showComboboxToggleButton: showComboboxToggleButton,
      handleClear: handleClear,
      disabled: restProps.disabled
    })]
  });
});

const getSelectedItemStyles = (theme, disabled) => {
  return /*#__PURE__*/css({
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
    ...(disabled && {
      pointerEvents: 'none'
    })
  }, process.env.NODE_ENV === "production" ? "" : ";label:getSelectedItemStyles;");
};
const getIconContainerStyles = (theme, disabled) => {
  return /*#__PURE__*/css({
    width: 16,
    height: 16,
    ':hover': {
      color: theme.colors.actionTertiaryTextHover,
      backgroundColor: theme.colors.tagHover
    },
    ...(disabled && {
      pointerEvents: 'none',
      color: theme.colors.actionDisabledText
    })
  }, process.env.NODE_ENV === "production" ? "" : ";label:getIconContainerStyles;");
};
const getXIconStyles = theme => {
  return /*#__PURE__*/css({
    fontSize: theme.typography.fontSizeSm,
    verticalAlign: '-1px',
    paddingLeft: theme.spacing.xs / 2,
    paddingRight: theme.spacing.xs / 2
  }, process.env.NODE_ENV === "production" ? "" : ";label:getXIconStyles;");
};
const TypeaheadComboboxSelectedItem = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    label,
    item,
    getSelectedItemProps,
    removeSelectedItem,
    disabled,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs("span", {
    ...getSelectedItemProps({
      selectedItem: item
    }),
    css: getSelectedItemStyles(theme, disabled),
    ref: ref,
    ...restProps,
    children: [jsx("span", {
      css: /*#__PURE__*/css({
        marginRight: 2,
        ...(disabled && {
          color: theme.colors.actionDisabledText
        })
      }, process.env.NODE_ENV === "production" ? "" : ";label:TypeaheadComboboxSelectedItem;"),
      children: label
    }), jsx("span", {
      css: getIconContainerStyles(theme, disabled),
      children: jsx(CloseIcon, {
        "aria-hidden": "false",
        onClick: e => {
          if (!disabled) {
            e.stopPropagation();
            removeSelectedItem(item);
          }
        },
        css: getXIconStyles(theme),
        role: "button",
        "aria-label": "Remove selected item"
      })
    })]
  });
});

const CountBadge = _ref => {
  let {
    countStartAt,
    totalCount,
    disabled
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: [getSelectedItemStyles(theme), {
      paddingInlineEnd: theme.spacing.xs,
      ...(disabled && {
        color: theme.colors.actionDisabledText
      })
    }, process.env.NODE_ENV === "production" ? "" : ";label:CountBadge;"],
    children: countStartAt ? `+${totalCount - countStartAt}` : totalCount
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$6() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const getContainerStyles = (theme, validationState, width, maxHeight, disabled) => {
  const validationColor = getValidationStateColor(theme, validationState);
  return /*#__PURE__*/css({
    cursor: 'text',
    display: 'inline-block',
    verticalAlign: 'top',
    border: `1px solid ${theme.colors.border}`,
    borderRadius: theme.general.borderRadiusBase,
    minHeight: 32,
    height: 'auto',
    minWidth: 0,
    ...(width ? {
      width
    } : {}),
    ...(maxHeight ? {
      maxHeight
    } : {}),
    padding: '5px 52px 5px 12px',
    position: 'relative',
    overflow: 'auto',
    textOverflow: 'ellipsis',
    '&:hover': {
      border: `1px solid ${theme.colors.actionPrimaryBackgroundHover}`
    },
    '&:focus-within': {
      outlineColor: theme.colors.actionPrimaryBackgroundDefault,
      outlineWidth: 2,
      outlineOffset: -2,
      outlineStyle: 'solid',
      boxShadow: 'none',
      borderColor: 'transparent'
    },
    '&&': {
      ...(validationState && {
        borderColor: validationColor
      }),
      '&:hover': {
        borderColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundHover
      },
      '&:focus': {
        outlineColor: validationState ? validationColor : theme.colors.actionPrimaryBackgroundDefault,
        outlineWidth: 2,
        outlineOffset: -2,
        outlineStyle: 'solid',
        boxShadow: 'none',
        borderColor: 'transparent'
      },
      ...(disabled && {
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
      })
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getContainerStyles;");
};
var _ref2$4 = process.env.NODE_ENV === "production" ? {
  name: "a9xlat",
  styles: "display:flex;flex:auto;flex-wrap:wrap;max-width:100%;position:relative"
} : {
  name: "azzs2i-getContentWrapperStyles",
  styles: "display:flex;flex:auto;flex-wrap:wrap;max-width:100%;position:relative;label:getContentWrapperStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$6
};
const getContentWrapperStyles = () => {
  return _ref2$4;
};
var _ref$3 = process.env.NODE_ENV === "production" ? {
  name: "qngrd1",
  styles: "display:inline-flex;position:relative;max-width:100%;align-self:auto;flex:none"
} : {
  name: "10x8s5t-getInputWrapperStyles",
  styles: "display:inline-flex;position:relative;max-width:100%;align-self:auto;flex:none;label:getInputWrapperStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$6
};
const getInputWrapperStyles = () => {
  return _ref$3;
};
const getInputStyles = theme => {
  return /*#__PURE__*/css({
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
  }, process.env.NODE_ENV === "production" ? "" : ";label:getInputStyles;");
};
var _ref4$1 = process.env.NODE_ENV === "production" ? {
  name: "1r88pt9",
  styles: "visibility:hidden;white-space:pre;position:absolute"
} : {
  name: "1noplic-content",
  styles: "visibility:hidden;white-space:pre;position:absolute;label:content;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$6
};
const TypeaheadComboboxMultiSelectInput = /*#__PURE__*/forwardRef((_ref3, ref) => {
  var _measureRef$current, _itemsRef$current, _containerRef$current, _innerRef$current2;
  let {
    comboboxState,
    multipleSelectionState,
    selectedItems,
    setSelectedItems,
    getSelectedItemLabel,
    allowClear = true,
    showTagAfterValueCount = 20,
    width,
    maxHeight,
    placeholder,
    validationState,
    showComboboxToggleButton,
    disableTooltip = false,
    ...restProps
  } = _ref3;
  const {
    isInsideTypeaheadCombobox
  } = useTypeaheadComboboxContext();
  if (!isInsideTypeaheadCombobox) {
    throw new Error('`TypeaheadComboboxMultiSelectInput` must be used within `TypeaheadCombobox`');
  }
  const {
    getInputProps,
    getToggleButtonProps,
    toggleMenu,
    inputValue,
    setInputValue
  } = comboboxState;
  const {
    getSelectedItemProps,
    getDropdownProps,
    removeSelectedItem
  } = multipleSelectionState;
  const {
    ref: downshiftRef,
    ...downshiftProps
  } = getInputProps(getDropdownProps({}, {
    suppressRefError: true
  }));
  const {
    floatingUiRefs,
    setInputWidth: setContextInputWidth,
    inputWidth: contextInputWidth
  } = useTypeaheadComboboxContext();
  const containerRef = useRef(null);
  const mergedContainerRef = useMergeRefs([containerRef, floatingUiRefs === null || floatingUiRefs === void 0 ? void 0 : floatingUiRefs.setReference]);
  const itemsRef = useRef(null);
  const measureRef = useRef(null);
  const innerRef = useRef(null);
  const mergedInputRef = useMergeRefs([ref, innerRef, downshiftRef]);
  const {
    theme
  } = useDesignSystemTheme();
  const [inputWidth, setInputWidth] = useState(0);
  const shouldShowCountBadge = selectedItems.length > showTagAfterValueCount;
  const [showTooltip, setShowTooltip] = useState(shouldShowCountBadge);
  const selectedItemsToRender = selectedItems.slice(0, showTagAfterValueCount);
  const handleClick = () => {
    if (!restProps.disabled) {
      var _innerRef$current;
      (_innerRef$current = innerRef.current) === null || _innerRef$current === void 0 || _innerRef$current.focus();
      toggleMenu();
    }
  };
  const handleClear = () => {
    setInputValue('');
    setSelectedItems([]);
  };

  // We measure width and set to the input immediately
  useLayoutEffect(() => {
    if (measureRef !== null && measureRef !== void 0 && measureRef.current) {
      const measuredWidth = measureRef.current.scrollWidth;
      setInputWidth(measuredWidth);
    }
  }, [measureRef === null || measureRef === void 0 || (_measureRef$current = measureRef.current) === null || _measureRef$current === void 0 ? void 0 : _measureRef$current.scrollWidth, selectedItems === null || selectedItems === void 0 ? void 0 : selectedItems.length]);

  // Gets the width of the input and sets it inside the context for rendering the dropdown when `matchTriggerWidth` is true on the menu
  useEffect(() => {
    // Use the DOM reference of the TypeaheadComboboxInput container div to get the width of the input
    if (floatingUiRefs !== null && floatingUiRefs !== void 0 && floatingUiRefs.domReference) {
      var _floatingUiRefs$domRe, _floatingUiRefs$domRe2;
      const width = (_floatingUiRefs$domRe = (_floatingUiRefs$domRe2 = floatingUiRefs.domReference.current) === null || _floatingUiRefs$domRe2 === void 0 ? void 0 : _floatingUiRefs$domRe2.getBoundingClientRect().width) !== null && _floatingUiRefs$domRe !== void 0 ? _floatingUiRefs$domRe : 0;
      // Only update context width when the input width updated
      if (width !== contextInputWidth) {
        setContextInputWidth === null || setContextInputWidth === void 0 || setContextInputWidth(width);
      }
    }
  }, [floatingUiRefs === null || floatingUiRefs === void 0 ? void 0 : floatingUiRefs.domReference, setContextInputWidth, contextInputWidth]);

  // Determine whether to show tooltip
  useEffect(() => {
    let isPartiallyHidden = false;
    if (itemsRef.current && containerRef.current) {
      const {
        clientHeight: innerHeight
      } = itemsRef.current;
      const {
        clientHeight: outerHeight
      } = containerRef.current;
      isPartiallyHidden = innerHeight > outerHeight;
    }
    setShowTooltip(!disableTooltip && (shouldShowCountBadge || isPartiallyHidden));
  }, [shouldShowCountBadge, (_itemsRef$current = itemsRef.current) === null || _itemsRef$current === void 0 ? void 0 : _itemsRef$current.clientHeight, (_containerRef$current = containerRef.current) === null || _containerRef$current === void 0 ? void 0 : _containerRef$current.clientHeight, disableTooltip]);
  const content = jsxs("div", {
    ...addDebugOutlineIfEnabled(),
    onClick: handleClick,
    ref: mergedContainerRef,
    css: getContainerStyles(theme, validationState, width, maxHeight, restProps.disabled),
    tabIndex: restProps.disabled ? -1 : 0,
    children: [jsxs("div", {
      ref: itemsRef,
      css: getContentWrapperStyles(),
      children: [selectedItemsToRender === null || selectedItemsToRender === void 0 ? void 0 : selectedItemsToRender.map((selectedItemForRender, index) => jsx(TypeaheadComboboxSelectedItem, {
        label: getSelectedItemLabel(selectedItemForRender),
        item: selectedItemForRender,
        getSelectedItemProps: getSelectedItemProps,
        removeSelectedItem: removeSelectedItem,
        disabled: restProps.disabled
      }, `selected-item-${index}`)), shouldShowCountBadge && jsx(CountBadge, {
        countStartAt: showTagAfterValueCount,
        totalCount: selectedItems.length,
        role: "status",
        "aria-label": "Selected options count",
        disabled: restProps.disabled
      }), jsxs("div", {
        css: getInputWrapperStyles(),
        children: [jsx("input", {
          ...downshiftProps,
          ref: mergedInputRef,
          css: [getInputStyles(theme), {
            width: inputWidth
          }, process.env.NODE_ENV === "production" ? "" : ";label:content;"],
          placeholder: selectedItems !== null && selectedItems !== void 0 && selectedItems.length ? undefined : placeholder,
          "aria-controls": comboboxState.isOpen ? downshiftProps['aria-controls'] : undefined,
          ...restProps
        }), jsxs("span", {
          ref: measureRef,
          "aria-hidden": true,
          css: _ref4$1,
          children: [(_innerRef$current2 = innerRef.current) !== null && _innerRef$current2 !== void 0 && _innerRef$current2.value ? innerRef.current.value : placeholder, "\xA0"]
        })]
      })]
    }), jsx(TypeaheadComboboxControls, {
      getDownshiftToggleButtonProps: getToggleButtonProps,
      showComboboxToggleButton: showComboboxToggleButton,
      showClearSelectionButton: allowClear && (Boolean(inputValue) || selectedItems && selectedItems.length > 0) && !restProps.disabled,
      handleClear: handleClear,
      disabled: restProps.disabled
    })]
  });
  if (showTooltip && selectedItems.length > 0) {
    return jsx(LegacyTooltip, {
      title: selectedItems.map(item => getSelectedItemLabel(item)).join(', '),
      children: content
    });
  }
  return content;
});

const TypeaheadComboboxSectionHeader = _ref => {
  let {
    children,
    ...props
  } = _ref;
  const {
    isInsideTypeaheadCombobox
  } = useTypeaheadComboboxContext();
  if (!isInsideTypeaheadCombobox) {
    throw new Error('`TypeaheadComboboxSectionHeader` must be used within `TypeaheadComboboxMenu`');
  }
  return jsx(SectionHeader, {
    ...props,
    children: children
  });
};

const TypeaheadComboboxSeparator = props => {
  const {
    isInsideTypeaheadCombobox
  } = useTypeaheadComboboxContext();
  if (!isInsideTypeaheadCombobox) {
    throw new Error('`TypeaheadComboboxSeparator` must be used within `TypeaheadComboboxMenu`');
  }
  return jsx(Separator, {
    ...props
  });
};

const DuboisTypeaheadComboboxFooter = _ref => {
  let {
    children,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    isInsideTypeaheadCombobox
  } = useTypeaheadComboboxContext();
  if (!isInsideTypeaheadCombobox) {
    throw new Error('`TypeaheadComboboxFooter` must be used within `TypeaheadComboboxMenu`');
  }
  return jsx("div", {
    ...restProps,
    css: getFooterStyles(theme),
    children: children
  });
};
DuboisTypeaheadComboboxFooter.defaultProps = {
  _TYPE: 'TypeaheadComboboxFooter'
};
const TypeaheadComboboxFooter = DuboisTypeaheadComboboxFooter;

const TypeaheadComboboxAddButton = _ref => {
  let {
    children,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    isInsideTypeaheadCombobox
  } = useTypeaheadComboboxContext();
  if (!isInsideTypeaheadCombobox) {
    throw new Error('`TypeaheadComboboxAddButton` must be used within `TypeaheadCombobox`');
  }
  return jsx(Button, {
    ...restProps,
    type: "tertiary",
    onClick: event => {
      var _restProps$onClick;
      event.stopPropagation();
      (_restProps$onClick = restProps.onClick) === null || _restProps$onClick === void 0 || _restProps$onClick.call(restProps, event);
    },
    onMouseUp: event => {
      var _restProps$onMouseUp;
      event.stopPropagation();
      (_restProps$onMouseUp = restProps.onMouseUp) === null || _restProps$onMouseUp === void 0 || _restProps$onMouseUp.call(restProps, event);
    },
    className: "combobox-footer-add-button",
    css: /*#__PURE__*/css({
      ...getComboboxOptionItemWrapperStyles(theme),
      ... /*#__PURE__*/css(importantify({
        width: '100%',
        padding: 0,
        display: 'flex',
        alignItems: 'center',
        borderRadius: 0,
        '&:focus': {
          background: theme.colors.actionTertiaryBackgroundHover,
          outline: 'none'
        }
      }), process.env.NODE_ENV === "production" ? "" : ";label:TypeaheadComboboxAddButton;")
    }, process.env.NODE_ENV === "production" ? "" : ";label:TypeaheadComboboxAddButton;"),
    icon: jsx(PlusIcon, {}),
    children: children
  });
};

function RHFControlledInput(_ref) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Input, {
    ...restProps,
    ...field,
    value: field.value,
    defaultValue: restProps.defaultValue
  });
}
function RHFControlledPasswordInput(_ref2) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref2;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Input.Password, {
    ...restProps,
    ...field,
    value: field.value,
    defaultValue: restProps.defaultValue
  });
}
function RHFControlledTextArea(_ref3) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref3;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Input.TextArea, {
    ...restProps,
    ...field,
    value: field.value,
    defaultValue: restProps.defaultValue
  });
}
/**
 * @deprecated Use `RHFControlledSelect` instead.
 */
function RHFControlledLegacySelect(_ref4) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref4;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(LegacySelect, {
    ...restProps,
    ...field,
    value: field.value,
    defaultValue: field.value
  });
}
/**
 * @deprecated This component is no longer necessary as `SimpleSelect` can be used uncontrolled by RHF.
 * Please consult the Forms Guide on go/dubois.
 */
function RHFControlledSelect(_ref5) {
  let {
    name,
    control,
    rules,
    options,
    validationState,
    children,
    width,
    triggerProps,
    contentProps,
    optionProps,
    ...restProps
  } = _ref5;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  const [selectedValueLabel, setSelectedValueLabel] = useState(field.value ? field.value.label ? field.value.label : field.value : '');
  const handleOnChange = option => {
    field.onChange(typeof option === 'object' ? option.value : option);
  };
  useEffect(() => {
    if (!field.value) {
      return;
    }

    // Find the appropriate label for the selected value
    if (!(options !== null && options !== void 0 && options.length) && children) {
      const renderedChildren = children({
        onChange: handleOnChange
      });
      const child = (Array.isArray(renderedChildren) ? renderedChildren : Children.toArray(renderedChildren.props.children)).find(child => /*#__PURE__*/React__default.isValidElement(child) && child.props.value === field.value);
      if (child) {
        var _props;
        if (((_props = child.props) === null || _props === void 0 ? void 0 : _props.children) !== field.value) {
          setSelectedValueLabel(child.props.children);
        } else {
          setSelectedValueLabel(field.value);
        }
      }
    } else if (options !== null && options !== void 0 && options.length) {
      const option = options.find(option => option.value === field.value);
      setSelectedValueLabel(option === null || option === void 0 ? void 0 : option.label);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [field.value]);
  return jsxs(Select, {
    ...restProps,
    value: field.value,
    children: [jsx(SelectTrigger, {
      ...triggerProps,
      width: width,
      onBlur: field.onBlur,
      validationState: validationState,
      children: selectedValueLabel
    }), jsx(SelectContent, {
      ...contentProps,
      side: "bottom",
      children: options && options.length > 0 ? options.map(option => createElement(SelectOption, {
        ...optionProps,
        key: option.value,
        value: option.value,
        onChange: handleOnChange
      }, option.label)) : // SelectOption out of the box gives users control over state and in this case RHF is controlling state
      // We expose onChange through a children renderer function to let users pass this down to SelectOption
      children === null || children === void 0 ? void 0 : children({
        onChange: handleOnChange
      })
    })]
  });
}
function RHFControlledDialogCombobox(_ref6) {
  let {
    name,
    control,
    rules,
    children,
    allowClear,
    validationState,
    placeholder,
    width,
    triggerProps,
    contentProps,
    optionListProps,
    ...restProps
  } = _ref6;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  const [valueMap, setValueMap] = useState({}); // Used for multi-select

  const updateValueMap = updatedValue => {
    if (updatedValue) {
      if (Array.isArray(updatedValue)) {
        setValueMap(updatedValue.reduce((acc, value) => {
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
  };
  const handleOnChangeSingleSelect = option => {
    let updatedValue = field.value;
    if (field.value === option) {
      updatedValue = undefined;
    } else {
      updatedValue = option;
    }
    field.onChange(updatedValue);
    updateValueMap(updatedValue);
  };
  const handleOnChangeMultiSelect = option => {
    var _field$value;
    let updatedValue;
    if ((_field$value = field.value) !== null && _field$value !== void 0 && _field$value.includes(option)) {
      updatedValue = field.value.filter(value => value !== option);
    } else if (!field.value) {
      updatedValue = [option];
    } else {
      updatedValue = [...field.value, option];
    }
    field.onChange(updatedValue);
    updateValueMap(updatedValue);
  };
  const handleOnChange = option => {
    if (restProps.multiSelect) {
      handleOnChangeMultiSelect(option);
    } else {
      handleOnChangeSingleSelect(option);
    }
  };
  useEffect(() => {
    if (field.value) {
      updateValueMap(field.value);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const isChecked = option => {
    return valueMap[option];
  };
  const handleOnClear = () => {
    field.onChange(Array.isArray(field.value) ? [] : '');
    setValueMap({});
  };
  return jsxs(DialogCombobox, {
    ...restProps,
    value: field.value ? Array.isArray(field.value) ? field.value : [field.value] : undefined,
    children: [jsx(DialogComboboxTrigger, {
      ...triggerProps,
      onBlur: field.onBlur,
      allowClear: allowClear,
      validationState: validationState,
      onClear: handleOnClear,
      withInlineLabel: false,
      placeholder: placeholder,
      width: width
    }), jsx(DialogComboboxContent, {
      ...contentProps,
      side: "bottom",
      width: width,
      children: jsx(DialogComboboxOptionList, {
        ...optionListProps,
        children: children === null || children === void 0 ? void 0 : children({
          onChange: handleOnChange,
          value: field.value,
          isChecked
        })
      })
    })]
  });
}
function RHFControlledTypeaheadCombobox(_ref7) {
  let {
    name,
    control,
    rules,
    allItems,
    itemToString,
    matcher,
    allowNewValue,
    children,
    validationState,
    inputProps,
    menuProps,
    ...props
  } = _ref7;
  const {
    field
  } = useController({
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
    formOnBlur: field.onBlur
  });
  useEffect(() => {
    setItems(allItems);
  }, [allItems]);
  return jsxs(TypeaheadComboboxRoot, {
    ...props,
    comboboxState: comboboxState,
    children: [jsx(TypeaheadComboboxInput, {
      ...inputProps,
      validationState: validationState,
      formOnChange: field.onChange,
      comboboxState: comboboxState
    }), jsx(TypeaheadComboboxMenu, {
      ...menuProps,
      comboboxState: comboboxState,
      children: children({
        comboboxState,
        items
      })
    })]
  });
}
function RHFControlledMultiSelectTypeaheadCombobox(_ref8) {
  let {
    name,
    control,
    rules,
    allItems,
    itemToString,
    matcher,
    children,
    validationState,
    inputProps,
    menuProps,
    ...props
  } = _ref8;
  const {
    field
  } = useController({
    name,
    control,
    rules
  });
  const [inputValue, setInputValue] = useState('');
  const [selectedItems, setSelectedItems] = useState(field.value || []);
  useEffect(() => {
    setSelectedItems(field.value || []);
  }, [field.value]);
  const items = React__default.useMemo(() => allItems.filter(item => matcher(item, inputValue)), [inputValue, matcher, allItems]);
  const multipleSelectionState = useMultipleSelectionState(selectedItems, setSelectedItems);
  const handleItemUpdate = item => {
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
    formOnBlur: field.onBlur
  });
  return jsxs(TypeaheadComboboxRoot, {
    ...props,
    comboboxState: comboboxState,
    children: [jsx(TypeaheadComboboxMultiSelectInput, {
      ...inputProps,
      multipleSelectionState: multipleSelectionState,
      selectedItems: selectedItems,
      setSelectedItems: handleItemUpdate,
      getSelectedItemLabel: itemToString,
      comboboxState: comboboxState,
      validationState: validationState
    }), jsx(TypeaheadComboboxMenu, {
      ...menuProps,
      comboboxState: comboboxState,
      children: children({
        comboboxState,
        items,
        selectedItems
      })
    })]
  });
}
function RHFControlledCheckboxGroup(_ref9) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref9;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Checkbox.Group, {
    ...restProps,
    ...field,
    value: field.value
  });
}
function RHFControlledCheckbox(_ref10) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref10;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: /*#__PURE__*/css({
      marginTop: theme.spacing.sm
    }, process.env.NODE_ENV === "production" ? "" : ";label:RHFControlledCheckbox;"),
    children: jsx(Checkbox, {
      ...restProps,
      ...field,
      isChecked: field.value
    })
  });
}
function RHFControlledRadioGroup(_ref11) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref11;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Radio.Group, {
    ...restProps,
    ...field
  });
}
function RHFControlledSwitch(_ref12) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref12;
  const {
    field
  } = useController({
    name: name,
    control: control,
    rules: rules
  });
  return jsx(Switch, {
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

const getHorizontalInputStyles = (theme, labelColWidth, inputColWidth) => {
  return /*#__PURE__*/css({
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
  }, process.env.NODE_ENV === "production" ? "" : ";label:getHorizontalInputStyles;");
};
const HorizontalFormRow = _ref => {
  let {
    children,
    labelColWidth = '33%',
    inputColWidth = '66%',
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
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

const Col = _ref => {
  let {
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Col$1, {
      ...props,
      ...dangerouslySetAntdProps
    })
  });
};

const ROW_GUTTER_SIZE = 8;
const Row = _ref => {
  let {
    gutter = ROW_GUTTER_SIZE,
    ...props
  } = _ref;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Row$1, {
      gutter: gutter,
      ...props
    })
  });
};

const Space = _ref => {
  let {
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Space$1, {
      ...props,
      ...dangerouslySetAntdProps
    })
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$5() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const getHeaderStyles = (clsPrefix, theme) => {
  const breadcrumbClass = `.${clsPrefix}-breadcrumb`;
  const styles = {
    [breadcrumbClass]: {
      lineHeight: theme.typography.lineHeightBase
    }
  };
  return /*#__PURE__*/css(importantify(styles), process.env.NODE_ENV === "production" ? "" : ";label:getHeaderStyles;");
};
var _ref$2 = process.env.NODE_ENV === "production" ? {
  name: "4y41qd",
  styles: "display:inline-flex;vertical-align:middle;align-items:center;flex-wrap:wrap"
} : {
  name: "aw04zu-titleAddOnsWrapper",
  styles: "display:inline-flex;vertical-align:middle;align-items:center;flex-wrap:wrap;label:titleAddOnsWrapper;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$5
};
var _ref2$3 = process.env.NODE_ENV === "production" ? {
  name: "1q4vxyr",
  styles: "margin-left:8px"
} : {
  name: "ozrfom-buttonContainer",
  styles: "margin-left:8px;label:buttonContainer;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$5
};
var _ref3$3 = process.env.NODE_ENV === "production" ? {
  name: "s079uh",
  styles: "margin-top:2px"
} : {
  name: "1ky5whb-titleIfOtherElementsPresent",
  styles: "margin-top:2px;label:titleIfOtherElementsPresent;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$5
};
var _ref4 = process.env.NODE_ENV === "production" ? {
  name: "fuxm9z",
  styles: "margin-top:0;margin-bottom:0 !important;align-self:stretch"
} : {
  name: "h5m2l9-title",
  styles: "margin-top:0;margin-bottom:0 !important;align-self:stretch;label:title;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$5
};
const Header$1 = _ref5 => {
  let {
    breadcrumbs,
    title,
    titleAddOns,
    dangerouslyAppendEmotionCSS,
    buttons,
    children,
    titleElementLevel,
    ...rest
  } = _ref5;
  const {
    classNamePrefix: clsPrefix,
    theme
  } = useDesignSystemTheme();
  const buttonsArray = Array.isArray(buttons) ? buttons : buttons ? [buttons] : [];

  // TODO: Move to getHeaderStyles for consistency, followup ticket: https://databricks.atlassian.net/browse/FEINF-1222
  const styles = {
    titleWrapper: /*#__PURE__*/css({
      display: 'flex',
      alignItems: 'flex-start',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      // Buttons have 32px height while Title level 2 elements used by this component have a height of 28px
      // These paddings enforce height to be the same without buttons too
      ...(buttonsArray.length === 0 && {
        paddingTop: breadcrumbs ? 0 : theme.spacing.xs / 2,
        paddingBottom: theme.spacing.xs / 2
      })
    }, process.env.NODE_ENV === "production" ? "" : ";label:titleWrapper;"),
    breadcrumbWrapper: /*#__PURE__*/css({
      lineHeight: theme.typography.lineHeightBase
    }, process.env.NODE_ENV === "production" ? "" : ";label:breadcrumbWrapper;"),
    title: _ref4,
    // TODO: Look into a more emotion-idomatic way of doing this.
    titleIfOtherElementsPresent: _ref3$3,
    buttonContainer: _ref2$3,
    titleAddOnsWrapper: _ref$2
  };
  return jsxs("div", {
    ...addDebugOutlineIfEnabled(),
    css: [getHeaderStyles(clsPrefix, theme), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Header;"],
    ...rest,
    children: [breadcrumbs && jsx("div", {
      css: styles.breadcrumbWrapper,
      children: breadcrumbs
    }), jsxs("div", {
      css: styles.titleWrapper,
      children: [jsxs(Title$2, {
        level: 2,
        elementLevel: titleElementLevel,
        css: [styles.title, (buttons || breadcrumbs) && styles.titleIfOtherElementsPresent, process.env.NODE_ENV === "production" ? "" : ";label:Header;"],
        children: [title, titleAddOns && jsxs(Fragment, {
          children: ["\u2002", jsx("span", {
            css: styles.titleAddOnsWrapper,
            children: titleAddOns
          })]
        })]
      }), buttons && jsx("div", {
        css: styles.buttonContainer,
        children: jsx(Space, {
          dangerouslySetAntdProps: {
            wrap: true
          },
          size: 8,
          children: buttonsArray.filter(Boolean).map((button, i) => {
            const defaultKey = `dubois-header-button-${i}`;
            return /*#__PURE__*/React__default.isValidElement(button) ? ( /*#__PURE__*/React__default.cloneElement(button, {
              key: button.key || defaultKey
            })) : jsx(React__default.Fragment, {
              children: button
            }, defaultKey);
          })
        })
      })]
    })]
  });
};

/**
 * The HoverCard component combines Radix's HoverCard primitives into a single, easy-to-use component.
 * It handles the setup of the trigger, content, and arrow elements, as well as applying custom styles
 * using Emotion CSS
 */
const HoverCard = _ref => {
  let {
    trigger,
    content,
    side = 'top',
    sideOffset = 4,
    align = 'center',
    minWidth = 220,
    maxWidth,
    ...props
  } = _ref;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const hoverCardStyles = useHoverCardStyles({
    minWidth,
    maxWidth
  });
  return jsxs(RadixHoverCard.Root, {
    ...props,
    children: [jsx(RadixHoverCard.Trigger, {
      asChild: true,
      children: trigger
    }), jsx(RadixHoverCard.Portal, {
      container: getPopupContainer && getPopupContainer(),
      children: jsxs(RadixHoverCard.Content, {
        side: side,
        sideOffset: sideOffset,
        align: align,
        css: hoverCardStyles['content'],
        children: [content, jsx(RadixHoverCard.Arrow, {
          css: hoverCardStyles['arrow']
        })]
      })
    })]
  });
};

// CONSTANTS used for defining the Arrow's appearance and behavior
const CONSTANTS = {
  arrowWidth: 12,
  arrowHeight: 6,
  arrowBottomLength() {
    // The built in arrow is a polygon: 0,0 30,0 15,10
    return 30;
  },
  arrowSide() {
    return 2 * (this.arrowHeight ** 2 * 2) ** 0.5;
  },
  arrowStrokeWidth() {
    // This is eyeballed b/c relative to the svg viewbox coordinate system
    return 2;
  }
};

/**
 * A custom hook to generate CSS styles for the HoverCard's content and arrow.
 * These styles are dynamically generated based on the theme and optional min/max width props.
 * The hook also applies necessary dark mode adjustments
 */
const useHoverCardStyles = _ref2 => {
  let {
    minWidth,
    maxWidth
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  return {
    content: {
      backgroundColor: theme.colors.backgroundPrimary,
      color: theme.colors.textPrimary,
      lineHeight: theme.typography.lineHeightBase,
      border: `1px solid ${theme.colors.borderDecorative}`,
      borderRadius: theme.borders.borderRadiusMd,
      padding: `${theme.spacing.sm}px`,
      boxShadow: theme.general.shadowLow,
      userSelect: 'none',
      zIndex: theme.options.zIndexBase + 30,
      minWidth,
      maxWidth,
      ...getDarkModePortalStyles(theme),
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
        outlineColor: theme.colors.primary
      }
    },
    arrow: {
      fill: theme.colors.backgroundPrimary,
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

const {
  Header,
  Footer,
  Sider,
  Content
} = Layout$1;
/**
 * @deprecated Use PageWrapper instead
 */
const Layout = /* #__PURE__ */(() => {
  const Layout = _ref => {
    let {
      children,
      dangerouslySetAntdProps,
      ...props
    } = _ref;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Layout$1, {
        ...addDebugOutlineIfEnabled(),
        ...props,
        ...dangerouslySetAntdProps,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    });
  };
  Layout.Header = _ref2 => {
    let {
      children,
      ...props
    } = _ref2;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Header, {
        ...addDebugOutlineIfEnabled(),
        ...props,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    });
  };
  Layout.Footer = _ref3 => {
    let {
      children,
      ...props
    } = _ref3;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Footer, {
        ...addDebugOutlineIfEnabled(),
        ...props,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    });
  };
  Layout.Sider = /*#__PURE__*/React__default.forwardRef((_ref4, ref) => {
    let {
      children,
      ...props
    } = _ref4;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Sider, {
        ...addDebugOutlineIfEnabled(),
        ...props,
        ref: ref,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    });
  });
  Layout.Content = _ref5 => {
    let {
      children,
      ...props
    } = _ref5;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Content, {
        ...addDebugOutlineIfEnabled(),
        ...props,
        children: jsx(RestoreAntDDefaultClsPrefix, {
          children: children
        })
      })
    });
  };
  return Layout;
})();

// Note: AntD only exposes context to notifications via the `useNotification` hook, and we need context to apply themes
// to AntD. As such you can currently only use notifications from within functional components.
/**
 * `useLegacyNotification` is deprecated in favor of the new `Notification` component.
 * @deprecated
 */
function useLegacyNotification() {
  const [notificationInstance, contextHolder] = notification.useNotification();
  const {
    getPrefixedClassName,
    theme
  } = useDesignSystemTheme();
  const {
    getPopupContainer: getContainer
  } = useDesignSystemContext();
  const clsPrefix = getPrefixedClassName('notification');
  const open = useCallback(args => {
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
    mergedArgs.icon = jsx(SeverityIcon, {
      severity: mergedArgs.type,
      className: iconClassName
    });
    mergedArgs.closeIcon = jsx(CloseIcon, {
      "aria-hidden": "false",
      css: /*#__PURE__*/css({
        cursor: 'pointer',
        fontSize: theme.general.iconSize
      }, process.env.NODE_ENV === "production" ? "" : ";label:mergedArgs-closeIcon;"),
      "aria-label": mergedArgs.closeLabel || 'Close notification'
    });
    notificationInstance.open(mergedArgs);
  }, [notificationInstance, getContainer, theme, clsPrefix]);
  const wrappedNotificationAPI = useMemo(() => {
    const error = args => open({
      ...args,
      type: 'error'
    });
    const warning = args => open({
      ...args,
      type: 'warning'
    });
    const info = args => open({
      ...args,
      type: 'info'
    });
    const success = args => open({
      ...args,
      type: 'success'
    });
    const close = key => notification.close(key);
    return {
      open,
      close,
      error,
      warning,
      info,
      success
    };
  }, [open]);

  // eslint-disable-next-line react/jsx-key -- TODO(FEINF-1756)
  return [wrappedNotificationAPI, jsx(DesignSystemAntDConfigProvider, {
    children: contextHolder
  })];
}
const defaultProps = {
  type: 'info',
  duration: 3
};

/**
 * A type wrapping given component interface with props returned by withNotifications() HOC
 *
 * @deprecated Please migrate components to functional components and use useNotification() hook instead.
 */

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
 */
const withNotifications = Component => /*#__PURE__*/forwardRef((props, ref) => {
  const [notificationAPI, notificationContextHolder] = useLegacyNotification();
  return jsx(Component, {
    ref: ref,
    notificationAPI: notificationAPI,
    notificationContextHolder: notificationContextHolder,
    ...props
  });
});

/**
 * `LegacyPopover` is deprecated in favor of the new `Popover` component.
 * @deprecated
 */
const LegacyPopover = _ref => {
  let {
    content,
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Popover$1, {
      zIndex: theme.options.zIndexBase + 30,
      ...props,
      content: jsx(RestoreAntDDefaultClsPrefix, {
        children: content
      })
    })
  });
};

/** @deprecated This component is deprecated. Use ParagraphSkeleton, TitleSkeleton, or GenericSkeleton instead. */
const LegacySkeleton = /* #__PURE__ */(() => {
  const LegacySkeleton = _ref => {
    var _loading;
    let {
      dangerouslySetAntdProps,
      label,
      loadingDescription = 'LegacySkeleton',
      ...props
    } = _ref;
    // There is a conflict in how the 'loading' prop is handled here, so we can't default it to true in
    // props destructuring above like we do for 'loadingDescription'. The 'loading' param is used both
    // for <LoadingState> and in <AntDSkeleton>. The intent is for 'loading' to default to true in
    // <LoadingState>, but if we do that, <AntDSkeleton> will not render the children. The workaround
    // here is to default 'loading' to true only when considering whether to render a <LoadingState>.
    // Also, AntDSkeleton looks at the presence of 'loading' in props, so We cannot explicitly destructure
    // 'loading' in the constructor since we would no longer be able to differentiate between it not being
    // passed in at all and being passed undefined.
    const loadingStateLoading = (_loading = props.loading) !== null && _loading !== void 0 ? _loading : true;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsxs(AccessibleContainer, {
        label: label,
        children: [loadingStateLoading && jsx(LoadingState, {
          description: loadingDescription
        }), jsx(Skeleton, {
          ...props,
          ...dangerouslySetAntdProps
        })]
      })
    });
  };
  LegacySkeleton.Button = Skeleton.Button;
  LegacySkeleton.Image = Skeleton.Image;
  LegacySkeleton.Input = Skeleton.Input;
  return LegacySkeleton;
})();

function _EMOTION_STRINGIFIED_CSS_ERROR__$4() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
function getPaginationEmotionStyles(clsPrefix, theme) {
  const classRoot = `.${clsPrefix}-pagination`;
  const classItem = `.${clsPrefix}-pagination-item`;
  const classLink = `.${clsPrefix}-pagination-item-link`;
  const classActive = `.${clsPrefix}-pagination-item-active`;
  const classEllipsis = `.${clsPrefix}-pagination-item-ellipsis`;
  const classNext = `.${clsPrefix}-pagination-next`;
  const classPrev = `.${clsPrefix}-pagination-prev`;
  const classJumpNext = `.${clsPrefix}-pagination-jump-next`;
  const classJumpPrev = `.${clsPrefix}-pagination-jump-prev`;
  const classSizeChanger = `.${clsPrefix}-pagination-options-size-changer`;
  const classOptions = `.${clsPrefix}-pagination-options`;
  const classDisabled = `.${clsPrefix}-pagination-disabled`;
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
      }
    }
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getPaginationEmotionStyles;");
}
const Pagination = function Pagination(_ref) {
  let {
    currentPageIndex,
    pageSize = 10,
    numTotal,
    onChange,
    style,
    hideOnSinglePage,
    dangerouslySetAntdProps,
    componentId,
    analyticsEvents
  } = _ref;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    pageSizeSelectAriaLabel,
    pageQuickJumperAriaLabel,
    ...restDangerouslySetAntdProps
  } = dangerouslySetAntdProps !== null && dangerouslySetAntdProps !== void 0 ? dangerouslySetAntdProps : {};
  const ref = useRef(null);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Pagination,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii: true
  });
  const onChangeWrapper = useCallback((pageIndex, pageSize) => {
    eventContext.onValueChange(pageIndex);
    onChange(pageIndex, pageSize);
  }, [eventContext, onChange]);
  useEffect(() => {
    if (ref && ref.current) {
      const selectDropdown = ref.current.querySelector(`.${classNamePrefix}-select-selection-search-input`);
      if (selectDropdown) {
        selectDropdown.setAttribute('aria-label', pageSizeSelectAriaLabel !== null && pageSizeSelectAriaLabel !== void 0 ? pageSizeSelectAriaLabel : 'Select page size');
      }
      const pageQuickJumper = ref.current.querySelector(`.${classNamePrefix}-pagination-options-quick-jumper > input`);
      if (pageQuickJumper) {
        pageQuickJumper.setAttribute('aria-label', pageQuickJumperAriaLabel !== null && pageQuickJumperAriaLabel !== void 0 ? pageQuickJumperAriaLabel : 'Go to page');
      }
    }
  }, [pageQuickJumperAriaLabel, pageSizeSelectAriaLabel, classNamePrefix]);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx("div", {
      ref: ref,
      children: jsx(Pagination$1, {
        ...addDebugOutlineIfEnabled(),
        css: getPaginationEmotionStyles(classNamePrefix, theme),
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
        ...restDangerouslySetAntdProps
      })
    })
  });
};
var _ref3$2 = process.env.NODE_ENV === "production" ? {
  name: "1u1zie3",
  styles: "width:120px"
} : {
  name: "1am9qog-CursorPagination",
  styles: "width:120px;label:CursorPagination;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
};
const CursorPagination = function CursorPagination(_ref2) {
  let {
    onNextPage,
    onPreviousPage,
    hasNextPage,
    hasPreviousPage,
    nextPageText = 'Next',
    previousPageText = 'Previous',
    pageSizeSelect: {
      options: pageSizeOptions,
      default: defaultPageSize,
      getOptionText: getPageSizeOptionText,
      onChange: onPageSizeChange,
      ariaLabel = 'Select page size'
    } = {},
    componentId = 'design_system.cursor_pagination',
    analyticsEvents,
    valueHasNoPii
  } = _ref2;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const [pageSizeValue, setPageSizeValue] = useState(defaultPageSize);
  const pageSizeEventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.LegacySelect,
    componentId: `${componentId}.page_size`,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii
  });
  const getPageSizeOptionTextDefault = pageSize => `${pageSize} / page`;
  return jsxs("div", {
    css: /*#__PURE__*/css({
      display: 'flex',
      flexDirection: 'row',
      gap: theme.spacing.sm,
      [`.${classNamePrefix}-select-selector::after`]: {
        content: 'none'
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:CursorPagination;"),
    children: [jsx(Button, {
      componentId: `${componentId}.previous_page`,
      icon: jsx(ChevronLeftIcon, {}),
      disabled: !hasPreviousPage,
      onClick: onPreviousPage,
      type: "tertiary",
      children: previousPageText
    }), jsx(Button, {
      componentId: `${componentId}.next_page`,
      endIcon: jsx(ChevronRightIcon, {}),
      disabled: !hasNextPage,
      onClick: onNextPage,
      type: "tertiary",
      children: nextPageText
    }), pageSizeOptions && jsx(LegacySelect, {
      "aria-label": ariaLabel,
      value: String(pageSizeValue),
      css: _ref3$2,
      onChange: pageSize => {
        const updatedPageSize = Number(pageSize);
        onPageSizeChange === null || onPageSizeChange === void 0 || onPageSizeChange(updatedPageSize);
        setPageSizeValue(updatedPageSize);
        // When this usage of LegacySelect is migrated to Select, this call
        // can be removed in favor of passing a componentId to Select
        pageSizeEventContext.onValueChange(pageSize);
      },
      children: pageSizeOptions.map(pageSize => jsx(LegacySelect.Option, {
        value: String(pageSize),
        children: (getPageSizeOptionText || getPageSizeOptionTextDefault)(pageSize)
      }, pageSize))
    })]
  });
};

const getTableEmotionStyles = (classNamePrefix, theme, scrollableInFlexibleContainer) => {
  const styles = [/*#__PURE__*/css({
    [`.${classNamePrefix}-table-pagination`]: {
      ...getPaginationEmotionStyles(classNamePrefix, theme)
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:styles;")];
  if (scrollableInFlexibleContainer) {
    styles.push(getScrollableInFlexibleContainerStyles(classNamePrefix));
  }
  return styles;
};
const getScrollableInFlexibleContainerStyles = clsPrefix => {
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
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getScrollableInFlexibleContainerStyles;");
};
const DEFAULT_LOADING_SPIN_PROPS = {
  indicator: jsx(Spinner, {})
};

/**
 * `LegacyTable` is deprecated in favor of the new `Table` component
 * For more information please join #dubois-table-early-adopters in Slack.
 * @deprecated
 */
// eslint-disable-next-line @typescript-eslint/ban-types
const LegacyTable = props => {
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    loading,
    scrollableInFlexibleContainer,
    children,
    ...tableProps
  } = props;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Table$1, {
      ...addDebugOutlineIfEnabled(),
      // NOTE(FEINF-1273): The default loading indicator from AntD does not animate
      // and the design system spinner is recommended over the AntD one. Therefore,
      // if `loading` is `true`, render the design system <Spinner/> component.
      loading: loading === true ? DEFAULT_LOADING_SPIN_PROPS : loading,
      scroll: scrollableInFlexibleContainer ? {
        y: 'auto'
      } : undefined,
      ...tableProps,
      css: getTableEmotionStyles(classNamePrefix, theme, Boolean(scrollableInFlexibleContainer))
      // ES-902549 this allows column names of "children", using a name that is less likely to be hit
      ,
      expandable: {
        childrenColumnName: '__antdChildren'
      },
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
};

/**
 * @deprecated Use `DropdownMenu` instead.
 */
const Menu = /* #__PURE__ */(() => {
  const Menu = _ref => {
    let {
      dangerouslySetAntdProps,
      ...props
    } = _ref;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Menu$1, {
        ...addDebugOutlineIfEnabled(),
        ...props,
        ...dangerouslySetAntdProps
      })
    });
  };
  Menu.Item = Menu$1.Item;
  Menu.ItemGroup = Menu$1.ItemGroup;
  Menu.SubMenu = function SubMenu(_ref2) {
    let {
      dangerouslySetAntdProps,
      ...props
    } = _ref2;
    const {
      theme
    } = useDesignSystemTheme();
    return jsx(ClassNames, {
      children: _ref3 => {
        let {
          css
        } = _ref3;
        return jsx(Menu$1.SubMenu, {
          ...addDebugOutlineIfEnabled(),
          popupClassName: css({
            zIndex: theme.options.zIndexBase + 50
          }),
          popupOffset: [-6, -10],
          ...props,
          ...dangerouslySetAntdProps
        });
      }
    });
  };
  return Menu;
})();

const hideAnimation = keyframes({
  from: {
    opacity: 1
  },
  to: {
    opacity: 0
  }
});
const slideInAnimation = keyframes({
  from: {
    transform: 'translateX(calc(100% + 12px))'
  },
  to: {
    transform: 'translateX(0)'
  }
});
const swipeOutAnimation = keyframes({
  from: {
    transform: 'translateX(var(--radix-toast-swipe-end-x))'
  },
  to: {
    transform: 'translateX(calc(100% + 12px))'
  }
});
const getToastRootStyle = (theme, classNamePrefix) => {
  return /*#__PURE__*/css({
    '&&': {
      position: 'relative',
      display: 'grid',
      background: theme.colors.backgroundPrimary,
      padding: 12,
      columnGap: 4,
      boxShadow: theme.general.shadowLow,
      borderRadius: theme.general.borderRadiusBase,
      lineHeight: '20px',
      gridTemplateRows: '[header] auto [content] auto',
      gridTemplateColumns: '[icon] auto [content] 1fr [close] auto',
      ...getDarkModePortalStyles(theme)
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
  }, process.env.NODE_ENV === "production" ? "" : ";label:getToastRootStyle;");
};
const Root = /*#__PURE__*/forwardRef(function (_ref, ref) {
  let {
    children,
    severity = 'info',
    componentId,
    analyticsEvents,
    ...props
  } = _ref;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Notification,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [],
    shouldStartInteraction: false
  });
  // A new ref was created rather than creating additional complexity of merging the refs, something to consider for the future to optimize
  const {
    elementRef
  } = useNotifyOnFirstView({
    onView: eventContext.onView
  });
  return jsxs(Toast.Root, {
    ref: ref,
    css: getToastRootStyle(theme, classNamePrefix),
    ...props,
    ...addDebugOutlineIfEnabled(),
    children: [jsx(SeverityIcon, {
      className: `${classNamePrefix}-notification-severity-icon ${classNamePrefix}-notification-${severity}-icon`,
      severity: severity,
      ref: elementRef
    }), children]
  });
});

// TODO: Support light and dark mode

const getViewportStyle = theme => {
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
const getTitleStyles = theme => {
  return /*#__PURE__*/css({
    fontWeight: theme.typography.typographyBoldFontWeight,
    color: theme.colors.textPrimary,
    gridRow: 'header / header',
    gridColumn: 'content / content',
    userSelect: 'text'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getTitleStyles;");
};
const Title = /*#__PURE__*/forwardRef(function (_ref2, ref) {
  let {
    children,
    ...props
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Toast.Title, {
    ref: ref,
    css: getTitleStyles(theme),
    ...props,
    children: children
  });
});
const getDescriptionStyles = theme => {
  return /*#__PURE__*/css({
    marginTop: 4,
    color: theme.colors.textPrimary,
    gridRow: 'content / content',
    gridColumn: 'content / content',
    userSelect: 'text'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getDescriptionStyles;");
};
const Description = /*#__PURE__*/forwardRef(function (_ref3, ref) {
  let {
    children,
    ...props
  } = _ref3;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Toast.Description, {
    ref: ref,
    css: getDescriptionStyles(theme),
    ...props,
    children: children
  });
});
const getCloseStyles = theme => {
  return /*#__PURE__*/css({
    color: theme.colors.textSecondary,
    position: 'absolute',
    // Offset close button position to align with the title, title uses 20px line height, button has 32px
    right: 6,
    top: 6
  }, process.env.NODE_ENV === "production" ? "" : ";label:getCloseStyles;");
};
const Close = /*#__PURE__*/forwardRef(function (props, ref) {
  var _ref4;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    closeLabel,
    componentId,
    analyticsEvents,
    ...restProps
  } = props;
  return (
    // Wrapper to keep close column width for content sizing, close button positioned absolute for alignment without affecting the grid's first row height (title)
    jsx("div", {
      style: {
        gridColumn: 'close / close',
        gridRow: 'header / content',
        width: 20
      },
      children: jsx(Toast.Close, {
        ref: ref,
        css: getCloseStyles(theme),
        ...restProps,
        asChild: true,
        children: jsx(Button, {
          componentId: componentId ? componentId : 'codegen_design-system_src_design-system_notification_notification.tsx_224',
          analyticsEvents: analyticsEvents,
          icon: jsx(CloseIcon, {}),
          "aria-label": (_ref4 = closeLabel !== null && closeLabel !== void 0 ? closeLabel : restProps['aria-label']) !== null && _ref4 !== void 0 ? _ref4 : 'Close notification'
        })
      })
    })
  );
});
const Provider = _ref5 => {
  let {
    children,
    ...props
  } = _ref5;
  return jsx(Toast.Provider, {
    ...props,
    children: children
  });
};
const Viewport = props => {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Toast.Viewport, {
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
  Root: Root,
  Title: Title,
  Viewport: Viewport
});

const colorMap = {
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
function getTagEmotionStyles(theme) {
  let color = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 'default';
  let clickable = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
  let textColor = theme.colors.tagText;
  const backgroundColor = theme.colors[colorMap[color]];
  let iconHover = theme.colors.tagIconHover;
  let iconPress = theme.colors.tagIconPress;
  let tagHover = theme.colors.tagHover;
  let tagPress = theme.colors.tagPress;

  // Because the default tag background color changes depending on system theme, so do its other variables.
  if (color === 'default') {
    textColor = theme.colors.textPrimary;
    iconHover = theme.colors.actionTertiaryTextHover;
    iconPress = theme.colors.actionTertiaryTextPress;
  }

  // Because lemon is a light yellow, all its variables pull from the light mode palette, regardless of system theme.
  if (color === 'lemon') {
    textColor = lightColorList.textPrimary;
    iconHover = lightColorList.actionTertiaryTextHover;
    iconPress = lightColorList.actionTertiaryTextPress;
    tagHover = lightColorList.actionTertiaryBackgroundHover;
    tagPress = lightColorList.actionTertiaryBackgroundPress;
  }
  return {
    tag: {
      border: 'none',
      color: textColor,
      padding: '2px 4px',
      backgroundColor,
      borderRadius: theme.borders.borderRadiusMd,
      marginRight: 8,
      display: 'inline-block',
      cursor: clickable ? 'pointer' : 'default'
    },
    content: {
      display: 'flex',
      alignItems: 'center'
    },
    close: {
      height: theme.general.iconFontSize,
      width: theme.general.iconFontSize,
      lineHeight: `${theme.general.iconFontSize}px`,
      padding: 0,
      color: textColor,
      fontSize: theme.general.iconFontSize,
      margin: '-2px -4px -2px 2px',
      borderTopRightRadius: theme.borders.borderRadiusMd,
      borderBottomRightRadius: theme.borders.borderRadiusMd,
      border: 'none',
      background: 'none',
      cursor: 'pointer',
      marginLeft: theme.spacing.xs,
      marginRight: -theme.spacing.xs,
      '&:hover': {
        backgroundColor: tagHover,
        color: iconHover
      },
      '&:active': {
        backgroundColor: tagPress,
        color: iconPress
      },
      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: 1,
        outlineOffset: 1,
        outlineColor: theme.colors.actionDefaultBorderFocus
      },
      '.anticon': {
        verticalAlign: 0
      }
    },
    text: {
      padding: 0,
      fontSize: theme.typography.fontSizeBase,
      fontWeight: theme.typography.typographyRegularFontWeight,
      lineHeight: theme.typography.lineHeightSm,
      '& .anticon': {
        verticalAlign: 'text-top'
      }
    }
  };
}
const Tag = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  const {
    theme
  } = useDesignSystemTheme();
  const {
    color,
    children,
    closable,
    onClose,
    role = 'status',
    closeButtonProps,
    analyticsEvents,
    componentId,
    ...attributes
  } = props;
  const isClickable = Boolean(props.onClick);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Tag,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : []
  });
  const {
    elementRef
  } = useNotifyOnFirstView({
    onView: eventContext.onView
  });
  const mergedRef = useMergeRefs([elementRef, forwardedRef]);
  const closeButtonEventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId: componentId ? `${componentId}.close` : undefined,
    analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick]
  });
  const handleClick = useCallback(e => {
    closeButtonEventContext.onClick();
    e.stopPropagation();
    if (onClose) {
      onClose();
    }
  }, [closeButtonEventContext, onClose]);
  const styles = getTagEmotionStyles(theme, color, isClickable);
  return jsx("div", {
    ref: mergedRef,
    role: role,
    ...attributes,
    css: styles.tag,
    ...addDebugOutlineIfEnabled(),
    children: jsxs("div", {
      css: [styles.content, styles.text, process.env.NODE_ENV === "production" ? "" : ";label:Tag;"],
      children: [children, closable && jsx("button", {
        css: styles.close,
        tabIndex: 0,
        onClick: handleClick,
        onMouseDown: e => {
          // Keeps dropdowns of any underlying select from opening.
          e.stopPropagation();
        },
        ...closeButtonProps,
        children: jsx(CloseIcon, {
          css: /*#__PURE__*/css({
            fontSize: theme.general.iconFontSize - 4
          }, process.env.NODE_ENV === "production" ? "" : ";label:Tag;")
        })
      })]
    })
  });
});

const Overflow = _ref => {
  let {
    children,
    noMargin = false,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const childrenList = children && Children.toArray(children);
  if (!childrenList || childrenList.length === 0) {
    return jsx(Fragment, {
      children: children
    });
  }
  const firstItem = childrenList[0];
  const additionalItems = childrenList.splice(1);
  const renderOverflowLabel = label => jsx(Tag, {
    componentId: "codegen_design-system_src_design-system_overflow_overflow.tsx_28",
    css: getTagStyles(theme),
    children: label
  });
  return additionalItems.length === 0 ? jsx(Fragment, {
    children: firstItem
  }) : jsxs("div", {
    css: /*#__PURE__*/css({
      display: 'inline-flex',
      alignItems: 'center',
      gap: noMargin ? 0 : theme.spacing.sm
    }, process.env.NODE_ENV === "production" ? "" : ";label:Overflow;"),
    children: [firstItem, additionalItems.length > 0 && jsx(OverflowPopover, {
      items: additionalItems,
      renderLabel: renderOverflowLabel,
      ...props
    })]
  });
};
const getTagStyles = theme => {
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
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getTagStyles;");
};

const PageWrapper = _ref => {
  let {
    children,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    ...addDebugOutlineIfEnabled(),
    css: /*#__PURE__*/css({
      paddingLeft: 16,
      paddingRight: 16,
      backgroundColor: theme.isDarkMode ? theme.colors.backgroundPrimary : 'transparent'
    }, process.env.NODE_ENV === "production" ? "" : ";label:PageWrapper;"),
    ...props,
    children: children
  });
};

const STATUS_TO_ICON = {
  online: _ref => {
    let {
      theme,
      ...props
    } = _ref;
    return jsx(CircleIcon$1, {
      color: "success",
      ...props
    });
  },
  disconnected: _ref2 => {
    let {
      theme,
      ...props
    } = _ref2;
    return jsx(CircleOutlineIcon$1, {
      css: /*#__PURE__*/css({
        color: theme.colors.grey500
      }, process.env.NODE_ENV === "production" ? "" : ";label:disconnected;"),
      ...props
    });
  },
  offline: _ref3 => {
    let {
      theme,
      ...props
    } = _ref3;
    return jsx(CircleOfflineIcon$1, {
      css: /*#__PURE__*/css({
        color: theme.colors.grey500
      }, process.env.NODE_ENV === "production" ? "" : ";label:offline;"),
      ...props
    });
  }
};
const ResourceStatusIndicator = props => {
  const {
    status
  } = props;
  const {
    theme
  } = useDesignSystemTheme();
  const StatusIcon = STATUS_TO_ICON[status];
  return jsx(StatusIcon, {
    theme: theme
  });
};

const SMALL_BUTTON_HEIGHT$1 = 24;
function getSegmentedControlGroupEmotionStyles(clsPrefix) {
  let spaced = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;
  const classGroup = `.${clsPrefix}-radio-group`;
  const classSmallGroup = `.${clsPrefix}-radio-group-small`;
  const classButtonWrapper = `.${clsPrefix}-radio-button-wrapper`;
  const styles = {
    [`&${classGroup}`]: spaced ? {
      display: 'flex',
      gap: 8,
      flexWrap: 'wrap'
    } : {},
    [`&${classSmallGroup} ${classButtonWrapper}`]: {
      padding: '0 12px'
    }
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSegmentedControlGroupEmotionStyles;");
}
function getSegmentedControlButtonEmotionStyles(clsPrefix, theme, size) {
  let spaced = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : false;
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
      boxShadow: 'none',
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
      padding: size === 'middle' ? '0 16px' : '0 8px',
      display: 'inline-flex',
      verticalAlign: 'middle',
      ...(spaced ? {
        borderWidth: 1,
        borderRadius: theme.general.borderRadiusBase
      } : {}),
      '&:focus-within': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '-2px',
        outlineColor: theme.colors.primary
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
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSegmentedControlButtonEmotionStyles;");
}
const SegmentedControlGroupContext = /*#__PURE__*/createContext({
  size: 'middle',
  spaced: false
});
const SegmentedControlButton = /*#__PURE__*/forwardRef(function SegmentedControlButton(_ref, ref) {
  let {
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  const {
    size,
    spaced
  } = useContext(SegmentedControlGroupContext);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Radio$1.Button, {
      css: getSegmentedControlButtonEmotionStyles(classNamePrefix, theme, size, spaced),
      ...props,
      ...dangerouslySetAntdProps,
      ref: ref
    })
  });
});
const SegmentedControlGroup = /*#__PURE__*/forwardRef(function SegmentedControlGroup(_ref2, ref) {
  let {
    dangerouslySetAntdProps,
    size = 'middle',
    spaced = false,
    onChange,
    componentId,
    analyticsEvents,
    valueHasNoPii,
    ...props
  } = _ref2;
  const {
    classNamePrefix
  } = useDesignSystemTheme();
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.SegmentedControlGroup,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii
  });
  const onChangeWrapper = useCallback(e => {
    eventContext.onValueChange(e.target.value);
    onChange === null || onChange === void 0 || onChange(e);
  }, [eventContext, onChange]);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(SegmentedControlGroupContext.Provider, {
      value: {
        size,
        spaced
      },
      children: jsx(Radio$1.Group, {
        ...addDebugOutlineIfEnabled(),
        ...props,
        css: getSegmentedControlGroupEmotionStyles(classNamePrefix, spaced),
        onChange: onChangeWrapper,
        ...dangerouslySetAntdProps,
        ref: ref
      })
    })
  });
});

const ButtonGroup = Button$1.Group;
const DropdownButton = props => {
  const {
    getPopupContainer: getContextPopupContainer,
    getPrefixCls
  } = useDesignSystemContext();
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
    icon = jsx(AntDIcon, {}),
    title,
    buttonsRender = buttons => buttons,
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
  const leftButton = jsxs(Button, {
    componentId: componentId ? `${componentId}.primary_button` : 'codegen_design-system_src_design-system_splitbutton_dropdown_dropdownbutton.tsx_148',
    type: type,
    danger: danger,
    disabled: disabled,
    loading: loading,
    onClick: onClick,
    htmlType: htmlType,
    href: href,
    title: title,
    icon: children && leftButtonIcon ? leftButtonIcon : undefined,
    "aria-label": ariaLabel,
    children: [leftButtonIcon && !children ? leftButtonIcon : undefined, children]
  });
  const rightButton = jsx(Button, {
    componentId: componentId ? `${componentId}.dropdown_button` : 'codegen_design-system_src_design-system_splitbutton_dropdown_dropdownbutton.tsx_166',
    type: type,
    danger: danger,
    disabled: disabled,
    "aria-label": menuButtonLabel,
    children: icon ? icon : jsx(ChevronDownIcon$1, {})
  });
  const [leftButtonToRender, rightButtonToRender] = buttonsRender([leftButton, rightButton]);
  return jsxs(ButtonGroup, {
    ...restProps,
    className: classnames(prefixCls, className),
    children: [leftButtonToRender, overlay !== undefined ? jsx(Dropdown$1, {
      ...dropdownProps,
      overlay: overlay,
      children: rightButtonToRender
    }) : jsxs(Root$4, {
      ...dropdownMenuRootProps,
      children: [jsx(Trigger$2, {
        disabled: disabled,
        asChild: true,
        children: rightButtonToRender
      }), menu && /*#__PURE__*/React.cloneElement(menu, {
        align: menu.props.align || 'end'
      })]
    })]
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$3() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const BUTTON_HORIZONTAL_PADDING = 12;
function getSplitButtonEmotionStyles(classNamePrefix, theme) {
  const classDefault = `.${classNamePrefix}-btn`;
  const classPrimary = `.${classNamePrefix}-btn-primary`;
  const classDropdownTrigger = `.${classNamePrefix}-dropdown-trigger`;
  const classSmall = `.${classNamePrefix}-btn-group-sm`;
  const styles = {
    [classDefault]: {
      ...getDefaultStyles(theme),
      boxShadow: 'none',
      height: theme.general.heightSm,
      padding: `4px ${BUTTON_HORIZONTAL_PADDING}px`,
      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '-2px',
        outlineColor: theme.colors.primary
      }
    },
    [classPrimary]: {
      ...getPrimaryStyles(theme),
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
    '[disabled]': {
      ...getDisabledSplitButtonStyles(theme),
      [`&:first-of-type`]: {
        borderRight: `1px solid ${theme.colors.actionPrimaryIcon}`,
        marginRight: 1
      },
      [classDropdownTrigger]: {
        borderLeft: `1px solid ${theme.colors.actionPrimaryIcon}`
      }
    },
    [`${classDefault}:not(:first-of-type)`]: {
      width: theme.general.heightSm,
      padding: '3px !important'
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSplitButtonEmotionStyles;");
}
var _ref$1 = process.env.NODE_ENV === "production" ? {
  name: "tp1ooh",
  styles: "display:inline-flex;position:relative;vertical-align:middle"
} : {
  name: "1kplxg4-SplitButton",
  styles: "display:inline-flex;position:relative;vertical-align:middle;label:SplitButton;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$3
};
const SplitButton = props => {
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    children,
    icon,
    deprecatedMenu,
    type,
    loading,
    loadingButtonStyles,
    placement,
    dangerouslySetAntdProps,
    ...dropdownButtonProps
  } = props;

  // Size of button when loading only icon is shown
  const LOADING_BUTTON_SIZE = theme.general.iconFontSize + 2 * BUTTON_HORIZONTAL_PADDING + 2 * theme.general.borderWidth;
  const [width, setWidth] = useState(LOADING_BUTTON_SIZE);

  // Set the width to the button's width in regular state to later use when in loading state
  // We do this to have just a loading icon in loading state at the normal width to avoid flicker and width changes in page
  const ref = useCallback(node => {
    if (node && !loading) {
      setWidth(node.getBoundingClientRect().width);
    }
  }, [loading]);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx("div", {
      ref: ref,
      css: _ref$1,
      children: loading ? jsx(Button, {
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
        children: children
      }) : jsx(DropdownButton, {
        ...dropdownButtonProps,
        overlay: deprecatedMenu,
        trigger: ['click'],
        css: getSplitButtonEmotionStyles(classNamePrefix, theme),
        icon: jsx(ChevronDownIcon$1, {
          css: /*#__PURE__*/css({
            fontSize: theme.general.iconSize
          }, process.env.NODE_ENV === "production" ? "" : ";label:SplitButton;"),
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

const Steps = /* #__PURE__ */(() => {
  function Steps(_ref) {
    let {
      dangerouslySetAntdProps,
      ...props
    } = _ref;
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Steps$1, {
        ...addDebugOutlineIfEnabled(),
        ...props,
        ...dangerouslySetAntdProps
      })
    });
  }
  Steps.Step = Steps$1.Step;
  return Steps;
})();

function _EMOTION_STRINGIFIED_CSS_ERROR__$2() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const getTableFilterInputStyles = (theme, defaultWidth) => {
  return /*#__PURE__*/css({
    [theme.responsive.mediaQueries.sm]: {
      width: 'auto'
    },
    [theme.responsive.mediaQueries.lg]: {
      width: '30%'
    },
    [theme.responsive.mediaQueries.xxl]: {
      width: defaultWidth
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getTableFilterInputStyles;");
};
var _ref2$2 = process.env.NODE_ENV === "production" ? {
  name: "7whenc",
  styles: "display:flex;width:100%"
} : {
  name: "3ktoz7-component",
  styles: "display:flex;width:100%;label:component;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$2
};
var _ref3$1 = process.env.NODE_ENV === "production" ? {
  name: "82a6rk",
  styles: "flex:1"
} : {
  name: "18ug1j7-component",
  styles: "flex:1;label:component;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$2
};
const TableFilterInput = /*#__PURE__*/forwardRef(function SearchInput(_ref, ref) {
  let {
    onSubmit,
    showSearchButton,
    className,
    containerProps,
    searchButtonProps,
    ...inputProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const DEFAULT_WIDTH = 400;
  let component = jsx(Input, {
    prefix: jsx(SearchIcon$1, {}),
    allowClear: true,
    ...inputProps,
    className: className,
    ref: ref
  });
  if (showSearchButton) {
    component = jsxs(Input.Group, {
      css: _ref2$2,
      className: className,
      children: [jsx(Input, {
        allowClear: true,
        ...inputProps,
        ref: ref,
        css: _ref3$1
      }), jsx(Button, {
        componentId: inputProps.componentId ? `${inputProps.componentId}.search_submit` : 'codegen_design-system_src_design-system_tableui_tablefilterinput.tsx_65',
        htmlType: "submit",
        "aria-label": "Search",
        ...searchButtonProps,
        children: jsx(SearchIcon$1, {})
      })]
    });
  }
  return jsx("div", {
    style: {
      height: theme.general.heightSm
    },
    css: getTableFilterInputStyles(theme, DEFAULT_WIDTH),
    ...containerProps,
    children: onSubmit ? jsx("form", {
      onSubmit: e => {
        e.preventDefault();
        onSubmit();
      },
      children: component
    }) : component
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$1() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref = process.env.NODE_ENV === "production" ? {
  name: "12vohjb",
  styles: "display:flex;flex-wrap:wrap;gap:var(--table-filter-layout-item-gap);align-self:flex-start"
} : {
  name: "djfh3-filterActions",
  styles: "display:flex;flex-wrap:wrap;gap:var(--table-filter-layout-item-gap);align-self:flex-start;label:filterActions;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$1
};
var _ref2$1 = process.env.NODE_ENV === "production" ? {
  name: "199bc0o",
  styles: "display:flex;flex-wrap:wrap;flex-direction:row;align-items:center;gap:var(--table-filter-layout-item-gap);margin-right:var(--table-filter-layout-group-margin);flex:1"
} : {
  name: "5r5dlb-filters",
  styles: "display:flex;flex-wrap:wrap;flex-direction:row;align-items:center;gap:var(--table-filter-layout-item-gap);margin-right:var(--table-filter-layout-group-margin);flex:1;label:filters;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$1
};
var _ref3 = process.env.NODE_ENV === "production" ? {
  name: "1fjm2n4",
  styles: "display:flex;flex-direction:row;justify-content:space-between;margin-bottom:var(--table-filter-layout-group-margin);column-gap:var(--table-filter-layout-group-margin);row-gap:var(--table-filter-layout-item-gap);flex-wrap:wrap"
} : {
  name: "181q9wu-layout",
  styles: "display:flex;flex-direction:row;justify-content:space-between;margin-bottom:var(--table-filter-layout-group-margin);column-gap:var(--table-filter-layout-group-margin);row-gap:var(--table-filter-layout-item-gap);flex-wrap:wrap;label:layout;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$1
};
const TableFilterLayout = /*#__PURE__*/forwardRef(function TableFilterLayout(_ref4, ref) {
  let {
    children,
    style,
    className,
    actions,
    ...rest
  } = _ref4;
  const {
    theme
  } = useDesignSystemTheme();
  const tableFilterLayoutStyles = {
    layout: _ref3,
    filters: _ref2$1,
    filterActions: _ref
  };
  return jsxs("div", {
    ...rest,
    ref: ref,
    style: {
      ['--table-filter-layout-item-gap']: `${theme.spacing.sm}px`,
      ['--table-filter-layout-group-margin']: `${theme.spacing.md}px`,
      ...style
    },
    css: tableFilterLayoutStyles.layout,
    className: className,
    children: [jsx("div", {
      css: tableFilterLayoutStyles.filters,
      children: children
    }), actions && jsx("div", {
      css: tableFilterLayoutStyles.filterActions,
      children: actions
    })]
  });
});

const TableHeaderResizeHandle = /*#__PURE__*/forwardRef(function TableHeaderResizeHandle(_ref, ref) {
  let {
    style,
    resizeHandler,
    children,
    ...rest
  } = _ref;
  const {
    isHeader
  } = useContext(TableRowContext);
  if (!isHeader) {
    throw new Error('`TableHeaderResizeHandle` must be used within a `TableRow` with `isHeader` set to true.');
  }
  return jsx("div", {
    ...rest,
    ref: ref,
    onPointerDown: resizeHandler,
    css: tableStyles$1.resizeHandleContainer,
    style: style,
    role: "separator",
    children: jsx("div", {
      css: tableStyles$1.resizeHandle
    })
  });
});
const TableHeader = /*#__PURE__*/forwardRef(function TableHeader(_ref2, ref) {
  let {
    children,
    ellipsis = false,
    multiline = false,
    sortable,
    sortDirection,
    onToggleSort,
    style,
    className,
    resizable,
    resizeHandler,
    isResizing = false,
    align = 'left',
    wrapContent = true,
    componentId,
    analyticsEvents,
    ...rest
  } = _ref2;
  const {
    size,
    grid
  } = useContext(TableContext);
  const {
    isHeader
  } = useContext(TableRowContext);
  const [currentSortDirection, setCurrentSortDirection] = useState(sortDirection);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.TableHeader,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii: true
  });
  if (!isHeader) {
    throw new Error('`TableHeader` must be used within a `TableRow` with `isHeader` set to true.');
  }
  let sortIcon = jsx(Fragment, {});
  // While most libaries use `asc` and `desc` for the sort value, the ARIA spec
  // uses `ascending` and `descending`.
  let ariaSort;
  if (sortable) {
    if (sortDirection === 'asc') {
      sortIcon = jsx(SortAscendingIcon$1, {});
      ariaSort = 'ascending';
    } else if (sortDirection === 'desc') {
      sortIcon = jsx(SortDescendingIcon$1, {});
      ariaSort = 'descending';
    } else if (sortDirection === 'none') {
      sortIcon = jsx(SortUnsortedIcon$1, {});
      ariaSort = 'none';
    }
  }
  useEffect(() => {
    if (sortDirection !== currentSortDirection) {
      setCurrentSortDirection(sortDirection);
      eventContext.onValueChange(sortDirection);
    }
  }, [sortDirection, currentSortDirection, eventContext]);
  const sortIconOnLeft = align === 'right';
  let typographySize = 'md';
  if (size === 'small') {
    typographySize = 'sm';
  }
  const content = wrapContent ? jsx(Typography.Text, {
    className: "table-header-text",
    ellipsis: !multiline,
    size: typographySize,
    title: !multiline && typeof children === 'string' && children || undefined,
    bold: true,
    children: children
  }) : children;
  const resizeHandle = resizable ? jsx(TableHeaderResizeHandle, {
    resizeHandler: resizeHandler
  }) : null;
  return jsxs("div", {
    ...rest,
    ref: ref
    // PE-259 Use more performance className for grid but keep css= for compatibility.
    ,
    css: !grid ? [repeatingElementsStyles.cell, repeatingElementsStyles.header] : undefined,
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
    children: [sortable && !isResizing ? jsxs("div", {
      css: [tableStyles$1.headerButtonTarget, process.env.NODE_ENV === "production" ? "" : ";label:TableHeader;"],
      role: "button",
      tabIndex: 0,
      onClick: onToggleSort,
      onKeyDown: event => {
        if (sortable && (event.key === 'Enter' || event.key === ' ')) {
          event.preventDefault();
          return onToggleSort === null || onToggleSort === void 0 ? void 0 : onToggleSort(event);
        }
      },
      children: [sortIconOnLeft ? jsx("span", {
        className: "table-header-icon-container",
        css: [tableStyles$1.sortHeaderIconOnLeft, process.env.NODE_ENV === "production" ? "" : ";label:TableHeader;"],
        children: sortIcon
      }) : null, content, !sortIconOnLeft ? jsx("span", {
        className: "table-header-icon-container",
        css: [tableStyles$1.sortHeaderIconOnRight, process.env.NODE_ENV === "production" ? "" : ";label:TableHeader;"],
        children: sortIcon
      }) : null]
    }) : content, resizeHandle]
  });
});

const TableRowActionHeader = _ref => {
  let {
    children
  } = _ref;
  return jsx(TableRowAction, {
    children: jsx("span", {
      css: visuallyHidden,
      children: children
    })
  });
};

const TableRowSelectCell = /*#__PURE__*/forwardRef(function TableRowSelectCell(_ref, ref) {
  let {
    onChange,
    checked,
    indeterminate,
    noCheckbox,
    children,
    isDisabled,
    checkboxLabel,
    componentId,
    analyticsEvents,
    ...rest
  } = _ref;
  const {
    isHeader
  } = useContext(TableRowContext);
  const {
    someRowsSelected
  } = useContext(TableContext);
  if (typeof someRowsSelected === 'undefined') {
    throw new Error('`TableRowSelectCell` cannot be used unless `someRowsSelected` has been provided to the `Table` component, see documentation.');
  }
  if (!isHeader && indeterminate) {
    throw new Error('`TableRowSelectCell` cannot be used with `indeterminate` in a non-header row.');
  }
  return jsx("div", {
    ...rest,
    ref: ref,
    css: tableStyles$1.checkboxCell,
    style: {
      ['--row-checkbox-opacity']: someRowsSelected ? 1 : 0
    },
    role: isHeader ? 'columnheader' : 'cell'
    // TODO: Ideally we shouldn't need to specify this `className`, but it allows for row-hovering to reveal
    // the checkbox in `TableRow`'s CSS without extra JS pointerin/out events.
    ,
    className: "table-row-select-cell",
    children: !noCheckbox && jsx(Checkbox, {
      componentId: componentId,
      analyticsEvents: analyticsEvents,
      isChecked: checked || indeterminate && null,
      onChange: (_checked, event) => onChange === null || onChange === void 0 ? void 0 : onChange(event.nativeEvent),
      isDisabled: isDisabled,
      "aria-label": checkboxLabel
    })
  });
});

const getTabEmotionStyles = (clsPrefix, theme) => {
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
      outlineColor: theme.colors.primary,
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
        borderColor: theme.colors.borderDecorative
      }
    },
    [classCloseButton]: {
      height: 24,
      width: 24,
      padding: 6,
      borderRadius: theme.borders.borderRadiusMd,
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
        outlineColor: theme.colors.primary
      }
    },
    [classAddButton]: {
      backgroundColor: 'transparent',
      color: theme.colors.textValidationInfo,
      border: 'none',
      borderRadius: theme.borders.borderRadiusMd,
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
        outlineColor: theme.colors.primary
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
const TabPane = _ref => {
  let {
    children,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Tabs$1.TabPane, {
      closeIcon: jsx(CloseIcon, {
        css: /*#__PURE__*/css({
          fontSize: theme.general.iconSize
        }, process.env.NODE_ENV === "production" ? "" : ";label:TabPane;")
      })
      // Note: this component must accept the entire `props` object and spread it here, because Ant's Tabs components
      // injects extra props here (at the time of writing, `prefixCls`, `tabKey` and `id`).
      // However, we use a restricted TS interface to still discourage consumers of the library from passing in these props.
      ,
      ...props,
      ...props.dangerouslySetAntdProps,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
};
const Tabs = /* #__PURE__ */(() => {
  const Tabs = _ref2 => {
    let {
      editable = false,
      activeKey,
      defaultActiveKey,
      onChange,
      onEdit,
      children,
      destroyInactiveTabPane = false,
      dangerouslySetAntdProps = {},
      dangerouslyAppendEmotionCSS = {},
      ...props
    } = _ref2;
    const {
      theme,
      classNamePrefix
    } = useDesignSystemTheme();
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Tabs$1, {
        ...addDebugOutlineIfEnabled(),
        activeKey: activeKey,
        defaultActiveKey: defaultActiveKey,
        onChange: onChange,
        onEdit: onEdit,
        destroyInactiveTabPane: destroyInactiveTabPane,
        type: editable ? 'editable-card' : 'card',
        addIcon: jsx(PlusIcon, {
          css: /*#__PURE__*/css({
            fontSize: theme.general.iconSize
          }, process.env.NODE_ENV === "production" ? "" : ";label:Tabs;")
        }),
        css: [getTabEmotionStyles(classNamePrefix, theme), importantify(dangerouslyAppendEmotionCSS), process.env.NODE_ENV === "production" ? "" : ";label:Tabs;"],
        ...dangerouslySetAntdProps,
        ...props,
        children: children
      })
    });
  };
  Tabs.TabPane = TabPane;
  return Tabs;
})();

const SMALL_BUTTON_HEIGHT = 24;
const getStyles = (theme, size, onlyIcon) => {
  return /*#__PURE__*/css({
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    whiteSpace: 'nowrap',
    border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
    borderRadius: theme.general.borderRadiusBase,
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
      '& > svg': {
        stroke: theme.colors.border
      }
    },
    ...(!onlyIcon && {
      '&&': {
        padding: '4px 12px',
        ...(size === 'small' && {
          padding: '0 8px'
        })
      }
    }),
    ...(onlyIcon && {
      width: theme.general.heightSm,
      border: 'none'
    }),
    ...(size === 'small' && {
      height: SMALL_BUTTON_HEIGHT,
      lineHeight: theme.typography.lineHeightBase,
      ...(onlyIcon && {
        width: SMALL_BUTTON_HEIGHT,
        paddingTop: 0,
        paddingBottom: 0,
        verticalAlign: 'middle'
      })
    })
  }, process.env.NODE_ENV === "production" ? "" : ";label:getStyles;");
};
const RectangleSvg = props => jsx("svg", {
  width: "16",
  height: "16",
  viewBox: "0 0 16 16",
  fill: "none",
  xmlns: "http://www.w3.org/2000/svg",
  ...props,
  children: jsx("rect", {
    x: "0.5",
    y: "0.5",
    width: "15",
    height: "15",
    rx: "3.5"
  })
});
const RectangleIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: RectangleSvg
  });
});
const ToggleButton = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    children,
    pressed,
    defaultPressed,
    icon,
    size = 'middle',
    componentId,
    analyticsEvents,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const [isPressed, setIsPressed] = React__default.useState(defaultPressed);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.ToggleButton,
    componentId: componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii: true
  });
  const handleOnPressedChange = useCallback(pressed => {
    var _props$onPressedChang;
    eventContext.onValueChange(pressed);
    (_props$onPressedChang = props.onPressedChange) === null || _props$onPressedChang === void 0 || _props$onPressedChang.call(props, pressed);
    setIsPressed(pressed);
  }, [eventContext, props]);
  useEffect(() => {
    setIsPressed(pressed);
  }, [pressed]);
  const iconOnly = !children && Boolean(icon);
  const iconStyle = iconOnly ? {} : {
    marginRight: theme.spacing.xs
  };
  const checkboxIcon = isPressed ? jsx(CheckIcon, {}) : jsx(RectangleIcon, {
    css: /*#__PURE__*/css({
      stroke: theme.colors.border
    }, process.env.NODE_ENV === "production" ? "" : ";label:checkboxIcon;")
  });
  return jsxs(Toggle.Root, {
    ...addDebugOutlineIfEnabled(),
    css: getStyles(theme, size, iconOnly),
    ...props,
    pressed: isPressed,
    onPressedChange: handleOnPressedChange,
    ref: ref,
    children: [jsx("span", {
      className: "togglebutton-icon-wrapper",
      style: {
        display: 'flex',
        ...iconStyle
      },
      children: icon ? icon : checkboxIcon
    }), children]
  });
});

const InfoTooltip = _ref => {
  let {
    content,
    iconTitle = 'More information',
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Tooltip$1, {
    content: content,
    ...props,
    children: jsx(InfoCircleOutlined, {
      tabIndex: 0,
      "aria-hidden": "false",
      "aria-label": iconTitle,
      alt: iconTitle,
      css: /*#__PURE__*/css({
        fontSize: theme.typography.fontSizeSm,
        color: theme.colors.textSecondary
      }, process.env.NODE_ENV === "production" ? "" : ";label:InfoTooltip;")
    })
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const hideLinesForSizes = ['x-small', 'xx-small'];
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
 */
function getTreeCheckboxEmotionStyles(clsPrefix, theme) {
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
function getTreeEmotionStyles(clsPrefix, theme, size) {
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
        borderRadius: theme.borders.borderRadiusMd
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
      outlineColor: theme.colors.primary,
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
    [`${classIndent}:before`]: {
      height: '100%'
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
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getTreeEmotionStyles;");
}
const SHOW_LINE_DEFAULT = {
  showLeafIcon: false
};

// @ts-expect-error: Tree doesn't expose a proper type
var _ref2 = process.env.NODE_ENV === "production" ? {
  name: "1gn554q",
  styles: "font-size:16px !important"
} : {
  name: "1afpwuk-Tree",
  styles: "font-size:16px !important;label:Tree;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__
};
const Tree = /*#__PURE__*/forwardRef(function Tree(_ref, ref) {
  let {
    treeData,
    defaultExpandedKeys,
    defaultSelectedKeys,
    defaultCheckedKeys,
    disabled = false,
    mode = 'default',
    size = 'default',
    showLine,
    dangerouslySetAntdProps,
    ...props
  } = _ref;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  let calculatedShowLine = showLine !== null && showLine !== void 0 ? showLine : false;
  if (hideLinesForSizes.includes(size)) {
    calculatedShowLine = false;
  } else {
    calculatedShowLine = showLine !== null && showLine !== void 0 ? showLine : SHOW_LINE_DEFAULT;
  }
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Tree$1, {
      ...addDebugOutlineIfEnabled(),
      treeData: treeData,
      defaultExpandedKeys: defaultExpandedKeys,
      defaultSelectedKeys: defaultSelectedKeys,
      defaultCheckedKeys: defaultCheckedKeys,
      disabled: disabled,
      css: getTreeEmotionStyles(classNamePrefix, theme, size),
      switcherIcon: jsx(ChevronDownIcon$1, {
        css: _ref2
      }),
      tabIndex: 0,
      selectable: mode === 'selectable' || mode === 'multiselectable',
      checkable: mode === 'checkable',
      multiple: mode === 'multiselectable'
      // With the library flag, defaults to showLine = true. The status quo default is showLine = false.
      ,
      showLine: calculatedShowLine,
      ...dangerouslySetAntdProps,
      ...props,
      ref: ref
    })
  });
});

export { AccessibleContainer, Accordion, AccordionPanel, Alert, AlignCenterIcon$1 as AlignCenterIcon, AlignLeftIcon$1 as AlignLeftIcon, AlignRightIcon$1 as AlignRightIcon, AppIcon$1 as AppIcon, ApplyDesignSystemContextOverrides, ApplyGlobalStyles, ArrowDownDotIcon$1 as ArrowDownDotIcon, ArrowDownIcon$1 as ArrowDownIcon, ArrowInIcon$1 as ArrowInIcon, ArrowLeftIcon$1 as ArrowLeftIcon, ArrowOverIcon$1 as ArrowOverIcon, ArrowRightIcon$1 as ArrowRightIcon, ArrowUpDotIcon$1 as ArrowUpDotIcon, ArrowUpIcon$1 as ArrowUpIcon, ArrowsUpDownIcon$1 as ArrowsUpDownIcon, AssistantAvatar, AssistantIcon$1 as AssistantIcon, AutoComplete, Avatar, AvatarGroup, BadgeCodeIcon$1 as BadgeCodeIcon, BadgeCodeOffIcon$1 as BadgeCodeOffIcon, BarChartIcon$1 as BarChartIcon, BarGroupedIcon$1 as BarGroupedIcon, BarStackedIcon$1 as BarStackedIcon, BarStackedPercentageIcon$1 as BarStackedPercentageIcon, BeakerIcon$1 as BeakerIcon, BinaryIcon$1 as BinaryIcon, BlockQuoteIcon$1 as BlockQuoteIcon, BoldIcon$1 as BoldIcon, BookIcon$1 as BookIcon, BookmarkFillIcon$1 as BookmarkFillIcon, BookmarkIcon$1 as BookmarkIcon, BooksIcon$1 as BooksIcon, BracketsCurlyIcon$1 as BracketsCurlyIcon, BracketsSquareIcon$1 as BracketsSquareIcon, BracketsXIcon$1 as BracketsXIcon, BranchIcon$1 as BranchIcon, Breadcrumb, BriefcaseFillIcon$1 as BriefcaseFillIcon, BriefcaseIcon$1 as BriefcaseIcon, BugIcon$1 as BugIcon, Button, CalendarClockIcon$1 as CalendarClockIcon, CalendarEventIcon$1 as CalendarEventIcon, CalendarIcon$1 as CalendarIcon, Card, CaretDownSquareIcon$1 as CaretDownSquareIcon, CaretUpSquareIcon$1 as CaretUpSquareIcon, CatalogCloudIcon$1 as CatalogCloudIcon, CatalogGearIcon$1 as CatalogGearIcon, CatalogHIcon$1 as CatalogHIcon, CatalogHomeIcon$1 as CatalogHomeIcon, CatalogIcon$1 as CatalogIcon, CatalogOffIcon$1 as CatalogOffIcon, CatalogSharedIcon$1 as CatalogSharedIcon, ChainIcon$1 as ChainIcon, ChartLineIcon$1 as ChartLineIcon, CheckCircleBadgeIcon$1 as CheckCircleBadgeIcon, CheckCircleFillIcon$1 as CheckCircleFillIcon, CheckCircleIcon$1 as CheckCircleIcon, CheckIcon, CheckLineIcon$1 as CheckLineIcon, CheckSmallIcon$1 as CheckSmallIcon, Checkbox, CheckboxIcon$1 as CheckboxIcon, ChecklistIcon$1 as ChecklistIcon, ChevronDoubleDownIcon$1 as ChevronDoubleDownIcon, ChevronDoubleLeftIcon$1 as ChevronDoubleLeftIcon, ChevronDoubleRightIcon$1 as ChevronDoubleRightIcon, ChevronDoubleUpIcon$1 as ChevronDoubleUpIcon, ChevronDownIcon$1 as ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon, ChevronUpIcon$1 as ChevronUpIcon, CircleIcon$1 as CircleIcon, CircleOfflineIcon$1 as CircleOfflineIcon, CircleOutlineIcon$1 as CircleOutlineIcon, ClipboardIcon$1 as ClipboardIcon, ClockKeyIcon$1 as ClockKeyIcon, CloseIcon, CloudDatabaseIcon$1 as CloudDatabaseIcon, CloudDownloadIcon$1 as CloudDownloadIcon, CloudIcon$1 as CloudIcon, CloudKeyIcon$1 as CloudKeyIcon, CloudModelIcon$1 as CloudModelIcon, CloudOffIcon$1 as CloudOffIcon, CloudUploadIcon$1 as CloudUploadIcon, CodeIcon$1 as CodeIcon, Col, ColorFillIcon$1 as ColorFillIcon, ColumnIcon$1 as ColumnIcon, ColumnsIcon$1 as ColumnsIcon, CommandPaletteIcon$1 as CommandPaletteIcon, ConnectIcon$1 as ConnectIcon, ContextMenu$1 as ContextMenu, CopyIcon$1 as CopyIcon, CursorIcon$1 as CursorIcon, CursorPagination, CursorTypeIcon$1 as CursorTypeIcon, DBAssistantAvatar, DIcon$1 as DIcon, DU_BOIS_ENABLE_ANIMATION_CLASSNAME, DagIcon$1 as DagIcon, DangerFillIcon$1 as DangerFillIcon, DangerIcon, DashIcon$1 as DashIcon, DashboardIcon$1 as DashboardIcon, DataIcon$1 as DataIcon, DatabaseIcon$1 as DatabaseIcon, DecimalIcon$1 as DecimalIcon, DesignSystemAntDConfigProvider, DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, DesignSystemEventSuppressInteractionProviderContext, DesignSystemEventSuppressInteractionTrueContextValue, DialogCombobox, DialogComboboxAddButton, DialogComboboxContent, DialogComboboxCountBadge, DialogComboboxCustomButtonTriggerWrapper, EmptyResults as DialogComboboxEmpty, DialogComboboxFooter, DialogComboboxHintRow, DialogComboboxOptionControlledList, DialogComboboxOptionList, DialogComboboxOptionListCheckboxItem, DialogComboboxOptionListSearch, DialogComboboxOptionListSelectItem, DialogComboboxSectionHeader, DialogComboboxSeparator, DialogComboboxTrigger, DotsCircleIcon$1 as DotsCircleIcon, DownloadIcon$1 as DownloadIcon, DragIcon$1 as DragIcon, Drawer, Dropdown, DropdownMenu, DuboisDatePicker, Empty, ErdIcon$1 as ErdIcon, ExpandLessIcon$1 as ExpandLessIcon, ExpandMoreIcon$1 as ExpandMoreIcon, FaceFrownIcon$1 as FaceFrownIcon, FaceNeutralIcon$1 as FaceNeutralIcon, FaceSmileIcon$1 as FaceSmileIcon, FileCodeIcon$1 as FileCodeIcon, FileDocumentIcon$1 as FileDocumentIcon, FileIcon$1 as FileIcon, FileImageIcon$1 as FileImageIcon, FileModelIcon$1 as FileModelIcon, FilterIcon$1 as FilterIcon, FlagPointerIcon$1 as FlagPointerIcon, FloatIcon$1 as FloatIcon, FolderBranchFillIcon$1 as FolderBranchFillIcon, FolderBranchIcon$1 as FolderBranchIcon, FolderCloudFilledIcon$1 as FolderCloudFilledIcon, FolderCloudIcon$1 as FolderCloudIcon, FolderFillIcon$1 as FolderFillIcon, FolderIcon$1 as FolderIcon, FontIcon$1 as FontIcon, ForkIcon$1 as ForkIcon, Form, FormDubois, FormUI, FullscreenExitIcon$1 as FullscreenExitIcon, FullscreenIcon$1 as FullscreenIcon, FunctionIcon$1 as FunctionIcon, GearFillIcon$1 as GearFillIcon, GearIcon$1 as GearIcon, GenericSkeleton, GiftIcon$1 as GiftIcon, GitCommitIcon$1 as GitCommitIcon, GlobeIcon$1 as GlobeIcon, GridDashIcon$1 as GridDashIcon, GridIcon$1 as GridIcon, H1Icon$1 as H1Icon, H2Icon$1 as H2Icon, H3Icon$1 as H3Icon, Header$1 as Header, HistoryIcon$1 as HistoryIcon, HomeIcon$1 as HomeIcon, HoverCard, Icon, ImageIcon$1 as ImageIcon, IndentDecreaseIcon$1 as IndentDecreaseIcon, IndentIncreaseIcon$1 as IndentIncreaseIcon, InfinityIcon$1 as InfinityIcon, InfoBookIcon$1 as InfoBookIcon, InfoFillIcon$1 as InfoFillIcon, InfoIcon, InfoPopover, InfoTooltip, IngestionIcon$1 as IngestionIcon, Input, ItalicIcon$1 as ItalicIcon, KeyIcon$1 as KeyIcon, KeyboardIcon$1 as KeyboardIcon, LayerGraphIcon$1 as LayerGraphIcon, LayerIcon$1 as LayerIcon, Layout, LeafIcon$1 as LeafIcon, LegacyDatePicker, LegacyInfoTooltip, LegacyOptGroup, LegacyOption, LegacyPopover, LegacySelect, LegacySelectOptGroup, LegacySelectOption, LegacySkeleton, LegacyTable, LegacyTooltip, LettersIcon$1 as LettersIcon, LibrariesIcon$1 as LibrariesIcon, LightningIcon$1 as LightningIcon, LinkIcon$1 as LinkIcon, LinkOffIcon$1 as LinkOffIcon, ListBorderIcon$1 as ListBorderIcon, ListClearIcon$1 as ListClearIcon, ListIcon, LoadingIcon, LoadingState, LockFillIcon$1 as LockFillIcon, LockIcon$1 as LockIcon, LockUnlockedIcon$1 as LockUnlockedIcon, MIcon$1 as MIcon, MailIcon$1 as MailIcon, Menu, MenuIcon$1 as MenuIcon, MinusBoxIcon$1 as MinusBoxIcon, MinusCircleFillIcon$1 as MinusCircleFillIcon, MinusCircleIcon$1 as MinusCircleIcon, ModelsIcon$1 as ModelsIcon, NoIcon$1 as NoIcon, NotebookIcon$1 as NotebookIcon, Notification, NotificationIcon$1 as NotificationIcon, NotificationOffIcon$1 as NotificationOffIcon, NumbersIcon$1 as NumbersIcon, OfficeIcon$1 as OfficeIcon, Overflow, OverflowIcon$1 as OverflowIcon, PageBottomIcon$1 as PageBottomIcon, PageFirstIcon$1 as PageFirstIcon, PageLastIcon$1 as PageLastIcon, PageTopIcon$1 as PageTopIcon, PageWrapper, Pagination, ParagraphSkeleton, PauseIcon$1 as PauseIcon, PencilIcon$1 as PencilIcon, PieChartIcon$1 as PieChartIcon, PinCancelIcon$1 as PinCancelIcon, PinFillIcon$1 as PinFillIcon, PinIcon$1 as PinIcon, PipelineIcon$1 as PipelineIcon, PlayCircleFillIcon$1 as PlayCircleFillIcon, PlayCircleIcon$1 as PlayCircleIcon, PlayIcon$1 as PlayIcon, PlugIcon$1 as PlugIcon, PlusCircleFillIcon$1 as PlusCircleFillIcon, PlusCircleIcon$1 as PlusCircleIcon, PlusIcon, PlusSquareIcon$1 as PlusSquareIcon, QueryEditorIcon$1 as QueryEditorIcon, QueryIcon$1 as QueryIcon, QuestionMarkFillIcon$1 as QuestionMarkFillIcon, QuestionMarkIcon$1 as QuestionMarkIcon, RHFControlledComponents, ROW_GUTTER_SIZE, Radio, ReaderModeIcon$1 as ReaderModeIcon, RedoIcon$1 as RedoIcon, RefreshIcon$1 as RefreshIcon, ReplyIcon$1 as ReplyIcon, ResourceStatusIndicator, RestoreAntDDefaultClsPrefix, Robot2Icon$1 as Robot2Icon, RobotIcon$1 as RobotIcon, RocketIcon$1 as RocketIcon, Row, RunningIcon$1 as RunningIcon, SaveIcon$1 as SaveIcon, SchemaIcon$1 as SchemaIcon, SchoolIcon$1 as SchoolIcon, SearchDataIcon$1 as SearchDataIcon, SearchIcon$1 as SearchIcon, SecurityIcon$1 as SecurityIcon, SegmentedControlButton, SegmentedControlGroup, Select, SelectContent, SelectOption, SelectOptionGroup, SelectTrigger, SendIcon$1 as SendIcon, ShareIcon$1 as ShareIcon, ShieldIcon$1 as ShieldIcon, ShortcutIcon$1 as ShortcutIcon, SidebarAutoIcon$1 as SidebarAutoIcon, SidebarCollapseIcon$1 as SidebarCollapseIcon, SidebarExpandIcon$1 as SidebarExpandIcon, SidebarIcon$1 as SidebarIcon, SimpleSelect, SimpleSelectOption, SimpleSelectOptionGroup, SlidersIcon$1 as SlidersIcon, SortAlphabeticalAscendingIcon$1 as SortAlphabeticalAscendingIcon, SortAlphabeticalDescendingIcon$1 as SortAlphabeticalDescendingIcon, SortAlphabeticalLeftIcon$1 as SortAlphabeticalLeftIcon, SortAlphabeticalRightIcon$1 as SortAlphabeticalRightIcon, SortAscendingIcon$1 as SortAscendingIcon, SortDescendingIcon$1 as SortDescendingIcon, SortUnsortedIcon$1 as SortUnsortedIcon, Space, Spacer, SparkleDoubleFillIcon$1 as SparkleDoubleFillIcon, SparkleDoubleIcon$1 as SparkleDoubleIcon, SparkleFillIcon$1 as SparkleFillIcon, SparkleIcon$1 as SparkleIcon, SparkleRectangleIcon$1 as SparkleRectangleIcon, SpeechBubbleIcon$1 as SpeechBubbleIcon, SpeechBubblePlusIcon$1 as SpeechBubblePlusIcon, SpeechBubbleQuestionMarkIcon$1 as SpeechBubbleQuestionMarkIcon, SpeechBubbleStarIcon$1 as SpeechBubbleStarIcon, Spinner, SplitButton, StarFillIcon$1 as StarFillIcon, StarIcon$1 as StarIcon, Steps, StopCircleFillIcon$1 as StopCircleFillIcon, StopCircleIcon$1 as StopCircleIcon, StopIcon$1 as StopIcon, StorefrontIcon$1 as StorefrontIcon, StreamIcon$1 as StreamIcon, Switch, SyncIcon$1 as SyncIcon, TabPane, Table, TableCell, TableContext, TableFilterInput, TableFilterLayout, TableGlassesIcon$1 as TableGlassesIcon, TableGlobeIcon$1 as TableGlobeIcon, TableHeader, TableIcon$1 as TableIcon, TableLightningIcon$1 as TableLightningIcon, TableModelIcon$1 as TableModelIcon, TableOnlineViewIcon$1 as TableOnlineViewIcon, TableRow, TableRowAction, TableRowActionHeader, TableRowContext, TableRowMenuContainer, TableRowSelectCell, TableSkeleton, TableSkeletonRows, TableStreamIcon$1 as TableStreamIcon, TableVectorIcon$1 as TableVectorIcon, TableViewIcon$1 as TableViewIcon, Tabs, Tag, TagIcon$1 as TagIcon, TargetIcon$1 as TargetIcon, TerminalIcon$1 as TerminalIcon, TextBoxIcon$1 as TextBoxIcon, TextUnderlineIcon$1 as TextUnderlineIcon, ThumbsDownIcon$1 as ThumbsDownIcon, ThumbsUpIcon$1 as ThumbsUpIcon, TitleSkeleton, ToggleButton, Tooltip$1 as Tooltip, TrashIcon$1 as TrashIcon, Tree, TreeIcon$1 as TreeIcon, TypeaheadComboboxAddButton, TypeaheadComboboxCheckboxItem, TypeaheadComboboxFooter, TypeaheadComboboxInput, TypeaheadComboboxMenu, TypeaheadComboboxMenuItem, TypeaheadComboboxMultiSelectInput, TypeaheadComboboxMultiSelectStateChangeTypes, TypeaheadComboboxRoot, TypeaheadComboboxSectionHeader, TypeaheadComboboxSelectedItem, TypeaheadComboboxSeparator, TypeaheadComboboxStateChangeTypes, TypeaheadComboboxToggleButton, Typography, UnderlineIcon$1 as UnderlineIcon, UndoIcon$1 as UndoIcon, UploadIcon$1 as UploadIcon, UsbIcon$1 as UsbIcon, UserBadgeIcon$1 as UserBadgeIcon, UserCircleIcon$1 as UserCircleIcon, UserGroupIcon$1 as UserGroupIcon, UserIcon$1 as UserIcon, VisibleIcon$1 as VisibleIcon, VisibleOffIcon$1 as VisibleOffIcon, WarningFillIcon$1 as WarningFillIcon, WarningIcon, WorkflowsIcon$1 as WorkflowsIcon, WorkspacesIcon$1 as WorkspacesIcon, WrenchIcon$1 as WrenchIcon, XCircleFillIcon$1 as XCircleFillIcon, XCircleIcon$1 as XCircleIcon, ZoomInIcon$1 as ZoomInIcon, ZoomMarqueeSelection$1 as ZoomMarqueeSelection, ZoomOutIcon$1 as ZoomOutIcon, __INTERNAL_DO_NOT_USE__FormItem, __INTERNAL_DO_NOT_USE__Group, __INTERNAL_DO_NOT_USE__HorizontalGroup, __INTERNAL_DO_NOT_USE__VerticalGroup, dialogComboboxLookAheadKeyDown, findClosestOptionSibling, findHighlightedOption, getAnimationCss, getComboboxOptionItemWrapperStyles, getComboboxOptionLabelStyles, getContentOptions, getDarkModePortalStyles, getDialogComboboxOptionLabelWidth, getGlobalStyles, getKeyboardNavigationFunctions, getPaginationEmotionStyles, getRadioStyles, getShadowScrollStyles, getTabEmotionStyles, getValidationStateColor, getWrapperStyle, highlightFirstNonDisabledOption, highlightOption, importantify, isOptionDisabled, resetTabIndexToFocusedElement, useComboboxState, useDesignSystemEventComponentCallbacks, useDesignSystemFlags, useDesignSystemTheme, useLegacyNotification, useModalContext, useMultipleSelectionState, useNotifyOnFirstView, useThemedStyles, useTypeaheadComboboxContext, visuallyHidden, withNotifications };
//# sourceMappingURL=index.js.map

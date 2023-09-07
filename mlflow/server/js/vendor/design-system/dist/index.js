import * as React from 'react';
import React__default, { useRef, useMemo, forwardRef, createContext, useContext, useState, useEffect, useImperativeHandle, Children, useCallback, useLayoutEffect } from 'react';
import { u as useDesignSystemTheme, I as Icon, D as DesignSystemAntDConfigProvider, R as RestoreAntDDefaultClsPrefix, g as getAnimationCss, C as CloseIcon, a as ChevronRightIcon, b as useDesignSystemFlags, i as importantify, S as Spinner, c as useDesignSystemContext, T as Tooltip, d as InfoIcon, e as Input, f as CheckIcon, B as Button, h as getValidationStateColor, A as ApplyDesignSystemContextOverrides, j as Typography, s as safex, L as LoadingState, k as LoadingIcon, l as dropdownContentStyles, m as Title$2, n as AccessibleContainer, o as ChevronLeftIcon, p as DU_BOIS_ENABLE_ANIMATION_CLASSNAME, v as visuallyHidden, q as Root$4, r as Trigger$2, t as getDefaultStyles, w as getPrimaryStyles, x as getDisabledStyles, y as lightColorList } from './DropdownMenu-d39d9283.js';
export { J as ApplyDesignSystemFlags, N as CursorIcon, F as DesignSystemContext, H as DesignSystemProvider, E as DesignSystemThemeContext, G as DesignSystemThemeProvider, M as DropdownMenu, O as FaceFrownIcon, P as FaceNeutralIcon, Q as FaceSmileIcon, Y as LoadingStateContext, U as NewWindowIcon, W as WithDesignSystemThemeHoc, X as __INTERNAL_DO_NOT_USE_DEDUPE__Group, V as __INTERNAL_DO_NOT_USE__Password, _ as __INTERNAL_DO_NOT_USE__TextArea, z as getButtonEmotionStyles, Z as getTypographyColor, K as useAntDConfigProviderContext } from './DropdownMenu-d39d9283.js';
import { jsx, jsxs, Fragment } from '@emotion/react/jsx-runtime';
import { css, Global, keyframes, ClassNames, createElement } from '@emotion/react';
import { Collapse, Alert as Alert$1, AutoComplete as AutoComplete$1, Breadcrumb as Breadcrumb$1, Checkbox as Checkbox$1, DatePicker, Dropdown as Dropdown$1, Form as Form$1, Radio as Radio$1, Select as Select$1, Col as Col$1, Row as Row$1, Space as Space$1, Layout as Layout$1, notification, Popover as Popover$2, Skeleton, Pagination as Pagination$1, Table as Table$1, Menu as Menu$1, Modal as Modal$1, Button as Button$1, Switch as Switch$1, Tabs as Tabs$1, Tree as Tree$1 } from 'antd';
import classnames from 'classnames';
import * as Popover$1 from '@radix-ui/react-popover';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { useController } from 'react-hook-form';
import _uniqueId from 'lodash/uniqueId';
import * as Toast from '@radix-ui/react-toast';
import { ResizableBox } from 'react-resizable';
import _times from 'lodash/times';
import _random from 'lodash/random';
import AntDIcon, { InfoCircleOutlined } from '@ant-design/icons';
import * as Toggle from '@radix-ui/react-toggle';
import { useFloating, autoUpdate, offset, flip, shift, useMergeRefs } from '@floating-ui/react';
import { createPortal } from 'react-dom';
import { useCombobox, useMultipleSelection } from 'downshift';
import '@radix-ui/react-dropdown-menu';
import 'chroma-js';
import 'lodash/isNil';
import 'lodash/endsWith';
import 'lodash/isBoolean';
import 'lodash/isNumber';
import 'lodash/isString';
import 'lodash/mapValues';
import '@emotion/unitless';

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
      d: "M1 2.5h14V1H1v1.5ZM11.5 5.75h-7v-1.5h7v1.5ZM15 8.75H1v-1.5h14v1.5ZM15 15H1v-1.5h14V15ZM4.5 11.75h7v-1.5h-7v1.5Z"
    })
  });
}
function AlignCenterIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgAlignCenterIcon
  });
}

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
      d: "M1 2.5h14V1H1v1.5ZM8 5.75H1v-1.5h7v1.5ZM1 8.75v-1.5h14v1.5H1ZM1 15v-1.5h14V15H1ZM1 11.75h7v-1.5H1v1.5Z"
    })
  });
}
function AlignLeftIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgAlignLeftIcon
  });
}

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
      d: "M1 2.5h14V1H1v1.5ZM15 5.75H8v-1.5h7v1.5ZM1 8.75v-1.5h14v1.5H1ZM1 15v-1.5h14V15H1ZM8 11.75h7v-1.5H8v1.5Z"
    })
  });
}
function AlignRightIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgAlignRightIcon
  });
}

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
      d: "M2.75 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5ZM8 1a1.75 1.75 0 1 0 0 3.5A1.75 1.75 0 0 0 8 1Zm5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5ZM2.75 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5Zm5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5Zm5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5ZM2.75 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5Zm5.25 0A1.75 1.75 0 1 0 8 15a1.75 1.75 0 0 0 0-3.5Zm5.25 0a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5Z",
      clipRule: "evenodd"
    })
  });
}
function AppIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgAppIcon
  });
}

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
      d: "M8.03 15.06 1 8.03l1.06-1.06 5.22 5.22V1h1.5v11.19L14 6.97l1.06 1.06-7.03 7.03Z",
      clipRule: "evenodd"
    })
  });
}
function ArrowDownIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgArrowDownIcon
  });
}

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
      d: "M1 8.03 8.03 1l1.061 1.06-5.22 5.22h11.19v1.5H3.87L9.091 14l-1.06 1.06L1 8.03Z",
      clipRule: "evenodd"
    })
  });
}
function ArrowLeftIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgArrowLeftIcon
  });
}

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
      d: "m15.06 8.03-7.03 7.03L6.97 14l5.22-5.22H1v-1.5h11.19L6.97 2.06 8.03 1l7.03 7.03Z",
      clipRule: "evenodd"
    })
  });
}
function ArrowRightIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgArrowRightIcon
  });
}

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
      d: "m8.03 1 7.03 7.03L14 9.091l-5.22-5.22v11.19h-1.5V3.87l-5.22 5.22L1 8.031 8.03 1Z",
      clipRule: "evenodd"
    })
  });
}
function ArrowUpIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgArrowUpIcon
  });
}

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
      d: "M5.03 1 1 5.03l1.06 1.061 2.22-2.22v6.19h1.5V3.87L8 6.091l1.06-1.06L5.03 1ZM11.03 15.121l4.03-4.03-1.06-1.06-2.22 2.219V6.06h-1.5v6.19l-2.22-2.22L7 11.091l4.03 4.03Z"
    })
  });
}
function ArrowsUpDownIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgArrowsUpDownIcon
  });
}

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
      d: "M1 1v13.25c0 .414.336.75.75.75H15v-1.5H2.5V1H1Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M7 1v11h1.5V1H7ZM10 5v7h1.5V5H10ZM4 5v7h1.5V5H4ZM13 12V8h1.5v4H13Z"
    })]
  });
}
function BarChartIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBarChartIcon
  });
}

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
      d: "M12.25 2a.75.75 0 0 0-.75.75V7H9.25a.75.75 0 0 0-.75.75v5.5c0 .414.336.75.75.75h6a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75h-3Zm-.75 10.5v-4H10v4h1.5Zm1.5 0h1.5v-9H13v9ZM3.75 5a.75.75 0 0 0-.75.75V9H.75a.75.75 0 0 0-.75.75v3.5c0 .414.336.75.75.75h6a.75.75 0 0 0 .75-.75v-7.5A.75.75 0 0 0 6.75 5h-3ZM3 12.5v-2H1.5v2H3Zm1.5 0H6v-6H4.5v6Z",
      clipRule: "evenodd"
    })
  });
}
function BarGroupedIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBarGroupedIcon
  });
}

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
      d: "M6.25 1a.75.75 0 0 0-.75.75V7H2.75a.75.75 0 0 0-.75.75v6.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-9.5a.75.75 0 0 0-.75-.75H10.5V1.75A.75.75 0 0 0 9.75 1h-3.5ZM9 8.5v5H7v-5h2ZM9 7V2.5H7V7h2Zm3.5 6.5h-2v-1.75h2v1.75Zm-2-8v4.75h2V5.5h-2Zm-5 4.75V8.5h-2v1.75h2Zm0 3.25v-1.75h-2v1.75h2Z",
      clipRule: "evenodd"
    })
  });
}
function BarStackedIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBarStackedIcon
  });
}

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
      d: "M5.75 1a.75.75 0 0 0-.75.75v6.089c0 .38-.173.739-.47.976l-2.678 2.143A2.27 2.27 0 0 0 3.27 15h9.46a2.27 2.27 0 0 0 1.418-4.042L11.47 8.815A1.25 1.25 0 0 1 11 7.839V1.75a.75.75 0 0 0-.75-.75h-4.5Zm.75 6.839V2.5h3v5.339c0 .606.2 1.188.559 1.661H5.942A2.75 2.75 0 0 0 6.5 7.839ZM4.2 11 2.79 12.13a.77.77 0 0 0 .48 1.37h9.461a.77.77 0 0 0 .481-1.37L11.8 11H4.201Z",
      clipRule: "evenodd"
    })
  });
}
function BeakerIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBeakerIcon
  });
}

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
      d: "M1 3a2 2 0 1 1 4 0v2a2 2 0 1 1-4 0V3Zm2-.5a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0V3a.5.5 0 0 0-.5-.5Zm3.378-.628c.482 0 .872-.39.872-.872h1.5v4.25H10v1.5H6v-1.5h1.25V3.206c-.27.107-.564.166-.872.166H6v-1.5h.378Zm5 0c.482 0 .872-.39.872-.872h1.5v4.25H15v1.5h-4v-1.5h1.25V3.206c-.27.107-.564.166-.872.166H11v-1.5h.378ZM6 11a2 2 0 1 1 4 0v2a2 2 0 1 1-4 0v-2Zm2-.5a.5.5 0 0 0-.5.5v2a.5.5 0 0 0 1 0v-2a.5.5 0 0 0-.5-.5Zm-6.622-.378c.482 0 .872-.39.872-.872h1.5v4.25H5V15H1v-1.5h1.25v-2.044c-.27.107-.564.166-.872.166H1v-1.5h.378Zm10 0c.482 0 .872-.39.872-.872h1.5v4.25H15V15h-4v-1.5h1.25v-2.044c-.27.107-.564.166-.872.166H11v-1.5h.378Z",
      clipRule: "evenodd"
    })
  });
}
function BinaryIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBinaryIcon
  });
}

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
      d: "M4.75 3a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h4.375a2.875 2.875 0 0 0 1.496-5.33A2.875 2.875 0 0 0 8.375 3H4.75Zm.75 5.75v2.75h3.625a1.375 1.375 0 0 0 0-2.75H5.5Zm2.877-1.5a1.375 1.375 0 0 0-.002-2.75H5.5v2.75h2.877Z",
      clipRule: "evenodd"
    })
  });
}
function BoldIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBoldIcon
  });
}

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
      d: "M2.75 1a.75.75 0 0 0-.75.75v13.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H2.75ZM7.5 2.5h-4v6.055l1.495-1.36a.75.75 0 0 1 1.01 0L7.5 8.555V2.5Zm-4 8.082 2-1.818 2.245 2.041A.75.75 0 0 0 9 10.25V2.5h3.5v12h-9v-3.918Z",
      clipRule: "evenodd"
    })
  });
}
function BookIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBookIcon
  });
}

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
      d: "M2.75 0A.75.75 0 0 0 2 .75v14.5a.75.75 0 0 0 1.28.53L8 11.06l4.72 4.72a.75.75 0 0 0 1.28-.53V.75a.75.75 0 0 0-.75-.75H2.75Z"
    })
  });
}
function BookmarkFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBookmarkFillIcon
  });
}

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
      d: "M2 .75A.75.75 0 0 1 2.75 0h10.5a.75.75 0 0 1 .75.75v14.5a.75.75 0 0 1-1.28.53L8 11.06l-4.72 4.72A.75.75 0 0 1 2 15.25V.75Zm1.5.75v11.94l3.97-3.97a.75.75 0 0 1 1.06 0l3.97 3.97V1.5h-9Z",
      clipRule: "evenodd"
    })
  });
}
function BookmarkIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBookmarkIcon
  });
}

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
      d: "M1.5 4.5v10h1v-10h-1ZM1 3a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1H1ZM6.5 1.5v13h2v-13h-2ZM6 0a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1H6Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m11.63 7.74 1.773 6.773.967-.254-1.773-6.771-.967.253Zm-.864-1.324a1 1 0 0 0-.714 1.221l2.026 7.74a1 1 0 0 0 1.22.713l1.936-.506a1 1 0 0 0 .714-1.22l-2.026-7.74a1 1 0 0 0-1.22-.714l-1.936.506Z",
      clipRule: "evenodd"
    })]
  });
}
function BooksIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBooksIcon
  });
}

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
      d: "M5.5 2a2.75 2.75 0 0 0-2.75 2.75v1C2.75 6.44 2.19 7 1.5 7H1v2h.5c.69 0 1.25.56 1.25 1.25v1A2.75 2.75 0 0 0 5.5 14H6v-1.5h-.5c-.69 0-1.25-.56-1.25-1.25v-1c0-.93-.462-1.752-1.168-2.25A2.747 2.747 0 0 0 4.25 5.75v-1c0-.69.56-1.25 1.25-1.25H6V2h-.5ZM13.25 4.75A2.75 2.75 0 0 0 10.5 2H10v1.5h.5c.69 0 1.25.56 1.25 1.25v1c0 .93.462 1.752 1.168 2.25a2.747 2.747 0 0 0-1.168 2.25v1c0 .69-.56 1.25-1.25 1.25H10V14h.5a2.75 2.75 0 0 0 2.75-2.75v-1c0-.69.56-1.25 1.25-1.25h.5V7h-.5c-.69 0-1.25-.56-1.25-1.25v-1Z"
    })
  });
}
function BracketsCurlyIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBracketsCurlyIcon
  });
}

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
      d: "M1 1.75A.75.75 0 0 1 1.75 1H5v1.5H2.5v11H5V15H1.75a.75.75 0 0 1-.75-.75V1.75Zm12.5.75H11V1h3.25a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H11v-1.5h2.5v-11Z",
      clipRule: "evenodd"
    })
  });
}
function BracketsSquareIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBracketsSquareIcon
  });
}

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
        d: "M1.75 4.75A2.75 2.75 0 0 1 4.5 2H5v1.5h-.5c-.69 0-1.25.56-1.25 1.25v1c0 .93-.462 1.752-1.168 2.25a2.747 2.747 0 0 1 1.168 2.25v1c0 .69.56 1.25 1.25 1.25H5V14h-.5a2.75 2.75 0 0 1-2.75-2.75v-1C1.75 9.56 1.19 9 .5 9H0V7h.5c.69 0 1.25-.56 1.25-1.25v-1ZM11.5 2a2.75 2.75 0 0 1 2.75 2.75v1c0 .69.56 1.25 1.25 1.25h.5v2h-.5c-.69 0-1.25.56-1.25 1.25v1A2.75 2.75 0 0 1 11.5 14H11v-1.5h.5c.69 0 1.25-.56 1.25-1.25v-1c0-.93.462-1.752 1.168-2.25a2.747 2.747 0 0 1-1.168-2.25v-1c0-.69-.56-1.25-1.25-1.25H11V2h.5Z"
      }), jsx("path", {
        d: "M4.97 6.03 6.94 8 4.97 9.97l1.06 1.06L8 9.06l1.97 1.97 1.06-1.06L9.06 8l1.97-1.97-1.06-1.06L8 6.94 6.03 4.97 4.97 6.03Z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "BracketsXIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function BracketsXIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBracketsXIcon
  });
}

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
      d: "M1 4a3 3 0 1 1 5.186 2.055 3.229 3.229 0 0 0 2 1.155 3.001 3.001 0 1 1-.152 1.494A4.73 4.73 0 0 1 4.911 6.86a2.982 2.982 0 0 1-.161.046v2.19a3.001 3.001 0 1 1-1.5 0v-2.19A3.001 3.001 0 0 1 1 4Zm3-1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3ZM2.5 12a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Zm7-3.75a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z",
      clipRule: "evenodd"
    })
  });
}
function BranchIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBranchIcon
  });
}

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
      d: "M5 4V2.75C5 1.784 5.784 1 6.75 1h2.5c.966 0 1.75.784 1.75 1.75V4h3.25a.75.75 0 0 1 .75.75v9.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75v-9.5A.75.75 0 0 1 1.75 4H5Zm1.5-1.25a.25.25 0 0 1 .25-.25h2.5a.25.25 0 0 1 .25.25V4h-3V2.75Zm-4 5.423V6.195A7.724 7.724 0 0 0 8 8.485c2.15 0 4.095-.875 5.5-2.29v1.978A9.211 9.211 0 0 1 8 9.985a9.21 9.21 0 0 1-5.5-1.812Z",
      clipRule: "evenodd"
    })
  });
}
function BriefcaseFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBriefcaseFillIcon
  });
}

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
      d: "M1.75 4H5V2.75C5 1.784 5.784 1 6.75 1h2.5c.966 0 1.75.784 1.75 1.75V4h3.25a.75.75 0 0 1 .75.75v9.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75v-9.5A.75.75 0 0 1 1.75 4Zm5-1.5a.25.25 0 0 0-.25.25V4h3V2.75a.25.25 0 0 0-.25-.25h-2.5ZM2.5 8.173V13.5h11V8.173A9.211 9.211 0 0 1 8 9.985a9.21 9.21 0 0 1-5.5-1.812Zm0-1.978A7.724 7.724 0 0 0 8 8.485c2.15 0 4.095-.875 5.5-2.29V5.5h-11v.695Z",
      clipRule: "evenodd"
    })
  });
}
function BriefcaseIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgBriefcaseIcon
  });
}

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
      d: "M4.5 0v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75H6v-1.5H2.5V7H15V2.75a.75.75 0 0 0-.75-.75H11.5V0H10v2H6V0H4.5Zm9 5.5v-2h-11v2h11Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M10.25 10.5V12c0 .199.079.39.22.53l1 1 1.06-1.06-.78-.78V10.5h-1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M7 12a4 4 0 1 1 8 0 4 4 0 0 1-8 0Zm4-2.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5Z",
      clipRule: "evenodd"
    })]
  });
}
function CalendarClockIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCalendarClockIcon
  });
}

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
      d: "M8.5 10.25a1.75 1.75 0 1 1 3.5 0 1.75 1.75 0 0 1-3.5 0Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M10 2H6V0H4.5v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H11.5V0H10v2ZM2.5 3.5v2h11v-2h-11Zm0 10V7h11v6.5h-11Z",
      clipRule: "evenodd"
    })]
  });
}
function CalendarEventIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCalendarEventIcon
  });
}

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
      d: "M4.5 0v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H11.5V0H10v2H6V0H4.5Zm9 3.5v2h-11v-2h11ZM2.5 7v6.5h11V7h-11Z",
      clipRule: "evenodd"
    })
  });
}
function CalendarIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCalendarIcon
  });
}

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
      d: "M8 10a.75.75 0 0 1-.59-.286l-2.164-2.75a.75.75 0 0 1 .589-1.214h4.33a.75.75 0 0 1 .59 1.214l-2.166 2.75A.75.75 0 0 1 8 10Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H1.75Zm.75 12.5v-11h11v11h-11Z",
      clipRule: "evenodd"
    })]
  });
}
function CaretDownSquareIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCaretDownSquareIcon
  });
}

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
      d: "M8 5.75a.75.75 0 0 1 .59.286l2.164 2.75A.75.75 0 0 1 10.165 10h-4.33a.75.75 0 0 1-.59-1.214l2.166-2.75A.75.75 0 0 1 8 5.75Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H1.75Zm.75 12.5v-11h11v11h-11Z",
      clipRule: "evenodd"
    })]
  });
}
function CaretUpSquareIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCaretUpSquareIcon
  });
}

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
      d: "M4.75 0A2.75 2.75 0 0 0 2 2.75V13.5A2.5 2.5 0 0 0 4.5 16h8.75a.75.75 0 0 0 .75-.75V.75a.75.75 0 0 0-.75-.75h-8.5Zm7.75 11V1.5H4.75c-.69 0-1.25.56-1.25 1.25v8.458a2.492 2.492 0 0 1 1-.208h8Zm-9 2.5a1 1 0 0 0 1 1h8v-2h-8a1 1 0 0 0-1 1Z",
      clipRule: "evenodd"
    })
  });
}
function CatalogIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCatalogIcon
  });
}

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
      d: "M14.03.75v10.69l-1.5-1.5V1.5H4.78c-.2 0-.39.047-.558.131L3.136.545A2.738 2.738 0 0 1 4.78 0h8.5a.75.75 0 0 1 .75.75Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M2.03 3.56 1 2.53l1.06-1.06 13 13L14 15.53l-.017-.017a.75.75 0 0 1-.703.487H4.53a2.5 2.5 0 0 1-2.5-2.5V3.56Zm8.94 8.94 1.56 1.56v.44h-8a1 1 0 1 1 0-2h6.44ZM9.47 11H4.53c-.355 0-.693.074-1 .208V5.061L9.47 11Z",
      clipRule: "evenodd"
    })]
  });
}
function CatalogOffIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCatalogOffIcon
  });
}

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
      d: "M1 1v13.25c0 .414.336.75.75.75H15v-1.5H2.5V1H1Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "m15.03 5.03-1.06-1.06L9.5 8.44 7 5.94 3.47 9.47l1.06 1.06L7 8.06l2.5 2.5 5.53-5.53Z"
    })]
  });
}
function ChartLineIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgChartLineIcon
  });
}

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
      d: "m10.47 5.47 1.06 1.06L7 11.06 4.47 8.53l1.06-1.06L7 8.94l3.47-3.47ZM16 12.5a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M1.5 8a6.5 6.5 0 0 1 13-.084c.54.236 1.031.565 1.452.967a8 8 0 1 0-7.07 7.07 5.008 5.008 0 0 1-.966-1.454A6.5 6.5 0 0 1 1.5 8Z"
    })]
  });
}
function CheckCircleBadgeIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCheckCircleBadgeIcon
  });
}

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
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm11.53-1.47-1.06-1.06L7 8.94 5.53 7.47 4.47 8.53l2 2 .53.53.53-.53 4-4Z",
      clipRule: "evenodd"
    })
  });
}
function CheckCircleFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCheckCircleFillIcon
  });
}

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
      d: "M11.53 6.53 7 11.06 4.47 8.53l1.06-1.06L7 8.94l3.47-3.47 1.06 1.06Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Z",
      clipRule: "evenodd"
    })]
  });
}
function CheckCircleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCheckCircleIcon
  });
}

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
      d: "M15.06 2.06 14 1 5.53 9.47 2.06 6 1 7.06l4.53 4.531 9.53-9.53ZM1.03 15.03h14v-1.5h-14v1.5Z",
      clipRule: "evenodd"
    })
  });
}
function CheckLineIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCheckLineIcon
  });
}

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
      d: "M1.75 2a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h11.5a.75.75 0 0 0 .75-.75V9h-1.5v4.5h-10v-10H10V2H1.75Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "m15.03 4.03-1.06-1.06L7.5 9.44 5.53 7.47 4.47 8.53l3.03 3.03 7.53-7.53Z"
    })]
  });
}
function CheckboxIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCheckboxIcon
  });
}

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
      d: "m5.5 2 1.06 1.06-3.53 3.531L1 4.561 2.06 3.5l.97.97L5.5 2ZM15.03 4.53h-7v-1.5h7v1.5ZM1.03 14.53v-1.5h14v1.5h-14ZM8.03 9.53h7v-1.5h-7v1.5ZM6.56 8.06 5.5 7 3.03 9.47l-.97-.97L1 9.56l2.03 2.031 3.53-3.53Z"
    })
  });
}
function ChecklistIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgChecklistIcon
  });
}

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
      d: "M10.947 7.954 8 10.891 5.056 7.954 3.997 9.016l4.004 3.993 4.005-3.993-1.06-1.062Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M10.947 3.994 8 6.931 5.056 3.994 3.997 5.056 8.001 9.05l4.005-3.993-1.06-1.062Z"
    })]
  });
}
function ChevronDoubleDownIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgChevronDoubleDownIcon
  });
}

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
      d: "M8.047 10.944 5.11 8l2.937-2.944-1.062-1.06L2.991 8l3.994 4.003 1.062-1.06Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M12.008 10.944 9.07 8l2.938-2.944-1.062-1.06L6.952 8l3.994 4.003 1.062-1.06Z"
    })]
  });
}
function ChevronDoubleLeftIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgChevronDoubleLeftIcon
  });
}

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
      d: "m7.954 5.056 2.937 2.946-2.937 2.945 1.062 1.059L13.01 8 9.016 3.998l-1.062 1.06Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "m3.994 5.056 2.937 2.946-2.937 2.945 1.062 1.059L9.05 8 5.056 3.998l-1.062 1.06Z"
    })]
  });
}
function ChevronDoubleRightIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgChevronDoubleRightIcon
  });
}

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
      d: "M5.056 8.047 8 5.11l2.944 2.937 1.06-1.062L8 2.991 3.997 6.985l1.059 1.062Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M5.056 12.008 8 9.07l2.944 2.937 1.06-1.062L8 6.952l-4.003 3.994 1.059 1.062Z"
    })]
  });
}
function ChevronDoubleUpIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgChevronDoubleUpIcon
  });
}

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
      d: "M8 8.917 10.947 6 12 7.042 8 11 4 7.042 5.053 6 8 8.917Z",
      clipRule: "evenodd"
    })
  });
}
function ChevronDownIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgChevronDownIcon
  });
}

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
      d: "M8 7.083 5.053 10 4 8.958 8 5l4 3.958L10.947 10 8 7.083Z",
      clipRule: "evenodd"
    })
  });
}
function ChevronUpIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgChevronUpIcon
  });
}

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
      d: "M12.5 8a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0Z"
    })
  });
}
function CircleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCircleIcon
  });
}

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
      d: "M5.5 0a.75.75 0 0 0-.75.75V1h-2a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75h-2V.75A.75.75 0 0 0 10.5 0h-5Zm5.75 2.5v.75a.75.75 0 0 1-.75.75h-5a.75.75 0 0 1-.75-.75V2.5H3.5v11h9v-11h-1.25Zm-5 0v-1h3.5v1h-3.5Z",
      clipRule: "evenodd"
    })
  });
}
function ClipboardIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgClipboardIcon
  });
}

function SvgClockIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.25 4v4c0 .199.079.39.22.53l2 2 1.06-1.06-1.78-1.78V4h-1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0ZM1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0Z",
      clipRule: "evenodd"
    })]
  });
}
function ClockIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgClockIcon
  });
}

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
      fillRule: "evenodd",
      d: "M15.25 12.5h-4.292a3 3 0 1 0-.13 1.5H12v2h1.5v-2h1v2H16v-2.75a.75.75 0 0 0-.75-.75ZM8 11.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z",
      clipRule: "evenodd"
    }), jsx("mask", {
      id: "ClockKeyIcon_svg__a",
      width: 17,
      height: 16,
      x: -1,
      y: 0,
      maskUnits: "userSpaceOnUse",
      style: {
        maskType: 'alpha'
      },
      children: jsx("path", {
        fill: "#D9D9D9",
        d: "M0 0h16v11.131h-2.848c-1.162 0-3.207-2.794-6.438-1.808-1.84.562-2.933 3.061-2.933 3.061V16H0V0Z"
      })
    }), jsxs("g", {
      fill: "currentColor",
      mask: "url(#ClockKeyIcon_svg__a)",
      children: [jsx("path", {
        fillRule: "evenodd",
        d: "M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0ZM8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0Z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M7.25 8V4h1.5v3.69l1.78 1.78-1.06 1.06-2-2A.75.75 0 0 1 7.25 8Z"
      })]
    })]
  });
}
function ClockKeyIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgClockKeyIcon
  });
}

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
      d: "M8 2a4.752 4.752 0 0 0-4.606 3.586 4.251 4.251 0 0 0 .427 8.393A.75.75 0 0 0 4 14v-1.511a2.75 2.75 0 0 1 .077-5.484.75.75 0 0 0 .697-.657 3.25 3.25 0 0 1 6.476.402v.5c0 .414.336.75.75.75h.25a2.25 2.25 0 1 1-.188 4.492.75.75 0 0 0-.062-.002V14a.757.757 0 0 0 .077-.004 3.75 3.75 0 0 0 .668-7.464A4.75 4.75 0 0 0 8 2Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M7.25 11.19 5.03 8.97l-1.06 1.06L8 14.06l4.03-4.03-1.06-1.06-2.22 2.22V6h-1.5v5.19Z"
    })]
  });
}
function CloudDownloadIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCloudDownloadIcon
  });
}

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
      d: "M3.394 5.586a4.752 4.752 0 0 1 9.351.946 3.75 3.75 0 0 1-.668 7.464A.757.757 0 0 1 12 14H4a.75.75 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393Zm.72 6.914h7.762a.745.745 0 0 1 .186-.008A2.25 2.25 0 1 0 12.25 8H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.476-.402.75.75 0 0 1-.697.657 2.75 2.75 0 0 0-.024 5.488.74.74 0 0 1 .062.007Z",
      clipRule: "evenodd"
    })
  });
}
function CloudIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCloudIcon
  });
}

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
      d: "M3.394 5.586a4.752 4.752 0 0 1 9.351.946A3.754 3.754 0 0 1 15.787 9H14.12a2.248 2.248 0 0 0-1.871-1H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.476-.402.75.75 0 0 1-.697.657A2.75 2.75 0 0 0 4 12.49V14a.75.75 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393ZM15.25 10.5h-4.291a3 3 0 1 0-.13 1.5H12v2h1.5v-2h1v2H16v-2.75a.75.75 0 0 0-.75-.75ZM8 9.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z",
      clipRule: "evenodd"
    })
  });
}
function CloudKeyIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCloudKeyIcon
  });
}

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
      d: "M3.394 5.586a4.752 4.752 0 0 1 9.351.946A3.754 3.754 0 0 1 15.787 9H14.12a2.248 2.248 0 0 0-1.871-1H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.476-.402.75.75 0 0 1-.697.657A2.75 2.75 0 0 0 4 12.49V14a.75.75 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 7a2.25 2.25 0 0 1 2.03 3.22l.5.5a2.25 2.25 0 1 1-1.06 1.06l-.5-.5A2.25 2.25 0 1 1 8 7Zm.75 2.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Zm3.5 3.5a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Z",
      clipRule: "evenodd"
    })]
  });
}
function CloudModelIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCloudModelIcon
  });
}

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
      d: "M13.97 14.53 2.47 3.03l-1 1 1.628 1.628a4.252 4.252 0 0 0 .723 8.32A.75.75 0 0 0 4 14h7.44l1.53 1.53 1-1ZM4.077 7.005a.748.748 0 0 0 .29-.078L9.939 12.5H4.115a.74.74 0 0 0-.062-.007 2.75 2.75 0 0 1 .024-5.488Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M4.8 3.24a4.75 4.75 0 0 1 7.945 3.293 3.75 3.75 0 0 1 1.928 6.58l-1.067-1.067A2.25 2.25 0 0 0 12.25 8H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-5.388-2.448L4.8 3.239Z"
    })]
  });
}
function CloudOffIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCloudOffIcon
  });
}

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
      d: "M8 2a4.752 4.752 0 0 0-4.606 3.586 4.251 4.251 0 0 0 .427 8.393A.75.75 0 0 0 4 14v-1.511a2.75 2.75 0 0 1 .077-5.484.75.75 0 0 0 .697-.657 3.25 3.25 0 0 1 6.476.402v.5c0 .414.336.75.75.75h.25a2.25 2.25 0 1 1-.188 4.492.75.75 0 0 0-.062-.002V14a.757.757 0 0 0 .077-.004 3.75 3.75 0 0 0 .668-7.464A4.75 4.75 0 0 0 8 2Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "m8.75 8.81 2.22 2.22 1.06-1.06L8 5.94 3.97 9.97l1.06 1.06 2.22-2.22V14h1.5V8.81Z"
    })]
  });
}
function CloudUploadIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCloudUploadIcon
  });
}

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
      d: "M4.03 12.06 5.091 11l-2.97-2.97 2.97-2.97L4.031 4 0 8.03l4.03 4.03ZM12.091 4l4.03 4.03-4.03 4.03-1.06-1.06L14 8.03l-2.97-2.97L12.091 4Z"
    })
  });
}
function CodeIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCodeIcon
  });
}

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
      d: "M7.5 1v1.59l4.88 4.88a.75.75 0 0 1 0 1.06l-4.242 4.243a2.75 2.75 0 0 1-3.89 0l-2.421-2.422a2.75 2.75 0 0 1 0-3.889L6 2.29V1h1.5ZM6 8V4.41L2.888 7.524a1.25 1.25 0 0 0 0 1.768l2.421 2.421a1.25 1.25 0 0 0 1.768 0L10.789 8 7.5 4.71V8H6Zm7.27 1.51a.76.76 0 0 0-1.092.001 8.53 8.53 0 0 0-1.216 1.636c-.236.428-.46.953-.51 1.501-.054.576.083 1.197.587 1.701a2.385 2.385 0 0 0 3.372 0c.505-.504.644-1.126.59-1.703-.05-.55-.274-1.075-.511-1.503a8.482 8.482 0 0 0-1.22-1.633Zm-.995 2.363c.138-.25.3-.487.451-.689.152.201.313.437.452.687.19.342.306.657.33.913.02.228-.03.377-.158.505a.885.885 0 0 1-1.25 0c-.125-.125-.176-.272-.155-.501.024-.256.14-.572.33-.915Z",
      clipRule: "evenodd"
    })
  });
}
function ColorFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgColorFillIcon
  });
}

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
      d: "M6.5 9V6h3v3h-3Zm3 1.5v3h-3v-3h3Zm1.5-.75v-9a.75.75 0 0 0-.75-.75h-4.5A.75.75 0 0 0 5 .75v13.5c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-4.5ZM6.5 4.5v-3h3v3h-3Z",
      clipRule: "evenodd"
    })
  });
}
function ColumnIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgColumnIcon
  });
}

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
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75V1.75ZM2.5 13.5v-11H5v11H2.5Zm4 0h3v-11h-3v11Zm4.5-11v11h2.5v-11H11Z",
      clipRule: "evenodd"
    })
  });
}
function ColumnsIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgColumnsIcon
  });
}

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
      d: "M7.78 3.97 5.03 1.22a.75.75 0 0 0-1.06 0L1.22 3.97a.75.75 0 0 0 0 1.06l2.75 2.75a.75.75 0 0 0 1.06 0l2.75-2.75a.75.75 0 0 0 0-1.06Zm-1.59.53L4.5 6.19 2.81 4.5 4.5 2.81 6.19 4.5ZM15 11.75a3.25 3.25 0 1 0-6.5 0 3.25 3.25 0 0 0 6.5 0ZM11.75 10a1.75 1.75 0 1 1 0 3.5 1.75 1.75 0 0 1 0-3.5Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M14.25 1H9v1.5h4.5V7H15V1.75a.75.75 0 0 0-.75-.75ZM1 9v5.25c0 .414.336.75.75.75H7v-1.5H2.5V9H1Z"
    })]
  });
}
function ConnectIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgConnectIcon
  });
}

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
      d: "M1.75 1a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75H5v3.25c0 .414.336.75.75.75h8.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75H11V1.75a.75.75 0 0 0-.75-.75h-8.5ZM9.5 5V2.5h-7v7H5V5.75A.75.75 0 0 1 5.75 5H9.5Zm-3 8.5v-7h7v7h-7Z",
      clipRule: "evenodd"
    })
  });
}
function CopyIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCopyIcon
  });
}

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
      d: "M8 3.75h1c.69 0 1.25.56 1.25 1.25v6c0 .69-.56 1.25-1.25 1.25H8v1.5h1c.788 0 1.499-.331 2-.863a2.742 2.742 0 0 0 2 .863h1v-1.5h-1c-.69 0-1.25-.56-1.25-1.25V5c0-.69.56-1.25 1.25-1.25h1v-1.5h-1c-.788 0-1.499.331-2 .863a2.742 2.742 0 0 0-2-.863H8v1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M5.936 8.003 3 5.058 4.062 4l3.993 4.004-3.993 4.005L3 10.948l2.936-2.945Z"
    })]
  });
}
function CursorTypeIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgCursorTypeIcon
  });
}

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
      d: "M8 1.75A.75.75 0 0 1 8.75 1h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5A.75.75 0 0 1 8 5.25v-1H5.5c-.69 0-1.25.56-1.25 1.25h2a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-2c0 .69.56 1.25 1.25 1.25H8v-1a.75.75 0 0 1 .75-.75h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5a.75.75 0 0 1-.75-.75v-1H5.5a2.75 2.75 0 0 1-2.75-2.75h-2A.75.75 0 0 1 0 9.75v-3.5a.75.75 0 0 1 .75-.75h2A2.75 2.75 0 0 1 5.5 2.75H8v-1Zm1.5.75v2h4v-2h-4ZM1.5 9V7h4v2h-4Zm8 4.5v-2h4v2h-4Z",
      clipRule: "evenodd"
    })
  });
}
function DagIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDagIcon
  });
}

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
      d: "M5.75 4.5a.75.75 0 0 0-.75.75v5.5c0 .414.336.75.75.75h2a3.5 3.5 0 1 0 0-7h-2ZM6.5 10V6h1.25a2 2 0 1 1 0 4H6.5Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H1.75Zm.75 12.5v-11h11v11h-11Z",
      clipRule: "evenodd"
    })]
  });
}
function DIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDIcon
  });
}

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
      d: "m15.78 11.533-4.242 4.243a.75.75 0 0 1-.53.22H4.996a.75.75 0 0 1-.53-.22L.224 11.533a.75.75 0 0 1-.22-.53v-6.01a.75.75 0 0 1 .22-.53L4.467.22a.75.75 0 0 1 .53-.22h6.01a.75.75 0 0 1 .53.22l4.243 4.242c.141.141.22.332.22.53v6.011a.75.75 0 0 1-.22.53Zm-8.528-.785a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Zm1.5-5.75v4h-1.5v-4h1.5Z",
      clipRule: "evenodd"
    })
  });
}
function DangerFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDangerFillIcon
  });
}

function SvgDangerIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.248 10.748a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0ZM8.748 4.998v4h-1.5v-4h1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m11.533 15.776 4.243-4.243a.75.75 0 0 0 .22-.53v-6.01a.75.75 0 0 0-.22-.53L11.533.22a.75.75 0 0 0-.53-.22h-6.01a.75.75 0 0 0-.53.22L.22 4.462a.75.75 0 0 0-.22.53v6.011c0 .199.079.39.22.53l4.242 4.243c.141.14.332.22.53.22h6.011a.75.75 0 0 0 .53-.22Zm2.963-10.473v5.39l-3.804 3.803H5.303L1.5 10.692V5.303L5.303 1.5h5.39l3.803 3.803Z",
      clipRule: "evenodd"
    })]
  });
}
function DangerIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDangerIcon
  });
}

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
      d: "M15 8.75H1v-1.5h14v1.5Z",
      clipRule: "evenodd"
    })
  });
}
function DashIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDashIcon
  });
}

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
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75V1.75Zm1.5 8.75v3h4.75v-3H2.5Zm0-1.5h4.75V2.5H2.5V9Zm6.25-6.5v3h4.75v-3H8.75Zm0 11V7h4.75v6.5H8.75Z",
      clipRule: "evenodd"
    })
  });
}
function DashboardIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDashboardIcon
  });
}

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
      d: "M8.646.368a.75.75 0 0 0-1.292 0l-3.25 5.5A.75.75 0 0 0 4.75 7h6.5a.75.75 0 0 0 .646-1.132l-3.25-5.5ZM8 2.224 9.936 5.5H6.064L8 2.224ZM8.5 9.25a.75.75 0 0 1 .75-.75h5a.75.75 0 0 1 .75.75v5a.75.75 0 0 1-.75.75h-5a.75.75 0 0 1-.75-.75v-5ZM10 10v3.5h3.5V10H10ZM1 11.75a3.25 3.25 0 1 1 6.5 0 3.25 3.25 0 0 1-6.5 0ZM4.25 10a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5Z",
      clipRule: "evenodd"
    })
  });
}
function DataIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDataIcon
  });
}

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
      d: "M2.727 3.695c-.225.192-.227.298-.227.305 0 .007.002.113.227.305.223.19.59.394 1.108.58C4.865 5.256 6.337 5.5 8 5.5c1.663 0 3.135-.244 4.165-.615.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305 0-.007-.002-.113-.227-.305-.223-.19-.59-.394-1.108-.58C11.135 2.744 9.663 2.5 8 2.5c-1.663 0-3.135.244-4.165.615-.519.186-.885.39-1.108.58ZM13.5 5.94a6.646 6.646 0 0 1-.826.358C11.442 6.74 9.789 7 8 7c-1.79 0-3.442-.26-4.673-.703a6.641 6.641 0 0 1-.827-.358V8c0 .007.002.113.227.305.223.19.59.394 1.108.58C4.865 9.256 6.337 9.5 8 9.5c1.663 0 3.135-.244 4.165-.615.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305V5.939ZM15 8V4c0-.615-.348-1.1-.755-1.447-.41-.349-.959-.63-1.571-.85C11.442 1.26 9.789 1 8 1c-1.79 0-3.442.26-4.673.703-.613.22-1.162.501-1.572.85C1.348 2.9 1 3.385 1 4v8c0 .615.348 1.1.755 1.447.41.349.959.63 1.572.85C4.558 14.74 6.21 15 8 15c1.79 0 3.441-.26 4.674-.703.612-.22 1.161-.501 1.571-.85.407-.346.755-.832.755-1.447V8Zm-1.5 1.939a6.654 6.654 0 0 1-.826.358C11.442 10.74 9.789 11 8 11c-1.79 0-3.442-.26-4.673-.703a6.649 6.649 0 0 1-.827-.358V12c0 .007.002.113.227.305.223.19.59.394 1.108.58 1.03.371 2.502.615 4.165.615 1.663 0 3.135-.244 4.165-.615.519-.186.885-.39 1.108-.58.225-.192.227-.298.227-.305V9.939Z",
      clipRule: "evenodd"
    })
  });
}
function DatabaseIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDatabaseIcon
  });
}

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
      d: "M3 10a3 3 0 1 0 6 0V6a3 3 0 0 0-6 0v4Zm3 1.5A1.5 1.5 0 0 1 4.5 10V6a1.5 1.5 0 1 1 3 0v4A1.5 1.5 0 0 1 6 11.5ZM10 10a3 3 0 1 0 6 0V6a3 3 0 1 0-6 0v4Zm3 1.5a1.5 1.5 0 0 1-1.5-1.5V6a1.5 1.5 0 0 1 3 0v4a1.5 1.5 0 0 1-1.5 1.5Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M1 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
    })]
  });
}
function DecimalIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDecimalIcon
  });
}

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
        d: "M6 8a.75.75 0 1 1-1.5 0A.75.75 0 0 1 6 8ZM8 8.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5ZM10.75 8.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0ZM1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0Z",
        clipRule: "evenodd"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "DotsCircleIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function DotsCircleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDotsCircleIcon
  });
}

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
      d: "M1 13.5h14V15H1v-1.5ZM12.53 6.53l-1.06-1.06-2.72 2.72V1h-1.5v7.19L4.53 5.47 3.47 6.53 8 11.06l4.53-4.53Z"
    })
  });
}
function DownloadIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDownloadIcon
  });
}

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
      d: "M5.25 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5ZM10.75 1a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5ZM5.25 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5ZM10.75 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5ZM5.25 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5ZM10.75 11.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5Z"
    })
  });
}
function DragIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgDragIcon
  });
}

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
      d: "M12.06 1.06 11 0 8.03 2.97 5.06 0 4 1.06l4.03 4.031 4.03-4.03ZM4 15l4.03-4.03L12.06 15 11 16.06l-2.97-2.969-2.97 2.97L4 15Z"
    })
  });
}
function ExpandLessIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgExpandLessIcon
  });
}

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
      d: "m4 4.03 1.06 1.061 2.97-2.97L11 5.091l1.06-1.06L8.03 0 4 4.03ZM12.06 12.091l-4.03 4.03L4 12.091l1.06-1.06L8.03 14 11 11.03l1.06 1.061Z"
    })
  });
}
function ExpandMoreIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgExpandMoreIcon
  });
}

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
      d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16H2V1.75Zm7.5 1.81 1.94 1.94H9.5V3.56Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M7.47 9.97 4.44 13l3.03 3.03 1.06-1.06L6.56 13l1.97-1.97-1.06-1.06ZM11.03 9.97l-1.06 1.06L11.94 13l-1.97 1.97 1.06 1.06L14.06 13l-3.03-3.03Z"
    })]
  });
}
function FileCodeIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFileCodeIcon
  });
}

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
      d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16H2V1.75Zm7.5 1.81 1.94 1.94H9.5V3.56Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M5 11.5V13h9v-1.5H5ZM14 16H5v-1.5h9V16Z"
    })]
  });
}
function FileDocumentIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFileDocumentIcon
  });
}

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
      d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53v9a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75V1.75Zm1.5.75v12h9V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5Zm6 1.06 1.94 1.94H9.5V3.56Z",
      clipRule: "evenodd"
    })
  });
}
function FileIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFileIcon
  });
}

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
      d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16H2V1.75Zm7.5 1.81 1.94 1.94H9.5V3.56Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M10.466 10a.75.75 0 0 0-.542.27l-3.75 4.5A.75.75 0 0 0 6.75 16h6.5a.75.75 0 0 0 .75-.75V13.5a.75.75 0 0 0-.22-.53l-2.75-2.75a.75.75 0 0 0-.564-.22Zm2.034 3.81v.69H8.351l2.2-2.639 1.949 1.95ZM6.5 7.25a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5ZM5.75 9.5a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0Z",
      clipRule: "evenodd"
    })]
  });
}
function FileImageIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFileImageIcon
  });
}

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
      d: "M2.75 1a.75.75 0 0 0-.75.75V16h1.5V2.5H8v3.75c0 .414.336.75.75.75h3.75v3H14V6.25a.75.75 0 0 0-.22-.53l-4.5-4.5A.75.75 0 0 0 8.75 1h-6Zm8.69 4.5L9.5 3.56V5.5h1.94Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M11.75 11.5a2.25 2.25 0 1 1-2.03 1.28l-.5-.5a2.25 2.25 0 1 1 1.06-1.06l.5.5c.294-.141.623-.22.97-.22Zm.75 2.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0ZM8.25 9.5a.75.75 0 1 1 0 1.5.75.75 0 0 1 0-1.5Z",
      clipRule: "evenodd"
    })]
  });
}
function FileModelIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFileModelIcon
  });
}

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
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75V4a.75.75 0 0 1-.22.53L10 9.31v4.94a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1-.75-.75V9.31L1.22 4.53A.75.75 0 0 1 1 4V1.75Zm1.5.75v1.19l4.78 4.78c.141.14.22.331.22.53v4.5h1V9a.75.75 0 0 1 .22-.53l4.78-4.78V2.5h-11Z",
      clipRule: "evenodd"
    })
  });
}
function FilterIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFilterIcon
  });
}

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
      d: "M0 5.25A2.25 2.25 0 0 0 2.25 3h1.5v8.5H6V13H0v-1.5h2.25V6c-.627.471-1.406.75-2.25.75v-1.5ZM10 5.75A2.75 2.75 0 0 1 12.75 3h.39a2.86 2.86 0 0 1 1.57 5.252l-2.195 1.44a2.25 2.25 0 0 0-1.014 1.808H16V13h-6v-1.426a3.75 3.75 0 0 1 1.692-3.135l2.194-1.44A1.36 1.36 0 0 0 13.14 4.5h-.389c-.69 0-1.25.56-1.25 1.25V6H10v-.25ZM8 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
    })
  });
}
function FloatIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFloatIcon
  });
}

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
      d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 0 1 .75.75V8h-1.5V5.5h-7a.75.75 0 0 1-.53-.22L5.555 3.866a1.25 1.25 0 0 0-.883-.366H1.5v9H5V14H.75a.75.75 0 0 1-.75-.75V2.75ZM9 8.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1ZM7 9a2 2 0 1 1 3.778.917c.376.58.888 1.031 1.414 1.227a2 2 0 1 1-.072 1.54c-.977-.207-1.795-.872-2.37-1.626v1.087a2 2 0 1 1-1.5 0v-1.29A2 2 0 0 1 7 9Zm7 2.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1Zm-5 2a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1Z",
      clipRule: "evenodd"
    })
  });
}
function FolderBranchIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFolderBranchIcon
  });
}

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
        d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 0 1 .75.75v5.02a4.4 4.4 0 0 0-.921-.607 5.11 5.11 0 0 0-9.512-.753A4.75 4.75 0 0 0 2.917 14H.75a.75.75 0 0 1-.75-.75V2.75Z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M6.715 9.595a3.608 3.608 0 0 1 7.056.688C15.07 10.572 16 11.739 16 13.107 16 14.688 14.757 16 13.143 16a2.795 2.795 0 0 1-.107 0H7.32a.757.757 0 0 1-.163-.018 3.25 3.25 0 0 1-.443-6.387Zm.703 4.905a1.75 1.75 0 0 1-.03-3.497.75.75 0 0 0 .7-.657 2.108 2.108 0 0 1 4.198.261v.357c0 .415.335.75.75.75h.107c.753 0 1.357.607 1.357 1.393s-.604 1.393-1.357 1.393c-.024 0-.047 0-.07-.002a.736.736 0 0 0-.1.002H7.418Z",
        clipRule: "evenodd"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "FolderCloudFilledIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function FolderCloudFilledIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFolderCloudFilledIcon
  });
}

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
        d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75H3v-1.5H1.5v-9h3.172c.331 0 .649.132.883.366L6.97 5.28c.14.141.331.22.53.22h7V8H16V4.75a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2H.75Z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M10.179 7a3.608 3.608 0 0 0-3.464 2.595 3.251 3.251 0 0 0 .443 6.387.757.757 0 0 0 .163.018h5.821C14.758 16 16 14.688 16 13.107c0-1.368-.931-2.535-2.229-2.824A3.608 3.608 0 0 0 10.18 7Zm-2.805 7.496c.015 0 .03.002.044.004h5.555a.736.736 0 0 1 .1-.002l.07.002c.753 0 1.357-.607 1.357-1.393s-.604-1.393-1.357-1.393h-.107a.75.75 0 0 1-.75-.75v-.357a2.107 2.107 0 0 0-4.199-.26.75.75 0 0 1-.698.656 1.75 1.75 0 0 0-.015 3.493Z",
        clipRule: "evenodd"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "FolderCloudIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function FolderCloudIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFolderCloudIcon
  });
}

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
      d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2H.75Z"
    })
  });
}
function FolderFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFolderFillIcon
  });
}

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
      d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 0 1 .75.75v8.5a.75.75 0 0 1-.75.75H.75a.75.75 0 0 1-.75-.75V2.75Zm1.5.75v9h13v-7h-7a.75.75 0 0 1-.53-.22L5.555 3.866a1.25 1.25 0 0 0-.883-.366H1.5Z",
      clipRule: "evenodd"
    })
  });
}
function FolderIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFolderIcon
  });
}

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
        d: "M5.197 3.473a.75.75 0 0 0-1.393-.001L-.006 13H1.61l.6-1.5h4.562l.596 1.5h1.614L5.197 3.473ZM6.176 10 4.498 5.776 2.809 10h3.367Zm4.07-2.385c.593-.205 1.173-.365 1.754-.365a1.5 1.5 0 0 1 1.42 1.014A3.764 3.764 0 0 0 12 8c-.741 0-1.47.191-2.035.607A2.301 2.301 0 0 0 9 10.5c0 .81.381 1.464.965 1.893.565.416 1.294.607 2.035.607.524 0 1.042-.096 1.5-.298V13H15V8.75a3 3 0 0 0-3-3c-.84 0-1.614.23-2.245.448l.49 1.417ZM13.5 10.5a.804.804 0 0 0-.353-.685C12.897 9.631 12.5 9.5 12 9.5c-.5 0-.897.131-1.146.315a.804.804 0 0 0-.354.685c0 .295.123.515.354.685.25.184.645.315 1.146.315.502 0 .897-.131 1.147-.315.23-.17.353-.39.353-.685Z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "FontIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function FontIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFontIcon
  });
}

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
      d: "M2 2.75a2.75 2.75 0 1 1 3.5 2.646V6.75h3.75A2.75 2.75 0 0 1 12 9.5v.104a2.751 2.751 0 1 1-1.5 0V9.5c0-.69-.56-1.25-1.25-1.25H5.5v1.354a2.751 2.751 0 1 1-1.5 0V5.396A2.751 2.751 0 0 1 2 2.75ZM4.75 1.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5ZM3.5 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0Zm6.5 0a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0Z",
      clipRule: "evenodd"
    })
  });
}
function ForkIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgForkIcon
  });
}

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
      d: "M6 1v4.25a.75.75 0 0 1-.75.75H1V4.5h3.5V1H6ZM10 15v-4.25a.75.75 0 0 1 .75-.75H15v1.5h-3.5V15H10ZM10.75 6H15V4.5h-3.5V1H10v4.25c0 .414.336.75.75.75ZM1 10h4.25a.75.75 0 0 1 .75.75V15H4.5v-3.5H1V10Z"
    })
  });
}
function FullscreenExitIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFullscreenExitIcon
  });
}

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
      d: "M6 1H1.75a.75.75 0 0 0-.75.75V6h1.5V2.5H6V1ZM10 2.5V1h4.25a.75.75 0 0 1 .75.75V6h-1.5V2.5H10ZM10 13.5h3.5V10H15v4.25a.75.75 0 0 1-.75.75H10v-1.5ZM2.5 10v3.5H6V15H1.75a.75.75 0 0 1-.75-.75V10h1.5Z"
    })
  });
}
function FullscreenIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFullscreenIcon
  });
}

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
        d: "M9.93 2.988c-.774-.904-2.252-.492-2.448.682L7.094 6h2.005a2.75 2.75 0 0 1 2.585 1.81l.073.202 2.234-2.063 1.018 1.102-2.696 2.489.413 1.137c.18.494.65.823 1.175.823H15V13h-1.1a2.75 2.75 0 0 1-2.585-1.81l-.198-.547-2.61 2.408-1.017-1.102 3.07-2.834-.287-.792A1.25 1.25 0 0 0 9.099 7.5H6.844l-.846 5.076c-.405 2.43-3.464 3.283-5.067 1.412l1.139-.976c.774.904 2.252.492 2.448-.682l.805-4.83H3V6h2.573l.43-2.576C6.407.994 9.465.14 11.068 2.012l-1.138.976Z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "FunctionIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M16 0H0v16h16z"
        })
      })
    })]
  });
}
function FunctionIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgFunctionIcon
  });
}

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
      d: "M7.965 0c-.34 0-.675.021-1.004.063a.75.75 0 0 0-.62.51l-.639 1.946c-.21.087-.413.185-.61.294L3.172 2.1a.75.75 0 0 0-.784.165c-.481.468-.903.996-1.255 1.572a.75.75 0 0 0 .013.802l1.123 1.713a5.898 5.898 0 0 0-.15.66L.363 8.07a.75.75 0 0 0-.36.716c.067.682.22 1.34.447 1.962a.75.75 0 0 0 .635.489l2.042.19c.13.184.271.36.422.529l-.27 2.032a.75.75 0 0 0 .336.728 7.97 7.97 0 0 0 1.812.874.75.75 0 0 0 .778-.192l1.422-1.478a5.924 5.924 0 0 0 .677 0l1.422 1.478a.75.75 0 0 0 .778.192 7.972 7.972 0 0 0 1.812-.874.75.75 0 0 0 .335-.728l-.269-2.032a5.94 5.94 0 0 0 .422-.529l2.043-.19a.75.75 0 0 0 .634-.49c.228-.621.38-1.279.447-1.961a.75.75 0 0 0-.36-.716l-1.756-1.056a5.89 5.89 0 0 0-.15-.661l1.123-1.713a.75.75 0 0 0 .013-.802 8.034 8.034 0 0 0-1.255-1.572.75.75 0 0 0-.784-.165l-1.92.713c-.197-.109-.4-.207-.61-.294L9.589.573a.75.75 0 0 0-.619-.51A8.07 8.07 0 0 0 7.965 0Zm.02 10.25a2.25 2.25 0 1 0 0-4.5 2.25 2.25 0 0 0 0 4.5Z",
      clipRule: "evenodd"
    })
  });
}
function GearFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgGearFillIcon
  });
}

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
        d: "M7.984 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm-1.5 3a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
      }), jsx("path", {
        d: "M7.965 0c-.34 0-.675.021-1.004.063a.75.75 0 0 0-.62.51l-.639 1.946c-.21.087-.413.185-.61.294L3.172 2.1a.75.75 0 0 0-.784.165c-.481.468-.903.996-1.255 1.572a.75.75 0 0 0 .013.802l1.123 1.713a5.898 5.898 0 0 0-.15.66L.363 8.07a.75.75 0 0 0-.36.716c.067.682.22 1.34.447 1.962a.75.75 0 0 0 .635.489l2.042.19c.13.184.271.36.422.529l-.27 2.032a.75.75 0 0 0 .336.728 7.97 7.97 0 0 0 1.812.874.75.75 0 0 0 .778-.192l1.422-1.478a5.924 5.924 0 0 0 .677 0l1.422 1.478a.75.75 0 0 0 .778.192 7.972 7.972 0 0 0 1.812-.874.75.75 0 0 0 .335-.728l-.269-2.032a5.94 5.94 0 0 0 .422-.529l2.043-.19a.75.75 0 0 0 .634-.49c.228-.621.38-1.279.447-1.961a.75.75 0 0 0-.36-.716l-1.756-1.056a5.89 5.89 0 0 0-.15-.661l1.123-1.713a.75.75 0 0 0 .013-.802 8.034 8.034 0 0 0-1.255-1.572.75.75 0 0 0-.784-.165l-1.92.713c-.197-.109-.4-.207-.61-.294L9.589.573a.75.75 0 0 0-.619-.51A8.071 8.071 0 0 0 7.965 0Zm-.95 3.328.598-1.819a6.62 6.62 0 0 1 .705 0l.597 1.819a.75.75 0 0 0 .472.476c.345.117.67.275.97.468a.75.75 0 0 0 .668.073l1.795-.668c.156.176.303.36.44.552l-1.05 1.6a.75.75 0 0 0-.078.667c.12.333.202.685.24 1.05a.75.75 0 0 0 .359.567l1.642.988c-.04.234-.092.463-.156.687l-1.909.178a.75.75 0 0 0-.569.353c-.19.308-.416.59-.672.843a.75.75 0 0 0-.219.633l.252 1.901a6.48 6.48 0 0 1-.635.306l-1.33-1.381a.75.75 0 0 0-.63-.225 4.483 4.483 0 0 1-1.08 0 .75.75 0 0 0-.63.225l-1.33 1.381a6.473 6.473 0 0 1-.634-.306l.252-1.9a.75.75 0 0 0-.219-.634 4.449 4.449 0 0 1-.672-.843.75.75 0 0 0-.569-.353l-1.909-.178a6.456 6.456 0 0 1-.156-.687L3.2 8.113a.75.75 0 0 0 .36-.567c.037-.365.118-.717.239-1.05a.75.75 0 0 0-.078-.666L2.67 4.229c.137-.192.284-.376.44-.552l1.795.668a.75.75 0 0 0 .667-.073c.3-.193.626-.351.97-.468a.75.75 0 0 0 .472-.476Z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "GearIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function GearIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgGearIcon
  });
}

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
      d: "M3 3.25A2.25 2.25 0 0 1 5.25 1C6.365 1 7.36 1.522 8 2.335A3.494 3.494 0 0 1 10.75 1a2.25 2.25 0 0 1 2.122 3h1.378a.75.75 0 0 1 .75.75v3a.75.75 0 0 1-.75.75H14v5.75a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75V8.5h-.25A.75.75 0 0 1 1 7.75v-3A.75.75 0 0 1 1.75 4h1.378A2.246 2.246 0 0 1 3 3.25ZM5.25 4h1.937A2 2 0 0 0 5.25 2.5a.75.75 0 0 0 0 1.5Zm2 1.5H2.5V7h4.75V5.5Zm0 3H3.5v5h3.75v-5Zm1.5 5v-5h3.75v5H8.75Zm0-6.5V5.5h4.75V7H8.75Zm.063-3h1.937a.75.75 0 0 0 0-1.5A2 2 0 0 0 8.813 4Z",
      clipRule: "evenodd"
    })
  });
}
function GiftIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgGiftIcon
  });
}

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
      d: "M8 5.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5ZM4.07 7.25a4.001 4.001 0 0 1 7.86 0H16v1.5h-4.07a4.001 4.001 0 0 1-7.86 0H0v-1.5h4.07Z",
      clipRule: "evenodd"
    })
  });
}
function GitCommitIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgGitCommitIcon
  });
}

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
        d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm5.354-5.393c.088-.231.184-.454.287-.666A6.506 6.506 0 0 0 1.543 7.25h2.971c.067-1.777.368-3.399.84-4.643Zm.661 4.643c.066-1.627.344-3.062.742-4.11.23-.607.485-1.046.73-1.32.247-.274.421-.32.513-.32.092 0 .266.046.512.32s.501.713.731 1.32c.398 1.048.676 2.483.742 4.11h-3.97Zm3.97 1.5h-3.97c.066 1.627.344 3.062.742 4.11.23.607.485 1.046.73 1.32.247.274.421.32.513.32.092 0 .266-.046.512-.32s.501-.713.731-1.32c.398-1.048.676-2.483.742-4.11Zm1.501-1.5c-.067-1.777-.368-3.399-.84-4.643a7.912 7.912 0 0 0-.287-.666 6.506 6.506 0 0 1 4.098 5.309h-2.971Zm2.971 1.5h-2.971c-.067 1.777-.368 3.399-.84 4.643a7.918 7.918 0 0 1-.287.666 6.506 6.506 0 0 0 4.098-5.309Zm-9.943 0H1.543a6.506 6.506 0 0 0 4.098 5.309 7.921 7.921 0 0 1-.287-.666c-.472-1.244-.773-2.866-.84-4.643Z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "GlobeIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 16h16V0H0z"
        })
      })
    })]
  });
}
function GlobeIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgGlobeIcon
  });
}

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
      d: "M1 1.75V4h1.5V2.5H4V1H1.75a.75.75 0 0 0-.75.75ZM15 14.25V12h-1.5v1.5H12V15h2.25a.75.75 0 0 0 .75-.75ZM12 1h2.25a.75.75 0 0 1 .75.75V4h-1.5V2.5H12V1ZM1.75 15H4v-1.5H2.5V12H1v2.25a.75.75 0 0 0 .75.75ZM10 2.5H6V1h4v1.5ZM6 15h4v-1.5H6V15ZM13.5 10V6H15v4h-1.5ZM1 6v4h1.5V6H1Z"
    })
  });
}
function GridDashIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgGridDashIcon
  });
}

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
      d: "M1.75 1a.75.75 0 0 0-.75.75v4.5c0 .414.336.75.75.75h4.5A.75.75 0 0 0 7 6.25v-4.5A.75.75 0 0 0 6.25 1h-4.5Zm.75 4.5v-3h3v3h-3ZM1.75 9a.75.75 0 0 0-.75.75v4.5c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-4.5A.75.75 0 0 0 6.25 9h-4.5Zm.75 4.5v-3h3v3h-3ZM9 1.75A.75.75 0 0 1 9.75 1h4.5a.75.75 0 0 1 .75.75v4.49a.75.75 0 0 1-.75.75h-4.5A.75.75 0 0 1 9 6.24V1.75Zm1.5.75v2.99h3V2.5h-3ZM9.75 9a.75.75 0 0 0-.75.75v4.5c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-4.5a.75.75 0 0 0-.75-.75h-4.5Zm.75 4.5v-3h3v3h-3Z",
      clipRule: "evenodd"
    })
  });
}
function GridIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgGridIcon
  });
}

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
      d: "M1 3v10h1.5V8.75H6V13h1.5V3H6v4.25H2.5V3H1ZM11.25 3A2.25 2.25 0 0 1 9 5.25v1.5c.844 0 1.623-.279 2.25-.75v5.5H9V13h6v-1.5h-2.25V3h-1.5Z"
    })
  });
}
function H1Icon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgH1Icon
  });
}

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
      d: "M1 3v10h1.5V8.75H6V13h1.5V3H6v4.25H2.5V3H1ZM11.75 3A2.75 2.75 0 0 0 9 5.75V6h1.5v-.25c0-.69.56-1.25 1.25-1.25h.39a1.36 1.36 0 0 1 .746 2.498L10.692 8.44A3.75 3.75 0 0 0 9 11.574V13h6v-1.5h-4.499a2.25 2.25 0 0 1 1.014-1.807l2.194-1.44A2.86 2.86 0 0 0 12.14 3h-.389Z"
    })
  });
}
function H2Icon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgH2Icon
  });
}

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
      d: "M1 3h1.5v4.25H6V3h1.5v10H6V8.75H2.5V13H1V3ZM9 5.75A2.75 2.75 0 0 1 11.75 3h.375a2.875 2.875 0 0 1 1.937 5 2.875 2.875 0 0 1-1.937 5h-.375A2.75 2.75 0 0 1 9 10.25V10h1.5v.25c0 .69.56 1.25 1.25 1.25h.375a1.375 1.375 0 1 0 0-2.75H11v-1.5h1.125a1.375 1.375 0 1 0 0-2.75h-.375c-.69 0-1.25.56-1.25 1.25V6H9v-.25Z"
    })
  });
}
function H3Icon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgH3Icon
  });
}

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
        d: "m3.507 7.73.963-.962 1.06 1.06-2.732 2.732L-.03 7.732l1.06-1.06.979.978a7 7 0 1 1 2.041 5.3l1.061-1.06a5.5 5.5 0 1 0-1.604-4.158Z"
      }), jsx("path", {
        d: "M8.25 8V4h1.5v3.69l1.78 1.78-1.06 1.06-2-2A.75.75 0 0 1 8.25 8Z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "HistoryIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function HistoryIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgHistoryIcon
  });
}

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
      d: "M7.625 1.1a.75.75 0 0 1 .75 0l6.25 3.61a.75.75 0 0 1 .375.65v8.89a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75V10H7v4.25a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75V5.355a.75.75 0 0 1 .375-.65L7.625 1.1ZM2.5 5.79V13.5h3V9.25a.75.75 0 0 1 .75-.75h3.5a.75.75 0 0 1 .75.75v4.25h3V5.792L8 2.616 2.5 5.789Z",
      clipRule: "evenodd"
    })
  });
}
function HomeIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgHomeIcon
  });
}

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
      d: "M6.25 3.998a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5Zm-.75 2.25a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.492a.75.75 0 0 1-.75.75H5.038l-.009.009-.008-.009H1.75a.75.75 0 0 1-.75-.75V1.75Zm12.5 11.742H6.544l4.455-4.436 2.47 2.469.031-.03v1.997Zm0-10.992v6.934l-1.97-1.968a.75.75 0 0 0-1.06-.001l-6.052 6.027H2.5V2.5h11Z",
      clipRule: "evenodd"
    })]
  });
}
function ImageIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgImageIcon
  });
}

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
      d: "M16 2H0v1.5h16V2ZM16 5.5H7V7h9V5.5ZM16 9H7v1.5h9V9ZM16 12.5H0V14h16v-1.5ZM3.97 11.03.94 8l3.03-3.03 1.06 1.06L3.06 8l1.97 1.97-1.06 1.06Z"
    })
  });
}
function IndentDecreaseIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgIndentDecreaseIcon
  });
}

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
      d: "M16 2H0v1.5h16V2ZM16 5.5H7V7h9V5.5ZM16 9H7v1.5h9V9ZM16 12.5H0V14h16v-1.5ZM2.03 4.97 5.06 8l-3.03 3.03L.97 9.97 2.94 8 .97 6.03l1.06-1.06Z"
    })
  });
}
function IndentIncreaseIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgIndentIncreaseIcon
  });
}

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
      d: "m8 6.94 1.59-1.592a3.75 3.75 0 1 1 0 5.304L8 9.06l-1.591 1.59a3.75 3.75 0 1 1 0-5.303L8 6.94Zm2.652-.531a2.25 2.25 0 1 1 0 3.182L9.06 8l1.59-1.591ZM6.939 8 5.35 6.409a2.25 2.25 0 1 0 0 3.182l1.588-1.589L6.939 8Z",
      clipRule: "evenodd"
    })
  });
}
function InfinityIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgInfinityIcon
  });
}

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
      d: "M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0Zm-8.75 3V7h1.5v4h-1.5ZM8 4.5A.75.75 0 1 1 8 6a.75.75 0 0 1 0-1.5Z",
      clipRule: "evenodd"
    })
  });
}
function InfoFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgInfoFillIcon
  });
}

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
      d: "M15 2.5a.75.75 0 0 0-.75-.75h-3a.75.75 0 0 0-.75.75V6H12V3.25h1.5v9.5H12V10h-1.5v3.5c0 .414.336.75.75.75h3a.75.75 0 0 0 .75-.75v-11Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M3.75 0c1.26 0 2.322.848 2.648 2.004A2.75 2.75 0 0 1 9 4.75v2.5h3v1.5H9v2.5a2.75 2.75 0 0 1-2.602 2.746 2.751 2.751 0 1 1-3.47-3.371 2.751 2.751 0 0 1 0-5.25A2.751 2.751 0 0 1 3.75 0ZM5 2.75a1.25 1.25 0 1 0-2.5 0 1.25 1.25 0 0 0 2.5 0Zm-.428 2.625a2.756 2.756 0 0 0 1.822-1.867A1.25 1.25 0 0 1 7.5 4.75v2.5H6.396a2.756 2.756 0 0 0-1.824-1.875ZM6.396 8.75H7.5v2.5a1.25 1.25 0 0 1-1.106 1.242 2.756 2.756 0 0 0-1.822-1.867A2.756 2.756 0 0 0 6.396 8.75ZM3.75 12a1.25 1.25 0 1 1 0 2.5 1.25 1.25 0 0 1 0-2.5Zm0-5.25a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5Z",
      clipRule: "evenodd"
    })]
  });
}
function IngestionIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgIngestionIcon
  });
}

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
      d: "M9.648 4.5H12V3H6v1.5h2.102l-1.75 7H4V13h6v-1.5H7.898l1.75-7Z",
      clipRule: "evenodd"
    })
  });
}
function ItalicIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgItalicIcon
  });
}

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
      d: "M0 8a4 4 0 0 1 7.93-.75h7.32A.75.75 0 0 1 16 8v3h-1.5V8.75H13V11h-1.5V8.75H7.93A4.001 4.001 0 0 1 0 8Zm4-2.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5Z",
      clipRule: "evenodd"
    })
  });
}
function KeyIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgKeyIcon
  });
}

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
      d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H.75Zm.75 10.5v-9h13v9h-13Zm2.75-8h-1.5V6h1.5V4.5Zm1.5 0V6h1.5V4.5h-1.5Zm3 0V6h1.5V4.5h-1.5Zm3 0V6h1.5V4.5h-1.5Zm-1.5 2.75h-1.5v1.5h1.5v-1.5Zm1.5 1.5v-1.5h1.5v1.5h-1.5Zm-4.5 0v-1.5h-1.5v1.5h1.5Zm-3 0v-1.5h-1.5v1.5h1.5ZM11 10H5v1.5h6V10Z",
      clipRule: "evenodd"
    })
  });
}
function KeyboardIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgKeyboardIcon
  });
}

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
      d: "M13.5 2.5H7V1h7.25a.75.75 0 0 1 .75.75V9h-1.5V2.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 7.75A.75.75 0 0 1 1.75 7h6.5a.75.75 0 0 1 .75.75v6.5a.75.75 0 0 1-.75.75h-6.5a.75.75 0 0 1-.75-.75v-6.5Zm1.5.75v5h5v-5h-5Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M4 5.32h6.5V12H12V4.57a.75.75 0 0 0-.75-.75H4v1.5Z"
    })]
  });
}
function LayerIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgLayerIcon
  });
}

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
      d: "M6.25 1h2.174a2.126 2.126 0 0 1 1.81 3.243 2.126 2.126 0 0 1-1.36 3.761H6.25a.75.75 0 0 1-.75-.75V1.75A.75.75 0 0 1 6.25 1ZM7 6.504V5.252h1.874a.626.626 0 1 1 0 1.252H7Zm2.05-3.378c0 .345-.28.625-.625.626H7.001L7 2.5h1.424c.346 0 .626.28.626.626ZM3.307 6a.75.75 0 0 1 .697.473L6.596 13H4.982l-.238-.6H1.855l-.24.6H0l2.61-6.528A.75.75 0 0 1 3.307 6Zm-.003 2.776.844 2.124H2.455l.85-2.124Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M12.5 15a2.5 2.5 0 0 0 2.5-2.5h-1.5a1 1 0 1 1-2 0v-1.947c0-.582.472-1.053 1.053-1.053.523 0 .947.424.947.947v.053H15v-.053A2.447 2.447 0 0 0 12.553 8 2.553 2.553 0 0 0 10 10.553V12.5a2.5 2.5 0 0 0 2.5 2.5Z"
    })]
  });
}
function LettersIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgLettersIcon
  });
}

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
      d: "m8.301 1.522 5.25 13.5 1.398-.544-5.25-13.5-1.398.544ZM1 15V1h1.5v14H1ZM5 15V1h1.5v14H5Z"
    })
  });
}
function LibrariesIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgLibrariesIcon
  });
}

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
      d: "M9.49.04a.75.75 0 0 1 .51.71V6h3.25a.75.75 0 0 1 .596 1.206l-6.5 8.5A.75.75 0 0 1 6 15.25V10H2.75a.75.75 0 0 1-.596-1.206l6.5-8.5A.75.75 0 0 1 9.491.04ZM4.269 8.5H6.75a.75.75 0 0 1 .75.75v3.785L11.732 7.5H9.25a.75.75 0 0 1-.75-.75V2.965L4.268 8.5Z",
      clipRule: "evenodd"
    })
  });
}
function LightningIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgLightningIcon
  });
}

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
      d: "M4 4h3v1.5H4a2.5 2.5 0 0 0 0 5h3V12H4a4 4 0 0 1 0-8ZM12 10.5H9V12h3a4 4 0 0 0 0-8H9v1.5h3a2.5 2.5 0 0 1 0 5Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M4 8.75h8v-1.5H4v1.5Z"
    })]
  });
}
function LinkIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgLinkIcon
  });
}

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
      d: "M14.035 11.444A4 4 0 0 0 12 4H9v1.5h3a2.5 2.5 0 0 1 .917 4.826l1.118 1.118ZM14 13.53 2.47 2l-1 1 1.22 1.22A4.002 4.002 0 0 0 4 12h3v-1.5H4a2.5 2.5 0 0 1-.03-5l1.75 1.75H4v1.5h3.22L13 14.53l1-1Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "m9.841 7.25 1.5 1.5H12v-1.5H9.841Z"
    })]
  });
}
function LinkOffIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgLinkOffIcon
  });
}

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
      d: "M12 8.75H7v-1.5h5v1.5ZM7 5.5h5V4H7v1.5ZM12 12H7v-1.5h5V12ZM4.75 5.5a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5ZM5.5 8A.75.75 0 1 1 4 8a.75.75 0 0 1 1.5 0ZM4.75 12a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75V1.75Zm1.5.75v11h11v-11h-11Z",
      clipRule: "evenodd"
    })]
  });
}
function ListBorderIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgListBorderIcon
  });
}

function SvgListIcon(props) {
  return jsx("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: jsx("path", {
      fill: "currentColor",
      d: "M1.5 2.75a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM3 2h13v1.5H3V2ZM3 5.5h13V7H3V5.5ZM3 9h13v1.5H3V9ZM3 12.5h13V14H3v-1.5ZM.75 7a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5ZM1.5 13.25a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM.75 10.5a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z"
    })
  });
}
function ListIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgListIcon
  });
}

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
      d: "M12 6V4a4 4 0 0 0-8 0v2H2.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75H12ZM5.5 6h5V4a2.5 2.5 0 0 0-5 0v2Zm1.75 7V9h1.5v4h-1.5Z",
      clipRule: "evenodd"
    })
  });
}
function LockFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgLockFillIcon
  });
}

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
      d: "M7.25 9v4h1.5V9h-1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M12 6V4a4 4 0 0 0-8 0v2H2.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75H12Zm.5 1.5v7h-9v-7h9ZM5.5 4v2h5V4a2.5 2.5 0 0 0-5 0Z",
      clipRule: "evenodd"
    })]
  });
}
function LockIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgLockIcon
  });
}

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
      d: "M10 11.75v-1.5H6v1.5h4Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M13.25 6H5.5V4a2.5 2.5 0 0 1 5 0v.5H12V4a4 4 0 0 0-8 0v2H2.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75ZM3.5 7.5h9v7h-9v-7Z",
      clipRule: "evenodd"
    })]
  });
}
function LockUnlockedIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgLockUnlockedIcon
  });
}

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
      d: "M6.42 5.415A.75.75 0 0 0 5 5.75V11h1.5V8.927l.83 1.658a.75.75 0 0 0 1.34 0l.83-1.658V11H11V5.75a.75.75 0 0 0-1.42-.335L8 8.573 6.42 5.415Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H1.75Zm.75 12.5v-11h11v11h-11Z",
      clipRule: "evenodd"
    })]
  });
}
function MIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgMIcon
  });
}

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
      d: "M15 4H1V2.5h14V4Zm0 4.75H1v-1.5h14v1.5Zm0 4.75H1V12h14v1.5Z",
      clipRule: "evenodd"
    })
  });
}
function MenuIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgMenuIcon
  });
}

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
      d: "M11.5 8.75h-7v-1.5h7v1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H1.75Zm.75 12.5v-11h11v11h-11Z",
      clipRule: "evenodd"
    })]
  });
}
function MinusBoxIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgMinusBoxIcon
  });
}

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
      d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16Zm3.5-7.25h-7v-1.5h7v1.5Z",
      clipRule: "evenodd"
    })
  });
}
function MinusCircleFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgMinusCircleFillIcon
  });
}

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
      d: "M4.5 8.75v-1.5h7v1.5h-7Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Z",
      clipRule: "evenodd"
    })]
  });
}
function MinusCircleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgMinusCircleIcon
  });
}

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
        d: "M0 4.75a2.75 2.75 0 0 1 5.145-1.353l4.372-.95a2.75 2.75 0 1 1 3.835 2.823l.282 2.257a2.75 2.75 0 1 1-2.517 4.46l-2.62 1.145.003.118a2.75 2.75 0 1 1-4.415-2.19L3.013 7.489A2.75 2.75 0 0 1 0 4.75ZM2.75 3.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5Zm2.715 1.688c.018-.11.029-.22.033-.333l4.266-.928a2.753 2.753 0 0 0 2.102 1.546l.282 2.257c-.377.165-.71.412-.976.719L5.465 5.188ZM4.828 6.55a2.767 2.767 0 0 1-.413.388l1.072 3.573a2.747 2.747 0 0 1 2.537 1.19l2.5-1.093a2.792 2.792 0 0 1 .01-.797l-5.706-3.26ZM12 10.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0ZM5.75 12a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5ZM11 2.75a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0Z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "ModelsIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function ModelsIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgModelsIcon
  });
}

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
      d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0ZM1.5 8a6.5 6.5 0 0 1 10.535-5.096l-9.131 9.131A6.472 6.472 0 0 1 1.5 8Zm2.465 5.096a6.5 6.5 0 0 0 9.131-9.131l-9.131 9.131Z",
      clipRule: "evenodd"
    })
  });
}
function NoIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgNoIcon
  });
}

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
      d: "M3 1.75A.75.75 0 0 1 3.75 1h10.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H3.75a.75.75 0 0 1-.75-.75V12.5H1V11h2V8.75H1v-1.5h2V5H1V3.5h2V1.75Zm1.5.75v11H6v-11H4.5Zm3 0v11h6v-11h-6Z",
      clipRule: "evenodd"
    })
  });
}
function NotebookIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgNotebookIcon
  });
}

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
      d: "M8 1a5 5 0 0 0-5 5v1.99c0 .674-.2 1.332-.573 1.892l-1.301 1.952A.75.75 0 0 0 1.75 13h3.5v.25a2.75 2.75 0 1 0 5.5 0V13h3.5a.75.75 0 0 0 .624-1.166l-1.301-1.952A3.41 3.41 0 0 1 13 7.99V6a5 5 0 0 0-5-5Zm1.25 12h-2.5v.25a1.25 1.25 0 1 0 2.5 0V13ZM4.5 6a3.5 3.5 0 1 1 7 0v1.99c0 .97.287 1.918.825 2.724l.524.786H3.15l.524-.786A4.91 4.91 0 0 0 4.5 7.99V6Z",
      clipRule: "evenodd"
    })
  });
}
function NotificationIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgNotificationIcon
  });
}

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
      d: "m14.47 13.53-12-12-1 1L3.28 4.342A4.992 4.992 0 0 0 3 6v1.99c0 .674-.2 1.332-.573 1.892l-1.301 1.952A.75.75 0 0 0 1.75 13h3.5v.25a2.75 2.75 0 1 0 5.5 0V13h1.19l1.53 1.53 1-1ZM13.038 8.5A3.409 3.409 0 0 1 13 7.99V6a5 5 0 0 0-7.965-4.026l1.078 1.078A3.5 3.5 0 0 1 11.5 6v1.99c0 .158.008.316.023.472l.038.038h1.477ZM4.5 6c0-.14.008-.279.024-.415L10.44 11.5H3.151l.524-.786A4.91 4.91 0 0 0 4.5 7.99V6Zm2.25 7.25V13h2.5v.25a1.25 1.25 0 1 1-2.5 0Z",
      clipRule: "evenodd"
    })
  });
}
function NotificationOffIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgNotificationOffIcon
  });
}

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
      d: "M7.889 1A2.389 2.389 0 0 0 5.5 3.389H7c0-.491.398-.889.889-.889h.371a.74.74 0 0 1 .292 1.42l-1.43.613A2.675 2.675 0 0 0 5.5 6.992V8h5V6.5H7.108c.12-.26.331-.472.604-.588l1.43-.613A2.24 2.24 0 0 0 8.26 1H7.89ZM2.75 6a1.5 1.5 0 0 1-1.5 1.5H1V9h.25c.546 0 1.059-.146 1.5-.401V11.5H1V13h5v-1.5H4.25V6h-1.5ZM10 12.85A2.15 2.15 0 0 0 12.15 15h.725a2.125 2.125 0 0 0 1.617-3.504 2.138 2.138 0 0 0-1.656-3.521l-.713.008A2.15 2.15 0 0 0 10 10.133v.284h1.5v-.284a.65.65 0 0 1 .642-.65l.712-.009a.638.638 0 1 1 .008 1.276H12v1.5h.875a.625.625 0 1 1 0 1.25h-.725a.65.65 0 0 1-.65-.65v-.267H10v.267Z"
    })
  });
}
function NumbersIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgNumbersIcon
  });
}

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
      d: "M4 8.75h8v-1.5H4v1.5ZM7 5.75H4v-1.5h3v1.5ZM4 11.75h8v-1.5H4v1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V5a.75.75 0 0 0-.75-.75H10v-2.5A.75.75 0 0 0 9.25 1h-7.5Zm.75 1.5h6V5c0 .414.336.75.75.75h4.25v7.75h-11v-11Z",
      clipRule: "evenodd"
    })]
  });
}
function OfficeIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgOfficeIcon
  });
}

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
      d: "M8 1a1.75 1.75 0 1 0 0 3.5A1.75 1.75 0 0 0 8 1ZM8 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5ZM8 11.5A1.75 1.75 0 1 0 8 15a1.75 1.75 0 0 0 0-3.5Z"
    })
  });
}
function OverflowIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgOverflowIcon
  });
}

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
      d: "M1 3.06 2.06 2l5.97 5.97L14 2l1.06 1.06-7.03 7.031L1 3.061Zm14.03 10.47v1.5h-14v-1.5h14Z",
      clipRule: "evenodd"
    })
  });
}
function PageBottomIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPageBottomIcon
  });
}

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
      d: "m12.97 1 1.06 1.06-5.97 5.97L14.03 14l-1.06 1.06-7.03-7.03L12.97 1ZM2.5 15.03H1v-14h1.5v14Z",
      clipRule: "evenodd"
    })
  });
}
function PageFirstIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPageFirstIcon
  });
}

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
      d: "M3.06 1 2 2.06l5.97 5.97L2 14l1.06 1.06 7.031-7.03L3.061 1Zm10.47 14.03h1.5v-14h-1.5v14Z",
      clipRule: "evenodd"
    })
  });
}
function PageLastIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPageLastIcon
  });
}

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
      d: "m1 12.97 1.06 1.06 5.97-5.97L14 14.03l1.06-1.06-7.03-7.03L1 12.97ZM15.03 2.5V1h-14v1.5h14Z",
      clipRule: "evenodd"
    })
  });
}
function PageTopIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPageTopIcon
  });
}

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
      d: "M13.487 1.513a1.75 1.75 0 0 0-2.474 0L1.22 11.306a.75.75 0 0 0-.22.53v2.5c0 .414.336.75.75.75h2.5a.75.75 0 0 0 .53-.22l9.793-9.793a1.75 1.75 0 0 0 0-2.475l-1.086-1.085Zm-1.414 1.06a.25.25 0 0 1 .354 0l1.086 1.086a.25.25 0 0 1 0 .354L12 5.525l-1.44-1.44 1.513-1.512ZM9.5 5.146l-7 7v1.44h1.44l7-7-1.44-1.44Z",
      clipRule: "evenodd"
    })
  });
}
function PencilIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPencilIcon
  });
}

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
      d: "M5.75 0A.75.75 0 0 0 5 .75v1.19l9 9V9a.75.75 0 0 0-.22-.53l-2.12-2.122a2.25 2.25 0 0 1-.66-1.59V.75a.75.75 0 0 0-.75-.75h-4.5ZM10.94 12l2.53 2.53 1.06-1.06-11.5-11.5-1.06 1.06 2.772 2.773c-.104.2-.239.383-.4.545L2.22 8.47A.75.75 0 0 0 2 9v2.25c0 .414.336.75.75.75h4.5v4h1.5v-4h2.19Z"
    })
  });
}
function PinCancelIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPinCancelIcon
  });
}

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
      d: "M5 .75A.75.75 0 0 1 5.75 0h4.5a.75.75 0 0 1 .75.75v4.007c0 .597.237 1.17.659 1.591L13.78 8.47c.141.14.22.331.22.53v2.25a.75.75 0 0 1-.75.75h-4.5v4h-1.5v-4h-4.5a.75.75 0 0 1-.75-.75V9a.75.75 0 0 1 .22-.53L4.34 6.348A2.25 2.25 0 0 0 5 4.758V.75Z"
    })
  });
}
function PinFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPinFillIcon
  });
}

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
      d: "M5.75 0A.75.75 0 0 0 5 .75v4.007a2.25 2.25 0 0 1-.659 1.591L2.22 8.47A.75.75 0 0 0 2 9v2.25c0 .414.336.75.75.75h4.5v4h1.5v-4h4.5a.75.75 0 0 0 .75-.75V9a.75.75 0 0 0-.22-.53L11.66 6.348A2.25 2.25 0 0 1 11 4.758V.75a.75.75 0 0 0-.75-.75h-4.5Zm.75 4.757V1.5h3v3.257a3.75 3.75 0 0 0 1.098 2.652L12.5 9.311V10.5h-9V9.31L5.402 7.41A3.75 3.75 0 0 0 6.5 4.757Z",
      clipRule: "evenodd"
    })
  });
}
function PinIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPinIcon
  });
}

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
      d: "M10.75 6.75A5.75 5.75 0 0 0 5 1H1.75a.75.75 0 0 0-.75.75V6c0 .414.336.75.75.75H5a.25.25 0 0 1 .25.25v2.25A5.75 5.75 0 0 0 11 15h3.25a.75.75 0 0 0 .75-.75V10a.75.75 0 0 0-.75-.75H11a.25.25 0 0 1-.25-.25V6.75ZM5.5 2.53a4.25 4.25 0 0 1 3.75 4.22V9a1.75 1.75 0 0 0 1.25 1.678v2.793A4.25 4.25 0 0 1 6.75 9.25V7A1.75 1.75 0 0 0 5.5 5.322V2.53ZM4 2.5v2.75H2.5V2.5H4Zm9.5 8.25H12v2.75h1.5v-2.75Z",
      clipRule: "evenodd"
    })
  });
}
function PipelineIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPipelineIcon
  });
}

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
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm7.125-2.815A.75.75 0 0 0 6 5.835v4.33a.75.75 0 0 0 1.125.65l3.75-2.166a.75.75 0 0 0 0-1.299l-3.75-2.165Z",
      clipRule: "evenodd"
    })
  });
}
function PlayCircleFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPlayCircleFillIcon
  });
}

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
      d: "M11.25 8a.75.75 0 0 1-.375.65l-3.75 2.165A.75.75 0 0 1 6 10.165v-4.33a.75.75 0 0 1 1.125-.65l3.75 2.165a.75.75 0 0 1 .375.65Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0ZM1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0Z",
      clipRule: "evenodd"
    })]
  });
}
function PlayCircleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPlayCircleIcon
  });
}

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
      d: "M12.125 8.864a.75.75 0 0 0 0-1.3l-6-3.464A.75.75 0 0 0 5 4.75v6.928a.75.75 0 0 0 1.125.65l6-3.464Z"
    })
  });
}
function PlayIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPlayIcon
  });
}

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
      d: "m14.168 2.953.893-.892L14 1l-.893.893a4.001 4.001 0 0 0-5.077.48l-.884.884a.75.75 0 0 0 0 1.061l4.597 4.596a.75.75 0 0 0 1.06 0l.884-.884a4.001 4.001 0 0 0 .48-5.077ZM12.627 6.97l-.354.353-3.536-3.535.354-.354a2.5 2.5 0 1 1 3.536 3.536ZM7.323 10.152 5.91 8.737l1.414-1.414-1.06-1.06-1.415 1.414-.53-.53a.75.75 0 0 0-1.06 0l-.885.883a4.001 4.001 0 0 0-.48 5.077L1 14l1.06 1.06.893-.892a4.001 4.001 0 0 0 5.077-.48l.884-.885a.75.75 0 0 0 0-1.06l-.53-.53 1.414-1.415-1.06-1.06-1.415 1.414Zm-3.889 2.475a2.5 2.5 0 0 0 3.536 0l.353-.354-3.535-3.536-.354.354a2.5 2.5 0 0 0 0 3.536Z",
      clipRule: "evenodd"
    })
  });
}
function PlugIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPlugIcon
  });
}

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
      d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16Zm-.75-4.5V8.75H4.5v-1.5h2.75V4.5h1.5v2.75h2.75v1.5H8.75v2.75h-1.5Z",
      clipRule: "evenodd"
    })
  });
}
function PlusCircleFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPlusCircleFillIcon
  });
}

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
      d: "M7.25 11.5V8.75H4.5v-1.5h2.75V4.5h1.5v2.75h2.75v1.5H8.75v2.75h-1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0ZM1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0Z",
      clipRule: "evenodd"
    })]
  });
}
function PlusCircleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPlusCircleIcon
  });
}

function SvgPlusIcon(props) {
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
      d: "M7.25 7.25V1h1.5v6.25H15v1.5H8.75V15h-1.5V8.75H1v-1.5h6.25Z",
      clipRule: "evenodd"
    })
  });
}
function PlusIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPlusIcon
  });
}

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
      d: "M7.25 7.25V4.5h1.5v2.75h2.75v1.5H8.75v2.75h-1.5V8.75H4.5v-1.5h2.75Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75V1.75Zm1.5.75v11h11v-11h-11Z",
      clipRule: "evenodd"
    })]
  });
}
function PlusSquareIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgPlusSquareIcon
  });
}

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
      d: "M12 12H8v-1.5h4V12ZM5.53 11.53 7.56 9.5 5.53 7.47 4.47 8.53l.97.97-.97.97 1.06 1.06Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H1.75Zm.75 3V2.5h11V4h-11Zm0 1.5v8h11v-8h-11Z",
      clipRule: "evenodd"
    })]
  });
}
function QueryEditorIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgQueryEditorIcon
  });
}

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
        d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16h-.75a.75.75 0 0 1-.75-.75V1.75Zm7.5 1.81 1.94 1.94H9.5V3.56Z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M5.53 9.97 8.56 13l-3.03 3.03-1.06-1.06L6.44 13l-1.97-1.97 1.06-1.06ZM14 14.5H9V16h5v-1.5Z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "QueryIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function QueryIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgQueryIcon
  });
}

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
      d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16Zm2.207-10.189a2.25 2.25 0 0 1-1.457 2.56V9h-1.5V7.75A.75.75 0 0 1 8 7a.75.75 0 1 0-.75-.75h-1.5a2.25 2.25 0 0 1 4.457-.439ZM7.25 10.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Z",
      clipRule: "evenodd"
    })
  });
}
function QuestionMarkFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgQuestionMarkFillIcon
  });
}

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
      d: "M7.25 10.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0ZM10.079 7.111A2.25 2.25 0 1 0 5.75 6.25h1.5A.75.75 0 1 1 8 7a.75.75 0 0 0-.75.75V9h1.5v-.629a2.25 2.25 0 0 0 1.329-1.26Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Z",
      clipRule: "evenodd"
    })]
  });
}
function QuestionMarkIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgQuestionMarkIcon
  });
}

function SvgQuestionMarkSpeechBubbleIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 18 18",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M9 3.5a5.5 5.5 0 0 0-3.89 9.39.75.75 0 0 1 0 1.06l-.55.55h3.916a.75.75 0 0 1 .18-.01A5.5 5.5 0 1 0 9 3.5Zm-.295 12.494a7 7 0 1 0-5.157-2.603L2.22 14.72A.75.75 0 0 0 2.75 16h5.86a.762.762 0 0 0 .095-.006Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M11.079 8.111A2.25 2.25 0 1 0 6.75 7.25h1.5A.75.75 0 1 1 9 8a.75.75 0 0 0-.75.75V10h1.5v-.629a2.249 2.249 0 0 0 1.329-1.26ZM9 12.5A.75.75 0 1 1 9 11a.75.75 0 0 1 0 1.5Z",
      clipRule: "evenodd"
    })]
  });
}
function QuestionMarkSpeechBubbleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgQuestionMarkSpeechBubbleIcon
  });
}

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
      d: "M13 4.5h-3V6h3V4.5ZM13 7.25h-3v1.5h3v-1.5ZM13 10h-3v1.5h3V10Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H.75Zm.75 10.5v-9h5.75v9H1.5Zm7.25 0h5.75v-9H8.75v9Z",
      clipRule: "evenodd"
    })]
  });
}
function ReaderModeIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgReaderModeIcon
  });
}

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
        d: "m13.19 5-2.72-2.72 1.06-1.06 4.53 4.53-4.53 4.53-1.06-1.06 2.72-2.72H4.5a3 3 0 1 0 0 6H9V14H4.5a4.5 4.5 0 0 1 0-9h8.69Z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "RedoIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 16h16V0H0z"
        })
      })
    })]
  });
}
function RedoIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgRedoIcon
  });
}

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
      d: "M1 8a7 7 0 0 1 11.85-5.047l.65.594V2H15v4h-4V4.5h1.32l-.496-.453-.007-.007a5.5 5.5 0 1 0 .083 7.839l1.063 1.058A7 7 0 0 1 1 8Z",
      clipRule: "evenodd"
    })
  });
}
function RefreshIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgRefreshIcon
  });
}

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
      d: "M8 0a.75.75 0 0 1 .75.75V3h5.5a.75.75 0 0 1 .75.75V6h.25a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-.75.75H15v2.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75V12H.75a.75.75 0 0 1-.75-.75v-4.5A.75.75 0 0 1 .75 6H1V3.75A.75.75 0 0 1 1.75 3h5.5V.75A.75.75 0 0 1 8 0ZM2.5 4.5v9h11v-9h-11ZM5 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2Zm7-1a1 1 0 1 1-2 0 1 1 0 0 1 2 0Zm-6.25 2.25a.75.75 0 0 0 0 1.5h4.5a.75.75 0 0 0 0-1.5h-4.5Z",
      clipRule: "evenodd"
    })
  });
}
function RobotIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgRobotIcon
  });
}

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
      d: "M10 9.25H6v1.5h4v-1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 1.75A.75.75 0 0 1 1.75 1H11a.75.75 0 0 1 .53.22l3.25 3.25c.141.14.22.331.22.53v9.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75V1.75Zm1.5.75H5v3.75c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75V2.81l2.5 2.5v8.19h-11v-11Zm4 0h3v3h-3v-3Z",
      clipRule: "evenodd"
    })]
  });
}
function SaveIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSaveIcon
  });
}

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
      d: "M16 7a.75.75 0 0 0-.37-.647l-7.25-4.25a.75.75 0 0 0-.76 0L.37 6.353a.75.75 0 0 0 0 1.294L3 9.188V12a.75.75 0 0 0 .4.663l4.25 2.25a.75.75 0 0 0 .7 0l4.25-2.25A.75.75 0 0 0 13 12V9.188l1.5-.879V12H16V7Zm-7.62 4.897 3.12-1.83v1.481L8 13.401l-3.5-1.853v-1.48l3.12 1.829a.75.75 0 0 0 .76 0ZM8 3.619 2.233 7 8 10.38 13.767 7 8 3.62Z",
      clipRule: "evenodd"
    })
  });
}
function SchoolIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSchoolIcon
  });
}

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
        d: "M8 1a7 7 0 1 0 4.39 12.453l2.55 2.55 1.06-1.06-2.55-2.55A7 7 0 0 0 8 1ZM2.5 8a5.5 5.5 0 1 1 11 0 5.5 5.5 0 0 1-11 0Z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "SearchIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function SearchIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSearchIcon
  });
}

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
      d: "M2 1.75A.75.75 0 0 1 2.75 1h10.5a.75.75 0 0 1 .75.75v7.465a5.75 5.75 0 0 1-2.723 4.889l-2.882 1.784a.75.75 0 0 1-.79 0l-2.882-1.784A5.75 5.75 0 0 1 2 9.214V1.75Zm1.5.75V7h3.75V2.5H3.5Zm5.25 0V7h3.75V2.5H8.75Zm3.75 6H8.75v5.404l1.737-1.076A4.25 4.25 0 0 0 12.5 9.215V8.5Zm-5.25 5.404V8.5H3.5v.715a4.25 4.25 0 0 0 2.013 3.613l1.737 1.076Z",
      clipRule: "evenodd"
    })
  });
}
function SecurityIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSecurityIcon
  });
}

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
      d: "M3.97 5.03 8 1l4.03 4.03-1.06 1.061-2.22-2.22v7.19h-1.5V3.87l-2.22 2.22-1.06-1.06Z"
    }), jsx("path", {
      fill: "currentColor",
      d: "M2.5 13.56v-6.5H1v7.25c0 .415.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V7.06h-1.5v6.5h-11Z"
    })]
  });
}
function ShareIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgShareIcon
  });
}

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
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1H1.75ZM4 2.5H2.5v11H4v-11Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "m9.06 8 1.97 1.97-1.06 1.06L6.94 8l3.03-3.03 1.06 1.06L9.06 8ZM11.97 6.03 13.94 8l-1.97 1.97 1.06 1.06L16.06 8l-3.03-3.03-1.06 1.06Z"
    })]
  });
}
function SidebarAutoIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSidebarAutoIcon
  });
}

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
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1H1.75ZM4 2.5H2.5v11H4v-11Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "m9.81 8.75 1.22 1.22-1.06 1.06L6.94 8l3.03-3.03 1.06 1.06-1.22 1.22H14v1.5H9.81Z"
    })]
  });
}
function SidebarCollapseIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSidebarCollapseIcon
  });
}

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
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1H1.75ZM4 2.5H2.5v11H4v-11Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      d: "M11.19 8.75 9.97 9.97l1.06 1.06L14.06 8l-3.03-3.03-1.06 1.06 1.22 1.22H7v1.5h4.19Z"
    })]
  });
}
function SidebarExpandIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSidebarExpandIcon
  });
}

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
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H1.75Zm.75 12.5v-11H4v11H2.5Zm3 0h8v-11h-8v11Z",
      clipRule: "evenodd"
    })
  });
}
function SidebarIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSidebarIcon
  });
}

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
      d: "M2 3.104V2h1.5v1.104a2.751 2.751 0 0 1 0 5.292V14H2V8.396a2.751 2.751 0 0 1 0-5.292ZM1.5 5.75a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0ZM12.5 2v1.104a2.751 2.751 0 0 0 0 5.292V14H14V8.396a2.751 2.751 0 0 0 0-5.292V2h-1.5Zm.75 2.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5ZM7.25 14v-1.104a2.751 2.751 0 0 1 0-5.292V2h1.5v5.604a2.751 2.751 0 0 1 0 5.292V14h-1.5ZM8 11.5A1.25 1.25 0 1 1 8 9a1.25 1.25 0 0 1 0 2.5Z",
      clipRule: "evenodd"
    })
  });
}
function SlidersIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSlidersIcon
  });
}

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
        d: "M4.307 0a.75.75 0 0 1 .697.473L7.596 7H5.982l-.238-.6H2.855l-.24.6H1L3.61.472A.75.75 0 0 1 4.307 0Zm-.852 4.9h1.693l-.844-2.124L3.455 4.9Z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M4.777 9.5H1.5V8h4.75a.75.75 0 0 1 .607 1.191L3.723 13.5H7V15H2.25a.75.75 0 0 1-.607-1.191L4.777 9.5ZM12 .94l4.03 4.03-1.06 1.06-2.22-2.22V10h-1.5V3.81L9.03 6.03 7.97 4.97 12 .94Z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "SortAlphabeticalAscendingIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function SortAlphabeticalAscendingIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSortAlphabeticalAscendingIcon
  });
}

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
        d: "M4.307 0a.75.75 0 0 1 .697.473L7.596 7H5.982l-.238-.6H2.855l-.24.6H1L3.61.472A.75.75 0 0 1 4.307 0Zm-.852 4.9h1.693l-.844-2.124L3.455 4.9Z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M4.777 9.5H1.5V8h4.75a.75.75 0 0 1 .607 1.191L3.723 13.5H7V15H2.25a.75.75 0 0 1-.607-1.191L4.777 9.5ZM12 15.06l-4.03-4.03 1.06-1.06 2.22 2.22V6h1.5v6.19l2.22-2.22 1.06 1.06L12 15.06Z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "SortAlphabeticalDescendingIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function SortAlphabeticalDescendingIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSortAlphabeticalDescendingIcon
  });
}

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
        d: "M.94 4 4.97-.03l1.06 1.06-2.22 2.22H10v1.5H3.81l2.22 2.22-1.06 1.06L.94 4Z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M4.307 9a.75.75 0 0 1 .697.473L7.596 16H5.982l-.238-.6H2.855l-.24.6H1l2.61-6.528A.75.75 0 0 1 4.307 9Zm-.852 4.9h1.693l-.844-2.124-.849 2.124Z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M11.777 10.5H8.5V9h4.75a.75.75 0 0 1 .607 1.191L10.723 14.5H14V16H9.25a.75.75 0 0 1-.607-1.191l3.134-4.309Z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "SortAlphabeticalLeftIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function SortAlphabeticalLeftIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSortAlphabeticalLeftIcon
  });
}

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
        d: "m14.06 4-4.03 4.03-1.06-1.06 2.22-2.22H5v-1.5h6.19L8.97 1.03l1.06-1.06L14.06 4Z"
      }), jsx("path", {
        fillRule: "evenodd",
        d: "M4.307 9a.75.75 0 0 1 .697.473L7.596 16H5.982l-.238-.6H2.855l-.24.6H1l2.61-6.528A.75.75 0 0 1 4.307 9Zm-.852 4.9h1.693l-.844-2.124-.849 2.124Z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M11.777 10.5H8.5V9h4.75a.75.75 0 0 1 .607 1.191l-3.134 4.31H14V16H9.25a.75.75 0 0 1-.607-1.192l3.134-4.309Z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "SortAlphabeticalRightIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function SortAlphabeticalRightIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSortAlphabeticalRightIcon
  });
}

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
      d: "m11.5.94 4.03 4.03-1.06 1.06-2.22-2.22V10h-1.5V3.81L8.53 6.03 7.47 4.97 11.5.94ZM1 4.5h4V6H1V4.5ZM1 12.5h10V14H1v-1.5ZM8 8.5H1V10h7V8.5Z"
    })
  });
}
function SortAscendingIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSortAscendingIcon
  });
}

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
      d: "M1 3.5h10V2H1v1.5Zm0 8h4V10H1v1.5Zm7-4H1V6h7v1.5Zm3.5 7.56 4.03-4.03-1.06-1.06-2.22 2.22V6h-1.5v6.19L8.53 9.97l-1.06 1.06 4.03 4.03Z",
      clipRule: "evenodd"
    })
  });
}
function SortDescendingIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSortDescendingIcon
  });
}

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
      d: "M11.5.94 7.47 4.97l1.06 1.06 2.22-2.22v8.38L8.53 9.97l-1.06 1.06 4.03 4.03 4.03-4.03-1.06-1.06-2.22 2.22V3.81l2.22 2.22 1.06-1.06L11.5.94ZM6 3.5H1V5h5V3.5ZM6 11.5H1V13h5v-1.5ZM1 7.5h5V9H1V7.5Z"
    })
  });
}
function SortUnsortedIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSortUnsortedIcon
  });
}

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
      d: "M8 8.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5ZM11.5 8A.75.75 0 1 1 10 8a.75.75 0 0 1 1.5 0ZM5.25 8.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 15c-.099 0-.197-.002-.295-.006A.762.762 0 0 1 7.61 15H1.75a.75.75 0 0 1-.53-1.28l1.328-1.329A7 7 0 1 1 8 15ZM2.5 8a5.5 5.5 0 1 1 5.156 5.49.75.75 0 0 0-.18.01H3.56l.55-.55a.75.75 0 0 0 0-1.06A5.48 5.48 0 0 1 2.5 8Z",
      clipRule: "evenodd"
    })]
  });
}
function SpeechBubbleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSpeechBubbleIcon
  });
}

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
      d: "M7.25 11V8.75H5v-1.5h2.25V5h1.5v2.25H11v1.5H8.75V11h-1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 15c-.099 0-.197-.002-.295-.006A.762.762 0 0 1 7.61 15H1.75a.75.75 0 0 1-.53-1.28l1.328-1.329A7 7 0 1 1 8 15ZM2.5 8a5.5 5.5 0 1 1 5.156 5.49.75.75 0 0 0-.18.01H3.56l.55-.55a.75.75 0 0 0 0-1.06A5.48 5.48 0 0 1 2.5 8Z",
      clipRule: "evenodd"
    })]
  });
}
function SpeechBubblePlusIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSpeechBubblePlusIcon
  });
}

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
      d: "M7.995 0a.75.75 0 0 1 .714.518l1.459 4.492h4.723a.75.75 0 0 1 .44 1.356l-3.82 2.776 1.459 4.492a.75.75 0 0 1-1.154.838l-3.82-2.776-3.821 2.776a.75.75 0 0 1-1.154-.838L4.48 9.142.66 6.366A.75.75 0 0 1 1.1 5.01h4.723L7.282.518A.75.75 0 0 1 7.995 0Z"
    })
  });
}
function StarFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgStarFillIcon
  });
}

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
      d: "M7.995 0a.75.75 0 0 1 .714.518l1.459 4.492h4.723a.75.75 0 0 1 .44 1.356l-3.82 2.776 1.459 4.492a.75.75 0 0 1-1.154.838l-3.82-2.776-3.821 2.776a.75.75 0 0 1-1.154-.838L4.48 9.142.66 6.366A.75.75 0 0 1 1.1 5.01h4.723L7.282.518A.75.75 0 0 1 7.995 0Zm0 3.177-.914 2.814a.75.75 0 0 1-.713.519h-2.96l2.394 1.739a.75.75 0 0 1 .273.839l-.915 2.814 2.394-1.74a.75.75 0 0 1 .882 0l2.394 1.74-.914-2.814a.75.75 0 0 1 .272-.839l2.394-1.74H9.623a.75.75 0 0 1-.713-.518l-.915-2.814Z",
      clipRule: "evenodd"
    })
  });
}
function StarIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgStarIcon
  });
}

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
      d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16ZM6.125 5.5a.625.625 0 0 0-.625.625v3.75c0 .345.28.625.625.625h3.75c.345 0 .625-.28.625-.625v-3.75a.625.625 0 0 0-.625-.625h-3.75Z",
      clipRule: "evenodd"
    })
  });
}
function StopCircleFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgStopCircleFillIcon
  });
}

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
      d: "M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0ZM8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0ZM5.5 6a.5.5 0 0 1 .5-.5h4a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-.5.5H6a.5.5 0 0 1-.5-.5V6Z",
      clipRule: "evenodd"
    })
  });
}
function StopCircleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgStopCircleIcon
  });
}

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
      d: "M4.5 4a.5.5 0 0 0-.5.5v7a.5.5 0 0 0 .5.5h7a.5.5 0 0 0 .5-.5v-7a.5.5 0 0 0-.5-.5h-7Z"
    })
  });
}
function StopIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgStopIcon
  });
}

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
      d: "M3.52 2.3a.75.75 0 0 1 .6-.3h7.76a.75.75 0 0 1 .6.3l2.37 3.158a.75.75 0 0 1 .15.45v.842c0 .04-.003.077-.009.115A2.311 2.311 0 0 1 14 8.567v5.683a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75V8.567A2.311 2.311 0 0 1 1 6.75v-.841a.75.75 0 0 1 .15-.45l2.37-3.16Zm7.605 6.068c.368.337.847.557 1.375.6V13.5h-9V8.968a2.303 2.303 0 0 0 1.375-.6c.411.377.96.607 1.563.607.602 0 1.15-.23 1.562-.607.411.377.96.607 1.563.607.602 0 1.15-.23 1.562-.607Zm2.375-2.21v.532l-.001.019a.813.813 0 0 1-1.623 0 .754.754 0 0 0-.008-.076.756.756 0 0 0 .012-.133V4L13.5 6.16Zm-3.113.445a.762.762 0 0 0-.013.106.813.813 0 0 1-1.624-.019V3.5h1.63v3c0 .035.002.07.007.103ZM7.25 3.5v3.19a.813.813 0 0 1-1.624.019.757.757 0 0 0-.006-.064V3.5h1.63ZM4.12 4 2.5 6.16v.531l.001.019a.813.813 0 0 0 1.619.045V4Z",
      clipRule: "evenodd"
    })
  });
}
function StorefrontIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgStorefrontIcon
  });
}

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
      d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0ZM1.52 7.48a6.5 6.5 0 0 1 12.722-1.298l-.09-.091a3.75 3.75 0 0 0-5.304 0L6.091 8.848a2.25 2.25 0 0 1-3.182 0L1.53 7.47l-.01.01Zm.238 2.338A6.5 6.5 0 0 0 14.48 8.52l-.01.01-1.379-1.378a2.25 2.25 0 0 0-3.182 0L7.152 9.909a3.75 3.75 0 0 1-5.304 0l-.09-.09Z",
      clipRule: "evenodd"
    })
  });
}
function StreamIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgStreamIcon
  });
}

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
      d: "M8 2.5a5.48 5.48 0 0 1 3.817 1.54l.009.009.5.451H11V6h4V2h-1.5v1.539l-.651-.588A7 7 0 0 0 1 8h1.5A5.5 5.5 0 0 1 8 2.5ZM1 10h4v1.5H3.674l.5.451.01.01A5.5 5.5 0 0 0 13.5 8h1.499a7 7 0 0 1-11.849 5.048L2.5 12.46V14H1v-4Z"
    })
  });
}
function SyncIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgSyncIcon
  });
}

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
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75V1.75Zm1.5.75v3h11v-3h-11Zm0 11V7H5v6.5H2.5Zm4 0h3V7h-3v6.5ZM11 7v6.5h2.5V7H11Z",
      clipRule: "evenodd"
    })
  });
}
function TableIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgTableIcon
  });
}

function SvgTableInfinityIcon(props) {
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
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75V9h-1.5V6h-11v7.5H4V15H1.75a.75.75 0 0 1-.75-.75V1.75Zm12.5.75v2H8.75v-2h4.75Zm-11 0h4.75v2H2.5v-2Zm10.693 9.295a1.01 1.01 0 0 0-1.417 0l-.002.002-.001.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.001-.002.002-.002.003-.002.002-.002.002-.002.002-.003.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.003-.003.002-.002.002-.002.002-.002.002-.003.003-.002.002-.002.002-.002.002-.003.003-.002.002-.002.002-.003.002-.002.003-.002.002-.003.002-.002.003-.002.002-.003.002-.002.003-.003.002-.002.002-.002.003-.003.002-.002.003-.003.002-.002.003-.003.002-.002.003-.003.002-.002.003-.003.002-.002.003-.003.002-.002.003-.003.002-.003.003-.002.002-.003.003-.002.002-.003.003-.003.003-.002.002-.003.003-.003.002-.002.003-.003.003-.003.002-.002.003-.003.003-.003.002-.002.003-.003.003-.003.002-.003.003-.002.003-.003.003-.003.002-.003.003-.002.003-.003.003-.003.002-.003.003-.003.003-.003.003-.002.002-.003.003-.003.003-.003.003-.003.003-.003.003-.002.002-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.002-.003.003-.003.003-.002.003-.004.003-.003.003-.003.003-.002.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.004.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.004.003-.003.003-.003.004-.003.003-.003.003-.003.003-.003.003-.004.003-.003.003-.003.003-.003.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.004.003-.003.003-.003.004-.003.003-.003.003-.004.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.004.003-.003.004-.003.003-.004.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.004.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.003.003-.004.003-.003.004-.003.003.003.003.003.003.003.003.003.003.003.003.003.004.004.003.003.003.003.003.003.003.003.003.004.003.003.004.003.003.003.003.003.003.004.003.003.003.003.004.003.003.003.003.004.003.003.003.003.004.003.003.003.003.004.003.003.003.003.003.003.004.004.003.003.003.003.003.003.003.004.004.003.003.003.003.003.003.004.003.003.004.003.003.003.003.004.003.003.003.003.004.003.003.003.003.004.003.003.003.003.004.003.003.004.003.003.003.003.003.003.004.004.003.003.003.003.003.003.004.004.003.003.003.003.003.003.003.004.004.003.003.003.003.003.003.004.003.003.004.003.003.003.003.003.003.004.003.003.004.003.003.003.003.004.003.003.003.003.003.003.004.003.003.004.003.003.003.003.003.003.004.004.003.003.003.003.003.003.003.003.003.004.004.003.003.003.003.003.003.003.003.004.003.003.003.003.003.003.004.003.003.003.003.004.003.003.003.003.003.003.003.003.004.003.003.004.003.003.003.003.003.003.003.003.003.003.003.003.003.003.004.004.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.004.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.002.002.003.003.003.003.003.003.003.003.003.003.003.003.002.003.003.003.003.003.003.002.003.003.003.003.002.003.003.003.003.002.003.003.003.003.002.003.003.003.003.002.003.003.002.003.003.003.003.003.002.002.003.003.003.003.003.002.002.003.003.003.003.003.002.002.003.003.003.003.002.002.003.003.003.003.002.002.003.003.002.002.003.003.002.003.003.002.003.003.002.002.003.003.002.003.003.002.002.003.003.002.002.003.003.002.002.003.003.002.002.003.003.002.002.003.003.002.002.002.002.003.003.002.002.003.003.002.002.002.002.003.003.002.002.003.002.002.003.002.002.003.002.002.003.002.002.002.002.003.002.002.003.002.002.003.002.002.002.002.003.002.002.002.002.003.002.002.002.002.002.002.002a1.01 1.01 0 0 0 1.417 0 .983.983 0 0 0 0-1.4Zm-3.192-.354-.003-.003-.003-.003-.003-.003-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003L9.556 11l-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.002-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.002-.003-.003-.003-.003-.002-.003-.003-.003-.003-.003-.003-.003-.003-.003-.002-.002-.003-.003-.003-.003-.003-.003-.003-.003-.002-.003-.003-.002-.003-.003-.003-.003-.003-.003-.002-.002-.003-.003-.003-.003-.003-.003-.002-.003-.003-.002-.003-.003-.002-.003-.003-.002-.003-.003-.002-.003-.003-.002-.003-.003-.002-.003-.003-.002-.003-.003-.002-.003-.003-.002-.003-.003-.002-.003-.003-.002-.002-.003-.003-.002-.002-.003-.003-.003-.002-.002-.003-.003-.003-.002-.002-.003-.003-.002-.002-.003-.003-.002-.002-.003-.003-.002-.002-.003-.002-.002-.003-.003-.002-.002-.003-.003-.002-.002-.003-.003-.002-.002-.002-.002-.003-.003-.002-.002-.002-.002-.003-.003-.002-.002-.003-.003-.002-.002-.002-.002-.002-.002-.003-.003-.002-.002-.002-.002-.002-.003-.003-.002-.002-.002-.002-.002-.002-.003-.002-.002-.003-.002-.002-.002-.002-.002-.002-.002-.002a2.53 2.53 0 0 0-3.55 0c-.981.97-.981 2.54 0 3.51.98.97 2.57.97 3.55 0l.001-.002.002-.002.002-.001.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.003-.002.002-.003.002-.002.002-.002.002-.002.002-.002.003-.002.002-.002.002-.003.002-.002.003-.002.002-.002.002-.003.002-.002.003-.002.002-.002.002-.003.003-.002.002-.002.002-.003.003-.002.002-.002.003-.003.002-.002.002-.003.003-.002.002-.002.003-.003.002-.002.003-.003.002-.002.003-.003.002-.002.003-.002.002-.003.003-.002.002-.003.003-.002.002-.003.003-.003.002-.002.003-.003.003-.002.002-.003.003-.003.002-.002.003-.003.003-.002.002-.003.003-.003.003-.002.002-.003.003-.002.003-.003.003-.003.002-.003.003-.002.003-.003.002-.003.003-.002.003-.003.003-.003.002-.003.003-.002.003-.003.003-.003.003-.003.002-.002.003-.003.003-.003.003-.003.003-.003.003-.002.003-.003.002-.003.003-.003.003-.003.003-.003.003-.003.003-.002.003-.003.003-.003.003-.003.003-.003.002-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.002.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.004-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.004.003-.003.003-.003.003-.003.004-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.003.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.003-.003.004-.003.003-.003.003-.004.003-.003.003-.003.004-.003.003-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.004-.003.003-.003.003-.004.003-.003.003-.003.004-.003.003-.003.003.002.003.004.003.003.004.003.003.003.003.003.003.003.003.004.003.003.004.003.003.003.003.003.003.003.003.003.004.004.003.003.003.003.003.003.003.003.004.003.003.004.003.003.003.003.003.003.004.003.003.004.003.003.003.003.004.003.003.003.003.003.003.004.004.003.003.003.003.003.003.003.003.004.004.003.003.003.003.003.003.003.004.004.003.003.003.003.003.003.004.003.003.004.003.003.003.003.004.003.003.004.003.003.003.003.004.003.003.003.003.004.003.003.004.003.003.003.003.003.003.004.003.003.004.003.003.003.003.003.003.004.004.003.003.003.003.003.003.003.004.004.003.003.003.003.003.003.003.003.004.004.003.003.003.003.003.003.004.003.003.003.003.004.003.003.003.003.004.003.003.003.003.004.003.003.003.003.004.003.003.003.003.003.003.004.003.003.004.003.003.003.003.003.003.003.003.003.004.003.003.004.003.003.003.003.003.003.003.003.004.003.003.003.003.003.003.004.003.003.003.003.003.003.004.003.003.003.003.003.003.003.003.003.003.003.003.004.003.003.003.003.004.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.002.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.003.002.002.003.003.003.003.003.003.003.003.003.003.003.003.002.003.003.002.003.003.003.003.003.003.002.003.003.002.003.003.003.003.002.003.003.003.003.002.003.003.003.003.002.003.003.002.003.003.002.003.003.003.003.002.002.003.003.003.003.002.002.003.003.003.003.002.002.003.003.003.002.002.003.003.003.002.002.003.003.003.002.002.003.003.002.002.003.003.002.003.003.002.003.002.002.003.002.002.003.003.002.002.003.003.002.002.003.003.002.002.003.003.002.002.002.003.003.002.002.002.003.003.002.002.002.003.003.002.002.002.002.003.003.002.002.002.002.003.002.002.003.002.002.002.002.003.003.002.002.002.002.002.002.003.002.002.003.002.002.002.002.002.002.003.002c.98.97 2.57.97 3.55 0s.98-2.54 0-3.51a2.53 2.53 0 0 0-3.55 0l-.002.002-.002.001-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.003.002-.002.003-.002.002-.002.002-.002.002-.002.002-.003.002-.002.003-.002.002-.002.002-.003.002-.002.003-.002.002-.003.002-.002.002-.002.003-.003.002-.002.002-.002.003-.003.002-.002.002-.002.003-.003.002-.002.002-.003.003-.002.002-.003.003-.002.002-.002.003-.003.002-.003.002-.002.003-.002.002-.003.003-.002.002-.003.003-.003.002-.002.003-.003.002-.002.003-.003.003-.002.002-.003.003-.003.002-.002.003-.003.003-.003.002-.002.003-.003.002-.003.003-.002.003-.003.002-.003.003-.002.003-.003.002-.003.003-.002.003-.003.003-.003.002-.003.003-.002.003-.003.002-.003.003-.003.003-.003.003-.003.002-.002.003-.003.003-.003.003-.003.003-.003.002-.002.003-.003.003-.003.003-.003.003-.003.003-.003.002-.003.003-.003.003-.003.003-.002.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.002-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.004.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.004-.003.003-.003.003-.003.003-.003.003-.003.003-.004.003-.003.003-.003.003-.003.003-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.003.003-.003.003-.004.003-.003.003-.003.004-.003.003-.003.003-.004.003-.003.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.004.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.004.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.004.003-.003.003-.003.004-.003.003-.003.003-.004.003-.003.003-.003.003-.003.004-.003.003-.004.003Zm-3.192.354a.983.983 0 0 0 0 1.402 1.01 1.01 0 0 0 1.417 0l.002-.002.002-.002.002-.002.001-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.003-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.002-.002.003-.003.002-.002.002-.002.002-.002.003-.002.002-.003.002-.002.002-.002.003-.002.002-.003.002-.002.002-.002.003-.002.002-.003.002-.002.003-.002.002-.003.003-.002.002-.003.002-.002.003-.002.002-.003.003-.002.002-.002.003-.003.002-.002.002-.003.003-.002.002-.003.003-.002.002-.003.003-.002.003-.003.002-.002.003-.003.002-.002.003-.003.002-.002.003-.003.003-.002.002-.003.003-.003.002-.002.003-.003.003-.003.002-.002.003-.003.003-.002.002-.003.003-.003.003-.002.003-.003.002-.003.003-.003.003-.002.003-.003.002-.003.003-.002.003-.003.003-.003.002-.003.003-.002.003-.003.003-.003.003-.003.002-.003.003-.002.003-.003.003-.003.003-.003.003-.003.003-.003.003-.002.002-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.002.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.003-.003.004-.003.003-.003.003-.003.003-.004.003-.003.003-.003.003-.003.003-.003.004-.003.003-.003.003-.003.003-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.004-.003.003-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.004.003-.003.004-.003.003-.003.003-.003.003-.004.004-.003.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.003-.003.004-.004.003-.003.003-.003.003-.003.004-.003.003-.004.003-.003.003-.003-.002-.003-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.003-.003-.003-.004-.004-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.004-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.004-.003-.003-.003-.003-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.004-.003-.003-.003-.003-.004-.003-.003-.003-.003-.003-.003-.003-.003-.003-.002-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.002-.002-.003-.003-.003-.003-.003-.003-.003-.003L8.433 12l-.003-.003-.003-.002-.003-.003-.002-.003-.003-.003-.003-.003-.003-.003-.003-.003-.003-.002-.003-.003-.002-.003-.003-.003-.003-.002-.003-.003-.003-.003-.002-.003-.003-.002-.003-.003-.003-.003-.002-.003-.003-.002-.003-.003-.003-.003-.002-.002-.003-.003-.003-.003-.002-.003-.003-.002-.003-.003-.002-.002-.003-.003-.003-.003-.002-.002-.003-.003-.003-.002-.002-.003-.003-.003-.003-.002-.002-.003-.003-.002-.002-.003-.003-.002-.002-.003-.003-.002-.002-.003-.003-.002-.003-.003-.002-.002-.003-.003-.002-.002-.002-.003-.003-.002-.002-.003-.003-.002-.002-.002-.003-.003-.002-.002-.003-.003-.002-.002-.002-.002-.003-.003-.002-.002-.002-.002-.003-.003-.002-.002-.002-.002-.003-.003-.002-.002-.002-.002-.003-.002-.002-.003-.002-.002-.002-.002-.003-.002-.002-.003-.002-.002-.002-.002-.003-.002-.002-.002-.002-.003-.002-.002a1.01 1.01 0 0 0-1.417 0Z",
      clipRule: "evenodd"
    })
  });
}
function TableInfinityIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgTableInfinityIcon
  });
}

function SvgTableLightningIcon(props) {
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
      d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75V9h-1.5V6h-11v7.5H8V15H1.75a.75.75 0 0 1-.75-.75V1.75Zm12.5.75v2H8.75v-2h4.75Zm-11 0h4.75v2H2.5v-2Zm8.43 4.512-3 3.5a.75.75 0 0 0 .57 1.238h3.37l-1.94 2.262 1.14.976 3-3.5a.75.75 0 0 0-.57-1.238h-3.37l1.94-2.262-1.14-.976Z",
      clipRule: "evenodd"
    })
  });
}
function TableLightningIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgTableLightningIcon
  });
}

function SvgTableViewIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("g", {
      clipPath: "url(#TableViewIcon_svg__a)",
      children: jsx("path", {
        fill: "currentColor",
        fillRule: "evenodd",
        d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75V9h-1.5V6h-11v7.5H4V15H1.75a.75.75 0 0 1-.75-.75V1.75Zm12.5.75v2H8.75v-2h4.75Zm-11 0h4.75v2H2.5v-2Zm4.048 9.953c.55 1.183 1.862 2.06 3.45 2.06 1.587 0 2.9-.877 3.449-2.06-.55-1.183-1.862-2.06-3.45-2.06-1.587 0-2.9.877-3.449 2.06ZM5.033 12.2c.701-1.955 2.688-3.307 4.964-3.307 2.277 0 4.264 1.352 4.965 3.307a.75.75 0 0 1 0 .507c-.701 1.954-2.688 3.307-4.965 3.307-2.276 0-4.263-1.353-4.964-3.307a.75.75 0 0 1 0-.507Zm6.217.3a1.25 1.25 0 1 1-2.5 0 1.25 1.25 0 0 1 2.5 0Z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "TableViewIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function TableViewIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgTableViewIcon
  });
}

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
      d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75H1.75Zm.75 12.5v-11h11v11h-11ZM5 6h2.25v5.5h1.5V6H11V4.5H5V6Z",
      clipRule: "evenodd"
    })
  });
}
function TextBoxIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgTextBoxIcon
  });
}

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
        d: "M13.655 2.274a.79.79 0 0 0-.528-.19h-1.044v5.833h1.044a.79.79 0 0 0 .79-.643V2.725a.79.79 0 0 0-.262-.451Zm-3.072 6.233V2.083H3.805a.583.583 0 0 0-.583.496v.001l-.92 6a.585.585 0 0 0 .583.67h3.782a.75.75 0 0 1 .75.75v2.667a1.25 1.25 0 0 0 .8 1.166l2.366-5.326Zm1.238.91L9.352 14.97a.75.75 0 0 1-.685.446 2.75 2.75 0 0 1-2.75-2.75V10.75h-3.02A2.082 2.082 0 0 1 .82 8.354l.92-6A2.085 2.085 0 0 1 3.816.584h9.29a2.29 2.29 0 0 1 2.303 1.982.751.751 0 0 1 .007.1v4.667a.751.751 0 0 1-.007.1 2.29 2.29 0 0 1-2.303 1.984h-1.286Z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "ThumbsDownIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function ThumbsDownIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgThumbsDownIcon
  });
}

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
        d: "M6.648 1.029a.75.75 0 0 1 .685-.446 2.75 2.75 0 0 1 2.75 2.75V5.25h3.02a2.083 2.083 0 0 1 2.079 2.396l-.92 6a2.085 2.085 0 0 1-2.08 1.77H2.668a2.083 2.083 0 0 1-2.084-2.082V8.667a2.083 2.083 0 0 1 2.084-2.083h1.512l2.469-5.555ZM3.917 8.084h-1.25a.583.583 0 0 0-.584.583v4.667a.583.583 0 0 0 .584.583h1.25V8.084Zm1.5 5.833h6.778a.583.583 0 0 0 .583-.496l.92-6a.584.584 0 0 0-.583-.67H9.333a.75.75 0 0 1-.75-.75V3.332a1.25 1.25 0 0 0-.8-1.166L5.417 7.493v6.424Z",
        clipRule: "evenodd"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "ThumbsUpIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function ThumbsUpIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgThumbsUpIcon
  });
}

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
      d: "M6 0a.75.75 0 0 0-.712.513L4.46 3H1v1.5h1.077l1.177 10.831A.75.75 0 0 0 4 16h8a.75.75 0 0 0 .746-.669L13.923 4.5H15V3h-3.46L10.712.513A.75.75 0 0 0 10 0H6Zm3.96 3-.5-1.5H6.54L6.04 3h3.92ZM3.585 4.5l1.087 10h6.654l1.087-10H3.586Z",
      clipRule: "evenodd"
    })
  });
}
function TrashIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgTrashIcon
  });
}

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
      d: "M2.004 9.602a2.751 2.751 0 1 0 3.371 3.47 2.751 2.751 0 0 0 5.25 0 2.751 2.751 0 1 0 3.371-3.47A2.75 2.75 0 0 0 11.25 7h-2.5v-.604a2.751 2.751 0 1 0-1.5 0V7h-2.5a2.75 2.75 0 0 0-2.746 2.602ZM2.75 11a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5Zm4.5-2.5h-2.5a1.25 1.25 0 0 0-1.242 1.106 2.756 2.756 0 0 1 1.867 1.822A2.756 2.756 0 0 1 7.25 9.604V8.5Zm1.5 0v1.104c.892.252 1.6.942 1.875 1.824a2.756 2.756 0 0 1 1.867-1.822A1.25 1.25 0 0 0 11.25 8.5h-2.5ZM12 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0Zm-5.25 0a1.25 1.25 0 1 0 2.5 0 1.25 1.25 0 0 0-2.5 0ZM8 5a1.25 1.25 0 1 1 0-2.5A1.25 1.25 0 0 1 8 5Z",
      clipRule: "evenodd"
    })
  });
}
function TreeIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgTreeIcon
  });
}

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
      d: "M4.544 6.466 4.6 2.988l1.5.024-.056 3.478A1.978 1.978 0 1 0 10 6.522V3h1.5v3.522a3.478 3.478 0 1 1-6.956-.056ZM12 13H4v-1.5h8V13Z",
      clipRule: "evenodd"
    })
  });
}
function UnderlineIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgUnderlineIcon
  });
}

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
        d: "M2.81 6.5h8.69a3 3 0 0 1 0 6H7V14h4.5a4.5 4.5 0 0 0 0-9H2.81l2.72-2.72-1.06-1.06-4.53 4.53 4.53 4.53 1.06-1.06L2.81 6.5Z"
      })
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "UndoIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M16 16H0V0h16z"
        })
      })
    })]
  });
}
function UndoIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgUndoIcon
  });
}

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
      d: "M1 13.56h14v1.5H1v-1.5ZM12.53 5.53l-1.06 1.061-2.72-2.72v7.19h-1.5V3.87l-2.72 2.72-1.06-1.06L8 1l4.53 4.53Z"
    })
  });
}
function UploadIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgUploadIcon
  });
}

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
      d: "M8 0a.75.75 0 0 1 .65.375l1.299 2.25a.75.75 0 0 1-.65 1.125H8.75V9.5h2.75V8h-.25a.75.75 0 0 1-.75-.75v-2a.75.75 0 0 1 .75-.75h2a.75.75 0 0 1 .75.75v2a.75.75 0 0 1-.75.75H13v2.25a.75.75 0 0 1-.75.75h-3.5v1.668a1.75 1.75 0 1 1-1.5 0V11h-3.5a.75.75 0 0 1-.75-.75V7.832a1.75 1.75 0 1 1 1.5 0V9.5h2.75V3.75h-.549a.75.75 0 0 1-.65-1.125l1.3-2.25A.75.75 0 0 1 8 0Z"
    })
  });
}
function UsbIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgUsbIcon
  });
}

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
      d: "M8 5.25a2.75 2.75 0 1 0 0 5.5 2.75 2.75 0 0 0 0-5.5ZM6.75 8a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "m4.401 2.5.386-.867A2.75 2.75 0 0 1 7.3 0h1.4a2.75 2.75 0 0 1 2.513 1.633l.386.867h1.651a.75.75 0 0 1 .75.75v12a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75v-12a.75.75 0 0 1 .75-.75h1.651Zm1.756-.258A1.25 1.25 0 0 1 7.3 1.5h1.4c.494 0 .942.29 1.143.742l.114.258H6.043l.114-.258ZM8 12a8.71 8.71 0 0 0-4.5 1.244V4h9v9.244A8.71 8.71 0 0 0 8 12Zm0 1.5c1.342 0 2.599.364 3.677 1H4.323A7.216 7.216 0 0 1 8 13.5Z",
      clipRule: "evenodd"
    })]
  });
}
function UserBadgeIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgUserBadgeIcon
  });
}

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
      d: "M5.25 6.75a2.75 2.75 0 1 1 5.5 0 2.75 2.75 0 0 1-5.5 0ZM8 5.5A1.25 1.25 0 1 0 8 8a1.25 1.25 0 0 0 0-2.5Z",
      clipRule: "evenodd"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 0 0-4.773 10.912A8.728 8.728 0 0 1 8 11c1.76 0 3.4.52 4.773 1.412A6.5 6.5 0 0 0 8 1.5Zm3.568 11.934A7.231 7.231 0 0 0 8 12.5a7.23 7.23 0 0 0-3.568.934A6.47 6.47 0 0 0 8 14.5a6.47 6.47 0 0 0 3.568-1.066Z",
      clipRule: "evenodd"
    })]
  });
}
function UserCircleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgUserCircleIcon
  });
}

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
      d: "M2.25 3.75a2.75 2.75 0 1 1 5.5 0 2.75 2.75 0 0 1-5.5 0ZM5 2.5A1.25 1.25 0 1 0 5 5a1.25 1.25 0 0 0 0-2.5ZM9.502 14H.75a.75.75 0 0 1-.75-.75V11a.75.75 0 0 1 .164-.469C1.298 9.114 3.077 8 5.125 8c1.76 0 3.32.822 4.443 1.952A5.545 5.545 0 0 1 11.75 9.5c1.642 0 3.094.745 4.041 1.73a.75.75 0 0 1 .209.52v1.5a.75.75 0 0 1-.75.75H9.502ZM1.5 12.5v-1.228C2.414 10.228 3.72 9.5 5.125 9.5c1.406 0 2.71.728 3.625 1.772V12.5H1.5Zm8.75 0h4.25v-.432A4.168 4.168 0 0 0 11.75 11c-.53 0-1.037.108-1.5.293V12.5ZM11.75 3.5a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5ZM11 5.75a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0Z",
      clipRule: "evenodd"
    })
  });
}
function UserGroupIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgUserGroupIcon
  });
}

function SvgUserIcon(props) {
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
      d: "M8 1a3.25 3.25 0 1 0 0 6.5A3.25 3.25 0 0 0 8 1ZM6.25 4.25a1.75 1.75 0 1 1 3.5 0 1.75 1.75 0 0 1-3.5 0ZM8 9a8.735 8.735 0 0 0-6.836 3.287.75.75 0 0 0-.164.469v1.494c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75v-1.494a.75.75 0 0 0-.164-.469A8.735 8.735 0 0 0 8 9Zm-5.5 4.5v-.474A7.232 7.232 0 0 1 8 10.5c2.2 0 4.17.978 5.5 2.526v.474h-11Z",
      clipRule: "evenodd"
    })
  });
}
function UserIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgUserIcon
  });
}

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
        d: "M8 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM6.5 8a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
      }), jsx("path", {
        d: "M8 2A8.389 8.389 0 0 0 .028 7.777a.75.75 0 0 0 0 .466 8.389 8.389 0 0 0 15.944 0 .75.75 0 0 0 0-.466A8.389 8.389 0 0 0 8 2Zm0 10.52a6.888 6.888 0 0 1-6.465-4.51 6.888 6.888 0 0 1 12.93 0A6.888 6.888 0 0 1 8 12.52Z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "VisibleIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function VisibleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgVisibleIcon
  });
}

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
        d: "m11.634 13.194 1.335 1.336 1.061-1.06-11.5-11.5-1.06 1.06 1.027 1.028a8.395 8.395 0 0 0-2.469 3.72.75.75 0 0 0 0 .465 8.389 8.389 0 0 0 11.606 4.951Zm-1.14-1.139-1.301-1.301a3 3 0 0 1-3.946-3.946L3.56 5.121A6.898 6.898 0 0 0 1.535 8.01a6.888 6.888 0 0 0 8.96 4.045Z",
        clipRule: "evenodd"
      }), jsx("path", {
        d: "M15.972 8.243a8.384 8.384 0 0 1-1.946 3.223l-1.06-1.06a6.887 6.887 0 0 0 1.499-2.396 6.888 6.888 0 0 0-8.187-4.293L5.082 2.522a8.389 8.389 0 0 1 10.89 5.256.75.75 0 0 1 0 .465Z"
      }), jsx("path", {
        d: "M11 8c0 .14-.01.277-.028.411L7.589 5.028A3 3 0 0 1 11 8Z"
      })]
    }), jsx("defs", {
      children: jsx("clipPath", {
        id: "VisibleOffIcon_svg__a",
        children: jsx("path", {
          fill: "#fff",
          d: "M0 0h16v16H0z"
        })
      })
    })]
  });
}
function VisibleOffIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgVisibleOffIcon
  });
}

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
      d: "M8.649 1.374a.75.75 0 0 0-1.298 0l-7.25 12.5A.75.75 0 0 0 .75 15h14.5a.75.75 0 0 0 .649-1.126l-7.25-12.5ZM7.25 10V6.5h1.5V10h-1.5Zm1.5 1.75a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0Z",
      clipRule: "evenodd"
    })
  });
}
function WarningFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgWarningFillIcon
  });
}

function SvgWarningIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.25 10V6.5h1.5V10h-1.5ZM8 12.5A.75.75 0 1 0 8 11a.75.75 0 0 0 0 1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 1a.75.75 0 0 1 .649.374l7.25 12.5A.75.75 0 0 1 15.25 15H.75a.75.75 0 0 1-.649-1.126l7.25-12.5A.75.75 0 0 1 8 1Zm0 2.245L2.052 13.5h11.896L8 3.245Z",
      clipRule: "evenodd"
    })]
  });
}
function WarningIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgWarningIcon
  });
}

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
      d: "M3.75 4a1.25 1.25 0 1 0 0-2.5 1.25 1.25 0 0 0 0 2.5Zm2.646-.5a2.751 2.751 0 1 1 0-1.5h5.229a3.375 3.375 0 0 1 .118 6.748L8.436 11.11a.75.75 0 0 1-.872 0l-3.3-2.357a1.875 1.875 0 0 0 .111 3.747h5.229a2.751 2.751 0 1 1 0 1.5H4.375a3.375 3.375 0 0 1-.118-6.748L7.564 4.89a.75.75 0 0 1 .872 0l3.3 2.357a1.875 1.875 0 0 0-.111-3.747H6.396Zm7.104 9.75a1.25 1.25 0 1 1-2.5 0 1.25 1.25 0 0 1 2.5 0ZM8 6.422 5.79 8 8 9.578 10.21 8 8 6.422Z",
      clipRule: "evenodd"
    })
  });
}
function WorkflowsIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgWorkflowsIcon
  });
}

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
      d: "M2.5 1a.75.75 0 0 0-.75.75v3c0 .414.336.75.75.75H6V4H3.25V2.5h9.5V4H10v1.5h3.5a.75.75 0 0 0 .75-.75v-3A.75.75 0 0 0 13.5 1h-11Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 12.25c0-1.26.848-2.322 2.004-2.648A2.75 2.75 0 0 1 4.75 7h2.5V4h1.5v3h2.5a2.75 2.75 0 0 1 2.746 2.602 2.751 2.751 0 1 1-3.371 3.47 2.751 2.751 0 0 1-5.25 0A2.751 2.751 0 0 1 0 12.25ZM2.75 11a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5Zm2.625.428a2.756 2.756 0 0 0-1.867-1.822A1.25 1.25 0 0 1 4.75 8.5h2.5v1.104c-.892.252-1.6.942-1.875 1.824ZM8.75 9.604V8.5h2.5c.642 0 1.17.483 1.242 1.106a2.756 2.756 0 0 0-1.867 1.822A2.756 2.756 0 0 0 8.75 9.604ZM12 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0Zm-5.25 0a1.25 1.25 0 1 0 2.5 0 1.25 1.25 0 0 0-2.5 0Z",
      clipRule: "evenodd"
    })]
  });
}
function WorkspacesIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgWorkspacesIcon
  });
}

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
      d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16Zm1.97-4.97L8 9.06l-1.97 1.97-1.06-1.06L6.94 8 4.97 6.03l1.06-1.06L8 6.94l1.97-1.97 1.06 1.06L9.06 8l1.97 1.97-1.06 1.06Z",
      clipRule: "evenodd"
    })
  });
}
function XCircleFillIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgXCircleFillIcon
  });
}

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
      d: "M6.94 8 4.97 6.03l1.06-1.06L8 6.94l1.97-1.97 1.06 1.06L9.06 8l1.97 1.97-1.06 1.06L8 9.06l-1.97 1.97-1.06-1.06L6.94 8Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Z",
      clipRule: "evenodd"
    })]
  });
}
function XCircleIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgXCircleIcon
  });
}

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
      d: "M8.75 7.25H11v1.5H8.75V11h-1.5V8.75H5v-1.5h2.25V5h1.5v2.25Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 1a7 7 0 1 0 4.39 12.453l2.55 2.55 1.06-1.06-2.55-2.55A7 7 0 0 0 8 1ZM2.5 8a5.5 5.5 0 1 1 11 0 5.5 5.5 0 0 1-11 0Z",
      clipRule: "evenodd"
    })]
  });
}
function ZoomInIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgZoomInIcon
  });
}

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
      d: "M11 7.25H5v1.5h6v-1.5Z"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M1 8a7 7 0 1 1 12.45 4.392l2.55 2.55-1.06 1.061-2.55-2.55A7 7 0 0 1 1 8Zm7-5.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11Z",
      clipRule: "evenodd"
    })]
  });
}
function ZoomOutIcon(props) {
  return jsx(Icon, {
    ...props,
    component: SvgZoomOutIcon
  });
}

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
      ...props
    } = _ref2;
    const {
      theme,
      getPrefixedClassName
    } = useDesignSystemTheme();
    // While this component is called `Accordion` for correctness, in AntD it is called `Collapse`.
    const clsPrefix = getPrefixedClassName('collapse');
    return jsx(DesignSystemAntDConfigProvider, {
      children: jsx(Collapse, {
        expandIcon: () => jsx(ChevronDownIcon, {}),
        expandIconPosition: "right",
        accordion: displayMode === 'single',
        ...props,
        ...dangerouslySetAntdProps,
        css: [getAccordionEmotionStyles(clsPrefix, theme), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Accordion;"]
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
function SeverityIcon(props) {
  const FilledIcon = filledIconsMap[props.severity];
  return jsx(FilledIcon, {
    ...props
  });
}

const Alert = _ref => {
  let {
    dangerouslySetAntdProps,
    closable = true,
    ...props
  } = _ref;
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const clsPrefix = getPrefixedClassName('alert');
  const mergedProps = {
    ...props,
    type: props.type || 'error',
    showIcon: true,
    closable
  };
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Alert$1, {
      ...mergedProps,
      className: classnames(mergedProps.className),
      css: getAlertEmotionStyles(clsPrefix, theme, mergedProps),
      icon: jsx(SeverityIcon, {
        severity: mergedProps.type
      })
      // Antd calls this prop `closeText` but we can use it to set any React element to replace the close icon.
      ,
      closeText: mergedProps.closable && jsx(CloseIcon, {
        "aria-label": "Close alert",
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
      [classMessage]: {
        margin: 0
      },
      [classDescription]: {
        display: 'none'
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
      color: theme.colors.textPrimary
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
        dropdownStyle: {
          boxShadow: theme.general.shadowLow
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

function getCheckboxEmotionStyles(clsPrefix, theme) {
  let isHorizontal = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
  let useNewCheckboxStyles = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : false;
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
    ...(useNewCheckboxStyles && {
      [`.${clsPrefix}`]: {
        top: 'unset',
        lineHeight: theme.typography.lineHeightBase
      },
      [`&${classWrapper}, ${classWrapper}`]: {
        alignItems: 'center',
        lineHeight: theme.typography.lineHeightBase
      }
    }),
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
            borderColor: theme.colors.actionDisabledBackground
          },
          // Disabled checked hover
          [hoverSelector]: {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBackground
          }
        },
        // Disabled indeterminate
        [`&${classIndeterminate}`]: {
          [classInner]: {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBackground
          },
          // Disabled indeterminate hover
          [hoverSelector]: {
            backgroundColor: theme.colors.actionDisabledBackground,
            borderColor: theme.colors.actionDisabledBackground
          }
        },
        // Disabled unchecked
        [classInner]: {
          backgroundColor: theme.colors.actionDisabledBackground,
          borderColor: theme.colors.actionDisabledBackground,
          // The after pseudo-element is used for the check image itself
          '&:after': {
            borderColor: theme.colors.white
          }
        },
        // Disabled hover
        [hoverSelector]: {
          backgroundColor: theme.colors.actionDisabledBackground,
          borderColor: theme.colors.actionDisabledBackground
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
    wrapperStyle = {},
    useNewStyles
  } = _ref;
  const styles = {
    height: theme.typography.lineHeightBase,
    lineHeight: theme.typography.lineHeightBase,
    ...(useNewStyles && {
      [`&& + .${clsPrefix}-hint, && + .${clsPrefix}-form-message`]: {
        paddingLeft: theme.spacing.lg,
        marginTop: 0
      }
    }),
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
    ...restProps
  } = _ref2;
  const {
    theme,
    classNamePrefix,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const {
    USE_NEW_CHECKBOX_STYLES
  } = useDesignSystemFlags();
  const clsPrefix = getPrefixedClassName('checkbox');
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx("div", {
      className: classnames(className, `${clsPrefix}-container`),
      css: getWrapperStyle({
        clsPrefix: classNamePrefix,
        theme,
        wrapperStyle,
        useNewStyles: USE_NEW_CHECKBOX_STYLES
      }),
      children: jsx(Checkbox$1, {
        checked: isChecked === null ? undefined : isChecked,
        ref: ref,
        onChange: onChange ? event => {
          onChange(event.target.checked, event);
        } : undefined,
        disabled: isDisabled,
        indeterminate: isChecked === null
        // Individual checkboxes don't depend on isHorizontal flag, orientation and spacing is handled by end users
        ,
        css: /*#__PURE__*/css(importantify(getCheckboxEmotionStyles(clsPrefix, theme, false, USE_NEW_CHECKBOX_STYLES)), process.env.NODE_ENV === "production" ? "" : ";label:DuboisCheckbox;"),
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
  const {
    USE_NEW_CHECKBOX_STYLES
  } = useDesignSystemFlags();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Checkbox$1.Group, {
      ref: ref,
      ...props,
      css: getCheckboxEmotionStyles(clsPrefix, theme, layout === 'horizontal', USE_NEW_CHECKBOX_STYLES),
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
const DuboisDatePicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker, {
      css: styles,
      ref: ref,
      ...props,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const RangePicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.RangePicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const TimePicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.TimePicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const QuarterPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.QuarterPicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const WeekPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.WeekPicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const MonthPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.MonthPicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
    })
  });
});
const YearPicker = /*#__PURE__*/forwardRef((props, ref) => {
  const styles = useDatePickerStyles();
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(DatePicker.YearPicker, {
      css: styles,
      ...props,
      ref: ref,
      popupStyle: {
        ...getDropdownStyles$1(theme),
        ...(props.popupStyle || {})
      }
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
    value = [],
    open,
    emptyText,
    scrollToSelectedElement = true,
    rememberLastScrollPosition = false,
    ...props
  } = _ref;
  // Used to avoid infinite loop when value is controlled from within the component (DialogComboboxOptionControlledList)
  // We can't remove setValue altogether because uncontrolled component users need to be able to set the value from root for trigger to update
  const [isControlled, setIsControlled] = useState(false);
  const [selectedValue, setSelectedValue] = useState(value);
  const [isOpen, setIsOpen] = useState(Boolean(open));
  const [contentWidth, setContentWidth] = useState();
  const [textOverflowMode, setTextOverflowMode] = useState('multiline');
  useEffect(() => {
    if ((!Array.isArray(selectedValue) || !Array.isArray(value)) && selectedValue !== value || selectedValue && value && selectedValue.length === value.length && selectedValue.every((v, i) => v === value[i])) {
      return;
    }
    if (!isControlled) {
      setSelectedValue(value);
    }
  }, [value, isControlled, selectedValue]);
  return jsx(DialogComboboxContextProvider, {
    value: {
      label,
      value: selectedValue,
      setValue: setSelectedValue,
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
      rememberLastScrollPosition
    },
    children: jsx(Root$3, {
      open: open !== undefined ? open : isOpen,
      ...props,
      children: children
    })
  });
};
const Root$3 = props => {
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
  return jsx(Popover$1.Root, {
    onOpenChange: handleOpenChange,
    ...restProps,
    children: children
  });
};

const getButtonContainerStyles = theme => {
  return /*#__PURE__*/css({
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'flex-end',
    alignItems: 'flex-end',
    padding: `${theme.spacing.sm}px ${theme.spacing.lg / 2}px ${theme.spacing.sm}px ${theme.spacing.lg / 2}px`,
    gap: theme.spacing.sm,
    alignSelf: 'stretch'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getButtonContainerStyles;");
};
const DialogComboboxButtonContainer = _ref => {
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
    throw new Error('`DialogComboboxButtonContainer` must be used within `DialogCombobox`');
  }
  return jsx("div", {
    ...restProps,
    css: getButtonContainerStyles(theme),
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
  return jsx(XCircleFillIcon, {
    "aria-hidden": "false",
    css: getButtonStyles(theme),
    role: "button",
    "aria-label": "Clear selection",
    ...restProps
  });
};

const EmptyResults = () => {
  const {
    theme
  } = useDesignSystemTheme();
  const {
    emptyText
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
    children: emptyText !== undefined ? emptyText : 'No results found'
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
    colorScheme: theme.isDarkMode ? 'dark' : 'light'
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
        marginTop: theme.spacing.xs / 2
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
    bottom: 0
  }, process.env.NODE_ENV === "production" ? "" : ";label:getFooterStyles;");
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$s() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$e = process.env.NODE_ENV === "production" ? {
  name: "1ij1o5n",
  styles: "display:flex;flex-direction:column;align-items:flex-start;justify-content:center"
} : {
  name: "189loa6-DialogComboboxContent",
  styles: "display:flex;flex-direction:column;align-items:flex-start;justify-content:center;label:DialogComboboxContent;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$s
};
const DialogComboboxContent = /*#__PURE__*/forwardRef((_ref, forwardedRef) => {
  let {
    children,
    loading,
    loadingDescription = 'DialogComboboxContent',
    matchTriggerWidth,
    textOverflowMode,
    maxHeight = 'var(--radix-popover-content-available-height)',
    maxWidth,
    minHeight,
    minWidth = 240,
    width,
    align = 'start',
    side = 'bottom',
    sideOffset = 4,
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
    rememberLastScrollPosition
  } = useDialogComboboxContext();
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const [lastScrollPosition, setLastScrollPosition] = useState(0);
  if (!isInsideDialogCombobox) {
    throw new Error('`DialogComboboxContent` must be used within `DialogCombobox`');
  }
  const contentRef = useRef(null);
  useImperativeHandle(forwardedRef, () => contentRef.current);
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
    if (contentWidth !== width) {
      setContentWidth(width);
    }
  }, [width, contentWidth, setContentWidth]);
  useEffect(() => {
    if (textOverflowMode !== contextTextOverflowMode) {
      setTextOverflowMode(textOverflowMode ? textOverflowMode : 'multiline');
    }
  }, [textOverflowMode, contextTextOverflowMode, setTextOverflowMode]);
  return jsx(Popover$1.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(Popover$1.Content, {
      "aria-label": `${label} options`,
      "aria-busy": loading,
      role: "listbox",
      "aria-multiselectable": multiSelect,
      css: getComboboxContentWrapperStyles(theme, {
        maxHeight,
        maxWidth,
        minHeight,
        minWidth,
        width: matchTriggerWidth ? 'var(--radix-popover-trigger-width)' : width
      }),
      align: align,
      side: side,
      sideOffset: sideOffset,
      ...restProps,
      ref: contentRef,
      children: jsx("div", {
        css: _ref2$e,
        children: loading ? jsx(LoadingSpinner, {
          "aria-label": "Loading",
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

const getDialogComboboxOptionLabelWidth = (theme, width) => {
  const paddingLeft = theme.spacing.xs + theme.spacing.sm;
  const iconWidth = theme.spacing.md;
  const labelMarginLeft = theme.spacing.sm;
  if (typeof width === 'string') {
    return `calc(${width} - ${paddingLeft + iconWidth + labelMarginLeft} px)`;
  }
  return width - paddingLeft + iconWidth + labelMarginLeft;
};
function findClosestOptionSibling(element, direction) {
  const nextSibling = direction === 'previous' ? element.previousElementSibling : element.nextElementSibling;
  if ((nextSibling === null || nextSibling === void 0 ? void 0 : nextSibling.getAttribute('role')) === 'option') {
    return nextSibling;
  } else if (nextSibling) {
    let nextOptionSibling = nextSibling;
    while (nextOptionSibling && nextOptionSibling.getAttribute('role') !== 'option') {
      nextOptionSibling = direction === 'previous' ? nextOptionSibling.previousElementSibling : nextOptionSibling.nextElementSibling;
    }
    return nextOptionSibling;
  }
  return null;
}
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
  (_currentSelection$scr = currentSelection.scrollIntoView) === null || _currentSelection$scr === void 0 ? void 0 : _currentSelection$scr.call(currentSelection, {
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
      onKeyDown === null || onKeyDown === void 0 ? void 0 : onKeyDown(e);
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
              highlightOption(firstOption, e.currentTarget);
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
              highlightOption(lastOption, e.currentTarget);
            }
          }
          break;
        case 'Enter':
          e.preventDefault();
          handleSelect();
          break;
        default:
          onDefaultKeyDown === null || onDefaultKeyDown === void 0 ? void 0 : onDefaultKeyDown(e);
          break;
      }
    },
    onMouseEnter: e => {
      onMouseEnter === null || onMouseEnter === void 0 ? void 0 : onMouseEnter(e);
      resetTabIndexToFocusedElement(e.currentTarget);
    }
  };
};
const resetTabIndexToFocusedElement = elem => {
  var _elem$closest;
  (_elem$closest = elem.closest('[role="list"]')) === null || _elem$closest === void 0 ? void 0 : _elem$closest.querySelectorAll('[role="option"]').forEach(el => {
    el.setAttribute('tabIndex', '-1');
  });
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$r() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$d = process.env.NODE_ENV === "production" ? {
  name: "1pgv7dg",
  styles: "display:flex;flex-direction:column;align-items:flex-start;width:100%"
} : {
  name: "1dtf9pj-DialogComboboxOptionList",
  styles: "display:flex;flex-direction:column;align-items:flex-start;width:100%;label:DialogComboboxOptionList;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$r
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
        highlightOption(firstOptionItem, undefined, false);
      }
    }
  }, [loading, withProgressiveLoading]);
  const handleOnMouseEnter = event => {
    const target = event.target;
    if (target) {
      var _target$closest;
      const options = target.hasAttribute('data-combobox-option-list') ? target.querySelectorAll('[role="option"]') : target === null || target === void 0 ? void 0 : (_target$closest = target.closest('[data-combobox-option-list="true"]')) === null || _target$closest === void 0 ? void 0 : _target$closest.querySelectorAll('[role="option"]');
      if (options) {
        options.forEach(option => option.removeAttribute('data-highlighted'));
      }
    }
  };
  return jsx("div", {
    ref: ref,
    "aria-busy": loading,
    "data-combobox-option-list": "true",
    css: _ref2$d,
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

const InfoTooltip = _ref => {
  let {
    title,
    tooltipProps,
    iconTitle,
    ...iconProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Tooltip, {
    title: title,
    ...tooltipProps,
    children: jsx(InfoCircleOutlined, {
      "aria-hidden": "false",
      title: iconTitle,
      "aria-label": iconTitle,
      css: /*#__PURE__*/css({
        fontSize: theme.typography.fontSizeSm,
        color: theme.colors.textSecondary
      }, process.env.NODE_ENV === "production" ? "" : ";label:InfoTooltip;"),
      ...iconProps
    })
  });
};

const useDialogComboboxOptionListContext = () => {
  return useContext(DialogComboboxOptionListContext);
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$q() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$c = process.env.NODE_ENV === "production" ? {
  name: "zjik7",
  styles: "display:flex"
} : {
  name: "tiu4as-content",
  styles: "display:flex;label:content;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$q
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
  const handleSelect = () => {
    if (onChange) {
      onChange(value);
    }
  };
  let content = children !== null && children !== void 0 ? children : value;
  if (props.disabled && disabledReason) {
    content = jsxs("div", {
      css: _ref2$c,
      children: [jsx("div", {
        children: content
      }), jsx("div", {
        children: jsx(Tooltip, {
          title: disabledReason,
          placement: "right",
          children: jsx("span", {
            css: getInfoIconStyles(theme),
            children: jsx(InfoIcon, {
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
        handleSelect();
      }
    },
    tabIndex: -1,
    ...getKeyboardNavigationFunctions(handleSelect, {
      onKeyDown: props.onKeyDown,
      onMouseEnter: props.onMouseEnter,
      onDefaultKeyDown: e => dialogComboboxLookAheadKeyDown(e, setLookAhead, lookAhead)
    }),
    children: jsx(Checkbox, {
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
        handleSelect();
      },
      children: content
    })
  });
});
DuboisDialogComboboxOptionListCheckboxItem.defaultProps = {
  _TYPE: 'DialogComboboxOptionListCheckboxItem'
};
const DialogComboboxOptionListCheckboxItem = DuboisDialogComboboxOptionListCheckboxItem;

function _EMOTION_STRINGIFIED_CSS_ERROR__$p() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
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
var _ref2$b = process.env.NODE_ENV === "production" ? {
  name: "1d3w5wq",
  styles: "width:100%"
} : {
  name: "csdki6-DialogComboboxOptionListSearch",
  styles: "width:100%;label:DialogComboboxOptionListSearch;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$p
};
const DialogComboboxOptionListSearch = /*#__PURE__*/forwardRef((_ref, forwardedRef) => {
  var _filteredChildren, _filteredChildren$pro, _filteredChildren2;
  let {
    onChange,
    onSearch,
    virtualized,
    children,
    hasWrapper,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
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
    onSearch === null || onSearch === void 0 ? void 0 : onSearch(event.target.value);
  };
  let filteredChildren = children;
  if (searchValue && !virtualized) {
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
        type: "search",
        name: "search",
        ref: forwardedRef,
        prefix: jsx(SearchIcon, {}),
        placeholder: "Search",
        onChange: handleOnChange,
        onKeyDown: event => {
          var _restProps$onKeyDown;
          handleOnKeyDown(event);
          (_restProps$onKeyDown = restProps.onKeyDown) === null || _restProps$onKeyDown === void 0 ? void 0 : _restProps$onKeyDown.call(restProps, event);
        },
        value: searchValue,
        ...restProps
      })
    }), virtualized ? children : (hasWrapper && (_filteredChildren = filteredChildren) !== null && _filteredChildren !== void 0 && (_filteredChildren$pro = _filteredChildren.props.children) !== null && _filteredChildren$pro !== void 0 && _filteredChildren$pro.length || !hasWrapper && (_filteredChildren2 = filteredChildren) !== null && _filteredChildren2 !== void 0 && _filteredChildren2.length) && childrenIsNotEmpty ? jsx("div", {
      "aria-live": "polite",
      css: _ref2$b,
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$o() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$a = process.env.NODE_ENV === "production" ? {
  name: "zjik7",
  styles: "display:flex"
} : {
  name: "tiu4as-content",
  styles: "display:flex;label:content;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
};
var _ref3$5 = process.env.NODE_ENV === "production" ? {
  name: "kjj0ot",
  styles: "padding-top:2px"
} : {
  name: "15osdio-DuboisDialogComboboxOptionListSelectItem",
  styles: "padding-top:2px;label:DuboisDialogComboboxOptionListSelectItem;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$o
};
const DuboisDialogComboboxOptionListSelectItem = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    value,
    checked,
    disabledReason,
    onChange,
    children,
    _TYPE,
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
            (_itemRef$current = itemRef.current) === null || _itemRef$current === void 0 ? void 0 : (_itemRef$current$scro = _itemRef$current.scrollIntoView) === null || _itemRef$current$scro === void 0 ? void 0 : _itemRef$current$scro.call(_itemRef$current, {
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
  const handleSelect = () => {
    if (onChange) {
      if (isSelect) {
        onChange({
          value,
          label: typeof children === 'string' ? children : value
        });
        return;
      }
      onChange(value);

      // On selecting a previously selected value, manually close the popup, top level logic will not be triggered
      if (!stayOpenOnSelection && existingValue !== null && existingValue !== void 0 && existingValue.includes(value)) {
        setIsOpen(false);
      }
    }
  };
  let content = children !== null && children !== void 0 ? children : value;
  if (props.disabled && disabledReason) {
    content = jsxs("div", {
      css: _ref2$a,
      children: [jsx("div", {
        children: content
      }), jsx(Tooltip, {
        title: disabledReason,
        placement: "right",
        children: jsx("span", {
          css: getInfoIconStyles(theme),
          children: jsx(InfoIcon, {
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
        handleSelect();
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
    children: [checked ? jsx(CheckIcon, {
      css: _ref3$5
    }) : jsx("div", {
      style: {
        width: 16,
        flexShrink: 0
      }
    }), jsx("label", {
      style: {
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
        }),
        ...(contentWidth ? {
          width: getDialogComboboxOptionLabelWidth(theme, contentWidth)
        } : {})
      },
      children: content
    })]
  });
});
DuboisDialogComboboxOptionListSelectItem.defaultProps = {
  _TYPE: 'DialogComboboxOptionListSelectItem'
};
const DialogComboboxOptionListSelectItem = DuboisDialogComboboxOptionListSelectItem;

function _EMOTION_STRINGIFIED_CSS_ERROR__$n() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$9 = process.env.NODE_ENV === "production" ? {
  name: "1pgv7dg",
  styles: "display:flex;flex-direction:column;align-items:flex-start;width:100%"
} : {
  name: "18t0chz-DialogComboboxOptionControlledList",
  styles: "display:flex;flex-direction:column;align-items:flex-start;width:100%;label:DialogComboboxOptionControlledList;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$n
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
    css: _ref2$9,
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
const getTriggerStyles = (theme, maxWidth, minWidth, removable, width, validationState, isBare, isSelect) => {
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
    [`&[disabled]`]: {
      background: theme.colors.actionDisabledBackground,
      color: theme.colors.actionDisabledText,
      pointerEvents: 'none',
      userSelect: 'none'
    },
    ...(isSelect && {
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
    })
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
    isBare = false,
    ...restProps
  } = _ref;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    label,
    value,
    isInsideDialogCombobox,
    multiSelect,
    setValue
  } = useDialogComboboxContext();
  const {
    isSelect,
    placeholder: selectPlaceholder
  } = useSelectContext();
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
      onClear === null || onClear === void 0 ? void 0 : onClear();
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
  const numValues = value.length;
  const concatenatedValues = Array.isArray(value) ? numValues > 10 ? `${value.slice(0, 10).join(', ')} + ${numValues - 10}` : value.join(', ') : value;
  const displayedValues = jsx("span", {
    children: concatenatedValues
  });
  const valuesBeforeBadge = Array.isArray(value) ? value.slice(0, showTagAfterValueCount).join(', ') : value;
  let ariaLabel = /*#__PURE__*/React__default.isValidElement(label) ? 'Dialog Combobox' : `${label}`;
  if (value !== null && value !== void 0 && value.length) {
    ariaLabel += multiSelect ? `, multiselectable, ${value.length} options selected: ${concatenatedValues}` : `, selected option: ${concatenatedValues}`;
  } else {
    ariaLabel += multiSelect ? ', multiselectable, 0 options selected' : ', no option selected';
  }
  const customSelectContent = isSelect && children ? children : null;
  const dialogComboboxClassname = !isSelect ? `${classNamePrefix}-dialogcombobox` : '';
  const selectV2Classname = isSelect ? `${classNamePrefix}-selectv2` : '';
  const triggerContent = isSelect ? jsxs(Popover$1.Trigger, {
    "aria-label": ariaLabel,
    ref: forwardedRef,
    role: "combobox",
    "aria-haspopup": "listbox",
    ...restProps,
    css: getTriggerStyles(theme, maxWidth, minWidth, removable, width, validationState, isBare, isSelect),
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
    }) : null, jsx(ChevronDownIcon, {
      css: /*#__PURE__*/css({
        color: theme.colors.textSecondary,
        marginLeft: theme.spacing.xs
      }, process.env.NODE_ENV === "production" ? "" : ";label:triggerContent;")
    })]
  }) : jsxs(Popover$1.Trigger, {
    "aria-label": ariaLabel,
    ref: forwardedRef,
    role: "combobox",
    "aria-haspopup": "listbox",
    ...restProps,
    css: getTriggerStyles(theme, maxWidth, minWidth, removable, width, validationState, isBare, isSelect),
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
    }) : null, withChevronIcon ? jsx(ChevronDownIcon, {
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
    children: [showTooltip && value !== null && value !== void 0 && value.length ? jsx(Tooltip, {
      title: customSelectContent !== null && customSelectContent !== void 0 ? customSelectContent : displayedValues,
      children: triggerContent
    }) : triggerContent, removable && jsx(Button, {
      "aria-label": `Remove ${label}`,
      onClick: handleRemove,
      dangerouslySetForceIconStyles: true,
      children: jsx(CloseIcon, {
        "aria-hidden": "false"
      })
    })]
  });
});

const Spacer = _ref => {
  let {
    size = 'md',
    shrinks,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const spacingValues = {
    xs: theme.spacing.xs,
    sm: theme.spacing.sm,
    md: theme.spacing.md,
    lg: theme.spacing.lg
  };
  return jsx("div", {
    css: /*#__PURE__*/css({
      height: spacingValues[size],
      ...(shrinks === false ? {
        flexShrink: 0
      } : undefined)
    }, process.env.NODE_ENV === "production" ? "" : ";label:Spacer;"),
    ...props
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$m() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const DEFAULT_WIDTH$1 = 320;
const MIN_WIDTH = 320;
const MAX_WIDTH = '90vw';
const DEFAULT_POSITION = 'right';
const ZINDEX_OVERLAY = 10;
const ZINDEX_CONTENT = ZINDEX_OVERLAY + 10;
var _ref2$8 = process.env.NODE_ENV === "production" ? {
  name: "zh83op",
  styles: "flex-grow:1;margin-bottom:0;margin-top:0;white-space:nowrap;overflow:hidden"
} : {
  name: "h5yqvj-Content",
  styles: "flex-grow:1;margin-bottom:0;margin-top:0;white-space:nowrap;overflow:hidden;label:Content;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$m
};
const Content$3 = _ref => {
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
    hideClose
  } = _ref;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const {
    theme
  } = useDesignSystemTheme();
  const horizontalContentPadding = theme.spacing.lg;
  const contentContainerRef = useRef(null);
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
    width: width !== null && width !== void 0 ? width : DEFAULT_WIDTH$1,
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
      css: dialogPrimitiveContentStyle,
      style: {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'flex-start',
        opacity: seeThrough ? 0 : 1
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
            css: _ref2$8,
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
              "aria-label": "Close",
              icon: jsx(CloseIcon, {})
            })
          })]
        }), jsxs("div", {
          css: /*#__PURE__*/css({
            // in order to have specific content in the drawer scroll with fixed title
            // hide overflow here and remove padding on the right side; content will be responsible for setting right padding
            // so that the scrollbar will appear in the padding right gutter
            paddingRight: useCustomScrollBehavior ? 0 : horizontalContentPadding,
            paddingLeft: horizontalContentPadding,
            overflowY: useCustomScrollBehavior ? 'hidden' : 'auto',
            height: expandContentToFullHeight ? '100%' : undefined,
            ...(theme.isDarkMode === false && !useCustomScrollBehavior ? {
              // Achieves an inner shadow on the content, but only when there is more left to scroll. When the content fits
              // in the container without scrolling, no shadow will be shown.
              // Taken from: https://css-tricks.com/scroll-shadows-with-javascript/
              background: `linear-gradient(
                    white 30%,
                    rgba(255, 255, 255, 0)
                  ) center top,
  
                  linear-gradient(
                    rgba(255, 255, 255, 0),
                    white 70%
                  ) center bottom,
      
                  radial-gradient(
                    farthest-side at 50% 0,
                    rgba(0, 0, 0, 0.2),
                    rgba(0, 0, 0, 0)
                  ) center top,
      
                  radial-gradient(
                    farthest-side at 50% 100%,
                    rgba(0, 0, 0, 0.2),
                    rgba(0, 0, 0, 0)
                  ) center bottom`,
              backgroundRepeat: 'no-repeat',
              backgroundSize: '100% 40px, 100% 40px, 100% 14px, 100% 14px',
              backgroundAttachment: 'local, local, scroll, scroll',
              backgroundOrigin: 'content-box'
            } : {})
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
function Root$2(props) {
  return jsx(DialogPrimitive.Root, {
    ...props
  });
}
function Trigger$1(props) {
  return jsx(DialogPrimitive.Trigger, {
    asChild: true,
    ...props
  });
}

var Drawer = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Content: Content$3,
  Root: Root$2,
  Trigger: Trigger$1
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$l() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
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
var _ref$5 = process.env.NODE_ENV === "production" ? {
  name: "zl1inp",
  styles: "display:flex;justify-content:center"
} : {
  name: "11tid6c-Empty",
  styles: "display:flex;justify-content:center;label:Empty;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$l
};
const Empty = props => {
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    title,
    description,
    image = jsx(GridDashIcon, {}),
    button,
    dangerouslyAppendEmotionCSS,
    ...dataProps
  } = props;
  return jsx("div", {
    ...dataProps,
    css: _ref$5,
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
      margin: 0
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
  success: CheckCircleIcon,
  warning: WarningIcon
};
function FormMessage(_ref) {
  let {
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
    className: wrapperClass,
    css: [getMessageStyles(classNamePrefix, theme), css, process.env.NODE_ENV === "production" ? "" : ";label:FormMessage;"],
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
    [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-select, && + .${classNamePrefix}-selectv2, && + .${classNamePrefix}-dialogcombobox, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group`]: {
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
    className: classnames(`${classNamePrefix}-hint`, className),
    css: getHintStyles(classNamePrefix, theme),
    ...restProps
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$k() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const getLabelStyles$1 = (classNamePrefix, theme, _ref) => {
  let {
    inline
  } = _ref;
  const styles = {
    '&&': {
      color: theme.colors.textPrimary,
      fontWeight: theme.typography.typographyBoldFontWeight,
      display: inline ? 'inline' : 'block',
      lineHeight: theme.typography.lineHeightBase
    },
    [`&& + .${classNamePrefix}-input, && + .${classNamePrefix}-select, && + .${classNamePrefix}-selectv2, && + .${classNamePrefix}-dialogcombobox, && + .${classNamePrefix}-checkbox-group, && + .${classNamePrefix}-radio-group`]: {
      marginTop: theme.spacing.sm
    }
  };
  return /*#__PURE__*/css(styles, process.env.NODE_ENV === "production" ? "" : ";label:getLabelStyles;");
};
var _ref2$7 = process.env.NODE_ENV === "production" ? {
  name: "s5xdrg",
  styles: "display:flex;align-items:center"
} : {
  name: "53ggpt-Label",
  styles: "display:flex;align-items:center;label:Label;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$k
};
const Label = props => {
  const {
    children,
    className,
    inline,
    ...restProps
  } = props;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  return jsx("label", {
    css: getLabelStyles$1(classNamePrefix, theme, {
      inline
    }),
    className: classnames(`${classNamePrefix}-label`, className),
    ...restProps,
    children: jsx("span", {
      css: _ref2$7,
      children: children
    })
  });
};

const getRadioInputStyles = _ref => {
  let {
    clsPrefix,
    theme,
    useNewStyles
  } = _ref;
  return {
    [`.${clsPrefix}`]: {
      alignSelf: 'start',
      // Unchecked Styles
      [`> .${clsPrefix}-input + .${clsPrefix}-inner`]: {
        width: theme.spacing.md,
        height: theme.spacing.md,
        background: useNewStyles ? theme.colors.actionDefaultBackgroundDefault : theme.colors.radioDefaultBackground,
        borderStyle: 'solid',
        borderColor: useNewStyles ? theme.colors.actionDefaultBorderDefault : theme.colors.radioDefaultBorder,
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
        borderColor: useNewStyles ? theme.colors.actionPrimaryBackgroundHover : theme.colors.radioInteractiveHover,
        background: useNewStyles ? theme.colors.actionDefaultBackgroundHover : theme.colors.radioInteractiveHoverSecondary
      },
      // Focus
      [`&:not(.${clsPrefix}-disabled)> .${clsPrefix}-input:focus + .${clsPrefix}-inner`]: {
        borderColor: useNewStyles ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.primary
      },
      // Active
      [`&:not(.${clsPrefix}-disabled)> .${clsPrefix}-input:active + .${clsPrefix}-inner`]: {
        borderColor: useNewStyles ? theme.colors.actionPrimaryBackgroundPress : theme.colors.radioInteractivePress,
        background: useNewStyles ? theme.colors.actionDefaultBackgroundPress : theme.colors.radioInteractivePressSecondary
      },
      // Disabled
      [`&.${clsPrefix}-disabled > .${clsPrefix}-input + .${clsPrefix}-inner`]: {
        ...(useNewStyles ? {
          border: `none !important`,
          // Ant uses !important
          background: theme.colors.actionDisabledBackground
        } : {
          borderColor: `${theme.colors.radioDisabled}!important` // Ant uses !important
        }),

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
          background: useNewStyles ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.primary,
          borderColor: theme.colors.primary,
          '&:after': {
            content: `""`,
            borderRadius: theme.spacing.xs,
            backgroundColor: useNewStyles ? theme.colors.white : theme.colors.radioDefaultBackground,
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
          background: useNewStyles ? theme.colors.actionPrimaryBackgroundHover : theme.colors.radioInteractiveHover,
          borderColor: useNewStyles ? theme.colors.actionPrimaryBackgroundPress : theme.colors.radioInteractiveHover
        },
        // Focus
        [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:focus-visible + .${clsPrefix}-inner`]: {
          background: useNewStyles ? theme.colors.actionDefaultBackgroundPress : theme.colors.primary,
          borderColor: useNewStyles ? theme.colors.actionDefaultBorderFocus : theme.colors.primary,
          boxShadow: `0 0 0 1px ${useNewStyles ? theme.colors.actionDefaultBackgroundDefault : theme.colors.radioDefaultBackground}, 0 0 0 3px ${theme.colors.primary}`
        },
        // Active
        [`&:not(.${clsPrefix}-disabled) > .${clsPrefix}-input:active + .${clsPrefix}-inner`]: {
          background: useNewStyles ? theme.colors.actionDefaultBackgroundPress : theme.colors.radioInteractivePress,
          borderColor: useNewStyles ? theme.colors.actionDefaultBorderPress : theme.colors.radioInteractivePress
        },
        // Disabled
        [`&.${clsPrefix}-disabled > .${clsPrefix}-input + .${clsPrefix}-inner`]: {
          background: useNewStyles ? theme.colors.actionDisabledBackground : theme.colors.radioDisabled,
          border: useNewStyles ? 'none !important' : `2px solid ${theme.colors.radioDisabled}!important`,
          // !important inherited from ant
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
    classNamePrefix,
    useNewStyles
  } = _ref2;
  return /*#__PURE__*/css({
    '& > label': {
      ...(useNewStyles && {
        [`&.${classNamePrefix}-radio-wrapper-disabled > span`]: {
          color: theme.colors.textPrimary
        }
      })
    },
    [`& > label + .${classNamePrefix}-hint`]: {
      paddingLeft: theme.spacing.lg
    },
    ...getRadioInputStyles({
      theme,
      clsPrefix,
      useNewStyles
    }),
    ...getAnimationCss(theme.options.enableAnimation)
  }, process.env.NODE_ENV === "production" ? "" : ";label:getCommonRadioGroupStyles;");
};
const getHorizontalRadioGroupStyles = _ref3 => {
  let {
    theme,
    classNamePrefix,
    useNewStyles
  } = _ref3;
  return /*#__PURE__*/css({
    '&&': {
      display: 'grid',
      gridTemplateRows: '[label] auto [hint] auto',
      gridAutoColumns: 'max-content',
      gridColumnGap: useNewStyles ? theme.spacing.md : theme.spacing.sm
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
    classNamePrefix,
    useNewStyles
  } = _ref4;
  return /*#__PURE__*/css({
    display: 'flex',
    flexDirection: 'column',
    flexWrap: 'wrap',
    '& > label': {
      fontWeight: 'normal',
      ...(useNewStyles && {
        paddingBottom: theme.spacing.sm
      })
    },
    [`& > label:last-of-type`]: {
      paddingBottom: 0
    },
    [`& > label + .${classNamePrefix}-hint`]: {
      marginBottom: theme.spacing.sm,
      paddingLeft: theme.spacing.lg,
      ...(useNewStyles && {
        marginTop: `-${theme.spacing.sm}px`
      })
    },
    [`& > label:last-of-type + .${classNamePrefix}-hint`]: {
      ...(useNewStyles && {
        marginTop: 0
      })
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getVerticalRadioGroupStyles;");
};
const getRadioStyles = _ref5 => {
  let {
    theme,
    clsPrefix,
    useNewStyles
  } = _ref5;
  // Default as bold for standalone radios
  const fontWeight = 'normal';
  const styles = {
    fontWeight
  };
  return /*#__PURE__*/css({
    ...getRadioInputStyles({
      theme,
      clsPrefix,
      useNewStyles
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
  const useNewStyles = safex('databricks.fe.designsystem.enableNewRadioStyles', false);
  const clsPrefix = getPrefixedClassName('radio');
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Radio$1, {
      css: getRadioStyles({
        theme,
        clsPrefix,
        useNewStyles
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
    ...props
  } = _ref7;
  const {
    theme,
    getPrefixedClassName,
    classNamePrefix
  } = useDesignSystemTheme();
  const useNewStyles = safex('databricks.fe.designsystem.enableNewRadioStyles', false);
  const clsPrefix = getPrefixedClassName('radio');
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Radio$1.Group, {
      ...props,
      css: getCommonRadioGroupStyles({
        theme,
        clsPrefix,
        classNamePrefix,
        useNewStyles
      }),
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
  const useNewStyles = safex('databricks.fe.designsystem.enableNewRadioStyles', false);
  return jsx(StyledRadioGroup, {
    css: getHorizontalRadioGroupStyles({
      theme,
      classNamePrefix,
      useNewStyles
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
  const useNewStyles = safex('databricks.fe.designsystem.enableNewRadioStyles', false);
  return jsx(StyledRadioGroup, {
    css: layout === 'horizontal' ? getHorizontalRadioGroupStyles({
      theme,
      classNamePrefix,
      useNewStyles
    }) : getVerticalRadioGroupStyles({
      theme,
      classNamePrefix,
      useNewStyles
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
      right: 24
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
  let useNewDropdownStyle = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
  const classItem = `.${clsPrefix}-item-option`;
  const classItemActive = `.${clsPrefix}-item-option-active`;
  const classItemSelected = `.${clsPrefix}-item-option-selected`;
  const classItemState = `.${clsPrefix}-item-option-state`;
  const classItemContent = `.${clsPrefix}-item-option-content`;
  const CONTENT_LEFT_PADDING = 28;
  const styles = {
    ...(useNewDropdownStyle ? dropdownContentStyles(theme) : {
      borderColor: theme.colors.borderDecorative,
      borderWidth: 1,
      borderStyle: 'solid',
      zIndex: theme.options.zIndexBase + 50,
      boxShadow: theme.general.shadowLow
    }),
    [classItem]: {
      height: theme.general.heightSm,
      ...(useNewDropdownStyle && {
        padding: '4px 8px',
        alignItems: 'center',
        lineHeight: theme.typography.lineHeightBase
      })
    },
    ...(useNewDropdownStyle && {
      [classItemContent]: {
        paddingLeft: CONTENT_LEFT_PADDING
      }
    }),
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
      ...(useNewDropdownStyle && {
        position: 'absolute'
      }),
      '& > span': {
        verticalAlign: 'middle'
      }
    },
    [`.${clsPrefix}-loading-options`]: {
      pointerEvents: 'none',
      margin: '0 auto',
      height: theme.general.heightSm,
      display: 'block',
      ...(useNewDropdownStyle && {
        left: -CONTENT_LEFT_PADDING / 2
      })
    },
    ...getAnimationCss(theme.options.enableAnimation)
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
    ...restProps
  } = _ref2;
  const {
    theme,
    getPrefixedClassName
  } = useDesignSystemTheme();
  const {
    USE_NEW_SELECT_DROPDOWN_STYLES
  } = useDesignSystemFlags();
  const clsPrefix = getPrefixedClassName('select');
  const [isOpen, setIsOpen] = useState(false);
  const [uniqueId, setUniqueId] = useState('');
  useEffect(() => {
    setUniqueId(id || _uniqueId('dubois-select-'));
  }, [id]);
  useEffect(() => {
    var _document$getElementB;
    // Ant doesn't populate aria-expanded on init (only on user interaction) so we need to do it ourselves
    // in order to pass accessibility tests (Microsoft Accessibility Insights). See: JOBS-11125
    (_document$getElementB = document.getElementById(uniqueId)) === null || _document$getElementB === void 0 ? void 0 : _document$getElementB.setAttribute('aria-expanded', 'false');
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
            onDropdownVisibleChange === null || onDropdownVisibleChange === void 0 ? void 0 : onDropdownVisibleChange(visible);
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
          clearIcon: jsx(XCircleFillIcon, {
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
          }) : jsx(ChevronDownIcon, {
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
            }),
            children: "No results found"
          }),
          dropdownClassName: css$1([getDropdownStyles(clsPrefix, theme, USE_NEW_SELECT_DROPDOWN_STYLES), dropdownClassName]),
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
            children: [children, jsx(Option, {
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

/** @deprecated Use `SelectOptionProps` */

const SelectOption = /*#__PURE__*/forwardRef(function Option(props, ref) {
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
SelectOption.isSelectOption = true;

/**
 * @deprecated use Select.Option instead
 */
const Option = SelectOption;

/** @deprecated Use `SelectOptGroupProps` */

const SelectOptGroup = /* #__PURE__ */(() => {
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
 * @deprecated use Select.OptGroup instead
 */
const OptGroup = SelectOptGroup;
const Select = /* #__PURE__ */(() => {
  const DuboisRefForwardedSelect = /*#__PURE__*/forwardRef(DuboisSelect);
  DuboisRefForwardedSelect.Option = SelectOption;
  DuboisRefForwardedSelect.OptGroup = SelectOptGroup;
  return DuboisRefForwardedSelect;
})();

const SelectV2 = props => {
  const {
    children,
    placeholder,
    value,
    ...restProps
  } = props;
  return jsx(SelectContextProvider, {
    value: {
      isSelect: true,
      placeholder
    },
    children: jsx(DialogCombobox, {
      value: value ? [value] : [],
      ...restProps,
      children: children
    })
  });
};

const SelectV2Content = /*#__PURE__*/forwardRef((_ref, ref) => {
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

const SelectV2Option = /*#__PURE__*/forwardRef((props, ref) => {
  const {
    value
  } = useDialogComboboxContext();
  return jsx(DialogComboboxOptionListSelectItem, {
    checked: value && value[0] === props.value,
    ...props,
    ref: ref
  });
});

const SelectV2OptionGroup = props => {
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

const SelectV2Trigger = /*#__PURE__*/forwardRef((props, ref) => {
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
function RHFControlledSelect(_ref4) {
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
  return jsx(Select, {
    ...restProps,
    ...field,
    value: field.value,
    defaultValue: field.value
  });
}
function RHFControlledSelectV2(_ref5) {
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
      var _props;
      const renderedChildren = children({
        onChange: handleOnChange
      });
      const child = Children.toArray(renderedChildren.props.children).find(child => /*#__PURE__*/React__default.isValidElement(child) && child.props.value === field.value);
      if (child && ((_props = child.props) === null || _props === void 0 ? void 0 : _props.children) !== field.value) {
        setSelectedValueLabel(child.props.children);
      }
    } else if (options !== null && options !== void 0 && options.length) {
      const option = options.find(option => option.value === field.value);
      setSelectedValueLabel(option === null || option === void 0 ? void 0 : option.label);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [field.value]);
  return jsxs(SelectV2, {
    ...restProps,
    value: field.value,
    children: [jsx(SelectV2Trigger, {
      ...triggerProps,
      width: width,
      onBlur: field.onBlur,
      validationState: validationState,
      children: selectedValueLabel
    }), jsx(SelectV2Content, {
      ...contentProps,
      side: "bottom",
      width: width,
      children: options && options.length > 0 ? options.map(option => createElement(SelectV2Option, {
        ...optionProps,
        key: option.value,
        value: option.value,
        onChange: handleOnChange
      }, option.label)) : // SelectV2Option out of the box gives users control over state and in this case RHF is controlling state
      // We expose onChange through a children renderer function to let users pass this down to SelectV2Option
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
  const hanldeOnChangeMultiSelect = option => {
    var _field$value;
    let updatedValue = field.value;
    if ((_field$value = field.value) !== null && _field$value !== void 0 && _field$value.includes(option)) {
      updatedValue = field.value.filter(value => value !== option);
    } else {
      updatedValue = [...field.value, option];
    }
    field.onChange(updatedValue);
    updateValueMap(updatedValue);
  };
  const handleOnChange = option => {
    if (restProps.multiSelect) {
      hanldeOnChangeMultiSelect(option);
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
function RHFControlledCheckboxGroup(_ref7) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref7;
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
function RHFControlledCheckbox(_ref8) {
  let {
    name,
    control,
    rules,
    ...restProps
  } = _ref8;
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
      checked: field.value
    })
  });
}
function RHFControlledRadioGroup(_ref9) {
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
  return jsx(Radio.Group, {
    ...restProps,
    ...field
  });
}
const RHFControlledComponents = {
  Input: RHFControlledInput,
  Password: RHFControlledPasswordInput,
  TextArea: RHFControlledTextArea,
  Select: RHFControlledSelect,
  SelectV2: RHFControlledSelectV2,
  DialogCombobox: RHFControlledDialogCombobox,
  Checkbox: RHFControlledCheckbox,
  CheckboxGroup: RHFControlledCheckboxGroup,
  RadioGroup: RHFControlledRadioGroup
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$j() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const getHeaderStyles = (clsPrefix, theme) => {
  const breadcrumbClass = `.${clsPrefix}-breadcrumb`;
  const styles = {
    [breadcrumbClass]: {
      lineHeight: theme.typography.lineHeightBase
    }
  };
  return /*#__PURE__*/css(importantify(styles), process.env.NODE_ENV === "production" ? "" : ";label:getHeaderStyles;");
};
var _ref$4 = process.env.NODE_ENV === "production" ? {
  name: "eh0igi",
  styles: "display:inline-flex;vertical-align:middle;align-items:center"
} : {
  name: "n13uil-titleAddOnsWrapper",
  styles: "display:inline-flex;vertical-align:middle;align-items:center;label:titleAddOnsWrapper;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$j
};
var _ref2$6 = process.env.NODE_ENV === "production" ? {
  name: "1q4vxyr",
  styles: "margin-left:8px"
} : {
  name: "ozrfom-buttonContainer",
  styles: "margin-left:8px;label:buttonContainer;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$j
};
var _ref3$4 = process.env.NODE_ENV === "production" ? {
  name: "s079uh",
  styles: "margin-top:2px"
} : {
  name: "1ky5whb-titleIfOtherElementsPresent",
  styles: "margin-top:2px;label:titleIfOtherElementsPresent;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$j
};
var _ref4$2 = process.env.NODE_ENV === "production" ? {
  name: "fuxm9z",
  styles: "margin-top:0;margin-bottom:0 !important;align-self:stretch"
} : {
  name: "h5m2l9-title",
  styles: "margin-top:0;margin-bottom:0 !important;align-self:stretch;label:title;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$j
};
var _ref5 = process.env.NODE_ENV === "production" ? {
  name: "1cl2a0e",
  styles: "display:flex;align-items:flex-start;justify-content:space-between"
} : {
  name: "m9gz0x-titleWrapper",
  styles: "display:flex;align-items:flex-start;justify-content:space-between;label:titleWrapper;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$j
};
const Header$1 = _ref6 => {
  let {
    breadcrumbs,
    title,
    titleAddOns,
    dangerouslyAppendEmotionCSS,
    buttons,
    children,
    ...rest
  } = _ref6;
  const {
    classNamePrefix: clsPrefix,
    theme
  } = useDesignSystemTheme();

  // TODO: Move to getHeaderStyles for consistency, followup ticket: https://databricks.atlassian.net/browse/FEINF-1222
  const styles = {
    titleWrapper: _ref5,
    breadcrumbWrapper: /*#__PURE__*/css({
      lineHeight: theme.typography.lineHeightBase
    }, process.env.NODE_ENV === "production" ? "" : ";label:breadcrumbWrapper;"),
    title: _ref4$2,
    // TODO: Look into a more emotion-idomatic way of doing this.
    titleIfOtherElementsPresent: _ref3$4,
    buttonContainer: _ref2$6,
    titleAddOnsWrapper: _ref$4
  };
  const buttonsArray = Array.isArray(buttons) ? buttons : [buttons];
  return jsxs("div", {
    css: [getHeaderStyles(clsPrefix, theme), dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Header;"],
    ...rest,
    children: [breadcrumbs && jsx("div", {
      css: styles.breadcrumbWrapper,
      children: breadcrumbs
    }), jsxs("div", {
      css: styles.titleWrapper,
      children: [jsxs(Title$2, {
        level: 2,
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
          size: 8,
          children: buttonsArray.filter(Boolean).map((button, i) => {
            const defaultKey = `dubois-header-button-${i}`;
            return /*#__PURE__*/React__default.isValidElement(button) ? /*#__PURE__*/React__default.cloneElement(button, {
              key: button.key || defaultKey
            }) : jsx(React__default.Fragment, {
              children: button
            }, defaultKey);
          })
        })
      })]
    })]
  });
};

const {
  Header,
  Footer,
  Sider,
  Content: Content$2
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
      children: jsx(Content$2, {
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
    children: jsx(Popover$2, {
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$i() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
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
    dangerouslySetAntdProps
  } = _ref;
  const {
    classNamePrefix,
    theme
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Pagination$1, {
      css: getPaginationEmotionStyles(classNamePrefix, theme),
      current: currentPageIndex,
      pageSize: pageSize,
      responsive: false,
      total: numTotal,
      onChange: onChange,
      showSizeChanger: false,
      showQuickJumper: false,
      size: 'small',
      style: style,
      ...dangerouslySetAntdProps
    })
  });
};
var _ref3$3 = process.env.NODE_ENV === "production" ? {
  name: "1u1zie3",
  styles: "width:120px"
} : {
  name: "1am9qog-CursorPagination",
  styles: "width:120px;label:CursorPagination;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$i
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
      onChange: onPageSizeChange
    } = {}
  } = _ref2;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const [pageSizeValue, setPageSizeValue] = useState(defaultPageSize);
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
      icon: jsx(ChevronLeftIcon, {}),
      disabled: !hasPreviousPage,
      onClick: onPreviousPage,
      type: "tertiary",
      children: previousPageText
    }), jsx(Button, {
      endIcon: jsx(ChevronRightIcon, {}),
      disabled: !hasNextPage,
      onClick: onNextPage,
      type: "tertiary",
      children: nextPageText
    }), pageSizeOptions && jsx(Select, {
      value: String(pageSizeValue),
      css: _ref3$3,
      onChange: pageSize => {
        const updatedPageSize = Number(pageSize);
        onPageSizeChange === null || onPageSizeChange === void 0 ? void 0 : onPageSizeChange(updatedPageSize);
        setPageSizeValue(updatedPageSize);
      },
      children: pageSizeOptions.map(pageSize => jsx(Select.Option, {
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
    children: jsx(Table$1
    // NOTE(FEINF-1273): The default loading indicator from AntD does not animate
    // and the design system spinner is recommended over the AntD one. Therefore,
    // if `loading` is `true`, render the design system <Spinner/> component.
    , {
      loading: loading === true ? DEFAULT_LOADING_SPIN_PROPS : loading,
      scroll: scrollableInFlexibleContainer ? {
        y: 'auto'
      } : undefined,
      ...tableProps,
      css: getTableEmotionStyles(classNamePrefix, theme, Boolean(scrollableInFlexibleContainer)),
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$h() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const SIZE_PRESETS = {
  normal: 640,
  wide: 880
};
const getModalEmotionStyles = args => {
  const {
    theme,
    clsPrefix,
    hasFooter = true,
    maxedOutHeight
  } = args;
  const classNameClose = `.${clsPrefix}-modal-close`;
  const classNameCloseX = `.${clsPrefix}-modal-close-x`;
  const classNameTitle = `.${clsPrefix}-modal-title`;
  const classNameContent = `.${clsPrefix}-modal-content`;
  const classNameBody = `.${clsPrefix}-modal-body`;
  const classNameHeader = `.${clsPrefix}-modal-header`;
  const classNameFooter = `.${clsPrefix}-modal-footer`;
  const classNameButton = `.${clsPrefix}-btn`;
  const classNameDropdownTrigger = `.${clsPrefix}-dropdown-trigger`;
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
  return /*#__PURE__*/css({
    [classNameHeader]: {
      background: 'transparent',
      paddingTop: theme.spacing.md,
      paddingLeft: theme.spacing.lg,
      paddingRight: theme.spacing.md,
      paddingBottom: theme.spacing.md
    },
    [classNameFooter]: {
      height: footerHeight,
      paddingTop: theme.spacing.lg - CONTENT_BUFFER,
      paddingLeft: MODAL_PADDING,
      paddingRight: MODAL_PADDING,
      marginTop: 'auto',
      [`${classNameButton} + ${classNameButton}:not(${classNameDropdownTrigger})`]: {
        marginLeft: theme.spacing.sm
      }
    },
    [classNameCloseX]: {
      fontSize: theme.general.iconSize,
      height: BUTTON_SIZE,
      width: BUTTON_SIZE,
      lineHeight: 'normal',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: theme.colors.textSecondary
    },
    [classNameClose]: {
      height: BUTTON_SIZE,
      width: BUTTON_SIZE,
      // Note: Ant has the close button absolutely positioned, rather than in a flex container with the title.
      // This magic number is eyeballed to get the close X to align with the title text.
      margin: '16px 16px 0 0',
      borderRadius: theme.borders.borderRadiusMd,
      backgroundColor: theme.colors.actionDefaultBackgroundDefault,
      borderColor: theme.colors.actionDefaultBackgroundDefault,
      color: theme.colors.actionDefaultTextDefault,
      '&:hover': {
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        borderColor: theme.colors.actionDefaultBackgroundHover,
        color: theme.colors.actionDefaultTextHover
      },
      '&:active': {
        backgroundColor: theme.colors.actionDefaultBackgroundPress,
        borderColor: theme.colors.actionDefaultBackgroundPress,
        color: theme.colors.actionDefaultTextPress
      },
      '&:focus-visible': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '1px',
        outlineColor: theme.colors.primary
      }
    },
    [classNameTitle]: {
      fontSize: theme.typography.fontSizeXl,
      lineHeight: theme.typography.lineHeightXl,
      fontWeight: theme.typography.typographyBoldFontWeight,
      paddingRight: MODAL_PADDING,
      minHeight: headerHeight / 2,
      display: 'flex',
      alignItems: 'center',
      overflowWrap: 'anywhere'
    },
    [classNameContent]: {
      backgroundColor: theme.colors.backgroundPrimary,
      maxHeight: modalMaxHeight,
      height: maxedOutHeight ? modalMaxHeight : '',
      overflow: 'hidden',
      paddingBottom: MODAL_PADDING,
      display: 'flex',
      flexDirection: 'column',
      boxShadow: theme.general.shadowHigh
    },
    [classNameBody]: {
      overflowY: 'auto',
      maxHeight: bodyMaxHeight,
      paddingLeft: MODAL_PADDING,
      paddingRight: MODAL_PADDING,
      paddingTop: CONTENT_BUFFER,
      paddingBottom: CONTENT_BUFFER,
      ...(theme.isDarkMode === false ? {
        // Achieves an inner shadow on the content, but only when there is more left to scroll. When the content fits
        // in the container without scrolling, no shadow will be shown.
        // Taken from: https://css-tricks.com/scroll-shadows-with-javascript/
        background: `linear-gradient(
              white 30%,
              rgba(255, 255, 255, 0)
            ) center top,

            linear-gradient(
              rgba(255, 255, 255, 0),
              white 70%
            ) center bottom,

            radial-gradient(
              farthest-side at 50% 0,
              rgba(0, 0, 0, 0.2),
              rgba(0, 0, 0, 0)
            ) center top,

            radial-gradient(
              farthest-side at 50% 100%,
              rgba(0, 0, 0, 0.2),
              rgba(0, 0, 0, 0)
            ) center bottom`,
        backgroundRepeat: 'no-repeat',
        backgroundSize: '100% 40px, 100% 40px, 100% 14px, 100% 14px',
        backgroundAttachment: 'local, local, scroll, scroll'
      } : {})
    },
    ...getAnimationCss(theme.options.enableAnimation)
  }, process.env.NODE_ENV === "production" ? "" : ";label:getModalEmotionStyles;");
};

/**
 * Render default footer with our buttons. Copied from AntD.
 */
function DefaultFooter(_ref) {
  let {
    onOk,
    onCancel,
    confirmLoading,
    okText,
    cancelText,
    okButtonProps,
    cancelButtonProps,
    autoFocusButton
  } = _ref;
  const handleCancel = e => {
    onCancel === null || onCancel === void 0 ? void 0 : onCancel(e);
  };
  const handleOk = e => {
    onOk === null || onOk === void 0 ? void 0 : onOk(e);
  };
  return jsxs(Fragment, {
    children: [cancelText && jsx(Button, {
      onClick: handleCancel,
      autoFocus: autoFocusButton === 'cancel',
      dangerouslyUseFocusPseudoClass: true,
      ...cancelButtonProps,
      children: cancelText
    }), okText && jsx(Button, {
      loading: confirmLoading,
      onClick: handleOk,
      type: "primary",
      autoFocus: autoFocusButton === 'ok',
      dangerouslyUseFocusPseudoClass: true,
      ...okButtonProps,
      children: okText
    })]
  });
}
function Modal(_ref2) {
  let {
    okButtonProps,
    cancelButtonProps,
    dangerouslySetAntdProps,
    children,
    title,
    footer,
    size = 'normal',
    verticalSizing = 'dynamic',
    autoFocusButton,
    truncateTitle,
    ...props
  } = _ref2;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Modal$1, {
      css: getModalEmotionStyles({
        theme,
        clsPrefix: classNamePrefix,
        hasFooter: footer !== null,
        maxedOutHeight: verticalSizing === 'maxed_out'
      }),
      title: jsx(RestoreAntDDefaultClsPrefix, {
        children: truncateTitle ? jsx("div", {
          css: /*#__PURE__*/css({
            textOverflow: 'ellipsis',
            marginRight: theme.spacing.md,
            overflow: 'hidden',
            whiteSpace: 'nowrap'
          }, process.env.NODE_ENV === "production" ? "" : ";label:Modal;"),
          title: typeof title === 'string' ? title : undefined,
          children: title
        }) : title
      }),
      footer: jsx(RestoreAntDDefaultClsPrefix, {
        children: footer === undefined ? jsx(DefaultFooter, {
          onOk: props.onOk,
          onCancel: props.onCancel,
          confirmLoading: props.confirmLoading,
          okText: props.okText,
          cancelText: props.cancelText,
          okButtonProps: okButtonProps,
          cancelButtonProps: cancelButtonProps,
          autoFocusButton: autoFocusButton
        }) : footer
      }),
      width: size ? SIZE_PRESETS[size] : undefined,
      closeIcon: jsx(CloseIcon, {}),
      centered: true,
      zIndex: theme.options.zIndexBase,
      ...props,
      ...dangerouslySetAntdProps,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: children
      })
    })
  });
}
var _ref3$2 = process.env.NODE_ENV === "production" ? {
  name: "b9hrb",
  styles: "position:relative;display:inline-flex;align-items:center"
} : {
  name: "1jkwrsj-titleComp",
  styles: "position:relative;display:inline-flex;align-items:center;label:titleComp;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$h
};
var _ref4$1 = process.env.NODE_ENV === "production" ? {
  name: "1o6wc9k",
  styles: "padding-left:6px"
} : {
  name: "i303lp-titleComp",
  styles: "padding-left:6px;label:titleComp;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$h
};
function DangerModal(props) {
  const {
    theme
  } = useDesignSystemTheme();
  const {
    title,
    onCancel,
    onOk,
    cancelText,
    okText,
    okButtonProps,
    cancelButtonProps,
    ...restProps
  } = props;
  const iconSize = 18;
  const iconFontSize = 18;
  const titleComp = jsxs("div", {
    css: _ref3$2,
    children: [jsx(DangerIcon, {
      css: /*#__PURE__*/css({
        color: theme.colors.textValidationDanger,
        left: 2,
        height: iconSize,
        width: iconSize,
        fontSize: iconFontSize
      }, process.env.NODE_ENV === "production" ? "" : ";label:titleComp;")
    }), jsx("div", {
      css: _ref4$1,
      children: title
    })]
  });
  return jsx(Modal, {
    title: titleComp,
    footer: [jsx(Button, {
      onClick: onCancel,
      ...cancelButtonProps,
      children: cancelText || 'Cancel'
    }, "cancel"), jsx(Button, {
      type: "primary",
      danger: true,
      onClick: onOk,
      loading: props.confirmLoading,
      ...okButtonProps,
      children: okText || 'Delete'
    }, "discard")],
    onOk: onOk,
    onCancel: onCancel,
    ...restProps
  });
}

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
      gridTemplateColumns: '[icon] auto [content] 1fr [close] auto'
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
const Root$1 = /*#__PURE__*/forwardRef(function (_ref, ref) {
  let {
    children,
    severity = 'info',
    ...props
  } = _ref;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  return jsxs(Toast.Root, {
    ref: ref,
    css: getToastRootStyle(theme, classNamePrefix),
    ...props,
    children: [jsx(SeverityIcon, {
      className: `${classNamePrefix}-notification-severity-icon ${classNamePrefix}-notification-${severity}-icon`,
      severity: severity
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
    outline: 'none'
  };
};
const getTitleStyles = theme => {
  return /*#__PURE__*/css({
    fontWeight: theme.typography.typographyBoldFontWeight,
    color: theme.colors.textPrimary,
    gridRow: 'header / header',
    gridColumn: 'content / content'
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
    gridColumn: 'content / content'
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
const Close$1 = /*#__PURE__*/forwardRef(function (props, ref) {
  var _ref4;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    closeLabel,
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
  Close: Close$1,
  Description: Description,
  Provider: Provider,
  Root: Root$1,
  Title: Title,
  Viewport: Viewport
});

const PageWrapper = _ref => {
  let {
    children,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: /*#__PURE__*/css({
      paddingLeft: 16,
      paddingRight: 16,
      backgroundColor: theme.isDarkMode ? theme.colors.backgroundPrimary : 'transparent'
    }, process.env.NODE_ENV === "production" ? "" : ";label:PageWrapper;"),
    ...props,
    children: children
  });
};

const Root = Popover$1.Root; // Behavioral component only
const Anchor = Popover$1.Anchor; // Behavioral component only

const Content$1 = /*#__PURE__*/forwardRef(function Content(_ref, ref) {
  let {
    children,
    minWidth = 220,
    ...props
  } = _ref;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  return jsx(Popover$1.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(Popover$1.Content, {
      ref: ref,
      css: [contentStyles, {
        minWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      sideOffset: 4,
      ...props,
      children: children
    })
  });
});
const Trigger = /*#__PURE__*/forwardRef(function Trigger(_ref2, ref) {
  let {
    children,
    ...props
  } = _ref2;
  return jsx(Popover$1.Trigger, {
    ref: ref,
    ...props,
    children: children
  });
});
const Close = /*#__PURE__*/forwardRef(function Close(_ref3, ref) {
  let {
    children,
    ...props
  } = _ref3;
  return jsx(Popover$1.Close, {
    ref: ref,
    ...props,
    children: children
  });
});
const Arrow = /*#__PURE__*/forwardRef(function Arrow(_ref4, ref) {
  let {
    children,
    ...props
  } = _ref4;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx(Popover$1.Arrow, {
    css: /*#__PURE__*/css({
      fill: theme.colors.backgroundPrimary,
      stroke: theme.colors.borderDecorative,
      strokeDashoffset: -CONSTANTS.arrowBottomLength(),
      strokeDasharray: CONSTANTS.arrowBottomLength() + 2 * CONSTANTS.arrowSide(),
      strokeWidth: CONSTANTS.arrowStrokeWidth(),
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

// CONSTANTS
const CONSTANTS = {
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
const popoverContentStyles = theme => ({
  backgroundColor: theme.colors.backgroundPrimary,
  color: theme.colors.textPrimary,
  lineHeight: theme.typography.lineHeightBase,
  border: `1px solid ${theme.colors.borderDecorative}`,
  borderRadius: theme.borders.borderRadiusMd,
  padding: `${theme.spacing.sm}px`,
  boxShadow: theme.general.shadowLow,
  userSelect: 'none',
  zIndex: theme.options.zIndexBase + 30,
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
});
const contentStyles = theme => ({
  ...popoverContentStyles(theme)
});

var Popover = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Anchor: Anchor,
  Arrow: Arrow,
  Close: Close,
  Content: Content$1,
  Root: Root,
  Trigger: Trigger
});

const SMALL_BUTTON_HEIGHT = 24;
function getSegmentedControlGroupEmotionStyles(clsPrefix) {
  const classSmallGroup = `.${clsPrefix}-radio-group-small`;
  const classButtonWrapper = `.${clsPrefix}-radio-button-wrapper`;
  const styles = {
    [`&${classSmallGroup} ${classButtonWrapper}`]: {
      padding: '0 12px'
    }
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSegmentedControlGroupEmotionStyles;");
}
function getSegmentedControlButtonEmotionStyles(clsPrefix, theme, size) {
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
    '::before': {
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
      }
    },
    [`&${classWrapperChecked}:focus-within`]: {
      '::before': {
        width: 0
      }
    },
    [`&${classWrapperDisabled}`]: {
      color: theme.colors.actionDisabledText,
      backgroundColor: theme.colors.actionDisabledBackground,
      '&:hover': {
        color: theme.colors.actionDisabledText,
        backgroundColor: theme.colors.actionDisabledBackground
      },
      '&:active': {
        color: theme.colors.actionDisabledText,
        backgroundColor: theme.colors.actionDisabledBackground
      }
    },
    [`&${classWrapper}`]: {
      padding: size === 'middle' ? '0 16px' : '0 8px',
      display: 'inline-flex',
      verticalAlign: 'middle',
      '&:focus-within': {
        outlineStyle: 'solid',
        outlineWidth: '2px',
        outlineOffset: '-2px',
        outlineColor: theme.colors.primary
      }
    },
    [`&${classWrapper}, ${classButton}`]: {
      height: size === 'middle' ? theme.general.heightSm : SMALL_BUTTON_HEIGHT,
      lineHeight: theme.typography.lineHeightBase,
      alignItems: 'center'
    },
    ...getAnimationCss(theme.options.enableAnimation)
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSegmentedControlButtonEmotionStyles;");
}
const SegmentedControlGroupContext = /*#__PURE__*/createContext({
  size: 'middle'
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
    size
  } = useContext(SegmentedControlGroupContext);
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Radio$1.Button, {
      css: getSegmentedControlButtonEmotionStyles(classNamePrefix, theme, size),
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
    ...props
  } = _ref2;
  const {
    classNamePrefix
  } = useDesignSystemTheme();
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(SegmentedControlGroupContext.Provider, {
      value: {
        size
      },
      children: jsx(Radio$1.Group, {
        ...props,
        css: getSegmentedControlGroupEmotionStyles(classNamePrefix),
        ...dangerouslySetAntdProps,
        ref: ref
      })
    })
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$g() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const DEFAULT_WIDTH = 200;
const ContentContextDefaults = {
  openPanelId: undefined,
  closable: true,
  destroyInactivePanels: false,
  setIsClosed: () => {}
};
const SidebarContextDefaults = {
  position: 'left'
};
const ContentContext = /*#__PURE__*/createContext(ContentContextDefaults);
const SidebarContext = /*#__PURE__*/createContext(SidebarContextDefaults);
function Nav(_ref) {
  let {
    children,
    dangerouslyAppendEmotionCSS
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("nav", {
    css: [{
      display: 'flex',
      flexDirection: 'column',
      gap: theme.spacing.xs,
      padding: theme.spacing.xs
    }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Nav;"],
    children: children
  });
}
function NavButton(_ref2) {
  let {
    active,
    disabled,
    icon,
    onClick,
    children,
    dangerouslyAppendEmotionCSS,
    'aria-label': ariaLabel
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: [active ? importantify({
      borderRadius: theme.borders.borderRadiusMd,
      background: theme.colors.actionDefaultBackgroundPress,
      button: {
        '&:enabled:not(:hover):not(:active) > .anticon': {
          color: theme.colors.actionTertiaryTextPress
        }
      }
    }) : undefined, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:NavButton;"],
    children: jsx(Button, {
      icon: icon,
      onClick: onClick,
      disabled: disabled,
      "aria-label": ariaLabel,
      children: children
    })
  });
}
function Content(_ref3) {
  let {
    disableResize,
    openPanelId,
    closable,
    onClose,
    onResizeStop,
    width,
    minWidth,
    maxWidth,
    destroyInactivePanels = false,
    children,
    dangerouslyAppendEmotionCSS
  } = _ref3;
  const openAnimation = keyframes`
  from { width: 50px }
  to   { width: ${width}px }`;
  const showAnimation = keyframes`
  from { opacity: 0 }
  80%  { opacity: 0 }
  to   { opacity: 1 }`;
  const {
    theme
  } = useDesignSystemTheme();
  const sidebarContext = useContext(SidebarContext);
  const onCloseRef = useRef(onClose);
  const resizeHandleStyle = sidebarContext.position === 'right' ? {
    left: 0
  } : {
    right: 0
  };
  const [dragging, setDragging] = useState(false);
  const isPanelClosed = openPanelId == null;
  const [animation, setAnimation] = useState(isPanelClosed ? {
    open: `${openAnimation} .2s cubic-bezier(0, 0, 0.2, 1)`,
    show: `${showAnimation} .25s linear`
  } : undefined);
  const hiddenPanelStyle = /*#__PURE__*/css(isPanelClosed && {
    display: 'none'
  }, process.env.NODE_ENV === "production" ? "" : ";label:hiddenPanelStyle;");
  const containerStyle = /*#__PURE__*/css({
    animation: animation === null || animation === void 0 ? void 0 : animation.open,
    direction: sidebarContext.position === 'right' ? 'rtl' : 'ltr',
    marginLeft: -1,
    position: 'relative',
    borderWidth: sidebarContext.position === 'right' ? `0 ${theme.general.borderWidth}px 0 0 ` : `0 0 0 ${theme.general.borderWidth}px`,
    borderStyle: 'inherit',
    borderColor: 'inherit',
    boxSizing: 'content-box'
  }, process.env.NODE_ENV === "production" ? "" : ";label:containerStyle;");
  const highlightedBorderStyle = sidebarContext.position === 'right' ? /*#__PURE__*/css({
    borderLeft: `2px solid ${theme.colors.actionDefaultBorderHover}`
  }, process.env.NODE_ENV === "production" ? "" : ";label:highlightedBorderStyle;") : /*#__PURE__*/css({
    borderRight: `2px solid ${theme.colors.actionDefaultBorderHover}`
  }, process.env.NODE_ENV === "production" ? "" : ";label:highlightedBorderStyle;");
  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);
  const value = useMemo(() => ({
    openPanelId,
    closable: closable === undefined ? true : closable,
    destroyInactivePanels,
    setIsClosed: () => {
      var _onCloseRef$current;
      (_onCloseRef$current = onCloseRef.current) === null || _onCloseRef$current === void 0 ? void 0 : _onCloseRef$current.call(onCloseRef);
      if (!animation) {
        setAnimation({
          open: `${openAnimation} .2s cubic-bezier(0, 0, 0.2, 1)`,
          show: `${showAnimation} .25s linear`
        });
      }
    }
  }), [openPanelId, closable, openAnimation, showAnimation, animation, destroyInactivePanels]);
  return jsx(ContentContext.Provider, {
    value: value,
    children: disableResize ? jsx("div", {
      css: [/*#__PURE__*/css(containerStyle, {
        width: width || '100%',
        height: '100%',
        overflow: 'hidden'
      }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"), dangerouslyAppendEmotionCSS, hiddenPanelStyle, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      "aria-hidden": isPanelClosed,
      children: jsx("div", {
        css: /*#__PURE__*/css({
          opacity: 1,
          height: '100%',
          animation: animation === null || animation === void 0 ? void 0 : animation.show,
          direction: 'ltr'
        }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"),
        children: children
      })
    }) : jsx(ResizableBox, {
      width: width || DEFAULT_WIDTH,
      height: undefined,
      axis: "x",
      resizeHandles: sidebarContext.position === 'right' ? ['w'] : ['e'],
      minConstraints: [minWidth !== null && minWidth !== void 0 ? minWidth : DEFAULT_WIDTH, 150],
      maxConstraints: [maxWidth !== null && maxWidth !== void 0 ? maxWidth : 800, 150],
      onResizeStart: () => setDragging(true),
      onResizeStop: (_, _ref4) => {
        let {
          size
        } = _ref4;
        onResizeStop === null || onResizeStop === void 0 ? void 0 : onResizeStop(size.width);
        setDragging(false);
      },
      handle: jsx("div", {
        css: /*#__PURE__*/css({
          width: 10,
          height: '100%',
          position: 'absolute',
          top: 0,
          cursor: sidebarContext.position === 'right' ? 'w-resize' : 'e-resize',
          '&:hover': highlightedBorderStyle,
          ...resizeHandleStyle
        }, dragging && highlightedBorderStyle, process.env.NODE_ENV === "production" ? "" : ";label:Content;")
      }),
      css: [containerStyle, hiddenPanelStyle, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
      "aria-hidden": isPanelClosed,
      children: jsx("div", {
        css: [{
          opacity: 1,
          animation: animation === null || animation === void 0 ? void 0 : animation.show,
          direction: 'ltr',
          height: '100%'
        }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
        children: children
      })
    })
  });
}
function Panel(_ref5) {
  let {
    panelId,
    children,
    forceRender = false,
    dangerouslyAppendEmotionCSS,
    ...delegated
  } = _ref5;
  const {
    openPanelId,
    destroyInactivePanels
  } = useContext(ContentContext);
  const hasOpenedPanelRef = useRef(false);
  const isPanelOpen = openPanelId === panelId;
  if (isPanelOpen && !hasOpenedPanelRef.current) {
    hasOpenedPanelRef.current = true;
  }
  if ((destroyInactivePanels || !hasOpenedPanelRef.current) && !isPanelOpen && !forceRender) return null;
  return jsx("div", {
    css: ["display:flex;height:100%;flex-direction:column;", dangerouslyAppendEmotionCSS, !isPanelOpen && {
      display: 'none'
    }, process.env.NODE_ENV === "production" ? "" : ";label:Panel;"],
    "aria-hidden": !isPanelOpen,
    ...delegated,
    children: children
  });
}
var _ref7 = process.env.NODE_ENV === "production" ? {
  name: "1066lcq",
  styles: "display:flex;justify-content:space-between;align-items:center"
} : {
  name: "fs19p8-PanelHeader",
  styles: "display:flex;justify-content:space-between;align-items:center;label:PanelHeader;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$g
};
function PanelHeader(_ref6) {
  let {
    children,
    dangerouslyAppendEmotionCSS
  } = _ref6;
  const {
    theme
  } = useDesignSystemTheme();
  const contentContext = useContext(ContentContext);
  return jsxs("div", {
    css: [{
      display: 'flex',
      paddingLeft: 8,
      paddingRight: 4,
      alignItems: 'center',
      minHeight: theme.general.heightSm,
      justifyContent: 'space-between',
      fontWeight: theme.typography.typographyBoldFontWeight,
      color: theme.colors.textPrimary
    }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:PanelHeader;"],
    children: [jsx("div", {
      css: /*#__PURE__*/css({
        width: contentContext.closable ? `calc(100% - ${theme.spacing.lg}px)` : '100%'
      }, process.env.NODE_ENV === "production" ? "" : ";label:PanelHeader;"),
      children: jsx("div", {
        css: _ref7,
        children: children
      })
    }), contentContext.closable ? jsx("div", {
      children: jsx(Button, {
        size: "small",
        icon: jsx(CloseIcon, {}),
        "aria-label": "Close",
        onClick: () => {
          contentContext.setIsClosed();
        }
      })
    }) : null]
  });
}
function PanelHeaderTitle(_ref8) {
  let {
    title,
    dangerouslyAppendEmotionCSS
  } = _ref8;
  return jsx("div", {
    title: title,
    css: ["align-self:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;", dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:PanelHeaderTitle;"],
    children: title
  });
}
function PanelHeaderButtons(_ref9) {
  let {
    children,
    dangerouslyAppendEmotionCSS
  } = _ref9;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: [{
      display: 'flex',
      alignItems: 'center',
      gap: theme.spacing.xs,
      paddingRight: theme.spacing.xs
    }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:PanelHeaderButtons;"],
    children: children
  });
}
function PanelBody(_ref10) {
  let {
    children,
    dangerouslyAppendEmotionCSS
  } = _ref10;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: [{
      height: '100%',
      overflowX: 'hidden',
      overflowY: 'auto',
      padding: '0 8px',
      colorScheme: theme.isDarkMode ? 'dark' : 'light'
    }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:PanelBody;"],
    children: children
  });
}
const Sidebar = /* #__PURE__ */(() => {
  function Sidebar(_ref11) {
    let {
      position,
      children,
      dangerouslyAppendEmotionCSS
    } = _ref11;
    const {
      theme
    } = useDesignSystemTheme();
    const value = useMemo(() => {
      return {
        position: position || 'left'
      };
    }, [position]);
    return jsx(SidebarContext.Provider, {
      value: value,
      children: jsx("div", {
        css: [{
          display: 'flex',
          height: '100%',
          backgroundColor: theme.colors.backgroundPrimary,
          flexDirection: position === 'right' ? 'row-reverse' : 'row',
          borderStyle: 'solid',
          borderColor: theme.colors.borderDecorative,
          borderWidth: theme.general.borderWidth,
          boxSizing: 'content-box'
        }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Sidebar;"],
        children: children
      })
    });
  }
  Sidebar.Content = Content;
  Sidebar.Nav = Nav;
  Sidebar.NavButton = NavButton;
  Sidebar.Panel = Panel;
  Sidebar.PanelHeader = PanelHeader;
  Sidebar.PanelHeaderTitle = PanelHeaderTitle;
  Sidebar.PanelHeaderButtons = PanelHeaderButtons;
  Sidebar.PanelBody = PanelBody;
  return Sidebar;
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$f() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const GenericContainerStyles = process.env.NODE_ENV === "production" ? {
  name: "12h7em6",
  styles: "cursor:progress;border-radius:var(--border-radius)"
} : {
  name: "19fx6jo-GenericContainerStyles",
  styles: "cursor:progress;border-radius:var(--border-radius);label:GenericContainerStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$f
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$e() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const paragraphContainerStyles = process.env.NODE_ENV === "production" ? {
  name: "sj05g9",
  styles: "cursor:progress;width:100%;height:20px;display:flex;justify-content:flex-start;align-items:center"
} : {
  name: "u3a3v7-paragraphContainerStyles",
  styles: "cursor:progress;width:100%;height:20px;display:flex;justify-content:flex-start;align-items:center;label:paragraphContainerStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$e
};
const paragraphFillStyles = process.env.NODE_ENV === "production" ? {
  name: "10nptxl",
  styles: "border-radius:var(--border-radius);height:8px"
} : {
  name: "h6xifd-paragraphFillStyles",
  styles: "border-radius:var(--border-radius);height:8px;label:paragraphFillStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$e
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$d() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const titleContainerStyles = process.env.NODE_ENV === "production" ? {
  name: "116rc6i",
  styles: "cursor:progress;width:100%;height:28px;display:flex;justify-content:flex-start;align-items:center"
} : {
  name: "1dar8xl-titleContainerStyles",
  styles: "cursor:progress;width:100%;height:28px;display:flex;justify-content:flex-start;align-items:center;label:titleContainerStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$d
};
const titleFillStyles = process.env.NODE_ENV === "production" ? {
  name: "9fmdbb",
  styles: "border-radius:var(--border-radius);height:12px;width:100%"
} : {
  name: "1vyd6dg-titleFillStyles",
  styles: "border-radius:var(--border-radius);height:12px;width:100%;label:titleFillStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$d
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$c() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
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
    name: "1n3v1dk",
    styles: "display:inline-grid;position:relative;flex:1;line-height:initial;box-sizing:border-box;padding-left:var(--table-spacing-sm);padding-right:var(--table-spacing-sm);word-break:break-word;overflow:hidden;& .anticon{vertical-align:text-bottom;}"
  } : {
    name: "q5g0tm-cell",
    styles: "display:inline-grid;position:relative;flex:1;line-height:initial;box-sizing:border-box;padding-left:var(--table-spacing-sm);padding-right:var(--table-spacing-sm);word-break:break-word;overflow:hidden;& .anticon{vertical-align:text-bottom;};label:cell;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
  },
  header: process.env.NODE_ENV === "production" ? {
    name: "ik7qgz",
    styles: "font-weight:bold;align-items:flex-end;display:flex;overflow:hidden;&[aria-sort]{cursor:pointer;user-select:none;}.table-header-text{color:var(--table-header-text-color);}.table-header-icon-container{color:var(--table-header-sort-icon-color);display:none;}&[aria-sort]:hover{.table-header-icon-container, .table-header-text{color:var(--table-header-focus-color);}}&[aria-sort]:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}}&:hover, &[aria-sort=\"ascending\"], &[aria-sort=\"descending\"]{.table-header-icon-container{display:inline;}}"
  } : {
    name: "1xg6jn4-header",
    styles: "font-weight:bold;align-items:flex-end;display:flex;overflow:hidden;&[aria-sort]{cursor:pointer;user-select:none;}.table-header-text{color:var(--table-header-text-color);}.table-header-icon-container{color:var(--table-header-sort-icon-color);display:none;}&[aria-sort]:hover{.table-header-icon-container, .table-header-text{color:var(--table-header-focus-color);}}&[aria-sort]:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}}&:hover, &[aria-sort=\"ascending\"], &[aria-sort=\"descending\"]{.table-header-icon-container{display:inline;}};label:header;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
  },
  row: process.env.NODE_ENV === "production" ? {
    name: "gf0r0k",
    styles: "display:flex;&.table-isHeader{> *{background-color:var(--table-header-background-color);}.table-isScrollable &{position:sticky;top:0;z-index:1;}}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:var(--row-checkbox-opacity, 0);}&:not(.table-row-isGrid)&:hover{&:not(.table-isHeader){background-color:var(--table-row-hover);}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:1;}}.table-row-select-cell input[type=\"checkbox\"]:focus ~ *{opacity:1;}> *{padding-top:var(--table-row-vertical-padding);padding-bottom:var(--table-row-vertical-padding);border-bottom:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid > *{border-right:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid > :first-of-type{border-left:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid.table-row-isHeader:first-of-type > *{border-top:1px solid;border-color:var(--table-separator-color);}"
  } : {
    name: "1nztrlu-row",
    styles: "display:flex;&.table-isHeader{> *{background-color:var(--table-header-background-color);}.table-isScrollable &{position:sticky;top:0;z-index:1;}}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:var(--row-checkbox-opacity, 0);}&:not(.table-row-isGrid)&:hover{&:not(.table-isHeader){background-color:var(--table-row-hover);}.table-row-select-cell input[type=\"checkbox\"] ~ *{opacity:1;}}.table-row-select-cell input[type=\"checkbox\"]:focus ~ *{opacity:1;}> *{padding-top:var(--table-row-vertical-padding);padding-bottom:var(--table-row-vertical-padding);border-bottom:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid > *{border-right:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid > :first-of-type{border-left:1px solid;border-color:var(--table-separator-color);}&.table-row-isGrid.table-row-isHeader:first-of-type > *{border-top:1px solid;border-color:var(--table-separator-color);};label:row;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
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
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
  },
  headerButtonTarget: process.env.NODE_ENV === "production" ? {
    name: "sezlox",
    styles: "align-items:flex-end;display:flex;overflow:hidden;width:100%;justify-content:inherit;&:focus{.table-header-text{color:var(--table-header-focus-color);}.table-header-icon-container{color:var(--table-header-focus-color);display:inline;}}&:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}}"
  } : {
    name: "1iv4trp-headerButtonTarget",
    styles: "align-items:flex-end;display:flex;overflow:hidden;width:100%;justify-content:inherit;&:focus{.table-header-text{color:var(--table-header-focus-color);}.table-header-icon-container{color:var(--table-header-focus-color);display:inline;}}&:active{.table-header-icon-container, .table-header-text{color:var(--table-header-active-color);}};label:headerButtonTarget;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
  },
  sortHeaderIconOnRight: process.env.NODE_ENV === "production" ? {
    name: "1hdiaor",
    styles: "margin-left:var(--table-spacing-xs)"
  } : {
    name: "evh3p3-sortHeaderIconOnRight",
    styles: "margin-left:var(--table-spacing-xs);label:sortHeaderIconOnRight;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
  },
  sortHeaderIconOnLeft: process.env.NODE_ENV === "production" ? {
    name: "d4plmt",
    styles: "margin-right:var(--table-spacing-xs)"
  } : {
    name: "1gr7edl-sortHeaderIconOnLeft",
    styles: "margin-right:var(--table-spacing-xs);label:sortHeaderIconOnLeft;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
  },
  checkboxCell: process.env.NODE_ENV === "production" ? {
    name: "4cdr0s",
    styles: "display:flex;align-items:center;flex:0;padding-left:var(--table-spacing-sm);padding-top:0;padding-bottom:0;min-width:var(--table-spacing-md);max-width:var(--table-spacing-md);box-sizing:content-box"
  } : {
    name: "17au6u2-checkboxCell",
    styles: "display:flex;align-items:center;flex:0;padding-left:var(--table-spacing-sm);padding-top:0;padding-bottom:0;min-width:var(--table-spacing-md);max-width:var(--table-spacing-md);box-sizing:content-box;label:checkboxCell;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
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
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
  },
  paginationContainer: process.env.NODE_ENV === "production" ? {
    name: "ehlmid",
    styles: "display:flex;justify-content:flex-end;padding-top:var(--table-spacing-sm);padding-bottom:var(--table-spacing-sm)"
  } : {
    name: "p324df-paginationContainer",
    styles: "display:flex;justify-content:flex-end;padding-top:var(--table-spacing-sm);padding-bottom:var(--table-spacing-sm);label:paginationContainer;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$c
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
  const flags = useDesignSystemFlags();
  return jsx(TableContext.Provider, {
    value: {
      size,
      someRowsSelected,
      grid
    },
    children: jsxs("div", {
      ...rest,
      // This is a performance optimization; we want to statically create the styles for the table,
      // but for the dynamic theme values, we need to use CSS variables.
      // See: https://emotion.sh/docs/best-practices#advanced-css-variables-with-style
      style: {
        ...style,
        ['--table-background-color']: theme.colors.backgroundPrimary,
        ['--table-header-active-color']: theme.colors.actionDefaultTextPress,
        ['colorScheme']: theme.isDarkMode ? 'dark' : undefined,
        // This hex is pulled directly from the old table as a temporary style-matching measure.
        ['--table-header-background-color']: flags.USE_UPDATED_TABLE_STYLES || theme.isDarkMode ? theme.colors.backgroundPrimary : '#F2F5F7',
        ['--table-header-focus-color']: theme.colors.actionDefaultTextHover,
        ['--table-header-sort-icon-color']: theme.colors.textSecondary,
        ['--table-header-text-color']: theme.colors.actionDefaultTextDefault,
        ['--table-row-hover']: theme.colors.tableRowHover,
        ['--table-separator-color']: theme.colors.borderDecorative,
        ['--table-resize-handle-color']: flags.USE_UPDATED_TABLE_STYLES ? theme.colors.borderDecorative : theme.colors.grey400,
        ['--table-spacing-md']: `${theme.spacing.md}px`,
        ['--table-spacing-sm']: `${theme.spacing.sm}px`,
        ['--table-spacing-xs']: `${theme.spacing.xs}px`
      },
      css: tableStyles$1.tableWrapper,
      className: classnames({
        'table-isScrollable': scrollable,
        'table-isGrid': grid
      }, className),
      children: [jsxs("div", {
        role: "table",
        ref: ref,
        css: tableStyles$1.table,
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
    value: {
      isHeader
    },
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$b() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const TableRowActionStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "gk361n",
    styles: "width:32px;padding-top:var(--vertical-padding);padding-bottom:var(--vertical-padding);display:flex;align-items:start;justify-content:center"
  } : {
    name: "q9pljs-container",
    styles: "width:32px;padding-top:var(--vertical-padding);padding-bottom:var(--vertical-padding);display:flex;align-items:start;justify-content:center;label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$b
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
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    ...rest,
    ref: ref,
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$a() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const TableSkeletonStyles = {
  container: process.env.NODE_ENV === "production" ? {
    name: "6kz1wu",
    styles: "display:flex;flex-direction:column;align-items:flex-start"
  } : {
    name: "1we0er9-container",
    styles: "display:flex;flex-direction:column;align-items:flex-start;label:container;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$a
  },
  cell: process.env.NODE_ENV === "production" ? {
    name: "1t820zr",
    styles: "width:100%;height:8px;border-radius:4px;background:var(--table-skeleton-color);margin-top:var(--table-skeleton-row-vertical-margin);margin-bottom:var(--table-skeleton-row-vertical-margin)"
  } : {
    name: "1m8dl5b-cell",
    styles: "width:100%;height:8px;border-radius:4px;background:var(--table-skeleton-color);margin-top:var(--table-skeleton-row-vertical-margin);margin-bottom:var(--table-skeleton-row-vertical-margin);label:cell;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$a
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
    // TODO: Re-enable this after Clusters fixes tests: https://databricks.slack.com/archives/C04LYE3F8HX/p1679597678339659
    // {...rest}
    css: TableSkeletonStyles.container,
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

const defaultGetPrefixCls = (suffixCls, customizePrefixCls) => {
  if (customizePrefixCls) return customizePrefixCls;
  return suffixCls ? `ant-${suffixCls}` : 'ant';
};
const ConfigContext = /*#__PURE__*/React.createContext({
  // We provide a default function for Context without provider
  getPrefixCls: defaultGetPrefixCls
});
const ButtonGroup = Button$1.Group;
const DropdownButton = props => {
  const {
    getPopupContainer: getContextPopupContainer,
    getPrefixCls,
    direction
  } = React.useContext(ConfigContext);
  const {
    prefixCls: customizePrefixCls,
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
    ...restProps
  } = props;
  const prefixCls = getPrefixCls('dropdown-button', customizePrefixCls);
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
    dropdownProps.placement = direction === 'rtl' ? 'bottomLeft' : 'bottomRight';
  }
  const leftButton = jsxs(Button, {
    type: type,
    danger: danger,
    disabled: disabled,
    loading: loading,
    onClick: onClick,
    htmlType: htmlType,
    href: href,
    title: title,
    icon: children && leftButtonIcon ? leftButtonIcon : undefined,
    children: [leftButtonIcon && !children ? leftButtonIcon : undefined, children]
  });
  const rightButton = jsx(Button, {
    type: type,
    danger: danger,
    disabled: disabled,
    "aria-label": menuButtonLabel,
    children: icon ? icon : jsx(ChevronDownIcon, {})
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$9() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
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
      ...getDisabledStyles(theme),
      [`&:first-of-type`]: {
        borderRight: `1px solid ${theme.colors.actionDisabledText}`,
        marginRight: 1
      },
      [classDropdownTrigger]: {
        borderLeft: `1px solid ${theme.colors.actionDisabledText}`
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
var _ref$3 = process.env.NODE_ENV === "production" ? {
  name: "tp1ooh",
  styles: "display:inline-flex;position:relative;vertical-align:middle"
} : {
  name: "1kplxg4-SplitButton",
  styles: "display:inline-flex;position:relative;vertical-align:middle;label:SplitButton;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$9
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
      css: _ref$3,
      children: loading ? jsx(Button, {
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
        icon: jsx(ChevronDownIcon, {
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

const getSwitchWithLabelStyles = _ref => {
  let {
    clsPrefix,
    theme
  } = _ref;
  // Default value
  const SWITCH_WIDTH = 28;
  const styles = {
    display: 'flex',
    alignItems: 'center',
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
      [`&:focus-visible`]: {
        border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
        boxShadow: 'none',
        outlineStyle: 'solid',
        outlineWidth: '1px',
        outlineOffset: '1px',
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
        backgroundColor: theme.colors.actionDisabledBackground,
        border: `1px solid ${theme.colors.actionDisabledBackground}`,
        [`.${clsPrefix}-switch-handle:before`]: {
          boxShadow: `0px 0px 0px 1px ${theme.colors.actionDisabledBackground}`
        }
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
    }
  };
  const importantStyles = importantify(styles);
  return /*#__PURE__*/css(importantStyles, process.env.NODE_ENV === "production" ? "" : ";label:getSwitchWithLabelStyles;");
};
const Switch = _ref2 => {
  let {
    dangerouslySetAntdProps,
    label,
    labelProps,
    ...props
  } = _ref2;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  return label ? jsx(DesignSystemAntDConfigProvider, {
    children: jsxs("div", {
      css: getSwitchWithLabelStyles({
        clsPrefix: classNamePrefix,
        theme
      }),
      children: [jsx(Switch$1, {
        ...props,
        ...dangerouslySetAntdProps,
        css: /*#__PURE__*/css({
          ... /*#__PURE__*/css(getAnimationCss(theme.options.enableAnimation), process.env.NODE_ENV === "production" ? "" : ";label:Switch;"),
          ...getSwitchWithLabelStyles({
            clsPrefix: classNamePrefix,
            theme
          })
        }, process.env.NODE_ENV === "production" ? "" : ";label:Switch;")
      }), jsx(Label, {
        inline: true,
        ...labelProps,
        children: label
      })]
    })
  }) : jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Switch$1, {
      ...props,
      ...dangerouslySetAntdProps,
      css: /*#__PURE__*/css({
        ... /*#__PURE__*/css(getAnimationCss(theme.options.enableAnimation), process.env.NODE_ENV === "production" ? "" : ";label:Switch;"),
        ...getSwitchWithLabelStyles({
          clsPrefix: classNamePrefix,
          theme
        })
      }, process.env.NODE_ENV === "production" ? "" : ";label:Switch;")
    })
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$8() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
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
var _ref2$5 = process.env.NODE_ENV === "production" ? {
  name: "7whenc",
  styles: "display:flex;width:100%"
} : {
  name: "3ktoz7-component",
  styles: "display:flex;width:100%;label:component;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$8
};
var _ref3$1 = process.env.NODE_ENV === "production" ? {
  name: "82a6rk",
  styles: "flex:1"
} : {
  name: "18ug1j7-component",
  styles: "flex:1;label:component;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$8
};
const TableFilterInput = /*#__PURE__*/forwardRef(function SearchInput(_ref, ref) {
  let {
    onSubmit,
    showSearchButton,
    className,
    ...inputProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const DEFAULT_WIDTH = 400;
  let component = jsx(Input, {
    prefix: jsx(SearchIcon, {}),
    allowClear: true,
    ...inputProps,
    className: className,
    ref: ref
  });
  if (showSearchButton) {
    component = jsxs(Input.Group, {
      css: _ref2$5,
      className: className,
      children: [jsx(Input, {
        allowClear: true,
        ...inputProps,
        ref: ref,
        css: _ref3$1
      }), jsx(Button, {
        htmlType: "submit",
        children: jsx(SearchIcon, {})
      })]
    });
  }
  const commonProps = {
    style: {
      height: theme.general.heightSm
    },
    css: getTableFilterInputStyles(theme, DEFAULT_WIDTH)
  };
  if (onSubmit) {
    return jsx("form", {
      ...commonProps,
      onSubmit: e => {
        e.preventDefault();
        onSubmit();
      },
      children: component
    });
  } else {
    return jsx("div", {
      ...commonProps,
      children: component
    });
  }
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$7() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const tableFilterLayoutStyles = {
  layout: process.env.NODE_ENV === "production" ? {
    name: "1yb0qmd",
    styles: "display:flex;flex-direction:row;justify-content:space-between;margin-bottom:var(--table-filter-layout-group-margin)"
  } : {
    name: "bmua0k-layout",
    styles: "display:flex;flex-direction:row;justify-content:space-between;margin-bottom:var(--table-filter-layout-group-margin);label:layout;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$7
  },
  filters: process.env.NODE_ENV === "production" ? {
    name: "2pdmyz",
    styles: "display:flex;flex-wrap:wrap;flex-direction:row;align-items:center;gap:var(--table-filter-layout-item-gap);flex:1"
  } : {
    name: "i28ows-filters",
    styles: "display:flex;flex-wrap:wrap;flex-direction:row;align-items:center;gap:var(--table-filter-layout-item-gap);flex:1;label:filters;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$7
  },
  filterActions: process.env.NODE_ENV === "production" ? {
    name: "1ol8kzq",
    styles: "display:flex;gap:var(--table-filter-layout-item-gap);margin-left:var(--table-filter-layout-group-margin)"
  } : {
    name: "bcekwq-filterActions",
    styles: "display:flex;gap:var(--table-filter-layout-item-gap);margin-left:var(--table-filter-layout-group-margin);label:filterActions;",
    toString: _EMOTION_STRINGIFIED_CSS_ERROR__$7
  }
};
const TableFilterLayout = /*#__PURE__*/forwardRef(function TableFilterLayout(_ref, ref) {
  let {
    children,
    style,
    className,
    actions,
    ...rest
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
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
    ...rest
  } = _ref2;
  const {
    size,
    grid
  } = useContext(TableContext);
  const {
    isHeader
  } = useContext(TableRowContext);
  if (!isHeader) {
    throw new Error('`TableHeader` must be used within a `TableRow` with `isHeader` set to true.');
  }
  let sortIcon = jsx(Fragment, {});
  // While most libaries use `asc` and `desc` for the sort value, the ARIA spec
  // uses `ascending` and `descending`.
  let ariaSort;
  if (sortable) {
    if (sortDirection === 'asc') {
      sortIcon = jsx(SortAscendingIcon, {});
      ariaSort = 'ascending';
    } else if (sortDirection === 'desc') {
      sortIcon = jsx(SortDescendingIcon, {});
      ariaSort = 'descending';
    } else if (sortDirection === 'none') {
      sortIcon = jsx(SortUnsortedIcon, {});
      ariaSort = 'none';
    }
  }
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

const TableRowSelectCell = /*#__PURE__*/forwardRef(function TableRowSelectCell(_ref, ref) {
  let {
    onChange,
    checked,
    indeterminate,
    noCheckbox,
    children,
    isDisabled,
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
      isChecked: checked || indeterminate && null,
      onChange: (_checked, event) => onChange === null || onChange === void 0 ? void 0 : onChange(event.nativeEvent),
      isDisabled: isDisabled
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

const colorMap = {
  default: 'tagDefault',
  brown: 'tagBrown',
  coral: 'tagCoral',
  charcoal: 'tagCharcoal',
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
  let centerIcons = arguments.length > 3 ? arguments[3] : undefined;
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
      lineHeight: theme.typography.lineHeightSm,
      ...(centerIcons && {
        '& .anticon': {
          verticalAlign: 'text-top'
        }
      })
    }
  };
}
function Tag(props) {
  const {
    theme
  } = useDesignSystemTheme();
  const {
    color,
    children,
    closable,
    onClose,
    role = 'status',
    ...attributes
  } = props;
  const isClickable = Boolean(props.onClick);
  const centerIconsInText = safex('databricks.fe.designsystem.centerIconsInText', false);
  const styles = getTagEmotionStyles(theme, color, isClickable, centerIconsInText);
  return jsx("div", {
    role: role,
    ...attributes,
    css: styles.tag,
    children: jsxs("div", {
      css: [styles.content, styles.text, process.env.NODE_ENV === "production" ? "" : ";label:Tag;"],
      children: [children, closable && jsx("button", {
        css: styles.close,
        tabIndex: 0,
        onClick: e => {
          e.stopPropagation();
          if (onClose) {
            onClose();
          }
        },
        onMouseDown: e => {
          // Keeps dropdowns of any underlying select from opening.
          e.stopPropagation();
        },
        children: jsx(CloseIcon, {
          css: /*#__PURE__*/css({
            fontSize: theme.general.iconFontSize - 4
          }, process.env.NODE_ENV === "production" ? "" : ";label:Tag;")
        })
      })]
    })
  });
}

const getStyles = theme => {
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
      border: 'transparent',
      color: theme.colors.actionDisabledText,
      backgroundColor: theme.colors.actionDisabledBackground,
      '& > svg': {
        stroke: theme.colors.border
      }
    }
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
const ToggleButton = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    children,
    pressed,
    defaultPressed,
    ...props
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const [isPressed, setIsPressed] = React__default.useState(defaultPressed);
  useEffect(() => {
    setIsPressed(pressed);
  }, [pressed]);
  return jsxs(Toggle.Root, {
    css: getStyles(theme),
    ...props,
    pressed: isPressed,
    onPressedChange: pressed => {
      var _props$onPressedChang;
      (_props$onPressedChang = props.onPressedChange) === null || _props$onPressedChang === void 0 ? void 0 : _props$onPressedChang.call(props, pressed);
      setIsPressed(pressed);
    },
    ref: ref,
    children: [isPressed ? jsx(CheckIcon, {
      css: /*#__PURE__*/css({
        marginRight: theme.spacing.xs
      }, process.env.NODE_ENV === "production" ? "" : ";label:ToggleButton;")
    }) : jsx(RectangleSvg, {
      css: /*#__PURE__*/css({
        height: theme.typography.lineHeightBase,
        width: theme.typography.lineHeightSm,
        marginRight: theme.spacing.xs,
        stroke: theme.colors.border
      }, process.env.NODE_ENV === "production" ? "" : ";label:ToggleButton;")
    }), children]
  });
});

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
function getTreeEmotionStyles(clsPrefix, theme, size, useNewTree) {
  const classNode = `.${clsPrefix}-tree-treenode`;
  const classNodeSelected = `.${clsPrefix}-tree-treenode-selected`;
  const classNodeActive = `.${clsPrefix}-tree-treenode-active`;
  const classNodeDisabled = `.${clsPrefix}-tree-treenode-disabled`;
  const classContent = `.${clsPrefix}-tree-node-content-wrapper`;
  const classSelected = `.${clsPrefix}-tree-node-selected`;
  const classSwitcher = `.${clsPrefix}-tree-switcher`;
  const classSwitcherNoop = `.${clsPrefix}-tree-switcher-noop`;
  const classFocused = `.${clsPrefix}-tree-focused`;
  const classCheckbox = `.${clsPrefix}-tree-checkbox`;
  const classUnselectable = `.${clsPrefix}-tree-unselectable`;
  const classIndent = `.${clsPrefix}-tree-indent-unit`;
  const classTreeList = `.${clsPrefix}-tree-list`;
  const classScrollbar = `.${clsPrefix}-tree-list-scrollbar`;
  const NODE_SIZE = sizeMap[size].nodeSize;
  const ICON_FONT_SIZE = useNewTree ? 16 : 24;
  const BORDER_WIDTH = 4;
  const styles = {
    // Basic node
    [classNode]: {
      height: NODE_SIZE,
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
      backgroundColor: theme.colors.actionTertiaryBackgroundPress,
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
      ...(!useNewTree && {
        padding: 0
      }),
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
    // TODO(FEINF-1595): Temporary style for now
    ...(useNewTree && {
      [`${classSwitcherNoop} + ${classContent}, ${classSwitcherNoop} + ${classCheckbox}`]: {
        marginLeft: NODE_SIZE + 4
      }
    }),
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
      ...(useNewTree && {
        display: 'none'
      }),
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
      marginTop: 0,
      marginBottom: 0,
      marginRight: theme.spacing.sm
    },
    // Vertical line
    ...(useNewTree && {
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
      }
    }),
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
  const {
    USE_NEW_TREE
  } = useDesignSystemFlags();
  let calculatedShowLine = showLine !== null && showLine !== void 0 ? showLine : false;
  if (hideLinesForSizes.includes(size)) {
    calculatedShowLine = false;
  } else if (USE_NEW_TREE) {
    calculatedShowLine = showLine !== null && showLine !== void 0 ? showLine : SHOW_LINE_DEFAULT;
  }
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Tree$1, {
      treeData: treeData,
      defaultExpandedKeys: defaultExpandedKeys,
      defaultSelectedKeys: defaultSelectedKeys,
      defaultCheckedKeys: defaultCheckedKeys,
      disabled: disabled,
      css: getTreeEmotionStyles(classNamePrefix, theme, size, USE_NEW_TREE),
      switcherIcon: jsx(ChevronDownIcon, {
        css: /*#__PURE__*/css({
          fontSize: `${USE_NEW_TREE ? 16 : 24}px!important`
        }, process.env.NODE_ENV === "production" ? "" : ";label:Tree;")
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

const typeaheadComboboxContextDefaults = {
  isInsideTypeaheadCombobox: false,
  multiSelect: false
};
const TypeaheadComboboxContext = /*#__PURE__*/createContext(typeaheadComboboxContextDefaults);
const TypeaheadComboboxContextProvider = _ref => {
  let {
    children,
    value
  } = _ref;
  return jsx(TypeaheadComboboxContext.Provider, {
    value: value,
    children: children
  });
};

const TypeaheadComboboxRoot = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    comboboxState,
    multiSelect = false,
    children,
    ...props
  } = _ref;
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
      multiSelect,
      isInsideTypeaheadCombobox: true,
      floatingUiRefs: refs,
      floatingStyles: floatingStyles
    },
    children: jsx("div", {
      ...comboboxState.getComboboxProps({}, {
        suppressRefError: true
      }),
      ...props,
      ref: ref,
      children: children
    })
  });
});

function useComboboxState(_ref) {
  let {
    allItems,
    items,
    itemToString,
    onIsOpenChange,
    ...props
  } = _ref;
  function getFilteredItems(inputValue) {
    var _inputValue$toLowerCa;
    const lowerCasedInputValue = (_inputValue$toLowerCa = inputValue === null || inputValue === void 0 ? void 0 : inputValue.toLowerCase()) !== null && _inputValue$toLowerCa !== void 0 ? _inputValue$toLowerCa : '';
    return allItems.filter(item => !inputValue || props.matcher(item, lowerCasedInputValue));
  }
  const comboboxState = useCombobox({
    onIsOpenChange: onIsOpenChange,
    onInputValueChange: props.multiSelect ? _ref2 => {
      let {
        inputValue
      } = _ref2;
      return inputValue !== undefined && props.setInputValue(inputValue);
    } : _ref3 => {
      let {
        inputValue
      } = _ref3;
      return props.setItems(getFilteredItems(inputValue));
    },
    items: items,
    itemToString(item) {
      return item ? itemToString ? itemToString(item) : item.toString() : '';
    },
    defaultHighlightedIndex: props.multiSelect ? 0 : undefined,
    // after selection for multiselect, highlight the first item.
    scrollIntoView: () => {},
    // disabling scroll because floating-ui repositions the menu
    selectedItem: props.multiSelect ? null : undefined,
    // useMultipleSelection will handle the item selection for multiselect
    stateReducer(state, actionAndChanges) {
      const {
        changes,
        type
      } = actionAndChanges;
      switch (type) {
        case useCombobox.stateChangeTypes.InputBlur:
          // If input is empty, clear selection
          if (!props.multiSelect && state.inputValue === '') {
            return {};
          } else {
            return changes;
          }
        case useCombobox.stateChangeTypes.InputKeyDownEnter:
        case useCombobox.stateChangeTypes.ItemClick:
          return {
            ...changes,
            highlightedIndex: 0,
            // with the first option highlighted.
            isOpen: props.multiSelect ? true : false // for multiselect, keep the menu open after selection.
          };

        default:
          return changes;
      }
    },
    onStateChange: props.multiSelect ? _ref4 => {
      let {
        type,
        selectedItem: newSelectedItem,
        inputValue: newInputValue
      } = _ref4;
      switch (type) {
        case useCombobox.stateChangeTypes.InputKeyDownEnter:
        case useCombobox.stateChangeTypes.ItemClick:
        case useCombobox.stateChangeTypes.InputBlur:
          if (newSelectedItem) {
            props.setSelectedItems([...props.selectedItems, newSelectedItem]);
            props.setInputValue('');
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
    } : undefined
  });
  return comboboxState;
}
function useMultipleSelectionState(selectedItems, setSelectedItems) {
  return useMultipleSelection({
    selectedItems,
    onStateChange(_ref5) {
      let {
        selectedItems: newSelectedItems,
        type
      } = _ref5;
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$6() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref$2 = process.env.NODE_ENV === "production" ? {
  name: "xv0ss6",
  styles: "padding:0;margin:0;display:flex;flex-direction:column;align-items:flex-start;position:absolute"
} : {
  name: "1sd733r-getTypeaheadComboboxMenuStyles",
  styles: "padding:0;margin:0;display:flex;flex-direction:column;align-items:flex-start;position:absolute;label:getTypeaheadComboboxMenuStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$6
};
const getTypeaheadComboboxMenuStyles = () => {
  return _ref$2;
};
const TypeaheadComboboxMenu = /*#__PURE__*/forwardRef((_ref2, ref) => {
  let {
    comboboxState,
    loading,
    width,
    minWidth = 240,
    maxWidth,
    minHeight,
    maxHeight,
    listWrapperHeight,
    virtualizerRef,
    children,
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
    isInsideTypeaheadCombobox
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
  const hasResults = children && Children.toArray(children).some(child => {
    if ( /*#__PURE__*/React__default.isValidElement(child)) {
      var _child$props$__EMOTIO, _child$props$__EMOTIO2;
      const childType = (_child$props$__EMOTIO = (_child$props$__EMOTIO2 = child.props['__EMOTION_TYPE_PLEASE_DO_NOT_USE__']) === null || _child$props$__EMOTIO2 === void 0 ? void 0 : _child$props$__EMOTIO2.defaultProps._TYPE) !== null && _child$props$__EMOTIO !== void 0 ? _child$props$__EMOTIO : child.props._TYPE;
      return ['TypeaheadComboboxMenuItem', 'TypeaheadComboboxCheckboxItem'].includes(childType);
    }
    return false;
  });
  const footer = Children.toArray(children).filter(child => /*#__PURE__*/React__default.isValidElement(child) && child.props._TYPE === 'TypeaheadComboboxFooter');
  return /*#__PURE__*/createPortal(jsx("ul", {
    "aria-busy": loading,
    ...downshiftProps,
    ref: mergedRef,
    css: [getComboboxContentWrapperStyles(theme, {
      maxHeight,
      maxWidth,
      minHeight,
      minWidth,
      width
    }), getTypeaheadComboboxMenuStyles(), process.env.NODE_ENV === "production" ? "" : ";label:TypeaheadComboboxMenu;"],
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
      children: [jsx(EmptyResults, {}), footer]
    })
  }), getPopupContainer ? getPopupContainer() : document.body);
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$5() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
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
var _ref2$4 = process.env.NODE_ENV === "production" ? {
  name: "kjj0ot",
  styles: "padding-top:2px"
} : {
  name: "1uez6s6-TypeaheadComboboxMenuItem",
  styles: "padding-top:2px;label:TypeaheadComboboxMenuItem;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$5
};
var _ref3 = process.env.NODE_ENV === "production" ? {
  name: "zjik7",
  styles: "display:flex"
} : {
  name: "1p9uv37-TypeaheadComboboxMenuItem",
  styles: "display:flex;label:TypeaheadComboboxMenuItem;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$5
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
    children,
    ...restProps
  } = _ref;
  const {
    selectedItem,
    highlightedIndex,
    getItemProps
  } = comboboxState;
  const isSelected = selectedItem === item;
  const isHighlighted = highlightedIndex === index;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs("li", {
    ref: ref,
    role: "option",
    "aria-selected": isSelected,
    disabled: isDisabled,
    css: [getComboboxOptionItemWrapperStyles(theme), getMenuItemStyles(theme, isHighlighted, isDisabled), process.env.NODE_ENV === "production" ? "" : ";label:TypeaheadComboboxMenuItem;"],
    ...getItemProps({
      item,
      index,
      disabled: isDisabled
    }),
    ...restProps,
    children: [isSelected ? jsx(CheckIcon, {
      css: _ref2$4
    }) : jsx("div", {
      style: {
        width: 16,
        flexShrink: 0
      }
    }), jsxs("label", {
      css: getLabelStyles(theme, textOverflowMode),
      children: [isDisabled && disabledReason ? jsxs("div", {
        css: _ref3,
        children: [jsx("div", {
          children: children
        }), jsx("div", {
          css: getInfoIconStyles(theme),
          children: jsx(InfoTooltip, {
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

function _EMOTION_STRINGIFIED_CSS_ERROR__$4() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$3 = process.env.NODE_ENV === "production" ? {
  name: "zjik7",
  styles: "display:flex"
} : {
  name: "1pneh3l-TypeaheadComboboxCheckboxItem",
  styles: "display:flex;label:TypeaheadComboboxCheckboxItem;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
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
    children,
    ...restProps
  } = _ref;
  const {
    highlightedIndex,
    getItemProps
  } = comboboxState;
  const isHighlighted = highlightedIndex === index;
  const {
    theme
  } = useDesignSystemTheme();
  const isSelected = selectedItems.includes(item);
  return jsx("li", {
    ref: ref,
    role: "option",
    "aria-selected": isSelected,
    disabled: isDisabled,
    css: [getComboboxOptionItemWrapperStyles(theme), getMenuItemStyles(theme, isHighlighted, isDisabled), process.env.NODE_ENV === "production" ? "" : ";label:TypeaheadComboboxCheckboxItem;"],
    ...getItemProps({
      item,
      index,
      disabled: isDisabled,
      onMouseUp: e => {
        var _restProps$onMouseUp;
        e.stopPropagation();
        (_restProps$onMouseUp = restProps.onMouseUp) === null || _restProps$onMouseUp === void 0 ? void 0 : _restProps$onMouseUp.call(restProps, e);
      }
    }),
    ...restProps,
    children: jsx(Checkbox, {
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
          css: _ref2$3,
          children: [jsx("div", {
            children: children
          }), jsx("div", {
            css: getInfoIconStyles(theme),
            children: jsx(InfoTooltip, {
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
    "aria-label": "toggle menu",
    ref: ref,
    css: getToggleButtonStyles(theme, disabled),
    ...restProps,
    onClick: handleClick,
    children: jsx(ChevronDownIcon, {})
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$3() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$2 = process.env.NODE_ENV === "production" ? {
  name: "l1fpjx",
  styles: "pointer-events:all;vertical-align:text-top"
} : {
  name: "4edanz-TypeaheadComboboxControls",
  styles: "pointer-events:all;vertical-align:text-top;label:TypeaheadComboboxControls;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$3
};
const TypeaheadComboboxControls = _ref => {
  let {
    getDownshiftToggleButtonProps,
    showClearSelectionButton,
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
      height: 16
    }, process.env.NODE_ENV === "production" ? "" : ";label:TypeaheadComboboxControls;"),
    children: [showClearSelectionButton && jsx(ClearSelectionButton, {
      onClick: handleClear,
      css: _ref2$2
    }), jsx(TypeaheadComboboxToggleButton, {
      ...getDownshiftToggleButtonProps(),
      disabled: disabled
    })]
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__$2() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref$1 = process.env.NODE_ENV === "production" ? {
  name: "1lb3761",
  styles: "display:flex;position:relative;width:fit-content"
} : {
  name: "rseyaw-getContainerStyles",
  styles: "display:flex;position:relative;width:fit-content;label:getContainerStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$2
};
const getContainerStyles$1 = () => {
  return _ref$1;
};
const getInputStyles$1 = theme => /*#__PURE__*/css({
  paddingRight: 52,
  width: '100%',
  minWidth: 72,
  '&:disabled': {
    border: 'none',
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
    ...restProps
  } = _ref2;
  const {
    isInsideTypeaheadCombobox,
    floatingUiRefs
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
    reset
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
  const handleClick = () => {
    toggleMenu();
  };
  const handleClear = () => {
    setInputValue('');
    reset();
  };
  return jsxs("div", {
    ref: floatingUiRefs === null || floatingUiRefs === void 0 ? void 0 : floatingUiRefs.setReference,
    css: getContainerStyles$1(),
    children: [jsx(Input, {
      ref: mergedRef,
      ...downshiftProps,
      onClick: handleClick,
      css: getInputStyles$1(theme),
      ...restProps
    }), jsx(TypeaheadComboboxControls, {
      getDownshiftToggleButtonProps: getToggleButtonProps,
      showClearSelectionButton: allowClear && Boolean(inputValue),
      handleClear: handleClear,
      disabled: restProps.disabled
    })]
  });
});

function _EMOTION_STRINGIFIED_CSS_ERROR__$1() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const getSelectedItemStyles = theme => {
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
    maxWidth: '100%'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getSelectedItemStyles;");
};
const getIconContainerStyles = theme => {
  return /*#__PURE__*/css({
    width: 16,
    height: 16,
    ':hover': {
      color: theme.colors.actionTertiaryTextHover,
      backgroundColor: theme.colors.tagHover
    }
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
var _ref2$1 = process.env.NODE_ENV === "production" ? {
  name: "2bhlo8",
  styles: "margin-right:2px"
} : {
  name: "d61dng-TypeaheadComboboxSelectedItem",
  styles: "margin-right:2px;label:TypeaheadComboboxSelectedItem;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$1
};
const TypeaheadComboboxSelectedItem = /*#__PURE__*/forwardRef((_ref, ref) => {
  let {
    item,
    getSelectedItemProps,
    removeSelectedItem,
    ...restProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs("span", {
    ...getSelectedItemProps({
      selectedItem: item
    }),
    css: getSelectedItemStyles(theme),
    ref: ref,
    ...restProps,
    children: [jsx("span", {
      css: _ref2$1,
      children: item.title
    }), jsx("span", {
      css: getIconContainerStyles(theme),
      children: jsx(CloseIcon, {
        "aria-hidden": "false",
        onClick: e => {
          e.stopPropagation();
          removeSelectedItem(item);
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
    totalCount
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: [getSelectedItemStyles(theme), {
      paddingInlineEnd: theme.spacing.xs
    }, process.env.NODE_ENV === "production" ? "" : ";label:CountBadge;"],
    children: countStartAt ? `+${totalCount - countStartAt}` : totalCount
  });
};

function _EMOTION_STRINGIFIED_CSS_ERROR__() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const getContainerStyles = (theme, width, maxHeight) => {
  return /*#__PURE__*/css({
    cursor: 'text',
    display: 'inline-block',
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
    '&:disabled': {
      backgroundColor: theme.colors.actionDisabledBackground,
      color: theme.colors.actionDisabledText
    }
  }, process.env.NODE_ENV === "production" ? "" : ";label:getContainerStyles;");
};
var _ref2 = process.env.NODE_ENV === "production" ? {
  name: "a9xlat",
  styles: "display:flex;flex:auto;flex-wrap:wrap;max-width:100%;position:relative"
} : {
  name: "azzs2i-getContentWrapperStyles",
  styles: "display:flex;flex:auto;flex-wrap:wrap;max-width:100%;position:relative;label:getContentWrapperStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__
};
const getContentWrapperStyles = () => {
  return _ref2;
};
var _ref = process.env.NODE_ENV === "production" ? {
  name: "qngrd1",
  styles: "display:inline-flex;position:relative;max-width:100%;align-self:auto;flex:none"
} : {
  name: "10x8s5t-getInputWrapperStyles",
  styles: "display:inline-flex;position:relative;max-width:100%;align-self:auto;flex:none;label:getInputWrapperStyles;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__
};
const getInputWrapperStyles = () => {
  return _ref;
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
var _ref4 = process.env.NODE_ENV === "production" ? {
  name: "1r88pt9",
  styles: "visibility:hidden;white-space:pre;position:absolute"
} : {
  name: "1noplic-content",
  styles: "visibility:hidden;white-space:pre;position:absolute;label:content;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__
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
    floatingUiRefs
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
    var _innerRef$current;
    (_innerRef$current = innerRef.current) === null || _innerRef$current === void 0 ? void 0 : _innerRef$current.focus();
    toggleMenu();
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
  }, [measureRef === null || measureRef === void 0 ? void 0 : (_measureRef$current = measureRef.current) === null || _measureRef$current === void 0 ? void 0 : _measureRef$current.scrollWidth, selectedItems === null || selectedItems === void 0 ? void 0 : selectedItems.length]);

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
    setShowTooltip(shouldShowCountBadge || isPartiallyHidden);
  }, [shouldShowCountBadge, (_itemsRef$current = itemsRef.current) === null || _itemsRef$current === void 0 ? void 0 : _itemsRef$current.clientHeight, (_containerRef$current = containerRef.current) === null || _containerRef$current === void 0 ? void 0 : _containerRef$current.clientHeight]);
  const content = jsxs("div", {
    onClick: handleClick,
    ref: mergedContainerRef,
    css: getContainerStyles(theme, width, maxHeight),
    children: [jsxs("div", {
      ref: itemsRef,
      css: getContentWrapperStyles(),
      children: [selectedItemsToRender === null || selectedItemsToRender === void 0 ? void 0 : selectedItemsToRender.map((selectedItemForRender, index) => jsx(TypeaheadComboboxSelectedItem, {
        label: getSelectedItemLabel(selectedItemForRender),
        item: selectedItemForRender,
        getSelectedItemProps: getSelectedItemProps,
        removeSelectedItem: removeSelectedItem
      }, `selected-item-${index}`)), shouldShowCountBadge && jsx(CountBadge, {
        countStartAt: showTagAfterValueCount,
        totalCount: selectedItems.length,
        role: "status",
        "aria-label": "Selected options count"
      }), jsxs("div", {
        css: getInputWrapperStyles(),
        children: [jsx("input", {
          ...downshiftProps,
          ref: mergedInputRef,
          css: [getInputStyles(theme), {
            width: inputWidth
          }, process.env.NODE_ENV === "production" ? "" : ";label:content;"],
          ...restProps
        }), jsxs("span", {
          ref: measureRef,
          "aria-hidden": true,
          css: _ref4,
          children: [selectedItems !== null && selectedItems !== void 0 && selectedItems.length ? (_innerRef$current2 = innerRef.current) === null || _innerRef$current2 === void 0 ? void 0 : _innerRef$current2.value : restProps.placeholder, "\xA0"]
        })]
      })]
    }), jsx(TypeaheadComboboxControls, {
      getDownshiftToggleButtonProps: getToggleButtonProps,
      showClearSelectionButton: allowClear && (Boolean(inputValue) || selectedItems && selectedItems.length > 0),
      handleClear: handleClear,
      disabled: restProps.disabled
    })]
  });
  if (showTooltip && selectedItems.length > 0) {
    return jsx(Tooltip, {
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

export { AccessibleContainer, Accordion, AccordionPanel, Alert, AlignCenterIcon, AlignLeftIcon, AlignRightIcon, AppIcon, ApplyDesignSystemContextOverrides, ApplyGlobalStyles, ArrowDownIcon, ArrowLeftIcon, ArrowRightIcon, ArrowUpIcon, ArrowsUpDownIcon, AutoComplete, BarChartIcon, BarGroupedIcon, BarStackedIcon, BeakerIcon, BinaryIcon, BoldIcon, BookIcon, BookmarkFillIcon, BookmarkIcon, BooksIcon, BracketsCurlyIcon, BracketsSquareIcon, BracketsXIcon, BranchIcon, Breadcrumb, BriefcaseFillIcon, BriefcaseIcon, Button, CalendarClockIcon, CalendarEventIcon, CalendarIcon, CaretDownSquareIcon, CaretUpSquareIcon, CatalogIcon, CatalogOffIcon, ChartLineIcon, CheckCircleBadgeIcon, CheckCircleFillIcon, CheckCircleIcon, CheckIcon, CheckLineIcon, Checkbox, CheckboxIcon, ChecklistIcon, ChevronDoubleDownIcon, ChevronDoubleLeftIcon, ChevronDoubleRightIcon, ChevronDoubleUpIcon, ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon, ChevronUpIcon, CircleIcon, ClipboardIcon, ClockIcon, ClockKeyIcon, CloseIcon, CloudDownloadIcon, CloudIcon, CloudKeyIcon, CloudModelIcon, CloudOffIcon, CloudUploadIcon, CodeIcon, Col, ColorFillIcon, ColumnIcon, ColumnsIcon, ConnectIcon, Content, CopyIcon, CursorPagination, CursorTypeIcon, DIcon, DU_BOIS_ENABLE_ANIMATION_CLASSNAME, DagIcon, DangerFillIcon, DangerIcon, DangerModal, DashIcon, DashboardIcon, DataIcon, DatabaseIcon, DecimalIcon, DesignSystemAntDConfigProvider, DialogCombobox, DialogComboboxButtonContainer, DialogComboboxContent, DialogComboboxCountBadge, EmptyResults as DialogComboboxEmpty, DialogComboboxFooter, DialogComboboxHintRow, DialogComboboxOptionControlledList, DialogComboboxOptionList, DialogComboboxOptionListCheckboxItem, DialogComboboxOptionListSearch, DialogComboboxOptionListSelectItem, DialogComboboxSectionHeader, DialogComboboxSeparator, DialogComboboxTrigger, DotsCircleIcon, DownloadIcon, DragIcon, Drawer, Dropdown, DuboisDatePicker, Empty, ExpandLessIcon, ExpandMoreIcon, FileCodeIcon, FileDocumentIcon, FileIcon, FileImageIcon, FileModelIcon, FilterIcon, FloatIcon, FolderBranchIcon, FolderCloudFilledIcon, FolderCloudIcon, FolderFillIcon, FolderIcon, FontIcon, ForkIcon, Form, FormDubois, FormUI, FullscreenExitIcon, FullscreenIcon, FunctionIcon, GearFillIcon, GearIcon, GenericSkeleton, GiftIcon, GitCommitIcon, GlobeIcon, GridDashIcon, GridIcon, H1Icon, H2Icon, H3Icon, Header$1 as Header, HistoryIcon, HomeIcon, Icon, ImageIcon, IndentDecreaseIcon, IndentIncreaseIcon, InfinityIcon, InfoFillIcon, InfoIcon, InfoTooltip, IngestionIcon, Input, ItalicIcon, KeyIcon, KeyboardIcon, LayerIcon, Layout, LegacyDatePicker, LegacyPopover, LegacySkeleton, LegacyTable, LettersIcon, LibrariesIcon, LightningIcon, LinkIcon, LinkOffIcon, ListBorderIcon, ListIcon, LoadingIcon, LoadingState, LockFillIcon, LockIcon, LockUnlockedIcon, MIcon, Menu, MenuIcon, MinusBoxIcon, MinusCircleFillIcon, MinusCircleIcon, Modal, ModelsIcon, Nav, NavButton, NoIcon, NotebookIcon, Notification, NotificationIcon, NotificationOffIcon, NumbersIcon, OfficeIcon, OptGroup, Option, OverflowIcon, PageBottomIcon, PageFirstIcon, PageLastIcon, PageTopIcon, PageWrapper, Pagination, Panel, PanelBody, PanelHeader, PanelHeaderButtons, PanelHeaderTitle, ParagraphSkeleton, PencilIcon, PinCancelIcon, PinFillIcon, PinIcon, PipelineIcon, PlayCircleFillIcon, PlayCircleIcon, PlayIcon, PlugIcon, PlusCircleFillIcon, PlusCircleIcon, PlusIcon, PlusSquareIcon, Popover, QueryEditorIcon, QueryIcon, QuestionMarkFillIcon, QuestionMarkIcon, QuestionMarkSpeechBubbleIcon, RHFControlledComponents, ROW_GUTTER_SIZE, Radio, ReaderModeIcon, RedoIcon, RefreshIcon, RestoreAntDDefaultClsPrefix, RobotIcon, Row, SaveIcon, SchoolIcon, SearchIcon, SecurityIcon, SegmentedControlButton, SegmentedControlGroup, Select, SelectOptGroup, SelectOption, SelectV2, SelectV2Content, SelectV2Option, SelectV2OptionGroup, SelectV2Trigger, ShareIcon, Sidebar, SidebarAutoIcon, SidebarCollapseIcon, SidebarExpandIcon, SidebarIcon, SlidersIcon, SortAlphabeticalAscendingIcon, SortAlphabeticalDescendingIcon, SortAlphabeticalLeftIcon, SortAlphabeticalRightIcon, SortAscendingIcon, SortDescendingIcon, SortUnsortedIcon, Space, Spacer, SpeechBubbleIcon, SpeechBubblePlusIcon, Spinner, SplitButton, StarFillIcon, StarIcon, StopCircleFillIcon, StopCircleIcon, StopIcon, StorefrontIcon, StreamIcon, Switch, SyncIcon, TabPane, Table, TableCell, TableContext, TableFilterInput, TableFilterLayout, TableHeader, TableIcon, TableInfinityIcon, TableLightningIcon, TableRow, TableRowAction, TableRowContext, TableRowMenuContainer, TableRowSelectCell, TableSkeleton, TableSkeletonRows, TableViewIcon, Tabs, Tag, TextBoxIcon, ThumbsDownIcon, ThumbsUpIcon, TitleSkeleton, ToggleButton, Tooltip, TrashIcon, Tree, TreeIcon, TypeaheadComboboxCheckboxItem, TypeaheadComboboxFooter, TypeaheadComboboxInput, TypeaheadComboboxMenu, TypeaheadComboboxMenuItem, TypeaheadComboboxMultiSelectInput, TypeaheadComboboxRoot, TypeaheadComboboxSectionHeader, TypeaheadComboboxSelectedItem, TypeaheadComboboxSeparator, TypeaheadComboboxToggleButton, Typography, UnderlineIcon, UndoIcon, UploadIcon, UsbIcon, UserBadgeIcon, UserCircleIcon, UserGroupIcon, UserIcon, VisibleIcon, VisibleOffIcon, WarningFillIcon, WarningIcon, WorkflowsIcon, WorkspacesIcon, XCircleFillIcon, XCircleIcon, ZoomInIcon, ZoomOutIcon, __INTERNAL_DO_NOT_USE__FormItem, __INTERNAL_DO_NOT_USE__Group, __INTERNAL_DO_NOT_USE__HorizontalGroup, __INTERNAL_DO_NOT_USE__VerticalGroup, dialogComboboxLookAheadKeyDown, findClosestOptionSibling, findHighlightedOption, getAnimationCss, getContentOptions, getDialogComboboxOptionLabelWidth, getGlobalStyles, getKeyboardNavigationFunctions, getPaginationEmotionStyles, getRadioStyles, getTabEmotionStyles, getValidationStateColor, getWrapperStyle, highlightOption, importantify, resetTabIndexToFocusedElement, useComboboxState, useDesignSystemFlags, useDesignSystemTheme, useLegacyNotification, useMultipleSelectionState, useThemedStyles, useTypeaheadComboboxContext, visuallyHidden, withNotifications };
//# sourceMappingURL=index.js.map

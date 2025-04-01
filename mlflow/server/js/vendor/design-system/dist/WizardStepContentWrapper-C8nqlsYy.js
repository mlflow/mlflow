import _isUndefined from 'lodash/isUndefined';
import React__default, { forwardRef, useContext, useMemo, createContext, useLayoutEffect, useEffect, useState, useRef, useCallback } from 'react';
import { css, Global, keyframes, createElement } from '@emotion/react';
import { I as Icon, t as DesignSystemEventSuppressInteractionProviderContext, v as DesignSystemEventSuppressInteractionTrueContextValue, d as DesignSystemEventProviderAnalyticsEventTypes, u as useDesignSystemTheme, a as useDesignSystemSafexFlags, f as useDesignSystemEventComponentCallbacks, h as DesignSystemEventProviderComponentTypes, l as useNotifyOnFirstView, w as safex, x as augmentWithDataComponentProps, D as DesignSystemAntDConfigProvider, b as addDebugOutlineIfEnabled, y as RestoreAntDDefaultClsPrefix, C as CloseIcon, j as DangerIcon, B as Button, z as addDebugOutlineStylesIfEnabled, A as getDarkModePortalStyles, E as getShadowScrollStyles, c as getAnimationCss, i as importantify, T as Typography, q as ChevronRightIcon, o as ChevronLeftIcon, G as useDesignSystemContext, S as Stepper, R as Root$1, m as Trigger$1, n as Content$2 } from './Stepper-4hx5KkUu.js';
import { jsx, jsxs, Fragment } from '@emotion/react/jsx-runtime';
import _noop from 'lodash/noop';
import { ResizableBox } from 'react-resizable';
import _pick from 'lodash/pick';
import _compact from 'lodash/compact';
import * as RadixTooltip from '@radix-ui/react-tooltip';
import { Modal as Modal$1 } from 'antd';

function SvgInfoIcon(props) {
  return jsxs("svg", {
    xmlns: "http://www.w3.org/2000/svg",
    width: "1em",
    height: "1em",
    fill: "none",
    viewBox: "0 0 16 16",
    ...props,
    children: [jsx("path", {
      fill: "currentColor",
      d: "M7.25 10.5v-3h1.5v3zM8 5a.75.75 0 1 1 0 1.5A.75.75 0 0 1 8 5"
    }), jsx("path", {
      fill: "currentColor",
      fillRule: "evenodd",
      d: "M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12m0-1.5a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9",
      clipRule: "evenodd"
    })]
  });
}
const InfoIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgInfoIcon
  });
});
InfoIcon.displayName = 'InfoIcon';

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
      d: "M1.5 2.75a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0M3 2h13v1.5H3zM3 5.5h13V7H3zM3 9h13v1.5H3zM3 12.5h13V14H3zM.75 7a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5M1.5 13.25a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0M.75 10.5a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5"
    })
  });
}
const ListIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgListIcon
  });
});
ListIcon.displayName = 'ListIcon';

function _EMOTION_STRINGIFIED_CSS_ERROR__$4() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
const ModalContext = /*#__PURE__*/createContext({
  isInsideModal: true
});
const useModalContext = () => useContext(ModalContext);
const SIZE_PRESETS = {
  normal: 640,
  wide: 880
};
const getModalEmotionStyles = args => {
  const {
    theme,
    clsPrefix,
    hasFooter = true,
    maxedOutHeight,
    useNewShadows,
    useNewBorderRadii
  } = args;
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
  return /*#__PURE__*/css({
    '&&': {
      ...addDebugOutlineStylesIfEnabled(theme)
    },
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
      [`${classNameButton} + ${classNameButton}`]: {
        marginLeft: theme.spacing.sm
      },
      // Needed to override AntD style for the SplitButton's dropdown button back to its original value
      [`${classNameDropdownTrigger} > ${classNameButton}:nth-of-type(2)`]: {
        marginLeft: -1
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
      borderRadius: useNewBorderRadii ? theme.borders.borderRadiusSm : theme.legacyBorders.borderRadiusMd,
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
        outlineColor: theme.colors.actionDefaultBorderFocus
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
      boxShadow: useNewShadows ? theme.shadows.xl : theme.general.shadowHigh,
      ...(useNewBorderRadii && {
        borderRadius: theme.borders.borderRadiusLg
      }),
      ...getDarkModePortalStyles(theme, useNewShadows)
    },
    [classNameBody]: {
      overflowY: 'auto',
      maxHeight: bodyMaxHeight,
      paddingLeft: MODAL_PADDING,
      paddingRight: MODAL_PADDING,
      paddingTop: CONTENT_BUFFER,
      paddingBottom: CONTENT_BUFFER,
      ...getShadowScrollStyles(theme)
    },
    ...getAnimationCss(theme.options.enableAnimation)
  }, process.env.NODE_ENV === "production" ? "" : ";label:getModalEmotionStyles;");
};
function closeButtonComponentId(componentId) {
  return componentId ? `${componentId}.footer.cancel` : 'codegen_design-system_src_design-system_modal_modal.tsx_260';
}

/**
 * Render default footer with our buttons. Copied from AntD.
 */
function DefaultFooter(_ref) {
  let {
    componentId,
    onOk,
    onCancel,
    confirmLoading,
    okText,
    cancelText,
    okButtonProps,
    cancelButtonProps,
    autoFocusButton,
    shouldStartInteraction
  } = _ref;
  const handleCancel = e => {
    onCancel === null || onCancel === void 0 || onCancel(e);
  };
  const handleOk = e => {
    onOk === null || onOk === void 0 || onOk(e);
  };
  return jsxs(Fragment, {
    children: [cancelText && jsx(Button, {
      componentId: closeButtonComponentId(componentId),
      onClick: handleCancel,
      autoFocus: autoFocusButton === 'cancel',
      dangerouslyUseFocusPseudoClass: true,
      shouldStartInteraction: shouldStartInteraction,
      ...cancelButtonProps,
      children: cancelText
    }), okText && jsx(Button, {
      componentId: componentId ? `${componentId}.footer.ok` : 'codegen_design-system_src_design-system_modal_modal.tsx_271',
      loading: confirmLoading,
      onClick: handleOk,
      type: "primary",
      autoFocus: autoFocusButton === 'ok',
      dangerouslyUseFocusPseudoClass: true,
      shouldStartInteraction: shouldStartInteraction,
      ...okButtonProps,
      children: okText
    })]
  });
}
function Modal(props) {
  return jsx(DesignSystemEventSuppressInteractionProviderContext.Provider, {
    value: DesignSystemEventSuppressInteractionTrueContextValue,
    children: jsx(ModalInternal, {
      ...props
    })
  });
}
function ModalInternal(_ref2) {
  let {
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView],
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
    shouldStartInteraction,
    ...props
  } = _ref2;
  const {
    theme,
    classNamePrefix
  } = useDesignSystemTheme();
  const {
    useNewShadows,
    useNewBorderRadii
  } = useDesignSystemSafexFlags();
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Modal,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    shouldStartInteraction
  });
  const {
    elementRef
  } = useNotifyOnFirstView({
    onView: eventContext.onView
  });

  // Need to simulate the close button being closed if the user clicks outside of the modal or clicks on the dismiss button
  // This should only be applied to the modal prop and not to the footer component
  const closeButtonEventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId: closeButtonComponentId(componentId),
    analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
    shouldStartInteraction
  });
  const onCancelWrapper = e => {
    var _props$onCancel;
    closeButtonEventContext.onClick(e);
    (_props$onCancel = props.onCancel) === null || _props$onCancel === void 0 || _props$onCancel.call(props, e);
  };

  // add data-component-* props to make modal discoverable by go/component-finder
  const augmentedChildren = safex('databricks.fe.observability.enableModalDataComponentProps', false) ? augmentWithDataComponentProps(children, eventContext.dataComponentProps) : children;
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Modal$1, {
      ...addDebugOutlineIfEnabled(),
      css: getModalEmotionStyles({
        theme,
        clsPrefix: classNamePrefix,
        hasFooter: footer !== null,
        maxedOutHeight: verticalSizing === 'maxed_out',
        useNewShadows,
        useNewBorderRadii
      }),
      title: jsx(RestoreAntDDefaultClsPrefix, {
        children: truncateTitle ? jsx("div", {
          css: /*#__PURE__*/css({
            textOverflow: 'ellipsis',
            marginRight: theme.spacing.md,
            overflow: 'hidden',
            whiteSpace: 'nowrap'
          }, process.env.NODE_ENV === "production" ? "" : ";label:ModalInternal;"),
          title: typeof title === 'string' ? title : undefined,
          children: title
        }) : title
      }),
      footer: footer === null ? null : jsx(RestoreAntDDefaultClsPrefix, {
        children: footer === undefined ? jsx(DefaultFooter, {
          componentId: componentId,
          onOk: props.onOk,
          onCancel: props.onCancel,
          confirmLoading: props.confirmLoading,
          okText: props.okText,
          cancelText: props.cancelText,
          okButtonProps: okButtonProps,
          cancelButtonProps: cancelButtonProps,
          autoFocusButton: autoFocusButton,
          shouldStartInteraction: shouldStartInteraction
        }) : footer
      }),
      width: size ? SIZE_PRESETS[size] : undefined,
      closeIcon: jsx(CloseIcon, {
        ref: elementRef
      }),
      centered: true,
      zIndex: theme.options.zIndexBase,
      maskStyle: {
        backgroundColor: theme.colors.overlayOverlay
      },
      ...props,
      onCancel: onCancelWrapper,
      ...dangerouslySetAntdProps,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: jsx(ModalContext.Provider, {
          value: {
            isInsideModal: true
          },
          children: augmentedChildren
        })
      })
    })
  });
}
var _ref3$1 = process.env.NODE_ENV === "production" ? {
  name: "b9hrb",
  styles: "position:relative;display:inline-flex;align-items:center"
} : {
  name: "1jkwrsj-titleComp",
  styles: "position:relative;display:inline-flex;align-items:center;label:titleComp;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
};
var _ref4 = process.env.NODE_ENV === "production" ? {
  name: "1o6wc9k",
  styles: "padding-left:6px"
} : {
  name: "i303lp-titleComp",
  styles: "padding-left:6px;label:titleComp;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$4
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
    css: _ref3$1,
    children: [jsx(DangerIcon, {
      css: /*#__PURE__*/css({
        color: theme.colors.textValidationDanger,
        left: 2,
        height: iconSize,
        width: iconSize,
        fontSize: iconFontSize
      }, process.env.NODE_ENV === "production" ? "" : ";label:titleComp;")
    }), jsx("div", {
      css: _ref4,
      children: title
    })]
  });
  return jsx(Modal, {
    shouldStartInteraction: props.shouldStartInteraction,
    title: titleComp,
    footer: [jsx(Button, {
      componentId: props.componentId ? `${props.componentId}.danger.footer.cancel` : 'codegen_design-system_src_design-system_modal_modal.tsx_386',
      onClick: onCancel,
      shouldStartInteraction: props.shouldStartInteraction,
      ...cancelButtonProps,
      children: cancelText || 'Cancel'
    }, "cancel"), jsx(Button, {
      componentId: props.componentId ? `${props.componentId}.danger.footer.ok` : 'codegen_design-system_src_design-system_modal_modal.tsx_395',
      type: "primary",
      danger: true,
      onClick: onOk,
      loading: props.confirmLoading,
      shouldStartInteraction: props.shouldStartInteraction,
      ...okButtonProps,
      children: okText || 'Delete'
    }, "discard")],
    onOk: onOk,
    onCancel: onCancel,
    ...restProps
  });
}

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

/**
 * Copied from usehooks-ts.
 * Custom hook for using either `useLayoutEffect` or `useEffect` based on the environment (client-side or server-side).
 *
 * [Documentation](https://usehooks-ts.com/react-hook/use-isomorphic-layout-effect)
 *
 * Example:
 * ```
 * useIsomorphicLayoutEffect(() => {
 * // Code to be executed during the layout phase on the client side
 * }, [dependency1, dependency2]);
 * ```
 */
const useIsomorphicLayoutEffect = typeof window !== 'undefined' ? useLayoutEffect : useEffect;

const IS_SERVER = typeof window === 'undefined';
/**
 * Copied from usehooks-ts.
 * Custom hook for tracking the state of a media query. Returns The current state of the media query (true if the query matches, false otherwise).
 *
 * [Documentation](https://usehooks-ts.com/react-hook/use-media-query)
 *
 * [MDN Match Media](https://developer.mozilla.org/en-US/docs/Web/API/Window/matchMedia)
 *
 * Example:
 *
 * `const isSmallScreen = useMediaQuery('(max-width: 600px)');`
 *
 * Use `isSmallScreen` to conditionally apply styles or logic based on the screen size.
 */
function useMediaQuery(_ref) {
  var _options$defaultValue, _options$initializeWi;
  let {
    query,
    options
  } = _ref;
  // TODO: Refactor this code after the deprecated signature has been removed.
  const defaultValue = typeof options === 'boolean' ? options : (_options$defaultValue = options === null || options === void 0 ? void 0 : options.defaultValue) !== null && _options$defaultValue !== void 0 ? _options$defaultValue : false;
  const initializeWithValue = typeof options === 'boolean' ? undefined : (_options$initializeWi = options === null || options === void 0 ? void 0 : options.initializeWithValue) !== null && _options$initializeWi !== void 0 ? _options$initializeWi : undefined;
  const [matches, setMatches] = useState(() => {
    if (initializeWithValue) {
      return getMatches(query);
    }
    return defaultValue;
  });
  const getMatches = query => {
    if (IS_SERVER) {
      return defaultValue;
    }
    return window.matchMedia(query).matches;
  };

  /** Handles the change event of the media query. */
  function handleChange() {
    setMatches(getMatches(query));
  }
  useIsomorphicLayoutEffect(() => {
    const matchMedia = window.matchMedia(query);

    // Triggered at the first client-side load and if query changes
    handleChange();

    // Use deprecated `addListener` and `removeListener` to support Safari < 14 (#135)
    if (matchMedia.addListener) {
      matchMedia.addListener(handleChange);
    } else {
      matchMedia.addEventListener('change', handleChange);
    }
    return () => {
      if (matchMedia.removeListener) {
        matchMedia.removeListener(handleChange);
      } else {
        matchMedia.removeEventListener('change', handleChange);
      }
    };
  }, [query]);
  return matches;
}

function _EMOTION_STRINGIFIED_CSS_ERROR__$3() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
/**
 * `ResizableBox` passes `handleAxis` to the element used as handle. We need to wrap the handle to prevent
 * `handleAxis becoming an attribute on the div element.
 */
const ResizablePanelHandle = /*#__PURE__*/forwardRef(function ResizablePanelHandle(_ref2, ref) {
  let {
    handleAxis,
    children,
    ...otherProps
  } = _ref2;
  return jsx("div", {
    ref: ref,
    ...otherProps,
    children: children
  });
});
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
function Nav(_ref3) {
  let {
    children,
    dangerouslyAppendEmotionCSS
  } = _ref3;
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
const NavButton = /*#__PURE__*/React__default.forwardRef((_ref4, ref) => {
  let {
    active,
    disabled,
    icon,
    onClick,
    children,
    dangerouslyAppendEmotionCSS,
    'aria-label': ariaLabel,
    ...restProps
  } = _ref4;
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: [active ? importantify({
      borderRadius: theme.borders.borderRadiusSm,
      background: theme.colors.actionDefaultBackgroundPress,
      button: {
        '&:enabled:not(:hover):not(:active) > .anticon': {
          color: theme.colors.actionTertiaryTextPress
        }
      }
    }) : undefined, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:NavButton;"],
    children: jsx(Button, {
      ref: ref,
      icon: icon,
      onClick: onClick,
      disabled: disabled,
      "aria-label": ariaLabel,
      ...restProps,
      children: children
    })
  });
});
const TOGGLE_BUTTON_Z_INDEX = 100;
const COMPACT_CONTENT_Z_INDEX = 50;
const ToggleButton = _ref5 => {
  let {
    isExpanded,
    position,
    toggleIsExpanded,
    componentId
  } = _ref5;
  const {
    theme
  } = useDesignSystemTheme();
  const positionStyle = useMemo(() => {
    if (position === 'right') {
      return isExpanded ? {
        right: DEFAULT_WIDTH,
        transform: 'translateX(+50%)'
      } : {
        left: 0,
        transform: 'translateX(-50%)'
      };
    } else {
      return isExpanded ? {
        left: DEFAULT_WIDTH,
        transform: 'translateX(-50%)'
      } : {
        right: 0,
        transform: 'translateX(+50%)'
      };
    }
  }, [isExpanded, position]);
  const ToggleIcon = useMemo(() => {
    if (position === 'right') {
      return isExpanded ? ChevronRightIcon : ChevronLeftIcon;
    } else {
      return isExpanded ? ChevronLeftIcon : ChevronRightIcon;
    }
  }, [isExpanded, position]);
  return jsxs("div", {
    css: /*#__PURE__*/css({
      position: 'absolute',
      top: 0,
      height: 46,
      display: 'flex',
      alignItems: 'center',
      zIndex: TOGGLE_BUTTON_Z_INDEX,
      ...positionStyle
    }, process.env.NODE_ENV === "production" ? "" : ";label:ToggleButton;"),
    children: [jsx("div", {
      css: /*#__PURE__*/css({
        borderRadius: '100%',
        width: theme.spacing.lg,
        height: theme.spacing.lg,
        backgroundColor: theme.colors.backgroundPrimary,
        position: 'absolute'
      }, process.env.NODE_ENV === "production" ? "" : ";label:ToggleButton;")
    }), jsx(Button, {
      componentId: componentId,
      css: /*#__PURE__*/css({
        borderRadius: '100%',
        '&&': {
          padding: '0px !important',
          width: `${theme.spacing.lg}px !important`
        }
      }, process.env.NODE_ENV === "production" ? "" : ";label:ToggleButton;"),
      onClick: toggleIsExpanded,
      size: "small",
      "aria-label": isExpanded ? 'hide sidebar' : 'expand sidebar',
      "aria-expanded": isExpanded,
      children: jsx(ToggleIcon, {})
    })]
  });
};
const getContentAnimation = width => {
  const showAnimation = keyframes`
  from { opacity: 0 }
  80%  { opacity: 0 }
  to   { opacity: 1 }`;
  const openAnimation = keyframes`
  from { width: 50px }
  to   { width: ${width}px }`;
  return {
    open: `${openAnimation} .2s cubic-bezier(0, 0, 0.2, 1)`,
    show: `${showAnimation} .25s linear`
  };
};
var _ref = process.env.NODE_ENV === "production" ? {
  name: "1h0bf8r",
  styles: "body, :host{user-select:none;}"
} : {
  name: "15c18a5-Content",
  styles: "body, :host{user-select:none;};label:Content;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$3
};
function Content$1(_ref6) {
  let {
    disableResize,
    openPanelId,
    closable = true,
    onClose,
    onResizeStart,
    onResizeStop,
    width,
    minWidth,
    maxWidth,
    destroyInactivePanels = false,
    children,
    dangerouslyAppendEmotionCSS,
    enableCompact,
    resizeBoxStyle,
    noSideBorder,
    componentId
  } = _ref6;
  const {
    theme
  } = useDesignSystemTheme();
  const isCompact = useMediaQuery({
    query: `not (min-width: ${theme.responsive.breakpoints.sm}px)`
  }) && enableCompact;
  const defaultAnimation = useMemo(() => getContentAnimation(isCompact ? DEFAULT_WIDTH : width || DEFAULT_WIDTH), [isCompact, width]);
  // specifically for non closable panel in compact mode
  const [isExpanded, setIsExpanded] = useState(true);
  // hide the panel in compact mode when the panel is not closable and collapsed
  const isNotExpandedStyle = /*#__PURE__*/css(isCompact && !closable && !isExpanded && {
    display: 'none'
  }, process.env.NODE_ENV === "production" ? "" : ";label:isNotExpandedStyle;");
  const sidebarContext = useContext(SidebarContext);
  const onCloseRef = useRef(onClose);
  const resizeHandleStyle = sidebarContext.position === 'right' ? {
    left: 0
  } : {
    right: 0
  };
  const [dragging, setDragging] = useState(false);
  const isPanelClosed = openPanelId == null;
  const [animation, setAnimation] = useState(isPanelClosed ? defaultAnimation : undefined);
  const compactStyle = /*#__PURE__*/css(isCompact && {
    position: 'absolute',
    zIndex: COMPACT_CONTENT_Z_INDEX,
    left: sidebarContext.position === 'left' && closable ? '100%' : undefined,
    right: sidebarContext.position === 'right' && closable ? '100%' : undefined,
    backgroundColor: theme.colors.backgroundPrimary,
    width: DEFAULT_WIDTH,
    // shift to the top due to border
    top: -1
  }, process.env.NODE_ENV === "production" ? "" : ";label:compactStyle;");
  const hiddenPanelStyle = /*#__PURE__*/css(isPanelClosed && {
    display: 'none'
  }, process.env.NODE_ENV === "production" ? "" : ";label:hiddenPanelStyle;");
  const containerStyle = /*#__PURE__*/css({
    animation: animation === null || animation === void 0 ? void 0 : animation.open,
    direction: sidebarContext.position === 'right' ? 'rtl' : 'ltr',
    position: 'relative',
    borderWidth: sidebarContext.position === 'right' ? `0 ${noSideBorder ? 0 : theme.general.borderWidth}px 0 0 ` : `0 0 0 ${noSideBorder ? 0 : theme.general.borderWidth}px`,
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

  // For non closable panel, reset expanded state to true so that the panel stays open
  // the next time the screen goes into compact mode.
  useEffect(() => {
    if (!closable && enableCompact && !isCompact) {
      setIsExpanded(true);
    }
  }, [isCompact, closable, defaultAnimation, enableCompact]);
  const value = useMemo(() => ({
    openPanelId,
    closable,
    destroyInactivePanels,
    setIsClosed: () => {
      var _onCloseRef$current;
      (_onCloseRef$current = onCloseRef.current) === null || _onCloseRef$current === void 0 || _onCloseRef$current.call(onCloseRef);
      if (!animation) {
        setAnimation(defaultAnimation);
      }
    }
  }), [openPanelId, closable, defaultAnimation, animation, destroyInactivePanels]);
  return jsx(ContentContext.Provider, {
    value: value,
    children: disableResize || isCompact ? jsxs(Fragment, {
      children: [jsx("div", {
        css: [/*#__PURE__*/css({
          width: width || '100%',
          height: '100%',
          overflow: 'hidden'
        }, containerStyle, compactStyle, process.env.NODE_ENV === "production" ? "" : ";label:Content;"), dangerouslyAppendEmotionCSS, hiddenPanelStyle, isNotExpandedStyle, process.env.NODE_ENV === "production" ? "" : ";label:Content;"],
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
      }), !closable && isCompact && jsx("div", {
        css: /*#__PURE__*/css({
          width: !isExpanded ? theme.spacing.md : undefined,
          marginRight: isExpanded ? theme.spacing.md : undefined,
          position: 'relative'
        }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"),
        children: jsx(ToggleButton, {
          componentId: componentId ? `${componentId}.toggle` : 'sidebar-toggle',
          isExpanded: isExpanded,
          position: sidebarContext.position || 'left',
          toggleIsExpanded: () => setIsExpanded(prev => !prev)
        })
      })]
    }) : jsxs(Fragment, {
      children: [dragging && jsx(Global, {
        styles: _ref
      }), jsx(ResizableBox, {
        style: resizeBoxStyle,
        width: width || DEFAULT_WIDTH,
        height: undefined,
        axis: "x",
        resizeHandles: sidebarContext.position === 'right' ? ['w'] : ['e'],
        minConstraints: [minWidth !== null && minWidth !== void 0 ? minWidth : DEFAULT_WIDTH, 150],
        maxConstraints: [maxWidth !== null && maxWidth !== void 0 ? maxWidth : 800, 150],
        onResizeStart: (_, _ref7) => {
          let {
            size
          } = _ref7;
          onResizeStart === null || onResizeStart === void 0 || onResizeStart(size.width);
          setDragging(true);
        },
        onResizeStop: (_, _ref8) => {
          let {
            size
          } = _ref8;
          onResizeStop === null || onResizeStop === void 0 || onResizeStop(size.width);
          setDragging(false);
        },
        handle: jsx(ResizablePanelHandle, {
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
      })]
    })
  });
}
function Panel(_ref9) {
  let {
    panelId,
    children,
    forceRender = false,
    dangerouslyAppendEmotionCSS,
    ...delegated
  } = _ref9;
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
var _ref11 = process.env.NODE_ENV === "production" ? {
  name: "1sn73n9",
  styles: "display:flex;justify-content:space-between;align-items:center;&&{margin:0;}"
} : {
  name: "z5j8kg-PanelHeader",
  styles: "display:flex;justify-content:space-between;align-items:center;&&{margin:0;};label:PanelHeader;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$3
};
function PanelHeader(_ref10) {
  let {
    children,
    dangerouslyAppendEmotionCSS,
    componentId
  } = _ref10;
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
      children: jsx(Typography.Title, {
        level: 4,
        css: _ref11,
        children: children
      })
    }), contentContext.closable ? jsx("div", {
      children: jsx(Button, {
        componentId: componentId ? `${componentId}.close` : 'codegen_design-system_src_design-system_sidebar_sidebar.tsx_427',
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
function PanelHeaderTitle(_ref12) {
  let {
    title,
    dangerouslyAppendEmotionCSS
  } = _ref12;
  return jsx("div", {
    title: title,
    css: ["align-self:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;", dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:PanelHeaderTitle;"],
    children: title
  });
}
function PanelHeaderButtons(_ref13) {
  let {
    children,
    dangerouslyAppendEmotionCSS
  } = _ref13;
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
function PanelBody(_ref14) {
  let {
    children,
    dangerouslyAppendEmotionCSS
  } = _ref14;
  const {
    theme
  } = useDesignSystemTheme();
  const [shouldBeFocusable, setShouldBeFocusable] = useState(false);
  const bodyRef = useRef(null);
  useEffect(() => {
    const ref = bodyRef.current;
    if (ref) {
      if (ref.scrollHeight > ref.clientHeight) {
        setShouldBeFocusable(true);
      } else {
        setShouldBeFocusable(false);
      }
    }
  }, []);
  return jsx("div", {
    ref: bodyRef
    // Needed to make panel body content focusable when scrollable for keyboard-only users to be able to focus & scroll
    // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
    ,
    tabIndex: shouldBeFocusable ? 0 : -1,
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
  function Sidebar(_ref15) {
    let {
      position,
      children,
      dangerouslyAppendEmotionCSS,
      ...dataProps
    } = _ref15;
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
        ...addDebugOutlineIfEnabled(),
        ...dataProps,
        css: [{
          display: 'flex',
          height: '100%',
          backgroundColor: theme.colors.backgroundPrimary,
          flexDirection: position === 'right' ? 'row-reverse' : 'row',
          borderStyle: 'solid',
          borderColor: theme.colors.borderDecorative,
          borderWidth: position === 'right' ? `0 0 0 ${theme.general.borderWidth}px` : `0px ${theme.general.borderWidth}px 0 0`,
          boxSizing: 'content-box',
          position: 'relative'
        }, dangerouslyAppendEmotionCSS, process.env.NODE_ENV === "production" ? "" : ";label:Sidebar;"],
        children: children
      })
    });
  }
  Sidebar.Content = Content$1;
  Sidebar.Nav = Nav;
  Sidebar.NavButton = NavButton;
  Sidebar.Panel = Panel;
  Sidebar.PanelHeader = PanelHeader;
  Sidebar.PanelHeaderTitle = PanelHeaderTitle;
  Sidebar.PanelHeaderButtons = PanelHeaderButtons;
  Sidebar.PanelBody = PanelBody;
  return Sidebar;
})();

const isInsideTest = () => typeof jest !== 'undefined';

const useTooltipStyles = _ref => {
  let {
    maxWidth
  } = _ref;
  const {
    theme,
    classNamePrefix: clsPrefix
  } = useDesignSystemTheme();
  const {
    useNewShadows
  } = useDesignSystemSafexFlags();
  const classTypography = `.${clsPrefix}-typography`;
  const {
    isDarkMode
  } = theme;
  const slideUpAndFade = keyframes({
    from: {
      opacity: 0,
      transform: 'translateY(2px)'
    },
    to: {
      opacity: 1,
      transform: 'translateY(0)'
    }
  });
  const slideRightAndFade = keyframes({
    from: {
      opacity: 0,
      transform: 'translateX(-2px)'
    },
    to: {
      opacity: 1,
      transform: 'translateX(0)'
    }
  });
  const slideDownAndFade = keyframes({
    from: {
      opacity: 0,
      transform: 'translateY(-2px)'
    },
    to: {
      opacity: 1,
      transform: 'translateY(0)'
    }
  });
  const slideLeftAndFade = keyframes({
    from: {
      opacity: 0,
      transform: 'translateX(2px)'
    },
    to: {
      opacity: 1,
      transform: 'translateX(0)'
    }
  });
  const linkColor = isDarkMode ? theme.colors.blue600 : theme.colors.blue500;
  const linkActiveColor = isDarkMode ? theme.colors.blue800 : theme.colors.blue300;
  const linkHoverColor = isDarkMode ? theme.colors.blue700 : theme.colors.blue400;
  return {
    content: {
      backgroundColor: theme.colors.tooltipBackgroundTooltip,
      color: theme.colors.actionPrimaryTextDefault,
      borderRadius: theme.borders.borderRadiusSm,
      fontSize: theme.typography.fontSizeMd,
      padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
      lineHeight: theme.typography.lineHeightLg,
      fontWeight: theme.typography.typographyRegularFontWeight,
      boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowHigh,
      maxWidth: maxWidth,
      wordWrap: 'break-word',
      whiteSpace: 'normal',
      zIndex: theme.options.zIndexBase + 70,
      willChange: 'transform, opacity',
      "&[data-state='delayed-open'][data-side='top']": {
        animation: `${slideDownAndFade} 400ms cubic-bezier(0.16, 1, 0.3, 1)`
      },
      "&[data-state='delayed-open'][data-side='right']": {
        animation: `${slideLeftAndFade} 400ms cubic-bezier(0.16, 1, 0.3, 1)`
      },
      "&[data-state='delayed-open'][data-side='bottom']": {
        animation: `${slideUpAndFade} 400ms cubic-bezier(0.16, 1, 0.3, 1)`
      },
      "&[data-state='delayed-open'][data-side='left']": {
        animation: `${slideRightAndFade} 400ms cubic-bezier(0.16, 1, 0.3, 1)`
      },
      [`& a${classTypography}`]: {
        '&, :focus': {
          color: linkColor,
          '.anticon': {
            color: linkColor
          }
        },
        ':active': {
          color: linkActiveColor,
          '.anticon': {
            color: linkActiveColor
          }
        },
        ':hover': {
          color: linkHoverColor,
          '.anticon': {
            color: linkHoverColor
          }
        }
      }
    },
    arrow: {
      fill: theme.colors.tooltipBackgroundTooltip,
      zIndex: theme.options.zIndexBase + 70,
      visibility: 'visible'
    }
  };
};

/**
 * If the tooltip is not displaying for you, it might be because the child does not accept the onMouseEnter, onMouseLeave, onPointerEnter,
 * onPointerLeave, onFocus, and onClick props. You can add these props to your child component, or wrap it in a `<span>` tag.
 *
 * See go/dubois.
 */
const Tooltip = _ref2 => {
  let {
    children,
    content,
    defaultOpen = false,
    delayDuration = 350,
    side = 'top',
    sideOffset = 4,
    align = 'center',
    maxWidth = 250,
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnView],
    zIndex,
    ...props
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    getPopupContainer
  } = useDesignSystemContext();
  const tooltipStyles = useTooltipStyles({
    maxWidth
  });
  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Tooltip,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents
  });
  const firstView = useRef(true);
  const handleOpenChange = useCallback(open => {
    if (open && firstView.current) {
      eventContext.onView();
      firstView.current = false;
    }
  }, [eventContext, firstView]);
  const IS_INSIDE_TEST = isInsideTest();
  return jsxs(RadixTooltip.Root, {
    defaultOpen: defaultOpen,
    delayDuration: IS_INSIDE_TEST ? 10 : delayDuration,
    onOpenChange: handleOpenChange,
    children: [jsx(RadixTooltip.Trigger, {
      asChild: true,
      children: children
    }), content ? jsx(RadixTooltip.Portal, {
      container: getPopupContainer && getPopupContainer(),
      children: jsxs(RadixTooltip.Content, {
        side: side,
        align: align,
        sideOffset: theme.spacing.sm,
        arrowPadding: theme.spacing.md,
        css: [tooltipStyles['content'], zIndex ? {
          zIndex
        } : undefined, process.env.NODE_ENV === "production" ? "" : ";label:Tooltip;"],
        ...props,
        ...eventContext.dataComponentProps,
        children: [content, jsx(RadixTooltip.Arrow, {
          css: tooltipStyles['arrow']
        })]
      })
    }) : null]
  });
};

function useStepperStepsFromWizardSteps(wizardSteps, currentStepIdx, hideDescriptionForFutureSteps) {
  return useMemo(() => wizardSteps.map((wizardStep, stepIdx) => ({
    ..._pick(wizardStep, ['title', 'status']),
    description: hideDescriptionForFutureSteps && !(stepIdx <= currentStepIdx || wizardStep.status === 'completed' || wizardStep.status === 'error' || wizardStep.status === 'warning') ? undefined : wizardStep.description,
    additionalVerticalContent: wizardStep.additionalHorizontalLayoutStepContent,
    clickEnabled: _isUndefined(wizardStep.clickEnabled) ? isWizardStepEnabled(wizardSteps, stepIdx, currentStepIdx, wizardStep.status) : wizardStep.clickEnabled
  })), [currentStepIdx, hideDescriptionForFutureSteps, wizardSteps]);
}
function isWizardStepEnabled(steps, stepIdx, currentStepIdx, status) {
  if (stepIdx < currentStepIdx || status === 'completed' || status === 'error' || status === 'warning') {
    return true;
  }

  // if every step before stepIdx is completed then the step is enabled
  return steps.slice(0, stepIdx).every(step => step.status === 'completed');
}

function HorizontalWizardStepsContent(_ref) {
  var _wizardSteps$currentS, _wizardSteps$currentS2;
  let {
    steps: wizardSteps,
    currentStepIndex,
    localizeStepNumber,
    enableClickingToSteps,
    goToStep,
    hideDescriptionForFutureSteps = false
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const stepperSteps = useStepperStepsFromWizardSteps(wizardSteps, currentStepIndex, hideDescriptionForFutureSteps);
  const expandContentToFullHeight = (_wizardSteps$currentS = wizardSteps[currentStepIndex].expandContentToFullHeight) !== null && _wizardSteps$currentS !== void 0 ? _wizardSteps$currentS : true;
  const disableDefaultScrollBehavior = (_wizardSteps$currentS2 = wizardSteps[currentStepIndex].disableDefaultScrollBehavior) !== null && _wizardSteps$currentS2 !== void 0 ? _wizardSteps$currentS2 : false;
  return jsxs(Fragment, {
    children: [jsx(Stepper, {
      currentStepIndex: currentStepIndex,
      direction: "horizontal",
      localizeStepNumber: localizeStepNumber,
      steps: stepperSteps,
      responsive: false,
      onStepClicked: enableClickingToSteps ? goToStep : undefined
    }), jsx("div", {
      css: /*#__PURE__*/css({
        marginTop: theme.spacing.md,
        flexGrow: expandContentToFullHeight ? 1 : undefined,
        overflowY: disableDefaultScrollBehavior ? 'hidden' : 'auto',
        ...(!disableDefaultScrollBehavior ? getShadowScrollStyles(theme) : {})
      }, process.env.NODE_ENV === "production" ? "" : ";label:HorizontalWizardStepsContent;"),
      children: wizardSteps[currentStepIndex].content
    })]
  });
}

/* eslint-disable react/no-unused-prop-types */
// eslint doesn't like passing props as object through to a function
// disabling to avoid a bunch of duplicate code.
function _EMOTION_STRINGIFIED_CSS_ERROR__$2() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2$1 = process.env.NODE_ENV === "production" ? {
  name: "1ff36h2",
  styles: "flex-grow:1"
} : {
  name: "1vyl5dv-getWizardFooterButtons",
  styles: "flex-grow:1;label:getWizardFooterButtons;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$2
};
// Buttons are returned in order with primary button last
function getWizardFooterButtons(_ref) {
  var _extraFooterButtonsRi;
  let {
    title,
    goToNextStepOrDone,
    isLastStep,
    currentStepIndex,
    goToPreviousStep,
    busyValidatingNextStep,
    nextButtonDisabled,
    nextButtonLoading,
    nextButtonContentOverride,
    previousButtonContentOverride,
    previousStepButtonHidden,
    previousButtonDisabled,
    previousButtonLoading,
    cancelButtonContent,
    cancelStepButtonHidden,
    nextButtonContent,
    previousButtonContent,
    doneButtonContent,
    extraFooterButtonsLeft,
    extraFooterButtonsRight,
    onCancel,
    moveCancelToOtherSide,
    componentId,
    tooltipContent
  } = _ref;
  return _compact([!cancelStepButtonHidden && (moveCancelToOtherSide ? jsx("div", {
    css: _ref2$1,
    children: jsx(CancelButton, {
      onCancel: onCancel,
      cancelButtonContent: cancelButtonContent,
      componentId: componentId ? `${componentId}.cancel` : undefined
    })
  }, "cancel") : jsx(CancelButton, {
    onCancel: onCancel,
    cancelButtonContent: cancelButtonContent,
    componentId: componentId ? `${componentId}.cancel` : undefined
  }, "cancel")), currentStepIndex > 0 && !previousStepButtonHidden && jsx(Button, {
    onClick: goToPreviousStep,
    type: "tertiary",
    disabled: previousButtonDisabled,
    loading: previousButtonLoading,
    componentId: componentId ? `${componentId}.previous` : 'dubois-wizard-footer-previous',
    children: previousButtonContentOverride ? previousButtonContentOverride : previousButtonContent
  }, "previous"), extraFooterButtonsLeft && extraFooterButtonsLeft.map((buttonProps, index) => createElement(ButtonWithTooltip, {
    ...buttonProps,
    type: undefined,
    key: `extra-left-${index}`
  })), jsx(ButtonWithTooltip, {
    onClick: goToNextStepOrDone,
    disabled: nextButtonDisabled,
    tooltipContent: tooltipContent,
    loading: nextButtonLoading || busyValidatingNextStep,
    type: ((_extraFooterButtonsRi = extraFooterButtonsRight === null || extraFooterButtonsRight === void 0 ? void 0 : extraFooterButtonsRight.length) !== null && _extraFooterButtonsRi !== void 0 ? _extraFooterButtonsRi : 0) > 0 ? undefined : 'primary',
    componentId: componentId ? `${componentId}.next` : 'dubois-wizard-footer-next',
    children: nextButtonContentOverride ? nextButtonContentOverride : isLastStep ? doneButtonContent : nextButtonContent
  }, "next"), extraFooterButtonsRight && extraFooterButtonsRight.map((buttonProps, index) => createElement(ButtonWithTooltip, {
    ...buttonProps,
    type: index === extraFooterButtonsRight.length - 1 ? 'primary' : undefined,
    key: `extra-right-${index}`
  }))]);
}
function CancelButton(_ref3) {
  let {
    onCancel,
    cancelButtonContent,
    componentId
  } = _ref3;
  return jsx(Button, {
    onClick: onCancel,
    type: "tertiary",
    componentId: componentId !== null && componentId !== void 0 ? componentId : 'dubois-wizard-footer-cancel',
    children: cancelButtonContent
  }, "cancel");
}
function ButtonWithTooltip(_ref4) {
  let {
    tooltipContent,
    disabled,
    children,
    ...props
  } = _ref4;
  return tooltipContent ? jsx(Tooltip, {
    componentId: "dubois-wizard-footer-tooltip",
    content: tooltipContent,
    children: jsx(Button, {
      ...props,
      disabled: disabled,
      children: children
    })
  }) : jsx(Button, {
    ...props,
    disabled: disabled,
    children: children
  });
}

function HorizontalWizardContent(_ref) {
  let {
    width,
    height,
    steps,
    currentStepIndex,
    localizeStepNumber,
    onStepsChange,
    enableClickingToSteps,
    hideDescriptionForFutureSteps,
    ...footerProps
  } = _ref;
  return jsxs("div", {
    css: /*#__PURE__*/css({
      width,
      height,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'flex-start'
    }, process.env.NODE_ENV === "production" ? "" : ";label:HorizontalWizardContent;"),
    ...addDebugOutlineIfEnabled(),
    children: [jsx(HorizontalWizardStepsContent, {
      steps: steps,
      currentStepIndex: currentStepIndex,
      localizeStepNumber: localizeStepNumber,
      enableClickingToSteps: Boolean(enableClickingToSteps),
      goToStep: footerProps.goToStep,
      hideDescriptionForFutureSteps: hideDescriptionForFutureSteps
    }), jsx(Spacer, {
      size: "lg"
    }), jsx(WizardFooter, {
      currentStepIndex: currentStepIndex,
      ...steps[currentStepIndex],
      ...footerProps,
      moveCancelToOtherSide: true
    })]
  });
}
function WizardFooter(props) {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: /*#__PURE__*/css({
      display: 'flex',
      flexDirection: 'row',
      justifyContent: 'flex-end',
      columnGap: theme.spacing.sm,
      paddingTop: theme.spacing.md,
      paddingBottom: theme.spacing.md,
      borderTop: `1px solid ${theme.colors.border}`
    }, process.env.NODE_ENV === "production" ? "" : ";label:WizardFooter;"),
    children: getWizardFooterButtons(props)
  });
}

function _EMOTION_STRINGIFIED_CSS_ERROR__$1() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
function Root(_ref) {
  let {
    children,
    initialContentId
  } = _ref;
  const [currentContentId, setCurrentContentId] = useState(initialContentId);
  return jsx(DocumentationSideBarContext.Provider, {
    value: useMemo(() => ({
      currentContentId,
      setCurrentContentId
    }), [currentContentId, setCurrentContentId]),
    children: children
  });
}
const DocumentationSideBarContext = /*#__PURE__*/React__default.createContext({
  currentContentId: undefined,
  setCurrentContentId: _noop
});
const useDocumentationSidebarContext = () => {
  const context = React__default.useContext(DocumentationSideBarContext);
  return context;
};
var _ref3 = process.env.NODE_ENV === "production" ? {
  name: "1b7w9np",
  styles: "border:none;background-color:transparent;padding:0;display:flex;height:var(--spacing-md);align-items:center;cursor:pointer"
} : {
  name: "1q7unyc-Trigger",
  styles: "border:none;background-color:transparent;padding:0;display:flex;height:var(--spacing-md);align-items:center;cursor:pointer;label:Trigger;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$1
};
function Trigger(_ref2) {
  let {
    contentId,
    label,
    tooltipContent,
    asChild,
    children,
    ...tooltipProps
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    setCurrentContentId
  } = useDocumentationSidebarContext();
  const triggerProps = useMemo(() => ({
    onClick: () => setCurrentContentId(contentId),
    [`aria-label`]: label
  }), [contentId, label, setCurrentContentId]);
  const renderAsChild = asChild && /*#__PURE__*/React__default.isValidElement(children);
  return jsx(Tooltip, {
    ...tooltipProps,
    content: tooltipContent,
    children: renderAsChild ? (/*#__PURE__*/React__default.cloneElement(children, triggerProps)) : jsx("button", {
      css: _ref3,
      ...triggerProps,
      children: jsx(InfoIcon, {
        css: /*#__PURE__*/css({
          fontSize: theme.typography.fontSizeSm,
          color: theme.colors.textSecondary
        }, process.env.NODE_ENV === "production" ? "" : ";label:Trigger;")
      })
    })
  });
}
var _ref5 = process.env.NODE_ENV === "production" ? {
  name: "fxdlmq",
  styles: "display:flex;flex-direction:row;justify-content:space-between;align-items:center;width:100%"
} : {
  name: "17fupas-Content",
  styles: "display:flex;flex-direction:row;justify-content:space-between;align-items:center;width:100%;label:Content;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__$1
};
function Content(_ref4) {
  let {
    title,
    modalTitleWhenCompact,
    width,
    children,
    closeLabel,
    displayModalWhenCompact
  } = _ref4;
  const {
    theme
  } = useDesignSystemTheme();
  const {
    currentContentId,
    setCurrentContentId
  } = useDocumentationSidebarContext();
  if (_isUndefined(currentContentId)) {
    return null;
  }
  const content = /*#__PURE__*/React__default.isValidElement(children) ? /*#__PURE__*/React__default.cloneElement(children, {
    contentId: currentContentId
  }) : children;
  if (displayModalWhenCompact) {
    return jsx(Modal, {
      componentId: `documentation-side-bar-compact-modal-${currentContentId}`,
      visible: true,
      size: "wide",
      onOk: () => setCurrentContentId(undefined),
      okText: closeLabel,
      okButtonProps: {
        type: undefined
      },
      onCancel: () => setCurrentContentId(undefined),
      title: modalTitleWhenCompact !== null && modalTitleWhenCompact !== void 0 ? modalTitleWhenCompact : title,
      children: content
    });
  }
  return jsx(Sidebar, {
    position: "right",
    dangerouslyAppendEmotionCSS: {
      border: 'none'
    },
    children: jsx(Sidebar.Content, {
      componentId: `documentation-side-bar-content-${currentContentId}`,
      openPanelId: 0,
      closable: true,
      disableResize: true,
      enableCompact: true,
      width: width,
      children: jsx(Sidebar.Panel, {
        panelId: 0,
        children: jsxs("div", {
          css: /*#__PURE__*/css({
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            rowGap: theme.spacing.md,
            borderRadius: theme.legacyBorders.borderRadiusLg,
            border: `1px solid ${theme.colors.backgroundSecondary}`,
            padding: `${theme.spacing.md}px ${theme.spacing.lg}px`,
            backgroundColor: theme.colors.backgroundSecondary
          }, process.env.NODE_ENV === "production" ? "" : ";label:Content;"),
          children: [jsxs("div", {
            css: _ref5,
            children: [jsx(Typography.Text, {
              color: "secondary",
              children: title
            }), jsx(Button, {
              "aria-label": closeLabel,
              icon: jsx(CloseIcon, {}),
              componentId: `documentation-side-bar-close-${currentContentId}`,
              onClick: () => setCurrentContentId(undefined)
            })]
          }), content]
        })
      })
    })
  });
}

var DocumentationSidebar = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Content: Content,
  Root: Root,
  Trigger: Trigger,
  useDocumentationSidebarContext: useDocumentationSidebarContext
});

const SMALL_FIXED_VERTICAL_STEPPER_WIDTH = 240;
const FIXED_VERTICAL_STEPPER_WIDTH = 280;
const MAX_VERTICAL_WIZARD_CONTENT_WIDTH = 920;
const DOCUMENTATION_SIDEBAR_WIDTH = 400;
const EXTRA_COMPACT_BUTTON_HEIGHT = 32 + 24; // button height + gap

function VerticalWizardContent(_ref) {
  var _wizardSteps$currentS, _wizardSteps$currentS2;
  let {
    width,
    height,
    steps: wizardSteps,
    currentStepIndex,
    localizeStepNumber,
    onStepsChange,
    title,
    padding,
    verticalCompactButtonContent,
    enableClickingToSteps,
    verticalDocumentationSidebarConfig,
    hideDescriptionForFutureSteps = false,
    contentMaxWidth,
    ...footerProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const stepperSteps = useStepperStepsFromWizardSteps(wizardSteps, currentStepIndex, hideDescriptionForFutureSteps);
  const expandContentToFullHeight = (_wizardSteps$currentS = wizardSteps[currentStepIndex].expandContentToFullHeight) !== null && _wizardSteps$currentS !== void 0 ? _wizardSteps$currentS : true;
  const disableDefaultScrollBehavior = (_wizardSteps$currentS2 = wizardSteps[currentStepIndex].disableDefaultScrollBehavior) !== null && _wizardSteps$currentS2 !== void 0 ? _wizardSteps$currentS2 : false;
  const displayDocumentationSideBar = Boolean(verticalDocumentationSidebarConfig);
  const Wrapper = displayDocumentationSideBar ? Root : React__default.Fragment;
  const displayCompactStepper = useMediaQuery({
    query: `(max-width: ${theme.responsive.breakpoints.lg}px)`
  }) && Boolean(verticalCompactButtonContent);
  const displayCompactSidebar = useMediaQuery({
    query: `(max-width: ${theme.responsive.breakpoints.xxl}px)`
  });
  return jsx(Wrapper, {
    initialContentId: verticalDocumentationSidebarConfig === null || verticalDocumentationSidebarConfig === void 0 ? void 0 : verticalDocumentationSidebarConfig.initialContentId,
    children: jsxs("div", {
      css: /*#__PURE__*/css({
        width,
        height: expandContentToFullHeight ? height : 'fit-content',
        maxHeight: '100%',
        display: 'flex',
        flexDirection: displayCompactStepper ? 'column' : 'row',
        gap: theme.spacing.lg,
        justifyContent: 'center'
      }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
      ...addDebugOutlineIfEnabled(),
      children: [!displayCompactStepper && jsxs("div", {
        css: /*#__PURE__*/css({
          display: 'flex',
          flexDirection: 'column',
          flexShrink: 0,
          rowGap: theme.spacing.lg,
          paddingTop: theme.spacing.lg,
          paddingBottom: theme.spacing.lg,
          height: 'fit-content',
          width: SMALL_FIXED_VERTICAL_STEPPER_WIDTH,
          [`@media (min-width: ${theme.responsive.breakpoints.xl}px)`]: {
            width: FIXED_VERTICAL_STEPPER_WIDTH
          },
          overflowX: 'hidden'
        }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
        children: [title, jsx(Stepper, {
          currentStepIndex: currentStepIndex,
          direction: "vertical",
          localizeStepNumber: localizeStepNumber,
          steps: stepperSteps,
          responsive: false,
          onStepClicked: enableClickingToSteps ? footerProps.goToStep : undefined
        })]
      }), displayCompactStepper && jsxs(Root$1, {
        componentId: "codegen_design-system_src_~patterns_wizard_verticalwizardcontent.tsx_93",
        children: [jsx(Trigger$1, {
          asChild: true,
          children: jsx("div", {
            children: jsx(Button, {
              icon: jsx(ListIcon, {}),
              componentId: "dubois-wizard-vertical-compact-show-stepper-popover",
              children: verticalCompactButtonContent === null || verticalCompactButtonContent === void 0 ? void 0 : verticalCompactButtonContent(currentStepIndex, stepperSteps.length)
            })
          })
        }), jsx(Content$2, {
          align: "start",
          side: "bottom",
          css: /*#__PURE__*/css({
            padding: theme.spacing.md
          }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
          children: jsx(Stepper, {
            currentStepIndex: currentStepIndex,
            direction: "vertical",
            localizeStepNumber: localizeStepNumber,
            steps: stepperSteps,
            responsive: false,
            onStepClicked: enableClickingToSteps ? footerProps.goToStep : undefined
          })
        })]
      }), jsxs("div", {
        css: /*#__PURE__*/css({
          display: 'flex',
          flexDirection: 'column',
          columnGap: theme.spacing.lg,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.legacyBorders.borderRadiusLg,
          flexGrow: 1,
          padding: padding !== null && padding !== void 0 ? padding : theme.spacing.lg,
          height: displayCompactStepper ? `calc(100% - ${EXTRA_COMPACT_BUTTON_HEIGHT}px)` : '100%',
          maxWidth: contentMaxWidth !== null && contentMaxWidth !== void 0 ? contentMaxWidth : MAX_VERTICAL_WIZARD_CONTENT_WIDTH
        }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
        children: [jsx("div", {
          css: /*#__PURE__*/css({
            flexGrow: expandContentToFullHeight ? 1 : undefined,
            overflowY: disableDefaultScrollBehavior ? 'hidden' : 'auto',
            ...(!disableDefaultScrollBehavior ? getShadowScrollStyles(theme) : {}),
            borderRadius: theme.legacyBorders.borderRadiusLg
          }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
          children: wizardSteps[currentStepIndex].content
        }), jsx("div", {
          css: /*#__PURE__*/css({
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'flex-end',
            columnGap: theme.spacing.sm,
            ...(padding !== undefined && {
              padding: theme.spacing.lg
            }),
            paddingTop: theme.spacing.md
          }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
          children: getWizardFooterButtons({
            currentStepIndex: currentStepIndex,
            ...wizardSteps[currentStepIndex],
            ...footerProps,
            moveCancelToOtherSide: true
          })
        })]
      }), displayDocumentationSideBar && verticalDocumentationSidebarConfig && jsx(Content, {
        width: displayCompactSidebar ? undefined : DOCUMENTATION_SIDEBAR_WIDTH,
        title: verticalDocumentationSidebarConfig.title,
        modalTitleWhenCompact: verticalDocumentationSidebarConfig.modalTitleWhenCompact,
        closeLabel: verticalDocumentationSidebarConfig.closeLabel,
        displayModalWhenCompact: displayCompactSidebar,
        children: verticalDocumentationSidebarConfig.content
      })]
    })
  });
}

function useWizardCurrentStep(_ref) {
  let {
    currentStepIndex,
    setCurrentStepIndex,
    totalSteps,
    onValidateStepChange,
    onStepChanged
  } = _ref;
  const [busyValidatingNextStep, setBusyValidatingNextStep] = useState(false);
  const isLastStep = useMemo(() => currentStepIndex === totalSteps - 1, [currentStepIndex, totalSteps]);
  const onStepsChange = useCallback(async function (step) {
    let completed = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;
    if (!completed && step === currentStepIndex) return;
    setCurrentStepIndex(step);
    onStepChanged({
      step,
      completed
    });
  }, [currentStepIndex, onStepChanged, setCurrentStepIndex]);
  const goToNextStepOrDone = useCallback(async () => {
    if (onValidateStepChange) {
      setBusyValidatingNextStep(true);
      try {
        const approvedStepChange = await onValidateStepChange(currentStepIndex);
        if (!approvedStepChange) {
          return;
        }
      } finally {
        setBusyValidatingNextStep(false);
      }
    }
    onStepsChange(Math.min(currentStepIndex + 1, totalSteps - 1), isLastStep);
  }, [currentStepIndex, isLastStep, onStepsChange, onValidateStepChange, totalSteps]);
  const goToPreviousStep = useCallback(() => {
    if (currentStepIndex > 0) {
      onStepsChange(currentStepIndex - 1);
    }
  }, [currentStepIndex, onStepsChange]);
  const goToStep = useCallback(step => {
    if (step > -1 && step < totalSteps) {
      onStepsChange(step);
    }
  }, [onStepsChange, totalSteps]);
  return {
    busyValidatingNextStep,
    isLastStep,
    onStepsChange,
    goToNextStepOrDone,
    goToPreviousStep,
    goToStep
  };
}

function Wizard(_ref) {
  let {
    steps,
    onStepChanged,
    onValidateStepChange,
    initialStep,
    ...props
  } = _ref;
  const [currentStepIndex, setCurrentStepIndex] = useState(initialStep !== null && initialStep !== void 0 ? initialStep : 0);
  const currentStepProps = useWizardCurrentStep({
    currentStepIndex,
    setCurrentStepIndex,
    totalSteps: steps.length,
    onStepChanged,
    onValidateStepChange
  });
  return jsx(WizardControlled, {
    ...currentStepProps,
    currentStepIndex: currentStepIndex,
    initialStep: initialStep,
    steps: steps,
    onStepChanged: onStepChanged,
    ...props
  });
}
function WizardControlled(_ref2) {
  let {
    initialStep = 0,
    layout = 'vertical',
    width = '100%',
    height = '100%',
    steps,
    title,
    ...restOfProps
  } = _ref2;
  if (steps.length === 0 || !_isUndefined(initialStep) && (initialStep < 0 || initialStep >= steps.length)) {
    return null;
  }
  if (layout === 'vertical') {
    return jsx(VerticalWizardContent, {
      width: width,
      height: height,
      steps: steps,
      title: title,
      ...restOfProps
    });
  }
  return jsx(HorizontalWizardContent, {
    width: width,
    height: height,
    steps: steps,
    ...restOfProps
  });
}

function WizardModal(_ref) {
  let {
    onStepChanged,
    onCancel,
    initialStep,
    steps,
    onModalClose,
    localizeStepNumber,
    cancelButtonContent,
    nextButtonContent,
    previousButtonContent,
    doneButtonContent,
    enableClickingToSteps,
    ...modalProps
  } = _ref;
  const [currentStepIndex, setCurrentStepIndex] = useState(initialStep !== null && initialStep !== void 0 ? initialStep : 0);
  const {
    onStepsChange,
    isLastStep,
    ...footerActions
  } = useWizardCurrentStep({
    currentStepIndex,
    setCurrentStepIndex,
    totalSteps: steps.length,
    onStepChanged
  });
  if (steps.length === 0 || !_isUndefined(initialStep) && (initialStep < 0 || initialStep >= steps.length)) {
    return null;
  }
  const footerButtons = getWizardFooterButtons({
    onCancel,
    isLastStep,
    currentStepIndex,
    doneButtonContent,
    previousButtonContent,
    nextButtonContent,
    cancelButtonContent,
    ...footerActions,
    ...steps[currentStepIndex]
  });
  return jsx(Modal, {
    ...modalProps,
    onCancel: onModalClose,
    size: "wide",
    footer: footerButtons,
    children: jsx(HorizontalWizardStepsContent, {
      steps: steps,
      currentStepIndex: currentStepIndex,
      localizeStepNumber: localizeStepNumber,
      enableClickingToSteps: Boolean(enableClickingToSteps),
      goToStep: footerActions.goToStep
    })
  });
}

function _EMOTION_STRINGIFIED_CSS_ERROR__() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2 = process.env.NODE_ENV === "production" ? {
  name: "13ku56z",
  styles: "display:flex;flex-direction:column;height:100%"
} : {
  name: "hucxjd-WizardStepContentWrapper",
  styles: "display:flex;flex-direction:column;height:100%;label:WizardStepContentWrapper;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__
};
function WizardStepContentWrapper(_ref) {
  let {
    header,
    title,
    description,
    children
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  return jsxs("div", {
    css: _ref2,
    children: [jsxs("div", {
      style: {
        backgroundColor: theme.colors.backgroundSecondary,
        padding: theme.spacing.lg,
        display: 'flex',
        flexDirection: 'column',
        borderTopLeftRadius: theme.legacyBorders.borderRadiusLg,
        borderTopRightRadius: theme.legacyBorders.borderRadiusLg
      },
      children: [jsx(Typography.Text, {
        size: "sm",
        style: {
          fontWeight: 500
        },
        children: header
      }), jsx(Typography.Title, {
        withoutMargins: true,
        style: {
          paddingTop: theme.spacing.lg
        },
        level: 3,
        children: title
      }), jsx(Typography.Text, {
        color: "secondary",
        children: description
      })]
    }), jsx("div", {
      css: /*#__PURE__*/css({
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        padding: `${theme.spacing.lg}px ${theme.spacing.lg}px 0`,
        overflowY: 'auto',
        ...getShadowScrollStyles(theme)
      }, process.env.NODE_ENV === "production" ? "" : ";label:WizardStepContentWrapper;"),
      children: children
    })]
  });
}

export { Content$1 as C, DocumentationSidebar as D, FIXED_VERTICAL_STEPPER_WIDTH as F, InfoIcon as I, ListIcon as L, MAX_VERTICAL_WIZARD_CONTENT_WIDTH as M, Nav as N, Panel as P, Spacer as S, Tooltip as T, Wizard as W, WizardControlled as a, WizardModal as b, WizardStepContentWrapper as c, Modal as d, useModalContext as e, DangerModal as f, NavButton as g, PanelHeader as h, PanelHeaderTitle as i, PanelHeaderButtons as j, PanelBody as k, Sidebar as l, useWizardCurrentStep as u };
//# sourceMappingURL=WizardStepContentWrapper-C8nqlsYy.js.map

import { forwardRef, useContext, createContext, useLayoutEffect, useEffect, useState } from 'react';
import { I as Icon, m as useDesignSystemContext, f as addDebugOutlineIfEnabled, u as useDesignSystemTheme, h as getDarkModePortalStyles, l as importantify, j as DesignSystemEventSuppressInteractionProviderContext, k as DesignSystemEventSuppressInteractionTrueContextValue, b as useDesignSystemEventComponentCallbacks, c as DesignSystemEventProviderComponentTypes, d as DesignSystemEventProviderAnalyticsEventTypes, e as useNotifyOnFirstView, D as DesignSystemAntDConfigProvider, R as RestoreAntDDefaultClsPrefix, C as CloseIcon, q as DangerIcon, B as Button, a as addDebugOutlineStylesIfEnabled, p as getShadowScrollStyles, g as getAnimationCss } from './Typography-78b12af3.js';
import { jsx, jsxs, Fragment } from '@emotion/react/jsx-runtime';
import { css } from '@emotion/react';
import * as Popover$1 from '@radix-ui/react-popover';
import { Modal as Modal$1 } from 'antd';

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
const ListIcon = /*#__PURE__*/forwardRef((props, forwardedRef) => {
  return jsx(Icon, {
    ref: forwardedRef,
    ...props,
    component: SvgListIcon
  });
});
ListIcon.displayName = 'ListIcon';
var ListIcon$1 = ListIcon;

const Root = Popover$1.Root; // Behavioral component only
const Anchor = Popover$1.Anchor; // Behavioral component only

const Content = /*#__PURE__*/forwardRef(function Content(_ref, ref) {
  let {
    children,
    minWidth = 220,
    maxWidth,
    ...props
  } = _ref;
  const {
    getPopupContainer
  } = useDesignSystemContext();
  return jsx(Popover$1.Portal, {
    container: getPopupContainer && getPopupContainer(),
    children: jsx(Popover$1.Content, {
      ...addDebugOutlineIfEnabled(),
      ref: ref,
      css: [contentStyles, {
        minWidth,
        maxWidth
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
    ...addDebugOutlineIfEnabled(),
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
});
const contentStyles = theme => ({
  ...popoverContentStyles(theme)
});

var Popover = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Anchor: Anchor,
  Arrow: Arrow,
  Close: Close,
  Content: Content,
  Root: Root,
  Trigger: Trigger
});

function _EMOTION_STRINGIFIED_CSS_ERROR__() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
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
      boxShadow: theme.general.shadowHigh,
      ...getDarkModePortalStyles(theme)
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
    analyticsEvents,
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
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Modal,
    componentId,
    analyticsEvents: analyticsEvents !== null && analyticsEvents !== void 0 ? analyticsEvents : [DesignSystemEventProviderAnalyticsEventTypes.OnView],
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
    closeButtonEventContext.onClick();
    (_props$onCancel = props.onCancel) === null || _props$onCancel === void 0 || _props$onCancel.call(props, e);
  };
  return jsx(DesignSystemAntDConfigProvider, {
    children: jsx(Modal$1, {
      ...addDebugOutlineIfEnabled(),
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
          }, process.env.NODE_ENV === "production" ? "" : ";label:ModalInternal;"),
          title: typeof title === 'string' ? title : undefined,
          children: title
        }) : title
      }),
      footer: jsx(RestoreAntDDefaultClsPrefix, {
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
      ...props,
      onCancel: onCancelWrapper,
      ...dangerouslySetAntdProps,
      children: jsx(RestoreAntDDefaultClsPrefix, {
        children: jsx(ModalContext.Provider, {
          value: {
            isInsideModal: true
          },
          children: children
        })
      })
    })
  });
}
var _ref3 = process.env.NODE_ENV === "production" ? {
  name: "b9hrb",
  styles: "position:relative;display:inline-flex;align-items:center"
} : {
  name: "1jkwrsj-titleComp",
  styles: "position:relative;display:inline-flex;align-items:center;label:titleComp;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__
};
var _ref4 = process.env.NODE_ENV === "production" ? {
  name: "1o6wc9k",
  styles: "padding-left:6px"
} : {
  name: "i303lp-titleComp",
  styles: "padding-left:6px;label:titleComp;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__
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
    css: _ref3,
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
    componentId: props.componentId,
    analyticsEvents: props.analyticsEvents,
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

export { Arrow as A, Content as C, DangerModal as D, ListIcon$1 as L, Modal as M, Popover as P, Root as R, Spacer as S, Trigger as T, useMediaQuery as a, useModalContext as u };
//# sourceMappingURL=useMediaQuery-72ec2d25.js.map

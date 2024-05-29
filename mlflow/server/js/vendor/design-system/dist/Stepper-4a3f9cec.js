import { css } from '@emotion/react';
import _isUndefined from 'lodash/isUndefined';
import { useRef, useState, useEffect } from 'react';
import { u as useDesignSystemTheme, f as addDebugOutlineIfEnabled, T as Typography, C as CloseIcon, r as LoadingIcon, n as CheckIcon, a0 as DEFAULT_SPACING_UNIT } from './Typography-af72332b.js';
import { jsx, jsxs } from '@emotion/react/jsx-runtime';

function Stepper(_ref) {
  let {
    direction: requestedDirection,
    currentStepIndex: currentStep,
    steps,
    localizeStepNumber,
    responsive = true
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const ref = useRef(null);
  const {
    direction
  } = useResponsiveDirection({
    ref,
    requestedDirection,
    responsive,
    enabled: steps.length > 0
  });
  if (steps.length === 0) {
    return null;
  }
  const isHorizontal = direction === 'horizontal';
  const currentStepIndex = currentStep ? Math.min(steps.length - 1, Math.max(0, currentStep)) : 0;
  return jsx("ol", {
    ...addDebugOutlineIfEnabled(),
    css: /*#__PURE__*/css(getStepsStyle(theme, isHorizontal), process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
    ref: ref,
    children: steps.map((step, index) => {
      const isCurrentStep = index === currentStepIndex;
      const isLastStep = index === steps.length - 1;
      const displayEndingDivider = !isLastStep;
      const contentTitleLevel = isCurrentStep ? 3 : 4;
      const StepIcon = step.icon;
      const {
        icon,
        iconBackgroundColor,
        iconTextColor,
        titleTextColor,
        descriptionTextColor,
        hasStepItemIconBorder
      } = getStepContentStyleFields(theme, isCurrentStep, step.status, StepIcon, step.additionalVerticalContent);
      return jsx("li", {
        "aria-current": isCurrentStep,
        css: /*#__PURE__*/css(getStepItemStyle(theme, isHorizontal, isLastStep), process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
        ...(step.status === 'error' && {
          'data-error': true
        }),
        children: jsxs(StepContentGrid, {
          isHorizontal: isHorizontal,
          isCurrentStep: isCurrentStep,
          children: [jsx("div", {
            css: /*#__PURE__*/css(getStepItemIconParentStyle(theme, isCurrentStep, iconBackgroundColor, hasStepItemIconBorder), process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
            children: StepIcon ? jsx(StepIcon, {
              statusColor: iconTextColor,
              status: step.status
            }) : icon ? jsx("span", {
              css: /*#__PURE__*/css({
                color: iconTextColor,
                display: 'flex'
              }, process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
              children: icon
            }) : jsx(Typography.Title, {
              level: contentTitleLevel,
              css: /*#__PURE__*/css({
                color: `${iconTextColor} !important`
              }, process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
              withoutMargins: true,
              children: localizeStepNumber(index + 1)
            })
          }), jsx(Typography.Title, {
            level: contentTitleLevel,
            withoutMargins: true,
            css: /*#__PURE__*/css({
              flexShrink: 0,
              color: `${titleTextColor} !important`
            }, process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
            children: step.title
          }), displayEndingDivider && jsx("div", {
            css: /*#__PURE__*/css(getStepEndingDividerStyles(theme, isHorizontal, isCurrentStep), process.env.NODE_ENV === "production" ? "" : ";label:Stepper;")
          }), (step.description || step.additionalVerticalContent && !isHorizontal) && jsxs("div", {
            css: /*#__PURE__*/css(getStepDescriptionStyles(theme, isHorizontal, isLastStep, descriptionTextColor), process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
            children: [step.description && jsx(Typography.Text, {
              css: /*#__PURE__*/css(getStepDescriptionTextStyles(descriptionTextColor), process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
              withoutMargins: true,
              size: "sm",
              children: step.description
            }), step.additionalVerticalContent && !isHorizontal && jsx("div", {
              css: /*#__PURE__*/css(getAdditionalVerticalStepContentStyles(theme, Boolean(step.description)), process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
              children: step.additionalVerticalContent
            })]
          })]
        })
      }, index);
    })
  });
}
function getStepsStyle(theme, isHorizontal) {
  return /*#__PURE__*/css({
    listStyle: 'none',
    display: 'flex',
    flexDirection: isHorizontal ? 'row' : 'column',
    flexWrap: 'wrap',
    alignItems: 'flex-start',
    gap: theme.spacing.md,
    width: '100%',
    margin: '0',
    padding: '0'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getStepsStyle;");
}
function getStepItemStyle(theme, isHorizontal, isLastStep) {
  return /*#__PURE__*/css({
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'flex-start',
    flexGrow: isLastStep ? 0 : 1,
    marginRight: isLastStep ? theme.spacing.lg + theme.spacing.md : 0,
    width: isHorizontal ? undefined : '100%'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getStepItemStyle;");
}
function getStepContentStyleFields(theme, isCurrentStep, status, icon, additionalVerticalContent) {
  const fields = getStepContentStyleFieldsFromStatus(theme, isCurrentStep, status, !_isUndefined(additionalVerticalContent));
  if (icon) {
    return {
      ...fields,
      icon: undefined,
      iconBackgroundColor: undefined,
      iconTextColor: getCustomIconColor(theme, isCurrentStep, status),
      hasStepItemIconBorder: false
    };
  }
  return fields;
}
function getCustomIconColor(theme, isCurrentStep, status) {
  switch (status) {
    case 'completed':
      return theme.colors.actionDefaultBackgroundPress;
    case 'loading':
      return theme.colors.textPlaceholder;
    case 'error':
      return theme.colors.textValidationDanger;
    default:
    case 'upcoming':
      return isCurrentStep ? theme.colors.actionLinkDefault : theme.colors.textPlaceholder;
  }
}
function getStepContentStyleFieldsFromStatus(theme, isCurrentStep, status, hasAdditionalVerticalContent) {
  switch (status) {
    case 'completed':
      return {
        icon: jsx(CheckIcon, {}),
        iconBackgroundColor: theme.colors.actionDefaultBackgroundPress,
        iconTextColor: theme.colors.textPlaceholder,
        titleTextColor: theme.colors.textPrimary,
        descriptionTextColor: theme.colors.textSecondary,
        hasStepItemIconBorder: true
      };
    case 'loading':
      return {
        icon: jsx(LoadingIcon, {
          spin: true,
          css: /*#__PURE__*/css({
            fontSize: isCurrentStep ? theme.typography.fontSizeXl : theme.typography.fontSizeLg
          }, process.env.NODE_ENV === "production" ? "" : ";label:icon;")
        }),
        iconBackgroundColor: undefined,
        iconTextColor: theme.colors.textPlaceholder,
        titleTextColor: isCurrentStep ? theme.colors.textPrimary : theme.colors.textSecondary,
        descriptionTextColor: theme.colors.textSecondary,
        hasStepItemIconBorder: false
      };
    case 'error':
      return {
        icon: jsx(CloseIcon, {}),
        iconBackgroundColor: theme.colors.textValidationDanger,
        iconTextColor: 'white',
        titleTextColor: theme.colors.textValidationDanger,
        descriptionTextColor: theme.colors.textValidationDanger,
        hasStepItemIconBorder: false
      };
    default:
    case 'upcoming':
      if (isCurrentStep) {
        return {
          icon: undefined,
          iconBackgroundColor: theme.colors.actionLinkDefault,
          iconTextColor: 'white',
          titleTextColor: theme.colors.textPrimary,
          descriptionTextColor: hasAdditionalVerticalContent ? theme.colors.textSecondary : theme.colors.textPrimary,
          hasStepItemIconBorder: false
        };
      }
      return {
        icon: undefined,
        iconBackgroundColor: undefined,
        iconTextColor: theme.colors.textPlaceholder,
        titleTextColor: theme.colors.textSecondary,
        descriptionTextColor: theme.colors.textSecondary,
        hasStepItemIconBorder: true
      };
  }
}
const MaxHorizontalStepDescriptionWidth = 140;
const CurrentStepIconSize = DEFAULT_SPACING_UNIT * 4;
const NonCurrentStepIconSize = DEFAULT_SPACING_UNIT * 3;
function getStepItemIconParentStyle(theme, isCurrentStep, iconBackgroundColor, hasStepItemIconBorder) {
  return /*#__PURE__*/css({
    width: isCurrentStep ? CurrentStepIconSize : NonCurrentStepIconSize,
    height: isCurrentStep ? CurrentStepIconSize : NonCurrentStepIconSize,
    backgroundColor: iconBackgroundColor,
    borderRadius: '50%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    fontSize: '20px',
    flexShrink: 0,
    border: hasStepItemIconBorder ? `1px solid ${theme.colors.textPlaceholder}` : undefined
  }, process.env.NODE_ENV === "production" ? "" : ";label:getStepItemIconParentStyle;");
}
function getStepEndingDividerStyles(theme, isHorizontal, isCurrentStep) {
  const backgroundColor = isCurrentStep ? theme.colors.actionLinkDefault : theme.colors.border;
  if (isHorizontal) {
    return /*#__PURE__*/css({
      backgroundColor,
      height: '1px',
      width: '100%',
      minWidth: theme.spacing.md
    }, process.env.NODE_ENV === "production" ? "" : ";label:getStepEndingDividerStyles;");
  }
  return /*#__PURE__*/css({
    backgroundColor,
    height: '100%',
    minHeight: theme.spacing.md,
    width: '1px',
    alignSelf: 'flex-start',
    marginLeft: theme.spacing.md
  }, process.env.NODE_ENV === "production" ? "" : ";label:getStepEndingDividerStyles;");
}
function getStepDescriptionStyles(theme, isHorizontal, isLastStep, textColor) {
  return /*#__PURE__*/css({
    alignSelf: 'flex-start',
    width: '100%',
    gridColumn: isHorizontal || isLastStep ? '2 / span 2' : undefined,
    maxWidth: isHorizontal ? MaxHorizontalStepDescriptionWidth : undefined,
    paddingBottom: isHorizontal ? undefined : theme.spacing.sm,
    color: `${textColor} !important`
  }, process.env.NODE_ENV === "production" ? "" : ";label:getStepDescriptionStyles;");
}
function getStepDescriptionTextStyles(textColor) {
  return /*#__PURE__*/css({
    color: `${textColor} !important`
  }, process.env.NODE_ENV === "production" ? "" : ";label:getStepDescriptionTextStyles;");
}
function getAdditionalVerticalStepContentStyles(theme, addTopPadding) {
  return /*#__PURE__*/css({
    paddingTop: addTopPadding ? theme.spacing.sm : 0
  }, process.env.NODE_ENV === "production" ? "" : ";label:getAdditionalVerticalStepContentStyles;");
}
function StepContentGrid(_ref2) {
  let {
    children,
    isHorizontal,
    isCurrentStep
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  if (isHorizontal) {
    return jsx("div", {
      css: /*#__PURE__*/css({
        display: 'grid',
        gridTemplateColumns: `${isCurrentStep ? CurrentStepIconSize : NonCurrentStepIconSize}px fit-content(100%) 1fr`,
        gridTemplateRows: `${CurrentStepIconSize}px auto`,
        alignItems: 'center',
        justifyItems: 'flex-start',
        gridColumnGap: theme.spacing.sm,
        width: '100%'
      }, process.env.NODE_ENV === "production" ? "" : ";label:StepContentGrid;"),
      children: children
    });
  }
  return jsx("div", {
    css: /*#__PURE__*/css({
      display: 'grid',
      gridTemplateColumns: `${CurrentStepIconSize}px auto`,
      alignItems: 'center',
      justifyItems: 'flex-start',
      gridColumnGap: theme.spacing.md,
      gridRowGap: theme.spacing.sm,
      width: '100%',
      '& > :first-child': {
        // horizontally center the first column (circle/icon)
        justifySelf: 'center'
      }
    }, process.env.NODE_ENV === "production" ? "" : ";label:StepContentGrid;"),
    children: children
  });
}

// Ant design uses the same value for their stepper and to works well for us as well.
const MinimumHorizonalDirectionWidth = 532;

// exported for unit test
function useResponsiveDirection(_ref3) {
  let {
    requestedDirection = 'horizontal',
    responsive,
    enabled,
    ref
  } = _ref3;
  const [direction, setDirection] = useState(requestedDirection);
  useEffect(() => {
    if (requestedDirection === 'vertical' || !enabled || !responsive || !ref.current) {
      return;
    }
    let timeoutId;
    const resizeObserver = new ResizeObserver(entries => {
      timeoutId = requestAnimationFrame(() => {
        if (entries.length === 1) {
          const width = entries[0].target.clientWidth || 0;
          setDirection(width < MinimumHorizonalDirectionWidth ? 'vertical' : 'horizontal');
        }
      });
    });
    if (ref.current) {
      resizeObserver.observe(ref.current);
    }
    return () => {
      resizeObserver.disconnect();
      cancelAnimationFrame(timeoutId);
    };
  }, [requestedDirection, enabled, ref, responsive]);
  return {
    direction
  };
}

export { Stepper as S };
//# sourceMappingURL=Stepper-4a3f9cec.js.map

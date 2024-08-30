import { css } from '@emotion/react';
import _isUndefined from 'lodash/isUndefined';
import { useRef, useState, useEffect } from 'react';
import { a as useDesignSystemTheme, h as addDebugOutlineIfEnabled, p as Typography, W as WarningIcon, y as DangerIcon, z as LoadingIcon, r as CheckIcon, a7 as DEFAULT_SPACING_UNIT } from './Typography-a18b0186.js';
import { jsx, jsxs } from '@emotion/react/jsx-runtime';

function Stepper(_ref) {
  let {
    direction: requestedDirection,
    currentStepIndex: currentStep,
    steps,
    localizeStepNumber,
    responsive = true,
    onStepClicked
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
      const contentTitleLevel = 4;
      const StepIcon = step.icon;
      const clickEnabled = step.clickEnabled && onStepClicked;
      const {
        icon,
        iconBackgroundColor,
        iconTextColor,
        titleTextColor,
        titleAsParagraph,
        descriptionTextColor,
        hasStepItemIconBorder,
        stepItemIconBorderColor
      } = getStepContentStyleFields(theme, isCurrentStep, step.status, StepIcon, step.additionalVerticalContent);
      return jsx("li", {
        "aria-current": isCurrentStep,
        css: /*#__PURE__*/css(getStepItemStyle(theme, isHorizontal, isLastStep), process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
        ...(step.status === 'error' && {
          'data-error': true
        }),
        ...(step.status === 'loading' && {
          'data-loading': true
        }),
        children: jsxs(StepContentGrid, {
          isHorizontal: isHorizontal,
          children: [jsx("div", {
            css: /*#__PURE__*/css(getStepItemIconParentStyle(theme, iconBackgroundColor, hasStepItemIconBorder, stepItemIconBorderColor, Boolean(clickEnabled)), process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
            onClick: clickEnabled ? () => onStepClicked(index) : undefined,
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
          }), jsx("span", {
            onClick: clickEnabled ? () => onStepClicked(index) : undefined,
            onKeyDown: event => {
              if (event.key === 'Enter' && clickEnabled) {
                onStepClicked(index);
              }
            },
            css: /*#__PURE__*/css({
              cursor: clickEnabled ? 'pointer' : undefined
            }, process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
            tabIndex: clickEnabled ? 0 : undefined,
            role: clickEnabled ? 'button' : undefined,
            children: titleAsParagraph ? jsx(Typography.Text, {
              withoutMargins: true,
              css: /*#__PURE__*/css({
                flexShrink: 0,
                color: `${titleTextColor} !important`
              }, process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
              children: step.title
            }) : jsx(Typography.Title, {
              level: 4,
              withoutMargins: true,
              css: /*#__PURE__*/css({
                flexShrink: 0,
                color: `${titleTextColor} !important`
              }, process.env.NODE_ENV === "production" ? "" : ";label:Stepper;"),
              children: step.title
            })
          }), displayEndingDivider && jsx("div", {
            css: /*#__PURE__*/css(getStepEndingDividerStyles(theme, isHorizontal, step.status === 'completed'), process.env.NODE_ENV === "production" ? "" : ";label:Stepper;")
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
    gap: isHorizontal ? theme.spacing.sm : theme.spacing.xs,
    width: '100%',
    margin: '0',
    padding: '0'
  }, process.env.NODE_ENV === "production" ? "" : ";label:getStepsStyle;");
}
function getStepItemStyle(theme, isHorizontal, isLastStep) {
  return /*#__PURE__*/css({
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'flex-start',
    flexGrow: isLastStep ? 0 : 1,
    marginRight: isLastStep && isHorizontal ? theme.spacing.lg + theme.spacing.md : 0,
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
      return theme.colors.actionLinkDefault;
    case 'loading':
      return theme.colors.textPlaceholder;
    case 'error':
      return theme.colors.textValidationDanger;
    case 'warning':
      return theme.colors.textValidationWarning;
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
        iconBackgroundColor: isCurrentStep ? theme.colors.actionLinkDefault : theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100,
        iconTextColor: isCurrentStep ? theme.colors.white : theme.colors.actionLinkDefault,
        titleAsParagraph: false,
        titleTextColor: theme.colors.actionLinkDefault,
        descriptionTextColor: theme.colors.textSecondary,
        hasStepItemIconBorder: true,
        stepItemIconBorderColor: theme.colors.actionDefaultBackgroundPress
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
        titleAsParagraph: false,
        titleTextColor: isCurrentStep ? theme.colors.textPrimary : theme.colors.textSecondary,
        descriptionTextColor: theme.colors.textSecondary,
        hasStepItemIconBorder: false
      };
    case 'error':
      return {
        icon: jsx(DangerIcon, {}),
        iconBackgroundColor: isCurrentStep ? theme.colors.textValidationDanger : theme.colors.backgroundDanger,
        iconTextColor: isCurrentStep ? theme.colors.white : theme.colors.textValidationDanger,
        titleAsParagraph: false,
        titleTextColor: theme.colors.textValidationDanger,
        descriptionTextColor: theme.colors.textValidationDanger,
        hasStepItemIconBorder: true,
        stepItemIconBorderColor: theme.colors.borderDanger
      };
    case 'warning':
      return {
        icon: jsx(WarningIcon, {}),
        iconBackgroundColor: isCurrentStep ? theme.colors.textValidationWarning : theme.colors.backgroundWarning,
        iconTextColor: isCurrentStep ? theme.colors.white : theme.colors.textValidationWarning,
        titleAsParagraph: false,
        titleTextColor: theme.colors.textValidationWarning,
        descriptionTextColor: theme.colors.textValidationWarning,
        hasStepItemIconBorder: true,
        stepItemIconBorderColor: theme.colors.borderWarning
      };
    default:
    case 'upcoming':
      if (isCurrentStep) {
        return {
          icon: undefined,
          iconBackgroundColor: theme.colors.actionLinkDefault,
          iconTextColor: 'white',
          titleAsParagraph: false,
          titleTextColor: theme.colors.actionLinkDefault,
          descriptionTextColor: hasAdditionalVerticalContent ? theme.colors.textSecondary : theme.colors.textPrimary,
          hasStepItemIconBorder: false
        };
      }
      return {
        icon: undefined,
        iconBackgroundColor: undefined,
        iconTextColor: theme.colors.textPlaceholder,
        titleAsParagraph: true,
        titleTextColor: theme.colors.textSecondary,
        descriptionTextColor: theme.colors.textSecondary,
        hasStepItemIconBorder: true,
        stepItemIconBorderColor: theme.colors.border
      };
  }
}
const MaxHorizontalStepDescriptionWidth = 140;
const StepIconSize = DEFAULT_SPACING_UNIT * 4;
function getStepItemIconParentStyle(theme, iconBackgroundColor, hasStepItemIconBorder, stepItemIconBorderColor, clickEnabled) {
  return /*#__PURE__*/css({
    width: StepIconSize,
    height: StepIconSize,
    backgroundColor: iconBackgroundColor,
    borderRadius: '50%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    fontSize: '20px',
    flexShrink: 0,
    border: hasStepItemIconBorder ? `1px solid ${stepItemIconBorderColor !== null && stepItemIconBorderColor !== void 0 ? stepItemIconBorderColor : theme.colors.textPlaceholder}` : undefined,
    boxSizing: 'border-box',
    cursor: clickEnabled ? 'pointer' : undefined
  }, process.env.NODE_ENV === "production" ? "" : ";label:getStepItemIconParentStyle;");
}
function getStepEndingDividerStyles(theme, isHorizontal, isCompleted) {
  const backgroundColor = isCompleted ? theme.colors.actionLinkDefault : theme.colors.border;
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
    minHeight: theme.spacing.lg,
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
    isHorizontal
  } = _ref2;
  const {
    theme
  } = useDesignSystemTheme();
  if (isHorizontal) {
    return jsx("div", {
      css: /*#__PURE__*/css({
        display: 'grid',
        gridTemplateColumns: `${StepIconSize}px fit-content(100%) 1fr`,
        gridTemplateRows: `${StepIconSize}px auto`,
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
      gridTemplateColumns: `${StepIconSize}px minmax(0, 1fr)`,
      alignItems: 'center',
      justifyItems: 'flex-start',
      gridColumnGap: theme.spacing.sm,
      gridRowGap: theme.spacing.xs,
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
//# sourceMappingURL=Stepper-56765495.js.map

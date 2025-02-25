import type { SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import { isUndefined } from 'lodash';
import { useState, type PropsWithChildren, useRef, useEffect } from 'react';

import { CheckIcon, DangerIcon, LoadingIcon, Typography, WarningIcon, useDesignSystemTheme } from '../../design-system';
import { addDebugOutlineIfEnabled } from '../../design-system/utils/debug';
import type { Theme } from '../../theme';
import { DEFAULT_SPACING_UNIT } from '../../theme/spacing';

export interface Step {
  /**
   * Title of the step
   */
  title: React.ReactNode;

  /**
   * Optional description of the step
   */
  description?: React.ReactNode;

  /**
   * Status of the step. This will change the icon and text color of the step.
   *
   * @default 'upcoming'
   */
  status?: 'completed' | 'loading' | 'upcoming' | 'error' | 'warning';

  /**
   * Custom icon to display in the step. If provided, the `icon` prop will be used instead of the default icon.
   */
  icon?: React.ComponentType<{ statusColor: string; status: Step['status'] }>;

  /**
   * Additional content displayed beneath the step description a vertical stepper
   *
   * This can be used to create a vertical wizard
   */
  additionalVerticalContent?: React.ReactNode;

  /**
   * If true, the step can be clicked and the `onStepClicked` callback will be called
   */
  clickEnabled?: boolean;
}

export interface StepperProps {
  /**
   * List of ordered steps in the stepper
   */
  steps: Step[];

  /**
   * Function to localize the step number; workaround for no react-intl support within dubois
   *
   * ex) localizeStepNumber={intl.formatNumber}
   */
  localizeStepNumber: (stepIndex: number) => string;

  /**
   * Direction of the stepper
   *
   * @default horizontal
   */
  direction?: 'horizontal' | 'vertical';

  /**
   * Current active step from the `steps` property (zero-indexed)
   *
   * @default 0
   */
  currentStepIndex?: number;

  /**
   * If true, and the stepper has a horizontal direction the stepper will be updated to be vertical if width is less than 532px.
   * Set this value to false to opt out of the responsive behavior.
   *
   * @default true
   */
  responsive?: boolean;

  /**
   * Callback when a step is clicked for steps with `clickEnabled` set to true
   *
   * @default 'undefined'
   */
  onStepClicked?: (stepIndex: number) => void;
}

export function Stepper({
  direction: requestedDirection,
  currentStepIndex: currentStep,
  steps,
  localizeStepNumber,
  responsive = true,
  onStepClicked,
}: StepperProps) {
  const { theme } = useDesignSystemTheme();
  const ref = useRef<HTMLOListElement>(null);
  const { direction } = useResponsiveDirection({ ref, requestedDirection, responsive, enabled: steps.length > 0 });

  if (steps.length === 0) {
    return null;
  }

  const isHorizontal = direction === 'horizontal';
  const currentStepIndex = currentStep ? Math.min(steps.length - 1, Math.max(0, currentStep)) : 0;

  return (
    <ol {...addDebugOutlineIfEnabled()} css={css(getStepsStyle(theme, isHorizontal))} ref={ref}>
      {steps.map((step, index) => {
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
          stepItemIconBorderColor,
        } = getStepContentStyleFields(theme, isCurrentStep, step.status, StepIcon, step.additionalVerticalContent);

        return (
          <li
            aria-current={isCurrentStep}
            key={index}
            css={css(getStepItemStyle(theme, isHorizontal, isLastStep))}
            {...(step.status === 'error' && { 'data-error': true })}
            {...(step.status === 'loading' && { 'data-loading': true })}
          >
            <StepContentGrid isHorizontal={isHorizontal}>
              <div
                css={css(
                  getStepItemIconParentStyle(
                    theme,
                    iconBackgroundColor,
                    hasStepItemIconBorder,
                    stepItemIconBorderColor,
                    Boolean(clickEnabled),
                  ),
                )}
                onClick={clickEnabled ? () => onStepClicked(index) : undefined}
              >
                {StepIcon ? (
                  <StepIcon statusColor={iconTextColor} status={step.status} />
                ) : icon ? (
                  <span css={{ color: iconTextColor, display: 'flex' }}>{icon}</span>
                ) : (
                  <Typography.Title
                    level={contentTitleLevel}
                    css={{ color: `${iconTextColor} !important` }}
                    withoutMargins
                  >
                    {localizeStepNumber(index + 1)}
                  </Typography.Title>
                )}
              </div>
              <span
                onClick={clickEnabled ? () => onStepClicked(index) : undefined}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && clickEnabled) {
                    onStepClicked(index);
                  }
                }}
                css={{ cursor: clickEnabled ? 'pointer' : undefined }}
                tabIndex={clickEnabled ? 0 : undefined}
                role={clickEnabled ? 'button' : undefined}
              >
                {titleAsParagraph ? (
                  <Typography.Text withoutMargins css={{ flexShrink: 0, color: `${titleTextColor} !important` }}>
                    {step.title}
                  </Typography.Text>
                ) : (
                  <Typography.Title
                    level={4}
                    withoutMargins
                    css={{ flexShrink: 0, color: `${titleTextColor} !important` }}
                  >
                    {step.title}
                  </Typography.Title>
                )}
              </span>
              {displayEndingDivider && (
                <div css={css(getStepEndingDividerStyles(theme, isHorizontal, step.status === 'completed'))} />
              )}
              {(step.description || (step.additionalVerticalContent && !isHorizontal)) && (
                <div css={css(getStepDescriptionStyles(theme, isHorizontal, isLastStep, descriptionTextColor))}>
                  {step.description && (
                    <Typography.Text
                      css={css(getStepDescriptionTextStyles(descriptionTextColor))}
                      withoutMargins
                      size="sm"
                    >
                      {step.description}
                    </Typography.Text>
                  )}
                  {step.additionalVerticalContent && !isHorizontal && (
                    <div css={css(getAdditionalVerticalStepContentStyles(theme, Boolean(step.description)))}>
                      {step.additionalVerticalContent}
                    </div>
                  )}
                </div>
              )}
            </StepContentGrid>
          </li>
        );
      })}
    </ol>
  );
}

function getStepsStyle(theme: Theme, isHorizontal: boolean): SerializedStyles {
  return css({
    listStyle: 'none',
    display: 'flex',
    flexDirection: isHorizontal ? 'row' : 'column',
    flexWrap: 'wrap',
    alignItems: 'flex-start',
    gap: isHorizontal ? theme.spacing.sm : theme.spacing.xs,
    width: '100%',
    margin: '0',
    padding: '0',
  });
}

function getStepItemStyle(theme: Theme, isHorizontal: boolean, isLastStep: boolean): SerializedStyles {
  return css({
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'flex-start',
    flexGrow: isLastStep ? 0 : 1,
    marginRight: isLastStep && isHorizontal ? theme.spacing.lg + theme.spacing.md : 0,
    width: isHorizontal ? undefined : '100%',
  });
}

function getStepContentStyleFields(
  theme: Theme,
  isCurrentStep: boolean,
  status: Step['status'],
  icon: Step['icon'],
  additionalVerticalContent: Step['additionalVerticalContent'],
) {
  const fields = getStepContentStyleFieldsFromStatus(
    theme,
    isCurrentStep,
    status,
    !isUndefined(additionalVerticalContent),
  );
  if (icon) {
    return {
      ...fields,
      icon: undefined,
      iconBackgroundColor: undefined,
      iconTextColor: getCustomIconColor(theme, isCurrentStep, status),
      hasStepItemIconBorder: false,
    };
  }

  return fields;
}

function getCustomIconColor(theme: Theme, isCurrentStep: boolean, status: Step['status']): string {
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

function getStepContentStyleFieldsFromStatus(
  theme: Theme,
  isCurrentStep: boolean,
  status: Step['status'],
  hasAdditionalVerticalContent: boolean,
): {
  icon: React.ReactNode | undefined;
  iconBackgroundColor: string | undefined;
  iconTextColor: string;
  titleAsParagraph: boolean;
  titleTextColor: string;
  descriptionTextColor: string;
  hasStepItemIconBorder: boolean;
  stepItemIconBorderColor?: string;
} {
  switch (status) {
    case 'completed':
      return {
        icon: <CheckIcon />,
        iconBackgroundColor: isCurrentStep
          ? theme.colors.actionLinkDefault
          : theme.isDarkMode
          ? theme.colors.blue800
          : theme.colors.blue100,
        iconTextColor: isCurrentStep ? theme.colors.white : theme.colors.actionLinkDefault,
        titleAsParagraph: false,
        titleTextColor: theme.colors.actionLinkDefault,
        descriptionTextColor: theme.colors.textSecondary,
        hasStepItemIconBorder: true,
        stepItemIconBorderColor: theme.colors.actionDefaultBackgroundPress,
      };
    case 'loading':
      return {
        icon: (
          <LoadingIcon
            spin
            css={{ fontSize: isCurrentStep ? theme.typography.fontSizeXl : theme.typography.fontSizeLg }}
          />
        ),
        iconBackgroundColor: undefined,
        iconTextColor: theme.colors.textPlaceholder,
        titleAsParagraph: false,
        titleTextColor: isCurrentStep ? theme.colors.textPrimary : theme.colors.textSecondary,
        descriptionTextColor: theme.colors.textSecondary,
        hasStepItemIconBorder: false,
      };
    case 'error':
      return {
        icon: <DangerIcon />,
        iconBackgroundColor: isCurrentStep ? theme.colors.textValidationDanger : theme.colors.backgroundDanger,
        iconTextColor: isCurrentStep ? theme.colors.white : theme.colors.textValidationDanger,
        titleAsParagraph: false,
        titleTextColor: theme.colors.textValidationDanger,
        descriptionTextColor: theme.colors.textValidationDanger,
        hasStepItemIconBorder: true,
        stepItemIconBorderColor: theme.colors.borderDanger,
      };
    case 'warning':
      return {
        icon: <WarningIcon />,
        iconBackgroundColor: isCurrentStep ? theme.colors.textValidationWarning : theme.colors.backgroundWarning,
        iconTextColor: isCurrentStep ? theme.colors.white : theme.colors.textValidationWarning,
        titleAsParagraph: false,
        titleTextColor: theme.colors.textValidationWarning,
        descriptionTextColor: theme.colors.textValidationWarning,
        hasStepItemIconBorder: true,
        stepItemIconBorderColor: theme.colors.borderWarning,
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
          hasStepItemIconBorder: false,
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
        stepItemIconBorderColor: theme.colors.border,
      };
  }
}

const MaxHorizontalStepDescriptionWidth = 140;
const StepIconSize = DEFAULT_SPACING_UNIT * 4;

function getStepItemIconParentStyle(
  theme: Theme,
  iconBackgroundColor: string | undefined,
  hasStepItemIconBorder: boolean,
  stepItemIconBorderColor: string | undefined,
  clickEnabled: boolean,
): SerializedStyles {
  return css({
    width: StepIconSize,
    height: StepIconSize,
    backgroundColor: iconBackgroundColor,
    borderRadius: '50%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    fontSize: '20px',
    flexShrink: 0,
    border: hasStepItemIconBorder ? `1px solid ${stepItemIconBorderColor ?? theme.colors.textPlaceholder}` : undefined,
    boxSizing: 'border-box',
    cursor: clickEnabled ? 'pointer' : undefined,
  });
}

function getStepEndingDividerStyles(theme: Theme, isHorizontal: boolean, isCompleted: boolean): SerializedStyles {
  const backgroundColor = isCompleted ? theme.colors.actionLinkDefault : theme.colors.border;

  if (isHorizontal) {
    return css({
      backgroundColor,
      height: '1px',
      width: '100%',
      minWidth: theme.spacing.md,
    });
  }
  return css({
    backgroundColor,
    height: '100%',
    minHeight: theme.spacing.lg,
    width: '1px',
    alignSelf: 'flex-start',
    marginLeft: theme.spacing.md,
  });
}

function getStepDescriptionStyles(
  theme: Theme,
  isHorizontal: boolean,
  isLastStep: boolean,
  textColor: string,
): SerializedStyles {
  return css({
    alignSelf: 'flex-start',
    width: '100%',
    gridColumn: isHorizontal || isLastStep ? '2 / span 2' : undefined,
    maxWidth: isHorizontal ? MaxHorizontalStepDescriptionWidth : undefined,
    paddingBottom: isHorizontal ? undefined : theme.spacing.sm,
    color: `${textColor} !important`,
  });
}

function getStepDescriptionTextStyles(textColor: string): SerializedStyles {
  return css({
    color: `${textColor} !important`,
  });
}

function getAdditionalVerticalStepContentStyles(theme: Theme, addTopPadding: boolean): SerializedStyles {
  return css({
    paddingTop: addTopPadding ? theme.spacing.sm : 0,
  });
}

function StepContentGrid({ children, isHorizontal }: PropsWithChildren<{ isHorizontal: boolean }>) {
  const { theme } = useDesignSystemTheme();
  if (isHorizontal) {
    return (
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: `${StepIconSize}px fit-content(100%) 1fr`,
          gridTemplateRows: `${StepIconSize}px auto`,
          alignItems: 'center',
          justifyItems: 'flex-start',
          gridColumnGap: theme.spacing.sm,
          width: '100%',
        }}
      >
        {children}
      </div>
    );
  }
  return (
    <div
      css={{
        display: 'grid',
        gridTemplateColumns: `${StepIconSize}px minmax(0, 1fr)`,
        alignItems: 'center',
        justifyItems: 'flex-start',
        gridColumnGap: theme.spacing.sm,
        gridRowGap: theme.spacing.xs,
        width: '100%',
        '& > :first-child': {
          // horizontally center the first column (circle/icon)
          justifySelf: 'center',
        },
      }}
    >
      {children}
    </div>
  );
}

// Ant design uses the same value for their stepper and to works well for us as well.
const MinimumHorizonalDirectionWidth = 532;

// exported for unit test
export function useResponsiveDirection({
  requestedDirection = 'horizontal',
  responsive,
  enabled,
  ref,
}: {
  requestedDirection: StepperProps['direction'];
  enabled: boolean;
  responsive: boolean;
  ref: React.RefObject<HTMLOListElement>;
}) {
  const [direction, setDirection] = useState(requestedDirection);

  useEffect(() => {
    if (requestedDirection === 'vertical' || !enabled || !responsive || !ref.current) {
      return;
    }

    let timeoutId: number;
    const resizeObserver = new ResizeObserver((entries) => {
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

  return { direction };
}

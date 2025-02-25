import type { CSSObject, Interpolation, SerializedStyles, Theme as EmotionTheme } from '@emotion/react';
import { css } from '@emotion/react';

import { getAnimationCss } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { Icon } from '../Icon';
import { CheckCircleIcon, DangerIcon, WarningIcon, InfoIcon } from '../Icon';
import type { ValidationState } from '../types';
import { useDesignSystemSafexFlags } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export interface FormMessageProps {
  id?: string;
  message: React.ReactNode;
  type: ValidationState;
  className?: string;
  css?: Interpolation<EmotionTheme>;
}

const getMessageStyles = (clsPrefix: string, theme: EmotionTheme, useNewFormUISpacing: boolean): SerializedStyles => {
  const errorClass = `.${clsPrefix}-form-error-message`;
  const infoClass = `.${clsPrefix}-form-info-message`;
  const successClass = `.${clsPrefix}-form-success-message`;
  const warningClass = `.${clsPrefix}-form-warning-message`;

  const styles: CSSObject = {
    '&&': {
      lineHeight: theme.typography.lineHeightSm,
      fontSize: theme.typography.fontSizeSm,
      ...(!useNewFormUISpacing && {
        marginTop: theme.spacing.sm,
      }),
      display: 'flex',
      alignItems: 'start',
    },

    [`&${errorClass}`]: {
      color: theme.colors.actionDangerPrimaryBackgroundDefault,
    },

    [`&${infoClass}`]: {
      color: theme.colors.textPrimary,
    },

    [`&${successClass}`]: {
      color: theme.colors.textValidationSuccess,
    },

    [`&${warningClass}`]: {
      color: theme.colors.textValidationWarning,
    },

    ...getAnimationCss(theme.options.enableAnimation),
  };

  return css(styles);
};

const VALIDATION_STATE_ICONS: Record<ValidationState, typeof Icon> = {
  error: DangerIcon,
  success: CheckCircleIcon,
  warning: WarningIcon,
  info: InfoIcon,
};

export function FormMessage({ id, message, type = 'error', className = '', css }: FormMessageProps): JSX.Element {
  const { classNamePrefix, theme } = useDesignSystemTheme();
  const { useNewFormUISpacing } = useDesignSystemSafexFlags();

  const stateClass = `${classNamePrefix}-form-${type}-message`;
  const StateIcon = VALIDATION_STATE_ICONS[type];

  const wrapperClass = `${classNamePrefix}-form-message ${className} ${stateClass}`.trim();

  return (
    <div
      {...(id && { id })}
      className={wrapperClass}
      {...addDebugOutlineIfEnabled()}
      css={[getMessageStyles(classNamePrefix, theme, useNewFormUISpacing), css]}
      role="alert"
    >
      <StateIcon />
      <div style={{ paddingLeft: theme.spacing.xs }}>{message}</div>
    </div>
  );
}

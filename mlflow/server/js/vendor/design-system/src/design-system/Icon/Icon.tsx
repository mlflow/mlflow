import AntDIcon from '@ant-design/icons';
import type { ReactElement } from 'react';
import React, { forwardRef, useMemo } from 'react';

import type { Theme } from '../../theme';
import { DesignSystemAntDConfigProvider } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import type { DangerouslySetAntdProps, HTMLDataAttributes } from '../types';
import { visuallyHidden } from '../utils';
import { addDebugOutlineIfEnabled } from '../utils/debug';
import { useUniqueId } from '../utils/useUniqueId';

type AntDIconProps = Parameters<typeof AntDIcon>[0];
export type IconColors = 'danger' | 'warning' | 'success' | 'ai';
export interface IconProps
  extends Omit<AntDIconProps, 'component'>,
    DangerouslySetAntdProps<AntDIconProps>,
    HTMLDataAttributes,
    React.HTMLAttributes<HTMLSpanElement> {
  component?: (props: React.SVGProps<SVGSVGElement>) => ReactElement | null;
  color?: IconColors;
}

const getIconVariantStyles = (theme: Theme, color?: IconColors) => {
  switch (color) {
    case 'success':
      return { color: theme.colors.textValidationSuccess };
    case 'warning':
      return { color: theme.colors.textValidationWarning };
    case 'danger':
      return { color: theme.colors.textValidationDanger };
    case 'ai':
      return {
        'svg *': {
          fill: 'var(--ai-icon-gradient)',
        },
      };
    default:
      return { color: color };
  }
};

export const Icon = forwardRef<HTMLSpanElement, IconProps>((props: IconProps, forwardedRef): JSX.Element => {
  const { component: Component, dangerouslySetAntdProps, color, style, ...otherProps } = props;
  const { theme } = useDesignSystemTheme();
  const linearGradientId = useUniqueId('ai-linear-gradient');

  const MemoizedComponent = useMemo(
    () =>
      Component
        ? ({ fill, ...iconProps }: React.SVGProps<SVGSVGElement>) => (
            // We don't rely on top-level fills for our colors. Fills are specified
            // with "currentColor" on children of the top-most svg.
            <>
              <Component
                fill="none"
                {...iconProps}
                style={
                  color === 'ai'
                    ? { ['--ai-icon-gradient' as any]: `url(#${linearGradientId})`, ...iconProps.style }
                    : iconProps.style
                }
              />
              {color === 'ai' && (
                <svg width="0" height="0" viewBox="0 0 0 0" css={visuallyHidden}>
                  <defs>
                    <linearGradient id={linearGradientId} x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="20.5%" stopColor={theme.colors.branded.ai.gradientStart} />
                      <stop offset="46.91%" stopColor={theme.colors.branded.ai.gradientMid} />
                      <stop offset="79.5%" stopColor={theme.colors.branded.ai.gradientEnd} />
                    </linearGradient>
                  </defs>
                </svg>
              )}
            </>
          )
        : undefined,
    [Component, color, linearGradientId, theme],
  );

  return (
    <DesignSystemAntDConfigProvider>
      <AntDIcon
        {...addDebugOutlineIfEnabled()}
        ref={forwardedRef}
        aria-hidden="true"
        css={{
          fontSize: theme.general.iconFontSize,
          ...getIconVariantStyles(theme, color),
        }}
        component={MemoizedComponent}
        style={{
          ...style,
        }}
        {...otherProps}
        {...dangerouslySetAntdProps}
      />
    </DesignSystemAntDConfigProvider>
  );
});

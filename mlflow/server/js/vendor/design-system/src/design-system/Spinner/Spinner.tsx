import type { SerializedStyles } from '@emotion/react';
import { css, keyframes } from '@emotion/react';
import type { ReactElement } from 'react';

import type { Theme } from '../../theme';
import { AccessibleContainer } from '../AccessibleContainer';
import { DU_BOIS_ENABLE_ANIMATION_CLASSNAME } from '../DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks';
import { LoadingIcon } from '../Icon';
import { LoadingState } from '../LoadingState/LoadingState';
import type { WithLoadingState } from '../LoadingState/LoadingState';
import type { HTMLDataAttributes } from '../types';
import { importantify } from '../utils/css-utils';

const rotate = keyframes({
  '0%': { transform: 'rotate(0deg) translate3d(0, 0, 0)' },
  '100%': { transform: 'rotate(360deg) translate3d(0, 0, 0)' },
});

const cssSpinner = (
  theme: Theme,
  frameRate = 60,
  delay = 0,
  animationDuration = 1,
  inheritColor = false,
): SerializedStyles => {
  const styles = {
    animation: `${rotate} ${animationDuration}s steps(${frameRate}, end) infinite`,
    ...(inheritColor
      ? {
          color: 'inherit',
        }
      : {
          color: theme.colors.textSecondary,
        }),
    animationDelay: `${delay}s`,

    '@media only percy': {
      animation: 'none',
    },
  };
  return css(importantify(styles));
};

export interface SpinnerProps extends HTMLDataAttributes, WithLoadingState {
  size?: 'small' | 'default' | 'large';
  className?: string;
  delay?: number;
  frameRate?: number;
  label?: React.ReactNode;
  animationDuration?: number;
  inheritColor?: boolean;
}

export const Spinner: React.FC<SpinnerProps> = ({
  frameRate,
  size = 'default',
  delay,
  className: propClass,
  label,
  animationDuration,
  inheritColor,
  loading = true,
  loadingDescription = 'Spinner',
  ...props
}): ReactElement => {
  const { classNamePrefix, theme } = useDesignSystemTheme();

  // We use Antd classes to keep styling unchanged
  // TODO(FEINF-1407): We want to move away from Antd classes and use Emotion for styling in the future
  const sizeSuffix = size === 'small' ? '-sm' : size === 'large' ? '-lg' : '';
  const sizeClass = sizeSuffix ? `${classNamePrefix}-spin${sizeSuffix}` : '';
  const wrapperClass = `${
    propClass || ''
  } ${classNamePrefix}-spin ${sizeClass} ${classNamePrefix}-spin-spinning ${DU_BOIS_ENABLE_ANIMATION_CLASSNAME}`.trim();
  const className = `${classNamePrefix}-spin-dot ${DU_BOIS_ENABLE_ANIMATION_CLASSNAME}`.trim();

  return (
    // className has to follow {...props}, otherwise is `css` prop is passed down it will overwrite our className
    <div {...props} className={wrapperClass}>
      {loading && <LoadingState description={loadingDescription} />}
      <AccessibleContainer label={label}>
        <LoadingIcon
          aria-hidden="false"
          css={cssSpinner(theme, frameRate, delay, animationDuration, inheritColor)}
          className={className}
        />
      </AccessibleContainer>
    </div>
  );
};

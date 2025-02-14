import type { CSSObject, SerializedStyles } from '@emotion/react';
import { css } from '@emotion/react';
import * as Progress from '@radix-ui/react-progress';
import React from 'react';

import { ProgressContext, ProgressContextProvider } from './providers/ProgressContex';
import { importantify, useDesignSystemTheme } from '../../design-system';
import type { Theme } from '../../theme';

export interface ProgressRootProps extends Progress.ProgressProps {
  minWidth?: number;
  maxWidth?: number;
}

const getProgressRootStyles = (
  theme: Theme,
  { minWidth, maxWidth }: { minWidth?: number; maxWidth?: number },
): SerializedStyles => {
  const styles: CSSObject = {
    position: 'relative',
    overflow: 'hidden',
    backgroundColor: theme.colors.progressTrack,
    height: theme.spacing.sm,
    width: '100%',
    borderRadius: theme.spacing.xs,

    ...(minWidth && { minWidth }),
    ...(maxWidth && { maxWidth }),

    /* Fix overflow clipping in Safari */
    /* https://gist.github.com/domske/b66047671c780a238b51c51ffde8d3a0 */
    transform: 'translateZ(0)',
  };

  return css(importantify(styles));
};

export const Root = (props: ProgressRootProps) => {
  const { children, value, minWidth, maxWidth, ...restProps } = props;
  const { theme } = useDesignSystemTheme();
  return (
    <ProgressContextProvider value={{ progress: value }}>
      <Progress.Root value={value} {...restProps} css={getProgressRootStyles(theme, { minWidth, maxWidth })}>
        {children}
      </Progress.Root>
    </ProgressContextProvider>
  );
};

export interface ProgressIndicatorProps extends Progress.ProgressIndicatorProps {}

const getProgressIndicatorStyles = (theme: Theme): SerializedStyles => {
  const styles = {
    backgroundColor: theme.colors.progressFill,
    height: '100%',
    width: '100%',
    transition: 'transform 300ms linear',
    borderRadius: theme.spacing.xs,
  };

  return css(importantify(styles));
};

export const Indicator = (props: ProgressIndicatorProps) => {
  const { progress } = React.useContext(ProgressContext);
  const { theme } = useDesignSystemTheme();
  return (
    <Progress.Indicator
      css={getProgressIndicatorStyles(theme)}
      style={{ transform: `translateX(-${100 - (progress ?? 100)}%)` }}
      {...props}
    />
  );
};

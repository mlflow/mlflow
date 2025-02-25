import { css } from '@emotion/react';
import * as RadixSlider from '@radix-ui/react-slider';
import { forwardRef } from 'react';

import { useDesignSystemSafexFlags } from '..';
import type { Theme } from '../../theme';
import { useDesignSystemTheme } from '../Hooks';
import { addDebugOutlineIfEnabled } from '../utils/debug';

export type SliderRootProps = Omit<RadixSlider.SliderProps, 'orientation'>;
export type SliderTrackProps = RadixSlider.SliderTrackProps;
export type SliderRangeProps = RadixSlider.SliderRangeProps;
export type SliderThumbProps = RadixSlider.SliderThumbProps;

const getRootStyles = () => {
  return css({
    position: 'relative',
    display: 'flex',
    alignItems: 'center',

    '&[data-orientation="vertical"]': {
      flexDirection: 'column',
      width: 20,
      height: 100,
    },

    '&[data-orientation="horizontal"]': {
      height: 20,
      width: 200,
    },
  });
};

export const Root = forwardRef<HTMLElement, SliderRootProps>((props: SliderRootProps, ref): JSX.Element => {
  return <RadixSlider.Root {...addDebugOutlineIfEnabled()} css={getRootStyles()} {...props} ref={ref} />;
});

const getTrackStyles = (theme: Theme) => {
  return css({
    backgroundColor: theme.colors.grey100,
    position: 'relative',
    flexGrow: 1,
    borderRadius: 9999,

    '&[data-orientation="vertical"]': {
      width: 3,
    },

    '&[data-orientation="horizontal"]': {
      height: 3,
    },
  });
};

export const Track = forwardRef<HTMLElement, SliderTrackProps>((props: SliderTrackProps, ref): JSX.Element => {
  const { theme } = useDesignSystemTheme();

  return <RadixSlider.Track css={getTrackStyles(theme)} {...props} ref={ref} />;
});

const getRangeStyles = (theme: Theme) => {
  return css({
    backgroundColor: theme.colors.primary,
    position: 'absolute',
    borderRadius: 9999,
    height: '100%',

    '&[data-disabled]': {
      backgroundColor: theme.colors.grey100,
    },
  });
};

export const Range = forwardRef<HTMLElement, SliderRangeProps>((props: SliderRangeProps, ref): JSX.Element => {
  const { theme } = useDesignSystemTheme();

  return <RadixSlider.Range css={getRangeStyles(theme)} {...props} ref={ref} />;
});

const getThumbStyles = (theme: Theme, useNewShadows: boolean) => {
  return css({
    display: 'block',
    width: 20,
    height: 20,
    backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
    boxShadow: useNewShadows ? theme.shadows.xs : `0 2px 4px 0 ${theme.colors.grey400}`,
    borderRadius: 10,
    outline: 'none',

    '&:hover': {
      backgroundColor: theme.colors.actionPrimaryBackgroundHover,
    },

    '&:focus': {
      backgroundColor: theme.colors.actionPrimaryBackgroundPress,
    },

    '&[data-disabled]': {
      backgroundColor: theme.colors.grey200,
      boxShadow: 'none',
    },
  });
};

export const Thumb = forwardRef<HTMLElement, SliderThumbProps>((props: SliderThumbProps, ref): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();

  return <RadixSlider.Thumb css={getThumbStyles(theme, useNewShadows)} {...props} ref={ref} />;
});

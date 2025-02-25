import { css } from '@emotion/react';
import * as RadixToolbar from '@radix-ui/react-toolbar';
import { forwardRef } from 'react';

import { useDesignSystemSafexFlags, useDesignSystemTheme } from '../../design-system';
import { addDebugOutlineIfEnabled } from '../../design-system/utils/debug';
import type { Theme } from '../../theme';

export type ToolbarRootProps = Omit<RadixToolbar.ToolbarProps, 'orientation'>;
export type ToolbarButtonProps = RadixToolbar.ToolbarButtonProps;
export type ToolbarSeparatorProps = RadixToolbar.ToolbarSeparatorProps;
export type ToolbarLinkProps = RadixToolbar.ToolbarLinkProps;
export type ToolbarToogleGroupProps =
  | RadixToolbar.ToolbarToggleGroupSingleProps
  | RadixToolbar.ToolbarToggleGroupMultipleProps;
export type ToolbarToggleItemProps = RadixToolbar.ToolbarToggleItemProps;

const getRootStyles = (theme: Theme, useNewShadows: boolean) => {
  return css({
    alignItems: 'center',
    backgroundColor: theme.colors.backgroundSecondary,
    border: `1px solid ${theme.colors.borderDecorative}`,
    borderRadius: theme.legacyBorders.borderRadiusMd,
    boxShadow: useNewShadows ? theme.shadows.lg : theme.general.shadowLow,
    display: 'flex',
    gap: theme.spacing.md,
    width: 'max-content',
    padding: theme.spacing.sm,
  });
};

export const Root = forwardRef<HTMLDivElement, ToolbarRootProps>((props: ToolbarRootProps, ref): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const { useNewShadows } = useDesignSystemSafexFlags();
  return (
    <RadixToolbar.Root {...addDebugOutlineIfEnabled()} css={getRootStyles(theme, useNewShadows)} {...props} ref={ref} />
  );
});

export const Button = forwardRef<HTMLButtonElement, ToolbarButtonProps>(
  (props: ToolbarButtonProps, ref): JSX.Element => {
    return <RadixToolbar.Button {...props} ref={ref} />;
  },
);

const getSeparatorStyles = (theme: Theme) => {
  return css({
    alignSelf: 'stretch',
    backgroundColor: theme.colors.borderDecorative,
    width: 1,
  });
};

export const Separator = forwardRef<HTMLDivElement, ToolbarSeparatorProps>(
  (props: ToolbarSeparatorProps, ref): JSX.Element => {
    const { theme } = useDesignSystemTheme();
    return <RadixToolbar.Separator css={getSeparatorStyles(theme)} {...props} ref={ref} />;
  },
);

export const Link = forwardRef<HTMLAnchorElement, ToolbarLinkProps>((props: ToolbarLinkProps, ref): JSX.Element => {
  return <RadixToolbar.Link {...props} ref={ref} />;
});

export const ToggleGroup = forwardRef<
  HTMLDivElement,
  RadixToolbar.ToolbarToggleGroupSingleProps | RadixToolbar.ToolbarToggleGroupMultipleProps
>((props: ToolbarToogleGroupProps, ref): JSX.Element => {
  return <RadixToolbar.ToggleGroup {...props} ref={ref} />;
});

const getToggleItemStyles = (theme: Theme) => {
  return css({
    background: 'none',
    color: theme.colors.textPrimary,
    border: 'none',
    cursor: 'pointer',
    '&:hover': {
      color: theme.colors.actionDefaultTextHover,
    },
    '&[data-state="on"]': {
      color: theme.colors.actionDefaultTextPress,
    },
  });
};

export const ToggleItem = forwardRef<HTMLButtonElement, ToolbarToggleItemProps>(
  (props: ToolbarToggleItemProps, ref): JSX.Element => {
    const { theme } = useDesignSystemTheme();
    return <RadixToolbar.ToggleItem css={getToggleItemStyles(theme)} {...props} ref={ref} />;
  },
);

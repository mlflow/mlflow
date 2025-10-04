import type { useDesignSystemTheme } from '@databricks/design-system';

type DesignSystemTheme = ReturnType<typeof useDesignSystemTheme>['theme'];

export const cardBaseStyles = (theme: DesignSystemTheme) => ({
  display: 'flex',
  flexDirection: 'column' as const,
  gap: theme.spacing.sm,
  padding: `${theme.spacing.xl}px ${theme.spacing.lg}px`,
  borderRadius: theme.borders.borderRadiusLg,
  border: `1px solid ${theme.colors.borderStrong}`,
  backgroundColor: theme.colors.backgroundPrimary,
  boxShadow: theme.shadows.md,
  textDecoration: 'none',
  color: theme.colors.textPrimary,
  transition: 'transform 150ms ease, box-shadow 150ms ease',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows.md,
    textDecoration: 'none',
  },
  '&:focus-visible': {
    outline: `2px solid ${theme.colors.actionPrimaryBackgroundHover}`,
    outlineOffset: 2,
  },
});

export const cardCtaStyles = (theme: DesignSystemTheme) => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: theme.spacing.xs,
  color: theme.colors.actionPrimaryTextDefault,
  fontWeight: theme.typography.typographyBoldFontWeight,
});

export const sectionHeaderStyles = { margin: 0 } as const;

export const getStartedCardLinkStyles = (theme: DesignSystemTheme) => ({
  textDecoration: 'none',
  color: theme.colors.textPrimary,
  display: 'block',
});

export const getStartedCardContainerStyles = (theme: DesignSystemTheme) => ({
  overflow: 'hidden',
  border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
  borderRadius: theme.borders.borderRadiusMd,
  background: theme.colors.backgroundPrimary,
  padding: theme.spacing.sm + theme.spacing.xs,
  display: 'flex',
  gap: theme.spacing.sm,
  width: 320,
  minWidth: 320,
  boxSizing: 'border-box' as const,
  boxShadow: theme.shadows.sm,
  cursor: 'pointer',
  transition: 'background 150ms ease',
  '&:hover': {
    background: theme.colors.actionDefaultBackgroundHover,
  },
  '&:active': {
    background: theme.colors.actionDefaultBackgroundPress,
  },
});

export const getStartedIconWrapperStyles = (theme: DesignSystemTheme) => ({
  borderRadius: theme.borders.borderRadiusSm,
  background: theme.colors.actionDefaultBackgroundHover,
  padding: theme.spacing.xs,
  color: theme.colors.blue500,
  height: 'min-content',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
});

export const getStartedCardContentStyles = (theme: DesignSystemTheme) => ({
  display: 'flex',
  flexDirection: 'column' as const,
  gap: theme.spacing.xs,
  flex: 1,
});

export const discoverNewsCardContainerStyles = (theme: DesignSystemTheme) => ({
  overflow: 'hidden',
  border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
  borderRadius: theme.borders.borderRadiusMd,
  background: theme.colors.backgroundPrimary,
  padding: theme.spacing.sm + theme.spacing.xs,
  display: 'flex',
  flexDirection: 'column' as const,
  gap: theme.spacing.sm,
  boxSizing: 'border-box' as const,
  boxShadow: theme.shadows.sm,
  cursor: 'pointer',
  transition: 'background 150ms ease',
  height: '100%',
  width: 320,
  minWidth: 320,
  '&:hover': {
    background: theme.colors.actionDefaultBackgroundHover,
  },
  '&:active': {
    background: theme.colors.actionDefaultBackgroundPress,
  },
});

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

import type { ThemeType } from '@databricks/design-system';

// Omits height: '100%' from the CLAUDE.md empty-state pattern so the empty state
// sits near the top in both grid and table views instead of centering vertically.
export const emptyCenterStyles = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  minHeight: 400,
  width: '100%',
  '& > div': {
    display: 'flex',
    flexDirection: 'column' as const,
    justifyContent: 'center',
    alignItems: 'center',
  },
};

export const cardGridStyles = (theme: ThemeType) => ({
  flex: '0 1 auto',
  overflow: 'auto',
  minHeight: 0,
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
  gap: theme.spacing.md,
  paddingTop: theme.spacing.md,
});

export const textClampStyles = (lines: number) => ({
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  display: '-webkit-box',
  WebkitLineClamp: lines,
  WebkitBoxOrient: 'vertical' as const,
});

export const textEllipsisStyles = {
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap' as const,
};

export const flexColumnContainerStyles = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column' as const,
  overflow: 'hidden',
};

export const headerIconStyles = (theme: ThemeType) => ({
  display: 'flex',
  borderRadius: theme.borders.borderRadiusSm,
  backgroundColor: theme.colors.backgroundSecondary,
  padding: theme.spacing.sm,
});

export const cardBodyStyles = (theme: ThemeType) => ({
  display: 'flex',
  flexDirection: 'column' as const,
  gap: theme.spacing.xs,
  overflow: 'hidden',
  flex: 1,
});

export const cardHeaderRowStyles = (theme: ThemeType) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: theme.spacing.sm,
});

export const inlineFlexRowStyles = (theme: ThemeType) => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: theme.spacing.sm,
  maxWidth: '100%',
});

export const flexRowStyles = (theme: ThemeType) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing.xs,
});

export const noShrinkStyles = {
  flexShrink: 0,
};

export const overlayButtonStyles = (theme: ThemeType) => ({
  position: 'absolute' as const,
  top: theme.spacing.xs,
  right: theme.spacing.xs,
  zIndex: 1,
});

export const selectedRowIndicatorStyles = (theme: ThemeType) => ({
  width: theme.spacing.md * 2,
  display: 'flex',
  alignItems: 'center',
  paddingRight: theme.spacing.sm,
});

export const tagListStyles = (theme: ThemeType) => ({
  display: 'flex',
  flexWrap: 'wrap' as const,
  alignItems: 'center',
  gap: theme.spacing.xs,
});

export const flexColumnGapStyles = (theme: ThemeType, gap = theme.spacing.sm) => ({
  display: 'flex',
  flexDirection: 'column' as const,
  gap,
});

export const flexRowWrapStyles = (theme: ThemeType) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing.sm,
  flexWrap: 'wrap' as const,
});

export const spaceBetweenRowStyles = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  width: '100%',
};

export const lineClampStyles = (lines = 1) => ({
  display: '-webkit-box' as const,
  WebkitLineClamp: lines,
  WebkitBoxOrient: 'vertical' as const,
  overflow: 'hidden' as const,
});

export const inlineCodeStyles = (theme: ThemeType) => ({
  display: 'inline-block' as const,
  width: 'fit-content',
  maxWidth: '100%',
  overflowWrap: 'anywhere' as const,
  verticalAlign: 'middle' as const,
  fontFamily: 'SFMono-Regular, Consolas, "Liberation Mono", Menlo, Courier, monospace',
  fontSize: theme.typography.fontSizeSm,
  lineHeight: theme.typography.lineHeightSm,
  backgroundColor: theme.colors.backgroundSecondary,
  color: theme.colors.textPrimary,
  padding: `${theme.spacing.xs / 2}px ${theme.spacing.xs}px`,
  borderRadius: theme.borders.borderRadiusSm,
});

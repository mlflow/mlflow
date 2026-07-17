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

export const mcpIconStyles = (theme: ThemeType) => ({
  flexShrink: 0,
  color: theme.colors.textSecondary,
});

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

export const expandableRowButtonStyles = (theme: ThemeType) => ({
  display: 'flex',
  alignItems: 'center',
  width: '100%',
  padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
  background: 'none' as const,
  border: 'none' as const,
  cursor: 'pointer' as const,
  gap: theme.spacing.sm,
  textAlign: 'left' as const,
  '&:hover': {
    backgroundColor: theme.colors.actionTertiaryBackgroundHover,
  },
});

export const chevronContainerStyles = (theme: ThemeType) => ({
  flexShrink: 0,
  width: theme.spacing.md,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
});

export const borderedListContainerStyles = (theme: ThemeType) => ({
  display: 'flex',
  flexDirection: 'column' as const,
  border: `1px solid ${theme.colors.border}`,
  borderRadius: theme.borders.borderRadiusSm,
  overflow: 'hidden' as const,
});

export const borderedListItemStyles = (theme: ThemeType, showTopBorder: boolean) => ({
  padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
  borderTop: showTopBorder ? `1px solid ${theme.colors.border}` : ('none' as const),
});

export const borderedSectionContainerStyles = (theme: ThemeType) => ({
  border: `1px solid ${theme.colors.border}`,
  borderRadius: theme.borders.borderRadiusMd,
  overflow: 'hidden' as const,
});

export const expandedContentPanelStyles = (theme: ThemeType) => ({
  padding: `${theme.spacing.xs}px ${theme.spacing.md}px ${theme.spacing.md}px`,
  paddingLeft: theme.spacing.md + theme.spacing.md + theme.spacing.sm,
  display: 'flex',
  flexDirection: 'column' as const,
  gap: theme.spacing.sm,
  textAlign: 'left' as const,
});

export const popoverTriggerStyles = (theme: ThemeType) => ({
  border: 0,
  background: 'none' as const,
  padding: 0,
  display: 'inline-flex',
  cursor: 'pointer' as const,
  color: theme.colors.textSecondary,
  '&:hover': { color: theme.colors.textPrimary },
});

export const selectedRowIndicatorStyles = (theme: ThemeType) => ({
  width: theme.spacing.md * 2,
  display: 'flex',
  alignItems: 'center',
  paddingRight: theme.spacing.sm,
});

export const inlineFlexRowStyles = (theme: ThemeType) => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: theme.spacing.sm,
  maxWidth: '100%',
});

export const showMoreRowStyles = (theme: ThemeType) => ({
  borderTop: `1px solid ${theme.colors.border}`,
  padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
  textAlign: 'center' as const,
});

export const tagListStyles = (theme: ThemeType) => ({
  display: 'flex',
  flexWrap: 'wrap' as const,
  alignItems: 'center',
  gap: theme.spacing.xs,
});

export const ellipsisStyles = (theme: ThemeType) => ({
  fontSize: theme.typography.fontSizeSm,
  overflow: 'hidden' as const,
  textOverflow: 'ellipsis' as const,
  whiteSpace: 'nowrap' as const,
  flex: 1,
  minWidth: 0,
});

export const lineClampStyles = (lines = 1) => ({
  display: '-webkit-box' as const,
  WebkitLineClamp: lines,
  WebkitBoxOrient: 'vertical' as const,
  overflow: 'hidden' as const,
});

export const sectionHeadingRowStyles = (theme: ThemeType) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing.xs,
  marginBottom: theme.spacing.sm,
});

export const flexRowStyles = (theme: ThemeType) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing.xs,
});

export const fallbackInfoBoxStyles = (theme: ThemeType) => ({
  display: 'flex',
  gap: theme.spacing.sm,
  padding: theme.spacing.sm,
  backgroundColor: theme.colors.backgroundSecondary,
  borderRadius: theme.borders.borderRadiusSm,
  border: `1px solid ${theme.colors.border}`,
});

export const jsonPreStyles = (theme: ThemeType, padding = theme.spacing.sm) =>
  ({
    margin: 0,
    padding,
    paddingTop: theme.spacing.xl,
    backgroundColor: theme.colors.backgroundSecondary,
    borderRadius: theme.borders.borderRadiusSm,
    overflow: 'auto' as const,
    fontSize: theme.typography.fontSizeSm,
    maxHeight: 400,
  }) as const;

export const overlayButtonStyles = (theme: ThemeType) => ({
  position: 'absolute' as const,
  top: theme.spacing.xs,
  right: theme.spacing.xs,
  zIndex: 1,
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

export const monoFontStyles = {
  fontFamily: 'monospace',
};

export const noShrinkStyles = {
  flexShrink: 0,
};

export const blockLabelStyles = (theme: ThemeType) => ({
  display: 'block' as const,
  marginBottom: theme.spacing.xs,
});

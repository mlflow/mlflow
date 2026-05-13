import { ChainIcon, useDesignSystemTheme } from '@databricks/design-system';

export const ExperimentSingleChatIcon = ({ displayLink = false }: { displayLink?: boolean }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        width: theme.general.iconSize,
        height: theme.general.iconSize,
        borderRadius: theme.borders.borderRadiusSm,
        border: `1px solid ${theme.colors.branded.ai.gradientStart}`,
        backgroundColor: theme.colors.backgroundSecondary,
        '::after': displayLink
          ? {
              content: '""',
              position: 'absolute',
              width: 1,
              height: theme.spacing.md,
              backgroundColor: theme.colors.branded.ai.gradientStart,
              top: '100%',
              left: theme.spacing.sm,
            }
          : undefined,
      }}
    >
      <ChainIcon color="ai" />
    </div>
  );
};

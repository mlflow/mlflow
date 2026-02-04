import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { FeatureDefinition } from './feature-definitions';
import { useHomePageViewState } from '../../HomePageViewStateContext';

interface FeatureCardProps {
  feature: FeatureDefinition;
}

export const FeatureCard = ({ feature }: FeatureCardProps) => {
  const { theme } = useDesignSystemTheme();
  const { openLogTracesDrawer } = useHomePageViewState();

  const containerStyles = {
    overflow: 'hidden',
    border: `1px solid ${theme.colors.borderDecorative}`,
    borderRadius: theme.borders.borderRadiusMd,
    background: theme.colors.backgroundPrimary,
    padding: theme.spacing.sm + theme.spacing.xs,
    display: 'flex',
    gap: theme.spacing.sm,
    width: 320,
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
  };

  const iconWrapperStyles = {
    borderRadius: theme.borders.borderRadiusSm,
    background: theme.colors.actionDefaultBackgroundHover,
    padding: theme.spacing.xs,
    color: theme.colors.blue500,
    height: 'min-content',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  };

  const contentStyles = {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: theme.spacing.xs,
    flex: 1,
    minWidth: 0,
  };

  const card = (
    <div css={containerStyles}>
      <div css={iconWrapperStyles}>
        <feature.icon />
      </div>
      <div css={contentStyles}>
        <span role="heading" aria-level={2}>
          <Typography.Text bold>{feature.title}</Typography.Text>
        </span>
        <Typography.Text
          color="secondary"
          size="sm"
          css={{
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
          }}
        >
          {feature.summary}
        </Typography.Text>
      </div>
    </div>
  );

  if (feature.hasDrawer) {
    return (
      <button
        type="button"
        onClick={openLogTracesDrawer}
        css={{
          textDecoration: 'none',
          color: theme.colors.textPrimary,
          display: 'block',
          border: 0,
          padding: 0,
          background: 'transparent',
          cursor: 'pointer',
          font: 'inherit',
          textAlign: 'left',
        }}
      >
        {card}
      </button>
    );
  }

  return (
    <a
      href={feature.docsLink}
      target="_blank"
      rel="noopener noreferrer"
      css={{
        textDecoration: 'none',
        color: theme.colors.textPrimary,
        display: 'block',
      }}
    >
      {card}
    </a>
  );
};

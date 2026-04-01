import { Button, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { FeatureDefinition } from './feature-definitions';
import { useHomePageViewState } from '../../HomePageViewStateContext';

interface FeatureCardProps {
  feature: FeatureDefinition;
  componentId: string;
}

export const FeatureCard = ({ feature, componentId }: FeatureCardProps) => {
  const { theme } = useDesignSystemTheme();
  const { openLogTracesDrawer } = useHomePageViewState();

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

  // Invisible Button overlay that captures clicks and handles telemetry
  // via built-in componentId support. Positioned on top of the card.
  const buttonOverlayStyles = {
    '&&&': {
      position: 'absolute' as const,
      inset: 0,
      opacity: 0,
      width: '100%',
      height: '100%',
      padding: 0,
      cursor: 'pointer',
    },
  };

  const card = (
    <div
      css={{
        overflow: 'hidden',
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusMd,
        background: theme.colors.backgroundPrimary,
        padding: theme.spacing.sm + theme.spacing.xs,
        display: 'flex',
        gap: theme.spacing.sm,
        boxSizing: 'border-box' as const,
        boxShadow: theme.shadows.sm,
        transition: 'background 150ms ease',
        height: '100%',
      }}
    >
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

  const wrapperStyles = {
    position: 'relative' as const,
    flex: 1,
    minWidth: 240,
    cursor: 'pointer',
    '&:hover > div': {
      background: theme.colors.actionDefaultBackgroundHover,
    },
    '&:active > div': {
      background: theme.colors.actionDefaultBackgroundPress,
    },
  };

  if (feature.hasDrawer) {
    return (
      <div css={wrapperStyles}>
        {card}
        <Button componentId={componentId} type="tertiary" onClick={openLogTracesDrawer} css={buttonOverlayStyles} />
      </div>
    );
  }

  return (
    <div css={wrapperStyles}>
      {card}
      <Button
        componentId={componentId}
        type="tertiary"
        href={feature.docsLink}
        target="_blank"
        rel="noopener noreferrer"
        css={buttonOverlayStyles}
      />
    </div>
  );
};

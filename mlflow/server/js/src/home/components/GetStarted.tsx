import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { homeQuickActions } from '../quick-actions';
import { useHomePageViewState } from '../HomePageViewStateContext';

type QuickAction = typeof homeQuickActions[number];

const GetStartedCard = ({ action }: { action: QuickAction }) => {
  const { theme } = useDesignSystemTheme();
  const linkStyles = {
    textDecoration: 'none',
    color: theme.colors.textPrimary,
    display: 'block',
  };
  const containerStyles = {
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
  };
  const contentStyles = {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: theme.spacing.xs,
    flex: 1,
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
  };

  const card = (
    <div css={containerStyles}>
      <div css={iconWrapperStyles}>
        <action.icon />
      </div>
      <div css={contentStyles}>
        <span role="heading" aria-level={2}>
          <Typography.Text bold>{action.title}</Typography.Text>
        </span>
        <Typography.Text color="secondary">{action.description}</Typography.Text>
      </div>
    </div>
  );

  const { openLogTracesDrawer } = useHomePageViewState();

  if (action.id === 'log-traces') {
    return (
      <button
        type="button"
        onClick={openLogTracesDrawer}
        css={{
          ...linkStyles,
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
    <a href={action.link} target="_blank" rel="noopener noreferrer" css={linkStyles}>
      {card}
    </a>
  );
};

export const GetStarted = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Typography.Title level={3} css={{ margin: 0 }}>
        <FormattedMessage defaultMessage="Get started" description="Home page quick action section title" />
      </Typography.Title>
      <section
        css={{
          marginBottom: 20,
          width: '100%',
          minWidth: 0,
          display: 'flex',
          gap: theme.spacing.sm + theme.spacing.xs,
          flexWrap: 'wrap',
        }}
      >
        {homeQuickActions.map((action) => (
          <GetStartedCard key={action.id} action={action} />
        ))}
      </section>
    </section>
  );
};

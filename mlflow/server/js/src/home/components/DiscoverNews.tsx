import type { ComponentType, ReactNode } from 'react';
import { Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { homeNewsItems } from '../news-items';

type NewsThumbnailProps = {
  gradient: {
    light: string;
    dark: string;
  };
  title: ReactNode;
  description: ReactNode;
  icon?: ComponentType<{ className?: string; css?: any }>;
};

const NewsThumbnail = ({ gradient, title, description, icon: IconComponent }: NewsThumbnailProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        borderRadius: theme.borders.borderRadiusMd,
        aspectRatio: '16 / 9',
        background: theme.isDarkMode ? gradient.dark : gradient.light,
        display: 'flex',
        flexDirection: 'column',
        padding: theme.spacing.md,
        gap: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        <Typography.Text bold color="primary">
          {title}
        </Typography.Text>
        {description ? <Typography.Text color="secondary">{description}</Typography.Text> : <Spacer size="sm" />}
      </div>
      {IconComponent && (
        <div css={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'flex-end', flex: 1 }}>
          <IconComponent css={{ color: theme.colors.white, fontSize: 24 }} />
        </div>
      )}
    </div>
  );
};

const DiscoverNewsCard = ({ title, description, link, thumbnail }: typeof homeNewsItems[number]) => {
  const { theme } = useDesignSystemTheme();
  const linkStyles = {
    textDecoration: 'none',
    color: theme.colors.textPrimary,
    display: 'block',
  };

  const card = (
    <div
      css={{
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
      }}
    >
      <NewsThumbnail gradient={thumbnail.gradient} title={title} description={description} icon={thumbnail.icon} />
    </div>
  );

  return (
    <a href={link} target="_blank" rel="noopener noreferrer" css={linkStyles}>
      {card}
    </a>
  );
};

export const DiscoverNews = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.md,
        }}
      >
        <Typography.Title level={3} css={{ margin: 0 }}>
          <FormattedMessage defaultMessage="Discover new features" description="Home page news section title" />
        </Typography.Title>
        <Typography.Link
          componentId="mlflow.home.news.view_more"
          href="https://mlflow.org/blog/"
          openInNewTab
          css={{ color: theme.colors.textSecondary }}
        >
          <FormattedMessage defaultMessage="View all" description="Home page news section view more link" />
        </Typography.Link>
      </div>
      <section css={{ width: '100%', minWidth: 0 }}>
        <div
          css={{
            width: '100%',
            display: 'flex',
            gap: theme.spacing.sm + theme.spacing.xs,
            flexWrap: 'wrap',
          }}
        >
          {homeNewsItems.map((item) => (
            <DiscoverNewsCard key={item.id} {...item} />
          ))}
        </div>
      </section>
    </section>
  );
};

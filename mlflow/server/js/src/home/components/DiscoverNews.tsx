import type { ComponentType, ReactNode } from 'react';
import { Spacer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../common/utils/RoutingUtils';
import { homeNewsItems } from '../news-items';
import { sectionHeaderStyles, discoverNewsCardContainerStyles, getStartedCardLinkStyles } from './cardStyles';

type NewsThumbnailProps = {
  gradient: string;
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
        background: gradient,
        display: 'flex',
        flexDirection: 'column',
        padding: theme.spacing.md,
        gap: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        <Typography.Text strong color="primary">
          {title}
        </Typography.Text>
        {description ? (
          <Typography.Text color="secondary">{description}</Typography.Text>
        ) : (
          <Spacer size="sm" />
        )}
      </div>
      {IconComponent && (
        <div css={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'flex-end', flex: 1 }}>
          <IconComponent css={{ color: theme.colors.white, fontSize: 24 }} />
        </div>
      )}
    </div>
  );
};

const DiscoverNewsCard = ({
  title,
  description,
  link,
  componentId,
  thumbnail,
}: (typeof homeNewsItems)[number]) => {
  const { theme } = useDesignSystemTheme();
  const linkStyles = getStartedCardLinkStyles(theme);
  const containerStyles = discoverNewsCardContainerStyles(theme);

  const card = (
    <div css={containerStyles}>
      <NewsThumbnail
        gradient={thumbnail.gradient}
        title={title}
        description={description}
        icon={thumbnail.icon}
      />
    </div>
  );

  if (link.type === 'internal') {
    return (
      <Link to={link.to} css={linkStyles} data-component-id={componentId}>
        {card}
      </Link>
    );
  }

  return (
    <a
      href={link.href}
      target={link.target ?? '_blank'}
      rel={link.rel ?? 'noopener noreferrer'}
      css={linkStyles}
      data-component-id={componentId}
    >
      {card}
    </a>
  );
};

export const DiscoverNews = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Typography.Title level={3} css={sectionHeaderStyles}>
        <FormattedMessage defaultMessage="Discover new features" description="Home page news section title" />
      </Typography.Title>
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
      <Typography.Link
        componentId="mlflow.home.news.view_more"
        href="https://mlflow.org/blog/"
        target="_blank"
        rel="noopener noreferrer"
        css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs, alignSelf: 'flex-end' }}
      >
        <span>{'>>>'}</span>
        <FormattedMessage defaultMessage="See more announcements" description="Home page news section view more link" />
      </Typography.Link>
    </section>
  );
};

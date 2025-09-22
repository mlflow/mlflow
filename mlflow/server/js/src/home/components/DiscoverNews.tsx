import { ArrowRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../common/utils/RoutingUtils';
import { homeNewsItems } from '../news-items';
import { cardBaseStyles, cardCtaStyles, sectionHeaderStyles } from './cardStyles';

const NewsThumbnail = ({ gradient, label }: { gradient: string; label: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        borderRadius: theme.borders.borderRadiusMd,
        aspectRatio: '16 / 9',
        background: gradient,
        display: 'flex',
        alignItems: 'flex-end',
        padding: theme.spacing.md,
        color: theme.colors.textSecondary,
        fontWeight: theme.typography.typographyMediumFontWeight,
      }}
    >
      <span>{label}</span>
    </div>
  );
};

const DiscoverNewsCard = ({
  title,
  description,
  ctaLabel,
  link,
  componentId,
  thumbnail,
}: (typeof homeNewsItems)[number]) => {
  const { theme } = useDesignSystemTheme();
  const baseStyles = cardBaseStyles(theme);
  const ctaStyles = cardCtaStyles(theme);

  const content = (
    <>
      <NewsThumbnail gradient={thumbnail.gradient} label={thumbnail.label} />
      <Typography.Title level={4} css={{ margin: 0 }}>
        {title}
      </Typography.Title>
      <Typography.Text css={{ color: theme.colors.textSecondary }}>{description}</Typography.Text>
      <span css={{ ...ctaStyles, marginTop: 'auto' }}>
        {ctaLabel}
        <ArrowRightIcon css={{ width: 16, height: 16 }} />
      </span>
    </>
  );

  if (link.type === 'internal') {
    return (
      <Link to={link.to} css={baseStyles} data-component-id={componentId}>
        {content}
      </Link>
    );
  }

  return (
    <a
      href={link.href}
      target={link.target ?? '_blank'}
      rel={link.rel ?? 'noopener noreferrer'}
      css={baseStyles}
      data-component-id={componentId}
    >
      {content}
    </a>
  );
};

export const DiscoverNews = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Typography.Title level={3} css={sectionHeaderStyles}>
        <FormattedMessage defaultMessage="Explore news" description="Home page news section title" />
      </Typography.Title>
      <div
        css={{
          display: 'grid',
          gap: theme.spacing.lg,
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        }}
      >
        {homeNewsItems.map((item) => (
          <DiscoverNewsCard key={item.id} {...item} />
        ))}
      </div>
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

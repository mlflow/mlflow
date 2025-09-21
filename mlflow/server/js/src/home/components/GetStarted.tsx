import { ArrowRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../common/utils/RoutingUtils';
import { homeQuickActions } from '../quick-actions';
import { cardBaseStyles, cardCtaStyles, sectionHeaderStyles } from './cardStyles';

type QuickAction = (typeof homeQuickActions)[number];

const GetStartedCard = ({ action }: { action: QuickAction }) => {
  const { theme } = useDesignSystemTheme();
  const baseStyles = cardBaseStyles(theme);
  const ctaStyles = cardCtaStyles(theme);

  const content = (
    <>
      <action.icon css={{ width: 36, height: 36 }} />
      <Typography.Title level={4} css={{ margin: 0 }}>
        {action.title}
      </Typography.Title>
      <Typography.Text css={{ color: theme.colors.textSecondary }}>{action.description}</Typography.Text>
      <span css={{ ...ctaStyles, marginTop: 'auto' }}>
        {action.ctaLabel}
        <ArrowRightIcon css={{ width: 16, height: 16 }} />
      </span>
    </>
  );

  if (action.link.type === 'internal') {
    return (
      <Link to={action.link.to} css={baseStyles} data-component-id={action.componentId}>
        {content}
      </Link>
    );
  }

  return (
    <a
      href={action.link.href}
      target={action.link.target ?? '_blank'}
      rel={action.link.rel ?? 'noopener noreferrer'}
      css={baseStyles}
      data-component-id={action.componentId}
    >
      {content}
    </a>
  );
};

export const GetStarted = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Typography.Title level={3} css={sectionHeaderStyles}>
        <FormattedMessage defaultMessage="Get started" description="Home page quick action section title" />
      </Typography.Title>
      <div
        css={{
          display: 'grid',
          gap: theme.spacing.lg,
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        }}
      >
        {homeQuickActions.map((action) => (
          <GetStartedCard key={action.id} action={action} />
        ))}
      </div>
    </section>
  );
};

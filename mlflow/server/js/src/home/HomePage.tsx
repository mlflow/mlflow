import { useMemo, useState } from 'react';
import {
  ArrowRightIcon,
  Button,
  ParagraphSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { ScrollablePageWrapper } from '../common/components/ScrollablePageWrapper';
import { Link } from '../common/utils/RoutingUtils';
import ExperimentTrackingRoutes from '../experiment-tracking/routes';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../experiment-tracking/sdk/MlflowService';
import type { SearchExperimentsApiResponse } from '../experiment-tracking/types';
import Utils from '../common/utils/Utils';
import { CreateExperimentModal } from '../experiment-tracking/components/modals/CreateExperimentModal';
import { useInvalidateExperimentList } from '../experiment-tracking/components/experiment-page/hooks/useExperimentListQuery';
import type { HomeExperimentRow } from './types';
import { homeQuickActions } from './quick-actions';
import { homeNewsItems } from './news-items';

type ExperimentQueryKey = ['home', 'recent-experiments'];

const RECENT_EXPERIMENTS_QUERY_KEY: ExperimentQueryKey = ['home', 'recent-experiments'];

const mapExperimentsToRows = (
  experiments: SearchExperimentsApiResponse['experiments'] | undefined,
): HomeExperimentRow[] => {
  if (!experiments) {
    return [];
  }
  return experiments
    .slice(0, 5)
    .map(({ experimentId, name, creationTime, lastUpdateTime, tags }) => ({
      experimentId,
      name,
      creationTime,
      lastUpdateTime,
      description: tags?.find(({ key }) => key === 'mlflow.note.content')?.value,
    }));
};

const cardCtaStyles = (theme: ReturnType<typeof useDesignSystemTheme>['theme']) => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: theme.spacing.xs,
  color: theme.colors.actionPrimaryTextDefault,
  fontWeight: theme.typography.typographyBoldFontWeight,
});

const sectionHeaderStyles = { margin: 0 };

const cardBaseStyles = (theme: ReturnType<typeof useDesignSystemTheme>['theme']) => ({
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing.sm,
  padding: `${theme.spacing.xl}px ${theme.spacing.lg}px`,
  borderRadius: theme.borders.borderRadiusLg,
  border: `1px solid ${theme.colors.borderStrong}`,
  backgroundColor: theme.colors.backgroundPrimary,
  boxShadow: theme.shadows.md,
  textDecoration: 'none',
  color: theme.colors.textPrimary,
  transition: 'transform 150ms ease, box-shadow 150ms ease',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows.md,
    textDecoration: 'none',
  },
  '&:focus-visible': {
    outline: `2px solid ${theme.colors.actionPrimaryBackgroundHover}`,
    outlineOffset: 2,
  },
});

const NewsCardWrapper = ({
  children,
  link,
  componentId,
}: {
  children: React.ReactNode;
  link: { type: 'internal'; to: string } | { type: 'external'; href: string; target?: string; rel?: string };
  componentId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const baseStyles = cardBaseStyles(theme);

  if (link.type === 'internal') {
    return (
      <Link to={link.to} css={baseStyles} data-component-id={componentId}>
        {children}
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
      {children}
    </a>
  );
};

const QuickActionCard = ({
  action,
}: {
  action: (typeof homeQuickActions)[number];
}) => {
  const { theme } = useDesignSystemTheme();
  const intlCtaStyles = cardCtaStyles(theme);
  const baseStyles = cardBaseStyles(theme);

  if (action.link.type === 'internal') {
    return (
      <Link to={action.link.to} css={baseStyles} data-component-id={action.componentId}>
        <action.icon css={{ width: 36, height: 36 }} />
        <Typography.Title level={4} css={{ margin: 0 }}>
          {action.title}
        </Typography.Title>
        <Typography.Text css={{ color: theme.colors.textSecondary }}>{action.description}</Typography.Text>
        <span css={{ ...intlCtaStyles, marginTop: 'auto' }}>
          {action.ctaLabel}
          <ArrowRightIcon css={{ width: 16, height: 16 }} />
        </span>
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
      <action.icon css={{ width: 36, height: 36 }} />
      <Typography.Title level={4} css={{ margin: 0 }}>
        {action.title}
      </Typography.Title>
      <Typography.Text css={{ color: theme.colors.textSecondary }}>{action.description}</Typography.Text>
      <span css={{ ...intlCtaStyles, marginTop: 'auto' }}>
        {action.ctaLabel}
        <ArrowRightIcon css={{ width: 16, height: 16 }} />
      </span>
    </a>
  );
};

const ExperimentsTableSkeleton = () => {
  const rows = Array.from({ length: 3 });
  const { theme } = useDesignSystemTheme();
  return (
    <tbody>
      {rows.map((_, idx) => (
        <tr key={idx}>
          {Array.from({ length: 4 }).map((__, cellIdx) => (
            <td
              key={cellIdx}
              css={{ padding: theme.spacing.sm, borderBottom: `1px solid ${theme.colors.borderSubtle}` }}
            >
              <ParagraphSkeleton css={{ width: cellIdx === 0 ? '60%' : '40%' }} />
            </td>
          ))}
        </tr>
      ))}
    </tbody>
  );
};

const ExperimentsEmptyState = ({ onCreate }: { onCreate: () => void }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        padding: theme.spacing.xxl,
        textAlign: 'center',
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
      }}
    >
      <Typography.Title level={4} css={{ margin: 0 }}>
        <FormattedMessage defaultMessage="Create your first experiment" description="Home page experiments empty state title" />
      </Typography.Title>
      <Typography.Text css={{ color: theme.colors.textSecondary }}>
        <FormattedMessage
          defaultMessage="Create your first experiment to start tracking ML workflows."
          description="Home page experiments empty state description"
        />
      </Typography.Text>
      <div>
        <Button componentId="mlflow.home.experiments.create" onClick={onCreate}>
          <FormattedMessage defaultMessage="Create experiment" description="Home page experiments empty state CTA" />
        </Button>
      </div>
    </div>
  );
};

const renderExperimentRow = (
  row: HomeExperimentRow,
  { theme, formatted }: { theme: ReturnType<typeof useDesignSystemTheme>['theme']; formatted: ReturnType<typeof useIntl> },
) => (
  <tr
    key={row.experimentId}
    css={{
      '&:hover': {
        backgroundColor: theme.colors.backgroundSecondary,
      },
    }}
  >
    <td css={{ padding: theme.spacing.sm, borderBottom: `1px solid ${theme.colors.borderSubtle}` }}>
      <Typography.Link to={ExperimentTrackingRoutes.getExperimentPageRoute(row.experimentId)} as={Link}>
        {row.name}
      </Typography.Link>
    </td>
    <td css={{ padding: theme.spacing.sm, borderBottom: `1px solid ${theme.colors.borderSubtle}` }}>
      {Utils.formatTimestamp(row.creationTime, formatted)}
    </td>
    <td css={{ padding: theme.spacing.sm, borderBottom: `1px solid ${theme.colors.borderSubtle}` }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        {Utils.formatTimestamp(row.lastUpdateTime, formatted)}
      </div>
    </td>
    <td css={{ padding: theme.spacing.sm, borderBottom: `1px solid ${theme.colors.borderSubtle}` }}>
      {row.description ?? '—'}
    </td>
  </tr>
);

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

const NewsCardContent = ({
  title,
  description,
  ctaLabel,
}: {
  title: React.ReactNode;
  description: React.ReactNode;
  ctaLabel: React.ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <>
      <Typography.Title level={4} css={{ margin: 0 }}>
        {title}
      </Typography.Title>
      <Typography.Text css={{ color: theme.colors.textSecondary }}>{description}</Typography.Text>
      <span css={{ ...cardCtaStyles(theme), marginTop: 'auto' }}>
        {ctaLabel}
        <ArrowRightIcon css={{ width: 16, height: 16 }} />
      </span>
    </>
  );
};

const HomePage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const invalidateExperiments = useInvalidateExperimentList();
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  const { data, isLoading, refetch } = useQuery<SearchExperimentsApiResponse, Error, SearchExperimentsApiResponse, ExperimentQueryKey>(
    RECENT_EXPERIMENTS_QUERY_KEY,
    {
      queryFn: () =>
        MlflowService.searchExperiments([
          ['max_results', '5'],
          ['order_by', 'last_update_time DESC'],
        ]),
      staleTime: 30_000,
    },
  );

  const experimentRows = useMemo(() => mapExperimentsToRows(data?.experiments), [data?.experiments]);

  const handleExperimentCreated = () => {
    setIsCreateModalOpen(false);
    invalidateExperiments();
    refetch();
  };

  const handleCreateClick = () => setIsCreateModalOpen(true);

  return (
    <ScrollablePageWrapper
      css={{
        padding: `${theme.spacing.xl}px ${theme.spacing.xxl}px`,
        display: 'flex',
        justifyContent: 'center',
      }}
    >
      <div
        css={{
          width: '100%',
          maxWidth: 1120,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xxl,
        }}
      >
        <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Title level={1} css={{ margin: 0 }}>
            <FormattedMessage defaultMessage="Welcome to MLflow" description="Home page hero title" />
          </Typography.Title>
          <Typography.Paragraph css={{ margin: 0, color: theme.colors.textSecondary }}>
            <FormattedMessage
              defaultMessage="Get started with experiment tracking, evaluations, tracing, and prompt management."
              description="Home page hero subtitle"
            />
          </Typography.Paragraph>
        </section>

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
              <QuickActionCard key={action.id} action={action} />
            ))}
          </div>
        </section>

        <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <Typography.Title level={3} css={sectionHeaderStyles}>
            <FormattedMessage defaultMessage="Experiments" description="Home page experiments preview title" />
          </Typography.Title>
          <div
            css={{
              border: `1px solid ${theme.colors.borderStrong}`,
              borderRadius: theme.borders.borderRadiusLg,
              overflow: 'hidden',
              backgroundColor: theme.colors.backgroundPrimary,
              boxShadow: theme.shadows.md,
            }}
          >
            <table
              css={{
                width: '100%',
                borderCollapse: 'collapse',
              }}
            >
              <thead>
                <tr css={{ backgroundColor: theme.colors.backgroundSecondary }}>
                  <th css={{ padding: theme.spacing.sm, textAlign: 'left' }}>
                    <FormattedMessage defaultMessage="Name" description="Home page experiments table column header name" />
                  </th>
                  <th css={{ padding: theme.spacing.sm, textAlign: 'left' }}>
                    <FormattedMessage defaultMessage="Time created" description="Home page experiments table column header time created" />
                  </th>
                  <th css={{ padding: theme.spacing.sm, textAlign: 'left' }}>
                    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                      <FormattedMessage defaultMessage="Last modified" description="Home page experiments table column header last modified" />
                    </div>
                  </th>
                  <th css={{ padding: theme.spacing.sm, textAlign: 'left' }}>
                    <FormattedMessage defaultMessage="Description" description="Home page experiments table column header description" />
                  </th>
                </tr>
              </thead>
              {isLoading ? (
                <ExperimentsTableSkeleton />
              ) : experimentRows.length > 0 ? (
                <tbody>
                  {experimentRows.map((row) => renderExperimentRow(row, { theme, formatted: intl }))}
                </tbody>
              ) : (
                <tbody>
                  <tr>
                    <td colSpan={4}>
                      <ExperimentsEmptyState onCreate={handleCreateClick} />
                    </td>
                  </tr>
                </tbody>
              )}
            </table>
          </div>
          <Typography.Link
            componentId="mlflow.home.experiments.view_all"
            to={ExperimentTrackingRoutes.experimentsObservatoryRoute}
            as={Link}
            css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}
          >
            <span>{'>>>'}</span>
            <FormattedMessage defaultMessage="See all experiments" description="Home page experiments view all link" />
          </Typography.Link>
        </section>

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
            {homeNewsItems.map(({ id, title, description, ctaLabel, link, componentId, thumbnail }) => (
              <NewsCardWrapper key={id} link={link} componentId={componentId}>
                <NewsThumbnail gradient={thumbnail.gradient} label={thumbnail.label} />
                <NewsCardContent title={title} description={description} ctaLabel={ctaLabel} />
              </NewsCardWrapper>
            ))}
          </div>
          <Typography.Link
            componentId="mlflow.home.news.view_more"
            href="https://mlflow.org/blog/"
            target="_blank"
            rel="noopener noreferrer"
            css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}
          >
            <span>{'>>>'}</span>
            <FormattedMessage defaultMessage="See more announcements" description="Home page news section view more link" />
          </Typography.Link>
        </section>
      </div>

      <CreateExperimentModal
        isOpen={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onExperimentCreated={handleExperimentCreated}
      />
    </ScrollablePageWrapper>
  );
};

export default HomePage;

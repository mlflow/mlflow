import { useMemo, useRef } from 'react';
import type { ReactNode } from 'react';
import invariant from 'invariant';
import { Alert, ParagraphSkeleton, TitleSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { PredefinedError } from '@databricks/web-shared/errors';
import { getGraphQLErrorMessage } from '../../../graphql/get-graphql-error';
import { Link, useLocation, useParams } from '../../../common/utils/RoutingUtils';
import { SETTINGS_RETURN_TO_PARAM, SETTINGS_SECTION_GENERAL } from '../../../settings/settingsSectionConstants';
import Routes from '../../routes';
import type { ExperimentEntity } from '../../types';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { useExperimentPageRevampContext } from '../experiment-page-tabs/ExperimentPageRevampContext';
import { ExperimentSettingsContent } from './ExperimentSettingsContent';
import { SETTING_CELL_CLASS_NAME } from './SettingCell';

const SETTINGS_MAX_WIDTH = '80ch';

const ExperimentSettingsPageContainer = ({
  children,
  revampEnabled,
}: {
  children: ReactNode;
  revampEnabled: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: revampEnabled ? 0 : theme.spacing.md }}>
      <div
        data-testid="experiment-settings-content"
        css={{
          display: 'flex',
          flexDirection: 'column',
          width: '100%',
          marginInline: 0,
          paddingBottom: theme.spacing.lg,
          boxSizing: 'border-box',
          '& > section + section': {
            borderBlockStart: `1px solid ${theme.colors.border}`,
            marginBlockStart: theme.spacing.lg,
            paddingBlockStart: theme.spacing.lg,
          },
          '& [data-settings-section-content] > h3': {
            maxWidth: SETTINGS_MAX_WIDTH,
          },
          '& [data-settings-section-content] > div': {
            borderBlockEnd: `1px solid ${theme.colors.border}`,
            maxWidth: SETTINGS_MAX_WIDTH,
          },
          '& [data-settings-section-content] > div:last-child': {
            borderBlockEnd: 0,
          },
          [`& [data-settings-section-content] > div > .${SETTING_CELL_CLASS_NAME}`]: {
            borderBlockEnd: 0,
          },
        }}
      >
        {children}
      </div>
    </div>
  );
};

/**
 * Says whose settings these are, so they aren't mistaken for MLflow's own Settings page, and links there.
 * The title is only needed in the GenAI layout: the Model training layout's header already shows it.
 */
const ExperimentSettingsIntro = ({ showTitle }: { showTitle: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const { pathname, search } = useLocation();
  // Like the sidebar's Settings link, so Settings can send the user back here.
  const mlflowSettingsRoute = `${Routes.getSettingsSectionRoute(SETTINGS_SECTION_GENERAL)}?${SETTINGS_RETURN_TO_PARAM}=${encodeURIComponent(pathname + search)}`;

  return (
    <header
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.xs,
        marginBlockEnd: theme.spacing.lg,
        maxWidth: SETTINGS_MAX_WIDTH,
      }}
    >
      {showTitle && (
        <Typography.Title level={2} withoutMargins>
          <FormattedMessage defaultMessage="Experiment settings" description="Title of the experiment Settings page" />
        </Typography.Title>
      )}
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="These settings apply to this experiment only. For settings across MLflow, go to <link>Settings</link>."
          description="Subtitle of the experiment Settings page, linking to MLflow's own Settings page"
          values={{
            link: (chunks: ReactNode) => (
              <Link componentId="mlflow.experiment_settings.mlflow_settings_link" to={mlflowSettingsRoute}>
                {chunks}
              </Link>
            ),
          }}
        />
      </Typography.Text>
    </header>
  );
};

const ExperimentSettingsSkeleton = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <TitleSkeleton seed="experiment-settings-title" />
      {Array.from({ length: 5 }, (_, index) => (
        <ParagraphSkeleton key={index} seed={`experiment-settings-${index}`} />
      ))}
    </div>
  );
};

export const ExperimentSettingsPage = () => {
  const { experimentId } = useParams();
  const { enabled: revampEnabled } = useExperimentPageRevampContext();
  invariant(experimentId, 'Experiment ID must be defined');

  // Load the latest experiment every time Settings opens. Some edits elsewhere, such as the experiments
  // list's tag editor, don't refresh the cached copy the experiment page would otherwise reuse.
  const { data, loading, refetch, apiError, apolloError } = useGetExperimentQuery({
    experimentId,
    options: { fetchPolicy: 'network-only' },
  });
  const error = apiError ?? apolloError;
  // The experiment page's own copy, from the cache. A network failure in Settings' separate request
  // can leave that page loaded and showing no error, so Settings has to report it itself.
  const { data: pageExperiment } = useGetExperimentQuery({
    experimentId,
    options: { fetchPolicy: 'cache-only' },
  });
  // Keep what this page last loaded, so a refresh that fails after a save doesn't replace the page with an error.
  // It's only used for the same experiment: the page stays mounted when the route changes to another one.
  const lastLoaded = useRef<{ experimentId: string; data: typeof data }>();
  if (data) {
    lastLoaded.current = { experimentId, data };
  }
  const shownData = data ?? (lastLoaded.current?.experimentId === experimentId ? lastLoaded.current.data : undefined);
  const experiment = useMemo(
    () =>
      shownData
        ? ({
            ...shownData,
            creationTime: Number(shownData.creationTime),
            lastUpdateTime: Number(shownData.lastUpdateTime),
          } as ExperimentEntity)
        : null,
    [shownData],
  );
  if (error instanceof PredefinedError) {
    throw error;
  }
  if (loading && !experiment) {
    return (
      <ExperimentSettingsPageContainer revampEnabled={revampEnabled}>
        <ExperimentSettingsSkeleton />
      </ExperimentSettingsPageContainer>
    );
  }

  const errorMessage = experiment ? undefined : getGraphQLErrorMessage(error);
  if (errorMessage && apolloError && pageExperiment) {
    return (
      <ExperimentSettingsPageContainer revampEnabled={revampEnabled}>
        <Alert
          componentId="mlflow.experiment_settings.load_error"
          type="error"
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="Experiment load error: {errorMessage}"
              description="Error shown when experiment Settings cannot load the experiment"
              values={{ errorMessage }}
            />
          }
        />
      </ExperimentSettingsPageContainer>
    );
  }
  // Otherwise the experiment page around every tab already shows this load error.
  if (errorMessage) {
    return null;
  }
  if (!experiment) {
    return null;
  }

  return (
    <ExperimentSettingsPageContainer revampEnabled={revampEnabled}>
      <ExperimentSettingsIntro showTitle={revampEnabled} />
      <ExperimentSettingsContent experiment={experiment} refetchExperiment={refetch} />
    </ExperimentSettingsPageContainer>
  );
};

export default ExperimentSettingsPage;

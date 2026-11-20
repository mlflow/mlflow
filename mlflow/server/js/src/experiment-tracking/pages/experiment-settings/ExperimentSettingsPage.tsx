import { useMemo, useRef } from 'react';
import type { ReactNode } from 'react';
import invariant from 'invariant';
import { Alert, ParagraphSkeleton, TitleSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { PredefinedError } from '@databricks/web-shared/errors';
import { getGraphQLErrorMessage } from '../../../graphql/get-graphql-error';
import { useParams } from '../../../common/utils/RoutingUtils';
import type { ExperimentEntity } from '../../types';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { useExperimentPermission } from '../../hooks/useExperimentPermission';
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
  const lastLoadedData = useRef(data);
  if (data) {
    lastLoadedData.current = data;
  }
  const shownData = data ?? lastLoadedData.current;
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
  // Look up permissions alongside the experiment, so the controls don't render disabled and then enable.
  useExperimentPermission(experimentId, ['EDIT', 'MANAGE']);

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
      <ExperimentSettingsContent experiment={experiment} refetchExperiment={refetch} />
    </ExperimentSettingsPageContainer>
  );
};

export default ExperimentSettingsPage;

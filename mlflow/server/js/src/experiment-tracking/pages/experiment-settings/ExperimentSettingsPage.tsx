import { useCallback, useRef } from 'react';
import type { ReactNode } from 'react';
import invariant from 'invariant';
import { Alert, ParagraphSkeleton, TitleSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { PredefinedError } from '@databricks/web-shared/errors';
import { FormattedMessage } from 'react-intl';
import { getGraphQLErrorMessage } from '../../../graphql/get-graphql-error';
import { useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import { ExperimentKind } from '../../constants';
import Routes from '../../routes';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { useExperimentKind } from '../../utils/ExperimentKindUtils';
import { getExperimentEntityFromQuery } from '../../components/experiment-page/utils/getExperimentEntityFromQuery';
import { useUpdateExperimentKind } from '../../components/experiment-page/hooks/useUpdateExperimentKind';
import { ExperimentViewHeaderKindSelector } from '../../components/experiment-page/components/header/ExperimentViewHeaderKindSelector';
import { useExperimentPageRevampContext } from '../experiment-page-tabs/ExperimentPageRevampContext';
import { ExperimentSettingsContent } from './ExperimentSettingsManagementActions';

const SETTINGS_MAX_WIDTH = '80ch';

export interface ExperimentSettingsPageContainerProps {
  children: ReactNode;
  revampEnabled: boolean;
}

const ExperimentSettingsPageContainer = ({ children, revampEnabled }: ExperimentSettingsPageContainerProps) => {
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
          '& [data-settings-section-content] > div > .__SettingCell__': {
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
  const navigate = useNavigate();
  const { enabled: revampEnabled, inferredExperimentKind } = useExperimentPageRevampContext();
  invariant(experimentId, 'Experiment ID must be defined');

  const { data, loading, refetch, apiError, apolloError } = useGetExperimentQuery({ experimentId });
  const error = apiError ?? apolloError;
  const experiment = getExperimentEntityFromQuery(data);
  const experimentKind = useExperimentKind(experiment?.tags);
  const updatedExperimentKind = useRef<ExperimentKind>();
  const onExperimentKindUpdated = useCallback(async () => {
    await refetch();
    if (updatedExperimentKind.current === ExperimentKind.CUSTOM_MODEL_DEVELOPMENT) {
      navigate(Routes.getExperimentPageRoute(experimentId), { replace: true });
    }
  }, [experimentId, navigate, refetch]);
  const { mutate: updateExperimentKind, isLoading: isUpdatingExperimentKind } =
    useUpdateExperimentKind(onExperimentKindUpdated);

  if (error instanceof PredefinedError) {
    throw error;
  }
  if (loading) {
    return (
      <ExperimentSettingsPageContainer revampEnabled={revampEnabled}>
        <ExperimentSettingsSkeleton />
      </ExperimentSettingsPageContainer>
    );
  }

  const errorMessage = getGraphQLErrorMessage(error);
  if (errorMessage) {
    return (
      <ExperimentSettingsPageContainer revampEnabled={revampEnabled}>
        <Alert
          componentId="mlflow.experiment_settings.load_error"
          type="error"
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="Experiment load error: {errorMessage}"
              description="Error shown when experiment Settings cannot load"
              values={{ errorMessage }}
            />
          }
        />
      </ExperimentSettingsPageContainer>
    );
  }
  if (!experiment) {
    return null;
  }

  return (
    <ExperimentSettingsPageContainer revampEnabled={revampEnabled}>
      <ExperimentSettingsContent
        experiment={experiment}
        experimentKindSelector={
          <ExperimentViewHeaderKindSelector
            value={experimentKind}
            inferredExperimentKind={inferredExperimentKind}
            onChange={(kind) => {
              updatedExperimentKind.current = kind;
              updateExperimentKind({ experimentId, kind });
            }}
            isUpdating={isUpdatingExperimentKind}
          />
        }
        refetchExperiment={refetch}
      />
    </ExperimentSettingsPageContainer>
  );
};

export default ExperimentSettingsPage;

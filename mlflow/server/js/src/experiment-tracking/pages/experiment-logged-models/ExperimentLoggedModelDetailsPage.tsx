import { Alert, PageWrapper, TableSkeleton, useDesignSystemTheme, Spacer } from '@databricks/design-system';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { ExperimentLoggedModelDetailsHeader } from '../../components/experiment-logged-models/ExperimentLoggedModelDetailsHeader';
import { ExperimentLoggedModelPageWrapper } from './ExperimentLoggedModelPageWrapper';
import { ExperimentLoggedModelDetailsNav } from '../../components/experiment-logged-models/ExperimentLoggedModelDetailsNav';
import { ExperimentLoggedModelDetailsOverview } from '../../components/experiment-logged-models/ExperimentLoggedModelDetailsOverview';
import { useGetLoggedModelQuery } from '../../hooks/logged-models/useGetLoggedModelQuery';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import React from 'react';
import { ExperimentLoggedModelDetailsArtifacts } from '../../components/experiment-logged-models/ExperimentLoggedModelDetailsArtifacts';
import { useUserActionErrorHandler } from '@databricks/web-shared/metrics';
import { FormattedMessage } from 'react-intl';
import { ExperimentLoggedModelDetailsTraces } from '../../components/experiment-logged-models/ExperimentLoggedModelDetailsTraces';
import { getExperimentKindFromTags } from '../../utils/ExperimentKindUtils';

/**
 * Temporary "in construction" placeholder box, to be removed after implementing the actual content.
 */
const PlaceholderBox = ({ children }: { children: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        paddingLeft: theme.spacing.md,
        border: `4px dashed ${theme.colors.border}`,
        marginBottom: theme.spacing.md,
        flex: 1,
      }}
    >
      {children}
    </div>
  );
};

const ExperimentLoggedModelDetailsPageImpl = () => {
  const { experimentId, loggedModelId, tabName } = useParams();
  const { clearUserActionError, currentUserActionError } = useUserActionErrorHandler();

  invariant(experimentId, 'Experiment ID must be defined');
  invariant(loggedModelId, 'Logged model ID must be defined');

  const { theme } = useDesignSystemTheme();
  const {
    data: loggedModel,
    isLoading: loggedModelLoading,
    error: loggedModelLoadError,
    refetch,
  } = useGetLoggedModelQuery({ loggedModelId });
  const {
    data: experimentData,
    loading: experimentLoading,
    apiError: experimentApiError,
    apolloError: experimentApolloError,
  } = useGetExperimentQuery({ experimentId });

  // If there is an unrecoverable error loading the model, throw it to be handled by the error boundary
  if (loggedModelLoadError) {
    throw loggedModelLoadError;
  }

  const experimentLoadError = experimentApiError ?? experimentApolloError;

  const renderSelectedTab = () => {
    if (loggedModelLoading) {
      return <TableSkeleton lines={12} />;
    }

    // TODO: implement error handling
    if (!loggedModel) {
      return null;
    }

    if (tabName === 'traces') {
      return <ExperimentLoggedModelDetailsTraces loggedModel={loggedModel} />;
    } else if (tabName === 'artifacts') {
      return <ExperimentLoggedModelDetailsArtifacts loggedModel={loggedModel} />;
    }

    const experiment = experimentData;
    const experimentKind = getExperimentKindFromTags(experiment?.tags);

    return (
      <ExperimentLoggedModelDetailsOverview
        onDataUpdated={refetch}
        loggedModel={loggedModel}
        experimentKind={experimentKind}
      />
    );
  };

  return (
    <>
      <ExperimentLoggedModelDetailsHeader
        experimentId={experimentId}
        experiment={experimentData}
        loggedModel={loggedModel}
        loading={loggedModelLoading || experimentLoading}
        onSuccess={refetch}
      />
      {currentUserActionError && (
        <Alert
          componentId="mlflow.logged_model.details.user-action-error"
          css={{ marginBottom: theme.spacing.sm }}
          type="error"
          message={currentUserActionError.displayMessage ?? currentUserActionError.message}
          onClose={clearUserActionError}
        />
      )}
      {experimentLoadError?.message && (
        <Alert
          componentId="mlflow.logged_model.details.experiment-error"
          css={{ marginBottom: theme.spacing.sm }}
          type="error"
          message={
            <FormattedMessage
              defaultMessage="Experiment load error: {errorMessage}"
              description="Error message displayed on logged models page when experiment data fails to load"
              values={{ errorMessage: experimentLoadError.message }}
            />
          }
          closable={false}
        />
      )}
      <ExperimentLoggedModelDetailsNav experimentId={experimentId} modelId={loggedModelId} activeTabName={tabName} />
      <div css={{ overflow: 'auto', flex: 1 }}>{renderSelectedTab()}</div>
    </>
  );
};

const ExperimentLoggedModelDetailsPage = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <ExperimentLoggedModelPageWrapper>
      <PageWrapper
        css={{
          display: 'flex',
          overflow: 'hidden',
          height: '100%',
          flexDirection: 'column',
        }}
      >
        <Spacer shrinks={false} />
        <ExperimentLoggedModelDetailsPageImpl />
        <Spacer shrinks={false} />
      </PageWrapper>
    </ExperimentLoggedModelPageWrapper>
  );
};

export default ExperimentLoggedModelDetailsPage;

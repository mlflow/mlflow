import React, { useState } from 'react';
import { useDesignSystemTheme, ParagraphSkeleton, Button, PlusIcon, Spacer } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import ScorerCardContainer from './ScorerCardContainer';
import ScorerModalRenderer from './ScorerModalRenderer';
import ScorerEmptyStateRenderer from './ScorerEmptyStateRenderer';
import { useGetScheduledScorers } from './hooks/useGetScheduledScorers';
import { COMPONENT_ID_PREFIX, SCORER_FORM_MODE } from './constants';

interface ExperimentScorersContentContainerProps {
  experimentId: string;
}

const ExperimentScorersContentContainer: React.FC<ExperimentScorersContentContainerProps> = ({ experimentId }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isModalVisible, setIsModalVisible] = useState(false);

  const scheduledScorersResult = useGetScheduledScorers(experimentId);
  const scorers = scheduledScorersResult.data?.scheduledScorers || [];

  const handleNewScorerClick = () => {
    setIsModalVisible(true);
  };

  // If no scorers exist and we're not currently showing the modal, show empty state
  const shouldShowEmptyState = scorers.length === 0 && !isModalVisible && !scheduledScorersResult.isLoading;

  const closeModal = () => {
    setIsModalVisible(false);
  };

  // Handle error state - throw error to be caught by PanelBoundary
  if (scheduledScorersResult.isError && scheduledScorersResult.error) {
    throw scheduledScorersResult.error;
  }

  // Handle loading state
  if (scheduledScorersResult.isLoading) {
    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          width: '100%',
          gap: theme.spacing.sm,
          padding: theme.spacing.lg,
        }}
      >
        {[...Array(3).keys()].map((i) => (
          <ParagraphSkeleton
            label={intl.formatMessage({
              defaultMessage: 'Loading judges...',
              description: 'Loading message while fetching experiment judges',
            })}
            key={i}
            seed={`scorer-${i}`}
          />
        ))}
      </div>
    );
  }

  // Show empty state when there are no scorers
  if (shouldShowEmptyState) {
    return <ScorerEmptyStateRenderer onAddScorerClick={handleNewScorerClick} />;
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'auto',
      }}
    >
      {/* Header with New scorer button */}
      <div
        css={{
          display: 'flex',
          justifyContent: 'flex-end',
          alignItems: 'center',
          padding: theme.spacing.sm,
        }}
      >
        <Button
          type="primary"
          icon={<PlusIcon />}
          componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_experimentscorerscontentcontainer_90"
          onClick={handleNewScorerClick}
        >
          <FormattedMessage defaultMessage="New judge" description="Button text to create a new judge" />
        </Button>
      </div>
      <Spacer size="sm" />
      {/* Content area */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            width: '100%',
          }}
        >
          {scorers.map((scorer) => (
            <ScorerCardContainer key={scorer.name} scorer={scorer} experimentId={experimentId} />
          ))}
        </div>
      </div>
      {/* New Scorer Modal */}
      <ScorerModalRenderer
        visible={isModalVisible}
        onClose={closeModal}
        experimentId={experimentId}
        mode={SCORER_FORM_MODE.CREATE}
      />
    </div>
  );
};

export default ExperimentScorersContentContainer;

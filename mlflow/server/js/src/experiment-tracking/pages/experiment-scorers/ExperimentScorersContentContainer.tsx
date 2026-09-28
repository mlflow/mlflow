import React, { useState } from 'react';
import { useDesignSystemTheme, ParagraphSkeleton, CodeIcon, Spacer, Button } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import ScorerCardContainer from './ScorerCardContainer';
import ScorerModalRenderer from './ScorerModalRenderer';
import ScorerEmptyStateRenderer from './ScorerEmptyStateRenderer';
import { useGetScheduledScorers } from './hooks/useGetScheduledScorers';
import { SCORER_FORM_MODE, ScorerEvaluationScope } from './constants';
import type { ScorerFormData } from './utils/scorerTransformUtils';
import NewLLMJudgeDropdown from './NewLLMJudgeDropdown';
import { LLM_TEMPLATE } from './types';

interface ExperimentScorersContentContainerProps {
  experimentId: string;
}

const ExperimentScorersContentContainer: React.FC<ExperimentScorersContentContainerProps> = ({ experimentId }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [initialScorerType, setInitialScorerType] = useState<ScorerFormData['scorerType']>('llm');
  const [initialTemplate, setInitialTemplate] = useState<LLM_TEMPLATE>(LLM_TEMPLATE.CUSTOM);
  const [initialScope, setInitialScope] = useState(ScorerEvaluationScope.TRACES);
  const scheduledScorersResult = useGetScheduledScorers(experimentId);
  const scorers = scheduledScorersResult.data?.scheduledScorers || [];
  const isLoading = scheduledScorersResult.isLoading;
  const isError = scheduledScorersResult.isError;
  const error = scheduledScorersResult.error;

  const handleNewLLMScorerClick = (template: LLM_TEMPLATE, scope: ScorerEvaluationScope) => {
    setInitialScorerType('llm');
    setInitialTemplate(template);
    setInitialScope(scope);
    setIsModalVisible(true);
  };

  const handleNewCustomCodeScorerClick = () => {
    setInitialScorerType('custom-code');
    setIsModalVisible(true);
  };

  // If no scorers exist and we're not currently showing the modal, show empty state
  const shouldShowEmptyState = scorers.length === 0 && !isModalVisible && !isLoading;

  const closeModal = () => {
    setIsModalVisible(false);
  };

  // Handle error state - throw error to be caught by PanelBoundary
  if (isError && error) {
    throw error;
  }

  // Handle loading state
  if (isLoading) {
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
    return (
      <ScorerEmptyStateRenderer
        onAddLLMScorerClick={handleNewLLMScorerClick}
        onAddCustomCodeScorerClick={handleNewCustomCodeScorerClick}
      />
    );
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
      {/* Header with new judge actions */}
      <div
        css={{
          display: 'flex',
          justifyContent: 'flex-end',
          alignItems: 'center',
          gap: theme.spacing.sm,
          padding: theme.spacing.sm,
        }}
      >
        <NewLLMJudgeDropdown onSelect={handleNewLLMScorerClick} />
        <Button
          icon={<CodeIcon />}
          componentId="mlflow.experiment-scorers.new-custom-code-scorer-button"
          onClick={handleNewCustomCodeScorerClick}
        >
          <FormattedMessage
            defaultMessage="New custom code judge"
            description="Button text to create a custom code judge"
          />
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
        initialScorerType={initialScorerType}
        initialTemplate={initialTemplate}
        initialScope={initialScope}
      />
    </div>
  );
};

export default ExperimentScorersContentContainer;

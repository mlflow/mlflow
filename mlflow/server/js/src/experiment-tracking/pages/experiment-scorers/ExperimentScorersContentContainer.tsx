import React, { useState } from 'react';
import {
  useDesignSystemTheme,
  ParagraphSkeleton,
  PlusIcon,
  CodeIcon,
  Spacer,
  SplitButton,
  DropdownMenu,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import ScorerCardContainer from './ScorerCardContainer';
import ScorerModalRenderer from './ScorerModalRenderer';
import ScorerEmptyStateRenderer from './ScorerEmptyStateRenderer';
import { useGetScheduledScorers } from './hooks/useGetScheduledScorers';
import { COMPONENT_ID_PREFIX, SCORER_FORM_MODE } from './constants';
import type { ScorerFormData } from './utils/scorerTransformUtils';

interface ExperimentScorersContentContainerProps {
  experimentId: string;
}

const ExperimentScorersContentContainer: React.FC<ExperimentScorersContentContainerProps> = ({ experimentId }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [initialScorerType, setInitialScorerType] = useState<ScorerFormData['scorerType']>('llm');

  const scheduledScorersResult = useGetScheduledScorers(experimentId);
  const scorers = scheduledScorersResult.data?.scheduledScorers || [];

  const handleNewLLMScorerClick = () => {
    setInitialScorerType('llm');
    setIsModalVisible(true);
  };

  const handleNewCustomCodeScorerClick = () => {
    setInitialScorerType('custom-code');
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
      {/* Header with New judge split button */}
      <div
        css={{
          display: 'flex',
          justifyContent: 'flex-end',
          alignItems: 'center',
          padding: theme.spacing.sm,
        }}
      >
        <SplitButton
          type="primary"
          icon={<PlusIcon />}
          componentId={`${COMPONENT_ID_PREFIX}.new-scorer-button`}
          onClick={handleNewLLMScorerClick}
          menu={
            <DropdownMenu.Content>
              <DropdownMenu.Item
                componentId={`${COMPONENT_ID_PREFIX}.new-custom-code-scorer-menu-item`}
                onClick={handleNewCustomCodeScorerClick}
                css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}
              >
                <CodeIcon />
                <FormattedMessage
                  defaultMessage="Custom code judge"
                  description="Menu item text to create a new custom code judge"
                />
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          }
        >
          <FormattedMessage defaultMessage="New LLM judge" description="Button text to create a new LLM judge" />
        </SplitButton>
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
      />
    </div>
  );
};

export default ExperimentScorersContentContainer;

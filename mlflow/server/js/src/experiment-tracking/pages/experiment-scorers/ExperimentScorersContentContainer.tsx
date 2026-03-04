import React, { useState, useCallback, useMemo } from 'react';
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
import JudgeSelectionActionBar from './JudgeSelectionActionBar';
import JudgeCategoryFilter from './JudgeCategoryFilter';
import { useGetScheduledScorers } from './hooks/useGetScheduledScorers';
import { useDeleteScheduledScorerMutation } from './hooks/useDeleteScheduledScorer';
import { COMPONENT_ID_PREFIX, SCORER_FORM_MODE } from './constants';
import type { ScorerFormData } from './utils/scorerTransformUtils';
import { type JudgeCategory, getScorerCategory } from './types';

interface ExperimentScorersContentContainerProps {
  experimentId: string;
}

const ExperimentScorersContentContainer: React.FC<ExperimentScorersContentContainerProps> = ({ experimentId }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [initialScorerType, setInitialScorerType] = useState<ScorerFormData['scorerType']>('llm');
  const [selectedScorerNames, setSelectedScorerNames] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState<JudgeCategory | 'all'>('all');

  const scheduledScorersResult = useGetScheduledScorers(experimentId);
  const scorers = scheduledScorersResult.data?.scheduledScorers || [];
  const deleteScorerMutation = useDeleteScheduledScorerMutation();

  const filteredScorers = useMemo(() => {
    return scorers.filter((scorer) => {
      if (searchQuery && !scorer.name.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }
      if (activeCategory !== 'all') {
        const category = getScorerCategory(scorer);
        if (category !== activeCategory) {
          return false;
        }
      }
      return true;
    });
  }, [scorers, searchQuery, activeCategory]);

  const selectedScorers = useMemo(
    () => scorers.filter((s) => selectedScorerNames.has(s.name)),
    [scorers, selectedScorerNames],
  );

  const handleSelectionChange = useCallback((scorerName: string, selected: boolean) => {
    setSelectedScorerNames((prev) => {
      const next = new Set(prev);
      if (selected) {
        next.add(scorerName);
      } else {
        next.delete(scorerName);
      }
      return next;
    });
  }, []);

  const handleClearSelection = useCallback(() => {
    setSelectedScorerNames(new Set());
  }, []);

  const handleBulkDelete = useCallback(() => {
    const names = Array.from(selectedScorerNames);
    if (names.length === 0) return;
    deleteScorerMutation.mutate(
      { experimentId, scorerNames: names },
      { onSuccess: () => setSelectedScorerNames(new Set()) },
    );
  }, [selectedScorerNames, deleteScorerMutation, experimentId]);

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
      {/* Selection action bar - shown when judges are selected */}
      <JudgeSelectionActionBar
        selectedScorers={selectedScorers}
        experimentId={experimentId}
        onDelete={handleBulkDelete}
        onClearSelection={handleClearSelection}
      />
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
          componentId="mlflow.experiment-scorers.new-scorer-button"
          onClick={handleNewLLMScorerClick}
          menu={
            <DropdownMenu.Content>
              <DropdownMenu.Item
                componentId="mlflow.experiment-scorers.new-custom-code-scorer-menu-item"
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
      {/* Search and category filter */}
      <div css={{ padding: `0 ${theme.spacing.sm}px` }}>
        <JudgeCategoryFilter
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          activeCategory={activeCategory}
          onCategoryChange={setActiveCategory}
        />
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
          {filteredScorers.map((scorer) => (
            <ScorerCardContainer
              key={scorer.name}
              scorer={scorer}
              experimentId={experimentId}
              isSelected={selectedScorerNames.has(scorer.name)}
              onSelectionChange={(selected) => handleSelectionChange(scorer.name, selected)}
            />
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

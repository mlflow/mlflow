import { useState, useCallback, useMemo } from 'react';
import { useIntl } from 'react-intl';
import {
  Button,
  Checkbox,
  CheckIcon,
  Input,
  Modal,
  Spinner,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useModelsQuery } from './useModelsQuery';
import type { Model } from '../types';

export interface ModelSelectorModalProps {
  provider: string;
  onSelect: (model: Model) => void;
}

interface FilterState {
  search: string;
  supportsTools: boolean;
  supportsReasoning: boolean;
}

export function useModelSelectorModal({ provider, onSelect }: ModelSelectorModalProps) {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    supportsTools: false,
    supportsReasoning: false,
  });

  const { data: models, isLoading } = useModelsQuery({ provider: provider || undefined });

  const filteredModels = useMemo(() => {
    if (!models) return [];

    return models.filter((model) => {
      // Search filter
      if (filters.search && !model.model.toLowerCase().includes(filters.search.toLowerCase())) {
        return false;
      }
      // Capability filters
      if (filters.supportsTools && !model.supports_function_calling) {
        return false;
      }
      if (filters.supportsReasoning && !model.supports_reasoning) {
        return false;
      }
      return true;
    });
  }, [models, filters]);

  const openModal = useCallback(() => {
    setIsModalVisible(true);
    setSelectedModel(null);
    setFilters({
      search: '',
      supportsTools: false,
      supportsReasoning: false,
    });
  }, []);

  const closeModal = useCallback(() => {
    setIsModalVisible(false);
    setSelectedModel(null);
  }, []);

  const handleConfirm = useCallback(() => {
    if (selectedModel) {
      onSelect(selectedModel);
      closeModal();
    }
  }, [selectedModel, onSelect, closeModal]);

  const formatCost = (cost: number | null) => {
    if (cost === null || cost === undefined) return '-';
    if (cost === 0) return 'Free';
    // Convert to cost per 1M tokens for readability
    const perMillion = cost * 1_000_000;
    if (perMillion < 0.01) return `$${perMillion.toFixed(4)}/1M`;
    return `$${perMillion.toFixed(2)}/1M`;
  };

  const formatTokens = (tokens: number | null) => {
    if (tokens === null || tokens === undefined) return '-';
    if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
    if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(0)}K`;
    return tokens.toString();
  };

  const modelSelectorModal = (
    <Modal
      componentId="mlflow.gateway.model-selector-modal"
      title={intl.formatMessage({
        defaultMessage: 'Select model',
        description: 'Model selector modal title',
      })}
      visible={isModalVisible}
      onCancel={closeModal}
      size="wide"
      footer={[
        <Button key="cancel" componentId="mlflow.gateway.model-selector-modal.cancel" onClick={closeModal}>
          {intl.formatMessage({
            defaultMessage: 'Cancel',
            description: 'Cancel button',
          })}
        </Button>,
        <Button
          key="confirm"
          componentId="mlflow.gateway.model-selector-modal.confirm"
          type="primary"
          disabled={!selectedModel}
          onClick={handleConfirm}
        >
          {intl.formatMessage({
            defaultMessage: 'Select',
            description: 'Select button',
          })}
        </Button>,
      ]}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {/* Search and filters */}
        <div css={{ display: 'flex', gap: theme.spacing.md, flexWrap: 'wrap', alignItems: 'center' }}>
          <Input
            componentId="mlflow.gateway.model-selector-modal.search"
            placeholder={intl.formatMessage({
              defaultMessage: 'Search models...',
              description: 'Search placeholder',
            })}
            value={filters.search}
            onChange={(e) => setFilters((f) => ({ ...f, search: e.target.value }))}
            css={{ width: 250 }}
          />
          <div css={{ display: 'flex', gap: theme.spacing.md, flexWrap: 'wrap' }}>
            <Checkbox
              componentId="mlflow.gateway.model-selector-modal.filter-tools"
              isChecked={filters.supportsTools}
              onChange={(checked) => setFilters((f) => ({ ...f, supportsTools: checked }))}
            >
              {intl.formatMessage({
                defaultMessage: 'Tools',
                description: 'Filter for tool/function calling support',
              })}
            </Checkbox>
            <Checkbox
              componentId="mlflow.gateway.model-selector-modal.filter-reasoning"
              isChecked={filters.supportsReasoning}
              onChange={(checked) => setFilters((f) => ({ ...f, supportsReasoning: checked }))}
            >
              {intl.formatMessage({
                defaultMessage: 'Reasoning',
                description: 'Filter for reasoning/thinking support',
              })}
            </Checkbox>
          </div>
        </div>

        {/* Model list */}
        {isLoading ? (
          <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
            <Spinner size="large" />
          </div>
        ) : (
          <div
            css={{
              maxHeight: 400,
              overflowY: 'auto',
              border: `1px solid ${theme.colors.borderDecorative}`,
              borderRadius: theme.general.borderRadiusBase,
            }}
          >
            {filteredModels.length === 0 ? (
              <div css={{ padding: theme.spacing.lg, textAlign: 'center' }}>
                <Typography.Text color="secondary">
                  {intl.formatMessage({
                    defaultMessage: 'No models match your filters',
                    description: 'Empty state message',
                  })}
                </Typography.Text>
              </div>
            ) : (
              filteredModels.map((model) => {
                const isSelected = selectedModel?.model === model.model;
                return (
                  <div
                    key={model.model}
                    onClick={() => setSelectedModel(model)}
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: theme.spacing.sm,
                      paddingLeft: theme.spacing.md,
                      cursor: 'pointer',
                      backgroundColor: isSelected ? theme.colors.actionTertiaryBackgroundPress : 'transparent',
                      borderLeft: isSelected
                        ? `3px solid ${theme.colors.actionPrimaryBackgroundDefault}`
                        : '3px solid transparent',
                      '&:hover': {
                        backgroundColor: theme.colors.actionTertiaryBackgroundHover,
                      },
                      borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                      '&:last-child': {
                        borderBottom: 'none',
                      },
                      transition: 'background-color 0.1s ease, border-left-color 0.1s ease',
                    }}
                  >
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                        <Typography.Text bold>{model.model}</Typography.Text>
                        {model.supports_function_calling && (
                          <Tag componentId="mlflow.gateway.model-selector.tag.tools">Tools</Tag>
                        )}
                        {model.supports_reasoning && (
                          <Tag componentId="mlflow.gateway.model-selector.tag.reasoning">Reasoning</Tag>
                        )}
                      </div>
                      <div
                        css={{
                          display: 'flex',
                          gap: theme.spacing.md,
                          color: theme.colors.textSecondary,
                          fontSize: theme.typography.fontSizeSm,
                        }}
                      >
                        <span>
                          {intl.formatMessage(
                            {
                              defaultMessage: 'Context: {tokens}',
                              description: 'Context window size',
                            },
                            { tokens: formatTokens(model.max_input_tokens) },
                          )}
                        </span>
                        <span>
                          {intl.formatMessage(
                            {
                              defaultMessage: 'Input: {cost}',
                              description: 'Input cost per token',
                            },
                            { cost: formatCost(model.input_cost_per_token) },
                          )}
                        </span>
                        <span>
                          {intl.formatMessage(
                            {
                              defaultMessage: 'Output: {cost}',
                              description: 'Output cost per token',
                            },
                            { cost: formatCost(model.output_cost_per_token) },
                          )}
                        </span>
                      </div>
                    </div>
                    {isSelected && (
                      <CheckIcon
                        css={{
                          color: theme.colors.actionPrimaryBackgroundDefault,
                          flexShrink: 0,
                          marginLeft: theme.spacing.sm,
                        }}
                      />
                    )}
                  </div>
                );
              })
            )}
          </div>
        )}

        {/* Results count */}
        <Typography.Text color="secondary" size="sm">
          {intl.formatMessage(
            {
              defaultMessage: '{count} models',
              description: 'Number of models shown',
            },
            { count: filteredModels.length },
          )}
        </Typography.Text>
      </div>
    </Modal>
  );

  return { openModal, modelSelectorModal, isModalVisible };
}

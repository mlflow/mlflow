import { useState, useCallback, useMemo } from 'react';
import { useIntl, FormattedMessage } from 'react-intl';
import {
  Button,
  Checkbox,
  ChevronDownIcon,
  FilterIcon,
  Input,
  Modal,
  Popover,
  Radio,
  SearchIcon,
  Spinner,
  Typography,
  useDesignSystemTheme,
  WarningFillIcon,
  XCircleFillIcon,
} from '@databricks/design-system';
import type { RadioChangeEvent } from '@databricks/design-system';
import { useModelsQuery } from '../../hooks/useModelsQuery';
import type { ProviderModel } from '../../types';
import { sortModelsByDate } from '../../utils/formatters';
import { ModelRow } from './ModelRow';

interface ModelSelectorModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (model: ProviderModel) => void;
  provider: string;
}

interface CapabilityFilter {
  tools: boolean;
  reasoning: boolean;
  promptCaching: boolean;
  structuredOutput: boolean;
}

interface FilterState {
  search: string;
  capabilities: CapabilityFilter;
}

export const ModelSelectorModal = ({ isOpen, onClose, onSelect, provider }: ModelSelectorModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [customModelName, setCustomModelName] = useState('');
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    capabilities: { tools: false, reasoning: false, promptCaching: false, structuredOutput: false },
  });
  const [isFilterOpen, setIsFilterOpen] = useState(false);

  const { data: models, isLoading } = useModelsQuery({ provider: provider || undefined });
  const isCustomMode = customModelName.trim().length > 0;

  const hasActiveFilters = Object.values(filters.capabilities).some(Boolean);
  const filterCount = Object.values(filters.capabilities).filter(Boolean).length;

  const filteredModels = useMemo(() => {
    if (!models) return [];

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const filtered = models.filter((model) => {
      // Exclude models that have already been deprecated
      if (model.deprecation_date) {
        const deprecationDate = new Date(model.deprecation_date);
        if (deprecationDate < today) {
          return false;
        }
      }
      if (filters.search && !model.model.toLowerCase().includes(filters.search.toLowerCase())) {
        return false;
      }
      if (filters.capabilities.tools && !model.supports_function_calling) {
        return false;
      }
      if (filters.capabilities.reasoning && !model.supports_reasoning) {
        return false;
      }
      if (filters.capabilities.promptCaching && !model.supports_prompt_caching) {
        return false;
      }
      if (filters.capabilities.structuredOutput && !model.supports_response_schema) {
        return false;
      }
      return true;
    });

    // Sort by extracted date from model name (newest first)
    return sortModelsByDate(filtered);
  }, [models, filters]);

  const selectedModel = useMemo(() => {
    return models?.find((m) => m.model === selectedModelId);
  }, [models, selectedModelId]);

  const handleClose = useCallback(() => {
    setSelectedModelId(null);
    setCustomModelName('');
    setFilters({
      search: '',
      capabilities: { tools: false, reasoning: false, promptCaching: false, structuredOutput: false },
    });
    onClose();
  }, [onClose]);

  const handleClearFilters = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setFilters((f) => ({
      ...f,
      capabilities: { tools: false, reasoning: false, promptCaching: false, structuredOutput: false },
    }));
  }, []);

  const handleCustomModelChange = useCallback((value: string) => {
    setCustomModelName(value);
    if (value.trim()) {
      setSelectedModelId(null);
    }
  }, []);

  const handleModelSelect = useCallback((modelId: string) => {
    setSelectedModelId(modelId);
    setCustomModelName('');
  }, []);

  const handleConfirm = useCallback(() => {
    if (isCustomMode) {
      onSelect({
        model: customModelName.trim(),
        provider: provider,
        supports_function_calling: false,
      });
      handleClose();
    } else if (selectedModel) {
      onSelect(selectedModel);
      handleClose();
    }
  }, [isCustomMode, customModelName, selectedModel, provider, onSelect, handleClose]);

  const isConfirmDisabled = isCustomMode ? !customModelName.trim() : !selectedModel;

  return (
    <Modal
      componentId="mlflow.gateway.model-selector-modal"
      title={intl.formatMessage({
        defaultMessage: 'Select model',
        description: 'Model selector modal title',
      })}
      visible={isOpen}
      onCancel={handleClose}
      size="wide"
      footer={
        <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
          <div>
            {selectedModel?.deprecation_date && (
              <Typography.Text color="warning" css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <WarningFillIcon />
                <FormattedMessage
                  defaultMessage="This model will be deprecated on {date}"
                  description="Deprecation warning in modal footer"
                  values={{ date: selectedModel.deprecation_date }}
                />
              </Typography.Text>
            )}
          </div>
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Button componentId="mlflow.gateway.model-selector-modal.cancel" onClick={handleClose}>
              {intl.formatMessage({
                defaultMessage: 'Cancel',
                description: 'Cancel button',
              })}
            </Button>
            <Button
              componentId="mlflow.gateway.model-selector-modal.confirm"
              type="primary"
              disabled={isConfirmDisabled}
              onClick={handleConfirm}
            >
              {intl.formatMessage({
                defaultMessage: 'Select',
                description: 'Select button',
              })}
            </Button>
          </div>
        </div>
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {/* Search and Filter row */}
        <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
          <Input
            componentId="mlflow.gateway.model-selector-modal.search"
            placeholder={intl.formatMessage({
              defaultMessage: 'Search',
              description: 'Search placeholder',
            })}
            prefix={<SearchIcon />}
            value={filters.search}
            onChange={(e) => setFilters((f) => ({ ...f, search: e.target.value }))}
            css={{ flexGrow: 1 }}
          />
          <Popover.Root
            componentId="mlflow.gateway.model-selector-modal.filter-popover"
            open={isFilterOpen}
            onOpenChange={setIsFilterOpen}
          >
            <Popover.Trigger asChild>
              <Button
                componentId="mlflow.gateway.model-selector-modal.filter-button"
                endIcon={<ChevronDownIcon />}
                css={{
                  border: hasActiveFilters ? `1px solid ${theme.colors.actionDefaultBorderFocus} !important` : '',
                  backgroundColor: hasActiveFilters ? `${theme.colors.actionDefaultBackgroundHover} !important` : '',
                }}
              >
                <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
                  <FilterIcon />
                  {intl.formatMessage(
                    {
                      defaultMessage: 'Capability{count}',
                      description: 'Capability filter button label with count',
                    },
                    { count: hasActiveFilters ? ` (${filterCount})` : '' },
                  )}
                  {hasActiveFilters && (
                    <XCircleFillIcon
                      css={{
                        fontSize: 12,
                        cursor: 'pointer',
                        color: theme.colors.grey400,
                        '&:hover': { color: theme.colors.grey600 },
                      }}
                      onClick={handleClearFilters}
                    />
                  )}
                </div>
              </Button>
            </Popover.Trigger>
            <Popover.Content align="end" css={{ padding: theme.spacing.md, minWidth: 200 }}>
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                <div css={{ fontWeight: theme.typography.typographyBoldFontWeight, marginBottom: theme.spacing.xs }}>
                  <FormattedMessage defaultMessage="Capabilities" description="Filter section label for capabilities" />
                </div>
                <Checkbox
                  componentId="mlflow.gateway.model-selector-modal.filter.tools"
                  isChecked={filters.capabilities.tools}
                  onChange={() =>
                    setFilters((f) => ({ ...f, capabilities: { ...f.capabilities, tools: !f.capabilities.tools } }))
                  }
                >
                  <FormattedMessage defaultMessage="Tools" description="Filter option for tool support" />
                </Checkbox>
                <Checkbox
                  componentId="mlflow.gateway.model-selector-modal.filter.reasoning"
                  isChecked={filters.capabilities.reasoning}
                  onChange={() =>
                    setFilters((f) => ({
                      ...f,
                      capabilities: { ...f.capabilities, reasoning: !f.capabilities.reasoning },
                    }))
                  }
                >
                  <FormattedMessage defaultMessage="Reasoning" description="Filter option for reasoning support" />
                </Checkbox>
                <Checkbox
                  componentId="mlflow.gateway.model-selector-modal.filter.promptCaching"
                  isChecked={filters.capabilities.promptCaching}
                  onChange={() =>
                    setFilters((f) => ({
                      ...f,
                      capabilities: { ...f.capabilities, promptCaching: !f.capabilities.promptCaching },
                    }))
                  }
                >
                  <FormattedMessage
                    defaultMessage="Prompt Caching"
                    description="Filter option for prompt caching support"
                  />
                </Checkbox>
                <Checkbox
                  componentId="mlflow.gateway.model-selector-modal.filter.structuredOutput"
                  isChecked={filters.capabilities.structuredOutput}
                  onChange={() =>
                    setFilters((f) => ({
                      ...f,
                      capabilities: { ...f.capabilities, structuredOutput: !f.capabilities.structuredOutput },
                    }))
                  }
                >
                  <FormattedMessage
                    defaultMessage="Structured Output"
                    description="Filter option for structured JSON output support"
                  />
                </Checkbox>
              </div>
            </Popover.Content>
          </Popover.Root>
        </div>

        {isLoading ? (
          <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.lg }}>
            <Spinner size="large" />
          </div>
        ) : (
          <div
            css={{
              border: `1px solid ${theme.colors.borderDecorative}`,
              borderRadius: theme.general.borderRadiusBase,
              overflow: 'hidden',
              opacity: isCustomMode ? 0.5 : 1,
              pointerEvents: isCustomMode ? 'none' : 'auto',
              transition: 'opacity 0.2s ease',
            }}
          >
            {/* Table header */}
            <div
              css={{
                display: 'grid',
                gridTemplateColumns: '40px 1fr 110px 100px',
                gap: theme.spacing.sm,
                padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                backgroundColor: theme.colors.backgroundSecondary,
                borderBottom: `1px solid ${theme.colors.borderDecorative}`,
                fontWeight: theme.typography.typographyBoldFontWeight,
                fontSize: theme.typography.fontSizeSm,
                color: theme.colors.textSecondary,
              }}
            >
              <div />
              <div>
                <FormattedMessage defaultMessage="Name" description="Table header for model name" />
              </div>
              <div css={{ textAlign: 'right' }}>
                <FormattedMessage defaultMessage="Max Input Tokens" description="Table header for max input tokens" />
              </div>
              <div css={{ textAlign: 'right', display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                <span>
                  <FormattedMessage defaultMessage="Input /1M" description="Table header for input cost" />
                </span>
                <span>
                  <FormattedMessage defaultMessage="Output /1M" description="Table header for output cost" />
                </span>
              </div>
            </div>

            {/* Table body */}
            <div css={{ maxHeight: 400, overflowY: 'auto' }}>
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
                <Radio.Group
                  name="model-selector"
                  componentId="mlflow.gateway.model-selector-modal.radio-group"
                  value={selectedModelId || ''}
                  onChange={(e: RadioChangeEvent) => handleModelSelect(e.target.value)}
                  css={{ width: '100%' }}
                >
                  {filteredModels.map((model) => (
                    <ModelRow
                      key={model.model}
                      model={model}
                      isSelected={selectedModelId === model.model}
                      onSelect={handleModelSelect}
                    />
                  ))}
                </Radio.Group>
              )}
            </div>
          </div>
        )}

        <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography.Text color="secondary" size="sm">
            {intl.formatMessage(
              {
                defaultMessage: '{count, plural, one {# model} other {# models}} available',
                description: 'Number of models shown',
              },
              { count: filteredModels.length },
            )}
          </Typography.Text>
          {selectedModelId && (
            <div
              onClick={() => setSelectedModelId(null)}
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.xs,
                cursor: 'pointer',
                color: theme.colors.textSecondary,
                '&:hover': {
                  color: theme.colors.textPrimary,
                },
              }}
            >
              <Typography.Text
                size="sm"
                css={{
                  color: 'inherit',
                }}
              >
                <FormattedMessage defaultMessage="Clear selection" description="Clear model selection" />
              </Typography.Text>
              <XCircleFillIcon css={{ width: 14, height: 14 }} />
            </div>
          )}
        </div>

        {/* Divider with "or" */}
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.md,
            marginTop: theme.spacing.sm,
          }}
        >
          <div css={{ flex: 1, height: 1, backgroundColor: theme.colors.borderDecorative }} />
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage defaultMessage="or" description="Divider between model list and custom input" />
          </Typography.Text>
          <div css={{ flex: 1, height: 1, backgroundColor: theme.colors.borderDecorative }} />
        </div>

        {/* Custom model input */}
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
            opacity: selectedModelId ? 0.5 : 1,
            transition: 'opacity 0.2s ease',
          }}
        >
          <Typography.Text bold>
            <FormattedMessage
              defaultMessage="Use a custom model name"
              description="Label for custom model input section"
            />
          </Typography.Text>
          <Input
            componentId="mlflow.gateway.model-selector-modal.custom-model"
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter model name...',
              description: 'Placeholder for custom model input',
            })}
            value={customModelName}
            onChange={(e) => handleCustomModelChange(e.target.value)}
            disabled={!!selectedModelId}
          />
          <Typography.Text color="secondary" size="sm">
            <FormattedMessage
              defaultMessage="Enter a model name not listed above. Capabilities may not be detected."
              description="Help text for custom model input"
            />
          </Typography.Text>
        </div>
      </div>
    </Modal>
  );
};

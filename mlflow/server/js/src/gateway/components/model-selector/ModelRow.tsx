import { useIntl } from 'react-intl';
import { Radio, Tooltip, Typography, useDesignSystemTheme, WarningFillIcon } from '@databricks/design-system';
import { CostIndicator } from './CostIndicator';
import { formatTokens } from '../../utils/formatters';
import type { ProviderModel } from '../../types';

interface ModelRowProps {
  model: ProviderModel;
  isSelected: boolean;
  costTier: number;
  onSelect: (modelId: string) => void;
}

export const ModelRow = ({ model, isSelected, costTier, onSelect }: ModelRowProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const contextWindow = formatTokens(model.max_input_tokens);

  return (
    <div
      onClick={() => onSelect(model.model)}
      css={{
        display: 'grid',
        gridTemplateColumns: '40px 1fr 110px 80px',
        gap: theme.spacing.sm,
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
        cursor: 'pointer',
        backgroundColor: isSelected ? theme.colors.actionTertiaryBackgroundPress : 'transparent',
        '&:hover': {
          backgroundColor: isSelected
            ? theme.colors.actionTertiaryBackgroundPress
            : theme.colors.actionTertiaryBackgroundHover,
        },
        borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        '&:last-child': {
          borderBottom: 'none',
        },
        alignItems: 'center',
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'center' }}>
        <Radio value={model.model} css={{ margin: 0 }} />
      </div>
      <div css={{ minWidth: 0 }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text bold>{model.model}</Typography.Text>
          {model.deprecation_date && (
            <Tooltip
              componentId="mlflow.gateway.model-selector-modal.deprecation-tooltip"
              content={intl.formatMessage(
                {
                  defaultMessage: 'This model will be deprecated on {date}',
                  description: 'Deprecation date warning tooltip',
                },
                { date: model.deprecation_date },
              )}
            >
              <WarningFillIcon
                css={{
                  fontSize: theme.typography.fontSizeSm,
                  color: theme.colors.textValidationWarning,
                }}
              />
            </Tooltip>
          )}
        </div>
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, display: 'block' }}>
          {[
            model.supports_function_calling && 'Tools',
            model.supports_reasoning && 'Reasoning',
            model.supports_prompt_caching && 'Caching',
            model.supports_response_schema && 'Structured',
          ]
            .filter(Boolean)
            .join(', ') || '\u00A0'}
        </Typography.Text>
      </div>
      <div css={{ textAlign: 'right' }}>
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          {contextWindow || '-'}
        </Typography.Text>
      </div>
      <div css={{ textAlign: 'right' }}>
        <CostIndicator
          tier={costTier}
          inputCost={model.input_cost_per_token}
          outputCost={model.output_cost_per_token}
        />
      </div>
    </div>
  );
};

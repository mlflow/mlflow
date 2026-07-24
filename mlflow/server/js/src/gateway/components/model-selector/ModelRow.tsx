import { useIntl } from 'react-intl';
import { Checkbox, Radio, Tooltip, Typography, useDesignSystemTheme, WarningFillIcon } from '@databricks/design-system';
import { CostIndicator } from './CostIndicator';
import { formatTokens } from '../../utils/formatters';
import type { ProviderModel } from '../../types';
import { getModelCapabilities } from '../../utils/getModelCapabilities';

interface ModelRowProps {
  model: ProviderModel;
  isSelected: boolean;
  onSelect: (modelId: string) => void;
  /** When true, render a checkbox instead of a radio (multi-select mode). */
  multiSelect?: boolean;
}

export const ModelRow = ({ model, isSelected, onSelect, multiSelect }: ModelRowProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const contextWindow = formatTokens(model.max_input_tokens);

  return (
    <div
      onClick={() => onSelect(model.model)}
      css={{
        display: 'grid',
        gridTemplateColumns: '40px 1fr 110px 100px',
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
        {multiSelect ? (
          <Checkbox
            componentId="mlflow.gateway.model-selector-modal.checkbox"
            isChecked={isSelected}
            onChange={() => onSelect(model.model)}
            css={{ margin: 0 }}
          />
        ) : (
          <Radio value={model.model} css={{ margin: 0 }} />
        )}
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
          {getModelCapabilities(model).join(', ') || '\u00A0'}
        </Typography.Text>
      </div>
      <div css={{ textAlign: 'right' }}>
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          {contextWindow || '-'}
        </Typography.Text>
      </div>
      <div css={{ textAlign: 'right' }}>
        <CostIndicator inputCost={model.input_cost_per_token} outputCost={model.output_cost_per_token} />
      </div>
    </div>
  );
};

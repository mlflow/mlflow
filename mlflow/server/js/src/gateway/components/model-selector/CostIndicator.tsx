import { useIntl } from 'react-intl';
import { Tooltip, useDesignSystemTheme } from '@databricks/design-system';

interface CostIndicatorProps {
  tier: number;
  inputCost?: number;
  outputCost?: number;
}

export const CostIndicator = ({ tier, inputCost, outputCost }: CostIndicatorProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const maxTier = 4;

  const formatCostPerMillion = (cost: number | undefined): string => {
    if (cost === undefined) return '-';
    const perMillion = cost * 1_000_000;
    if (perMillion < 0.01) return `$${perMillion.toFixed(4)}`;
    return `$${perMillion.toFixed(2)}`;
  };

  const tooltipContent =
    inputCost !== undefined || outputCost !== undefined ? (
      <>
        {intl.formatMessage(
          {
            defaultMessage: 'Input: {input}/1M tokens',
            description: 'Input cost per million tokens',
          },
          { input: formatCostPerMillion(inputCost) },
        )}
        <br />
        {intl.formatMessage(
          {
            defaultMessage: 'Output: {output}/1M tokens',
            description: 'Output cost per million tokens',
          },
          { output: formatCostPerMillion(outputCost) },
        )}
      </>
    ) : (
      intl.formatMessage({
        defaultMessage: 'Cost data unavailable',
        description: 'Tooltip when cost data is not available',
      })
    );

  return (
    <Tooltip componentId="mlflow.gateway.model-selector-modal.cost-tooltip" content={tooltipContent}>
      <span css={{ fontFamily: 'monospace', letterSpacing: '-1px', fontSize: theme.typography.fontSizeSm }}>
        {Array.from({ length: maxTier }, (_, i) => (
          <span key={i} css={{ color: i < tier ? theme.colors.textPrimary : theme.colors.textPlaceholder }}>
            $
          </span>
        ))}
      </span>
    </Tooltip>
  );
};

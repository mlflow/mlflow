import { Typography, useDesignSystemTheme } from '@databricks/design-system';

interface CostIndicatorProps {
  inputCost?: number;
  outputCost?: number;
}

export const CostIndicator = ({ inputCost, outputCost }: CostIndicatorProps) => {
  const { theme } = useDesignSystemTheme();

  const formatCostPerMillion = (cost: number | undefined): string => {
    if (cost === undefined) return '-';
    const perMillion = cost * 1_000_000;
    if (perMillion < 0.01) return `$${perMillion.toFixed(4)}`;
    return `$${perMillion.toFixed(2)}`;
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2 }}>
      <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
        {formatCostPerMillion(inputCost)}
      </Typography.Text>
      <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
        {formatCostPerMillion(outputCost)}
      </Typography.Text>
    </div>
  );
};

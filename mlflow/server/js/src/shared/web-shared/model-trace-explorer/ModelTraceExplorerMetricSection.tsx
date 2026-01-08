import type { TagColors } from '@databricks/design-system';
import { Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';

export const ModelTraceHeaderMetricSection = ({
  label,
  value,
  displayValue = value,
  icon,
  color,
  getTruncatedLabel,
  onCopy,
}: {
  label: React.ReactNode;
  /**
   * Actual value used to copy to clipboard and show in tooltip
   */
  value: string;
  /**
   * Optional display value to show in the tag. If not provided, `value` will be used.
   */
  displayValue?: string;
  icon?: React.ReactNode;
  color?: TagColors;
  getTruncatedLabel: (label: string) => string;
  onCopy: () => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const handleClick = () => {
    navigator.clipboard.writeText(value);
    onCopy();
  };

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        gap: theme.spacing.sm,
      }}
    >
      <Typography.Text size="md" color="secondary">
        {label}
      </Typography.Text>
      <Tooltip componentId="shared.model-trace-explorer.header-details.tooltip" content={value} maxWidth={400}>
        <Tag
          componentId="shared.model-trace-explorer.header-details.tag"
          color={color}
          onClick={handleClick}
          css={{ cursor: 'pointer' }}
        >
          <span css={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: theme.spacing.xs }}>
            {icon && <span>{icon}</span>}
            <span>{getTruncatedLabel(displayValue)}</span>
          </span>
        </Tag>
      </Tooltip>
    </div>
  );
};

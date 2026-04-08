import { Button, useDesignSystemTheme } from '@databricks/design-system';
import { CloseIcon } from '@databricks/design-system';
import type { getRangeColor } from './rangeColors';

interface RangeBadgeProps {
  label: string;
  color: ReturnType<typeof getRangeColor>;
  onDelete?: () => void;
  onClick?: () => void;
  isSelected?: boolean;
}

export const RangeBadge = ({ label, color, onDelete, onClick, isSelected }: RangeBadgeProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick ? (e: React.MouseEvent) => { e.stopPropagation(); onClick(); } : undefined}
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        padding: `2px ${theme.spacing.sm}px`,
        backgroundColor: color.background,
        border: `1px solid ${isSelected ? color.primary : `${color.primary}33`}`,
        borderRadius: theme.borders.borderRadiusMd,
        marginBottom: theme.spacing.xs,
        cursor: onClick ? 'pointer' : undefined,
        '&:hover': onClick ? { borderColor: color.primary } : undefined,
      }}
    >
      <div
        css={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          backgroundColor: color.primary,
        }}
      />
      <span
        css={{
          color: color.primary,
          fontSize: theme.typography.fontSizeSm,
          fontWeight: 600,
        }}
      >
        {label}
      </span>
      {onDelete && (
        <Button
          componentId="range-badge.delete"
          type="tertiary"
          size="small"
          icon={<CloseIcon />}
          onClick={(e: React.MouseEvent) => {
            e.stopPropagation();
            onDelete();
          }}
          aria-label="Delete range"
          css={{ marginLeft: theme.spacing.xs }}
        />
      )}
    </div>
  );
};

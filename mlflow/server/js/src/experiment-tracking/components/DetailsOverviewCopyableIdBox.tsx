import { CopyIcon, useDesignSystemTheme } from '@databricks/design-system';
import { CopyButton } from '../../shared/building_blocks/CopyButton';

export const DetailsOverviewCopyableIdBox = ({
  value,
  className,
  element,
}: {
  value: string;
  element?: React.ReactNode;
  className?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }} className={className}>
      <span css={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{element ?? value}</span>
      <CopyButton showLabel={false} copyText={value} icon={<CopyIcon />} size="small" css={{ flexShrink: 0 }} />
    </div>
  );
};

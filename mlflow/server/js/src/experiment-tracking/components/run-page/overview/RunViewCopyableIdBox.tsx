import { CopyIcon, useDesignSystemTheme } from '@databricks/design-system';
import { CopyButton } from '../../../../shared/building_blocks/CopyButton';

export const RunViewCopyableIdBox = ({ value }: { value: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
      {value}
      <CopyButton showLabel={false} copyText={value} icon={<CopyIcon />} size="small" />
    </div>
  );
};

import { TableSkeleton, TitleSkeleton, useDesignSystemTheme } from '@databricks/design-system';

export const ModelTraceExplorerSkeleton = ({ label }: { label?: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', height: '100%' }}>
      <div css={{ flex: 1 }}>
        <div css={{ padding: theme.spacing.sm, borderBottom: `1px solid ${theme.colors.border}` }}>
          <TitleSkeleton label={label} />
        </div>
        <div
          css={{
            borderRadius: theme.legacyBorders.borderRadiusMd,
            overflow: 'hidden',
            display: 'flex',
          }}
        >
          <div css={{ flex: 1, padding: theme.spacing.sm, borderRight: `1px solid ${theme.colors.border}` }}>
            <TableSkeleton lines={5} />
          </div>
          <div css={{ flex: 2, padding: theme.spacing.sm }}>
            <TableSkeleton lines={5} />
          </div>
        </div>
      </div>
      <div css={{ padding: theme.spacing.md, overflowY: 'auto', flex: 1 }}>
        <TableSkeleton lines={12} />
      </div>
    </div>
  );
};

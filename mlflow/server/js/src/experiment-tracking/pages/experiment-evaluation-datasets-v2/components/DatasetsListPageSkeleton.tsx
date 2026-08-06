import { GenericSkeleton, useDesignSystemTheme } from '@databricks/design-system';

// Enough rows to cover any reasonable laptop/desktop viewport; the container clips
// extras so this never introduces page scroll on shorter viewports.
const ROW_COUNT = 20;
const ROW_HEIGHT = 32;

/**
 * Full-page loading placeholder for the V2 datasets list. Rendered in place of the
 * toolbar + table on the first fetch, before we know whether the workspace has any
 * datasets. Intentionally omits a search/buttons placeholder — for empty workspaces
 * the resolved state has no toolbar, so the skeleton shouldn't imply one is coming.
 */
export const DatasetsListPageSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      role="status"
      aria-label="Loading datasets"
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        flex: 1,
        minHeight: 0,
        overflow: 'hidden',
      }}
    >
      {Array.from({ length: ROW_COUNT }, (_, i) => (
        <GenericSkeleton key={i} style={{ height: ROW_HEIGHT }} />
      ))}
    </div>
  );
};

import { GenericSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';

const ROW_COUNT = 6;
const ROW_HEIGHT = 32;

/**
 * Full-area loading placeholder for the V2 dataset records area. Rendered in place of the
 * toolbar + records table on the first fetch so the toolbar never briefly appears for
 * datasets that turn out to be empty (which would otherwise flash before the
 * `DatasetRecordsEmptyState` slides in). Subsequent search/refresh fetches keep the
 * toolbar visible and let the records table show its own in-table skeleton — only the
 * first load takes this path.
 */
export const DatasetRecordsLoadingSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  return (
    <div
      role="status"
      aria-label={intl.formatMessage({
        defaultMessage: 'Loading records',
        description: 'Aria label for the full-area loading placeholder on the V2 dataset records area',
      })}
      css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, flex: 1, minHeight: 0 }}
    >
      {Array.from({ length: ROW_COUNT }, (_, i) => (
        <GenericSkeleton key={i} style={{ height: ROW_HEIGHT }} />
      ))}
    </div>
  );
};

import { Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export interface DatasetRecordsCountProps {
  /** Count of records currently passing the active search filter. */
  filtered: number;
  /** Total count of records in the dataset (unfiltered). */
  total: number;
  /** When true, renders the "X of Y records" form; otherwise just "Y records". */
  hasActiveSearch: boolean;
}

export const DatasetRecordsCount = ({ filtered, total, hasActiveSearch }: DatasetRecordsCountProps) => (
  // aria-live so screen readers announce the count as it changes via search.
  <Typography.Text color="secondary" aria-live="polite" aria-atomic="true" css={{ whiteSpace: 'nowrap' }}>
    {hasActiveSearch ? (
      <FormattedMessage
        defaultMessage="{filtered, number} of {total, plural, one {# record} other {# records}}"
        description="Record count shown in the V2 dataset records toolbar when a search filter is active. {filtered} is the count matching the search; {total} is the dataset's full record count."
        values={{ filtered, total }}
      />
    ) : (
      <FormattedMessage
        defaultMessage="{count, plural, one {# record} other {# records}}"
        description="Total record count shown in the V2 dataset records toolbar when no search filter is active"
        values={{ count: total }}
      />
    )}
  </Typography.Text>
);

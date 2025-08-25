import { useEffect, useState } from 'react';

import { TableFilterInput } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

const SEARCH_QUERY_FILTER_DEBOUNCE_MS = 400;

export function GenAiTracesTableSearchInput({
  searchQuery,
  setSearchQuery,
}: {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
}) {
  const intl = useIntl();

  const [pendingUserQuery, setPendingUserQuery] = useState(searchQuery);
  // When the search query changes, update the pending user query.
  useEffect(() => {
    setPendingUserQuery(searchQuery);
  }, [searchQuery]);

  // Debounce adding the filter search query to the URL so we don't over push to the URL.
  useEffect(() => {
    const timeout = setTimeout(() => {
      setSearchQuery(pendingUserQuery);
    }, SEARCH_QUERY_FILTER_DEBOUNCE_MS);
    return () => clearTimeout(timeout);
  }, [pendingUserQuery, setSearchQuery]);

  return (
    <TableFilterInput
      componentId="mlflow.evaluations_review.table_ui.filter_input"
      placeholder={intl.formatMessage({
        defaultMessage: 'Search traces by request',
        description: 'Placeholder text for the search input in the trace results table',
      })}
      value={pendingUserQuery}
      onChange={(e) => setPendingUserQuery(e.target.value)}
    />
  );
}

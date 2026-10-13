import { Empty, SearchIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const IssuesTabEmptyState = () => {
  return (
    <Empty
      image={<SearchIcon />}
      title={
        <FormattedMessage
          defaultMessage="No issues found"
          description="Issue detection run details > Issues tab > Empty state title"
        />
      }
      description={
        <FormattedMessage
          defaultMessage="Issues identified from traces will appear here."
          description="Issue detection run details > Issues tab > Empty state description"
        />
      }
    />
  );
};

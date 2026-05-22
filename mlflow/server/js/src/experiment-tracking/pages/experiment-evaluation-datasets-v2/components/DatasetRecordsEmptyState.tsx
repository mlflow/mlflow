import {
  Button,
  Empty,
  PlusIcon,
  SearchIcon,
  TableIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { getEvalMonitorDocsLink } from '../utils/docsLinks';

interface NoRecordsEmptyStateProps {
  onAddRecord: () => void;
}

/**
 * Initial empty state for a freshly-created dataset. Per spec: one primary "Add record"
 * button and a single link to the SDK docs for programmatic record creation.
 */
export const DatasetRecordsEmptyState = ({ onAddRecord }: NoRecordsEmptyStateProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: 400,
        width: '100%',
        padding: theme.spacing.md,
      }}
    >
      <div
        aria-hidden
        css={{
          color: theme.colors.textSecondary,
          marginBottom: theme.spacing.md,
          // Match the visual weight of the list page's hero illustration so the two empty
          // states feel like a pair.
          fontSize: 96,
          lineHeight: 1,
        }}
      >
        <TableIcon />
      </div>
      <Typography.Title level={3} css={{ marginTop: 0 }}>
        <FormattedMessage
          defaultMessage="Add dataset records"
          description="Title for the V2 dataset records empty state"
        />
      </Typography.Title>
      <Typography.Paragraph color="secondary" css={{ maxWidth: 520, textAlign: 'center' }}>
        <FormattedMessage
          defaultMessage="Add records to start evaluating your app. {sdkLink}"
          description="Description text for the V2 dataset records empty state"
          values={{
            sdkLink: (
              <Typography.Link
                componentId="mlflow.eval-datasets-v2.records.empty.sdk-docs-link"
                href={getEvalMonitorDocsLink('build-eval-dataset')}
                openInNewTab
              >
                <FormattedMessage
                  defaultMessage="See SDK docs to add records programmatically"
                  description="Link text to SDK docs for programmatic dataset record creation"
                />
              </Typography.Link>
            ),
          }}
        />
      </Typography.Paragraph>
      <Button
        componentId="mlflow.eval-datasets-v2.records.empty.add-record"
        type="primary"
        icon={<PlusIcon />}
        onClick={onAddRecord}
        css={{ marginTop: theme.spacing.md }}
      >
        <FormattedMessage
          defaultMessage="Add record"
          description="Primary button text for adding a new dataset record from the empty state"
        />
      </Button>
      {/* Counterbalance the ~112px icon block above the title so flex centering
       * lands the text near the optical center instead of below it. The
       * spacer is intentionally shorter than the icon block (xl * 2.5 = 80px
       * vs ~112px), preserving a small downward bias that keeps the icon's
       * presence felt. */}
      <div aria-hidden css={{ height: theme.spacing.xl * 2.5 }} />
    </div>
  );
};

interface NoRecordsSearchResultsProps {
  searchQuery: string;
  onClearSearch: () => void;
}

export const DatasetRecordsNoResultsEmptyState = ({ searchQuery, onClearSearch }: NoRecordsSearchResultsProps) => {
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: 320,
        width: '100%',
        '& > div': {
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
        },
      }}
    >
      <Empty
        image={<SearchIcon />}
        title={
          <FormattedMessage
            defaultMessage='No records match "{query}"'
            description="Title for the empty state when a search returns no records in the V2 dataset detail page"
            values={{ query: searchQuery }}
          />
        }
        description={
          <Typography.Link
            componentId="mlflow.eval-datasets-v2.records.empty.clear-search-link"
            onClick={onClearSearch}
          >
            <FormattedMessage
              defaultMessage="Clear the search"
              description="Link text to clear the search filter on the V2 dataset records list"
            />
          </Typography.Link>
        }
      />
    </div>
  );
};

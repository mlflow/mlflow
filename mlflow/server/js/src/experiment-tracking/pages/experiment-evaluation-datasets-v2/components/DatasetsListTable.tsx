import {
  Button,
  CursorPagination,
  DropdownMenu,
  OverflowIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowAction,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { getTimeAgoStrings } from '@databricks/web-shared/browse';
import { Link } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import type { Dataset } from '../hooks/useDatasetsQueries';
import { truncateCss } from './DatasetRecordCell';

interface DatasetsListTableProps {
  experimentId: string;
  datasets: Dataset[];
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  onDeleteDataset: (dataset: Dataset) => void;
  // Pagination footer is hidden when there's only a single page; passes through to CursorPagination.
  showPagination: boolean;
  /**
   * True whenever a user-initiated query change (search submit/clear or pagination
   * next/prev) is in flight. The data rows are replaced with skeleton rows so the user has
   * feedback that the previous result is stale; the header row stays intact. Refresh
   * refetches keep prior rows visible instead.
   */
  isLoadingRows: boolean;
}

const cellStyles = { verticalAlign: 'middle' as const };
// Headers share the same `verticalAlign` so single-line header text sits on the same
// baseline as the vertically-centered cell content below. Without this, headers default
// to top-aligned and read as misaligned against the centered cell rows.
const headerStyles = { verticalAlign: 'middle' as const };
// Dataset names can be long; give Name a heavier flex weight so it claims roughly twice
// the width of the other columns. Applied to header and cells so the columns align.
const nameCellStyles = { ...cellStyles, flex: 2 };
const nameHeaderStyles = { ...headerStyles, flex: 2 };
const SKELETON_ROW_COUNT = 5;

export const DatasetsListTable = ({
  experimentId,
  datasets,
  hasNextPage,
  hasPreviousPage,
  onNextPage,
  onPreviousPage,
  onDeleteDataset,
  showPagination,
  isLoadingRows,
}: DatasetsListTableProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  // Match the in-flight skeleton count to the rows we're replacing so the table doesn't
  // shrink during a pagination transition (a full page on either side of the boundary
  // shows the same height). Fall back to SKELETON_ROW_COUNT for the search-from-empty
  // case where there's no prior data to anchor the height.
  const skeletonRowCount = datasets.length || SKELETON_ROW_COUNT;

  const renderDate = (iso?: string) => {
    if (!iso) return <Typography.Text color="secondary">-</Typography.Text>;
    const { displayText, tooltipTitle } = getTimeAgoStrings({ date: new Date(iso), intl });
    return (
      <span css={truncateCss} title={tooltipTitle}>
        {displayText}
      </span>
    );
  };

  return (
    <div
      css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}
      role="region"
      aria-busy={isLoadingRows}
      aria-label={intl.formatMessage({
        defaultMessage: 'Datasets',
        description: 'Region label for the V2 evaluation datasets table — wraps the table and pagination',
      })}
    >
      <Table>
        <TableRow isHeader>
          <TableHeader componentId="mlflow.eval-datasets-v2.list.header.name" css={nameHeaderStyles}>
            <FormattedMessage defaultMessage="Name" description="Header for the dataset name column" />
          </TableHeader>
          <TableHeader componentId="mlflow.eval-datasets-v2.list.header.created-by" css={headerStyles}>
            <FormattedMessage defaultMessage="Created by" description="Header for the dataset created-by column" />
          </TableHeader>
          <TableHeader componentId="mlflow.eval-datasets-v2.list.header.last-updated" css={headerStyles}>
            <FormattedMessage defaultMessage="Last updated" description="Header for the dataset last-updated column" />
          </TableHeader>
          <TableHeader componentId="mlflow.eval-datasets-v2.list.header.created-at" css={headerStyles}>
            <FormattedMessage defaultMessage="Created at" description="Header for the dataset created-at column" />
          </TableHeader>
          {/* Blank actions slot matches the v1 column — DS TableRowAction renders the column header role and gives the slot the same 32px width as the body rows, keeping every column aligned. */}
          <TableRowAction />
        </TableRow>
        {isLoadingRows
          ? Array.from({ length: skeletonRowCount }, (_, i) => (
              <TableRow key={`skeleton-${i}`}>
                <TableCell css={nameCellStyles}>
                  <TableSkeleton seed={`datasets-name-${i}`} />
                </TableCell>
                <TableCell css={cellStyles}>
                  <TableSkeleton seed={`datasets-created-by-${i}`} />
                </TableCell>
                <TableCell css={cellStyles}>
                  <TableSkeleton seed={`datasets-last-updated-${i}`} />
                </TableCell>
                <TableCell css={cellStyles}>
                  <TableSkeleton seed={`datasets-created-at-${i}`} />
                </TableCell>
                <TableRowAction />
              </TableRow>
            ))
          : datasets.map((dataset) => (
              <TableRow key={dataset.dataset_id}>
                <TableCell css={nameCellStyles}>
                  <Link componentId="mlflow.eval-datasets-v2.list.dataset-link"
                    to={Routes.getExperimentPageDatasetDetailRoute(experimentId, dataset.dataset_id)}
                    css={truncateCss}
                  >
                    {dataset.name ?? dataset.dataset_id}
                  </Link>
                </TableCell>
                <TableCell css={cellStyles}>
                  {dataset.created_by ? (
                    <span css={truncateCss}>{dataset.created_by}</span>
                  ) : (
                    <Typography.Text color="secondary">-</Typography.Text>
                  )}
                </TableCell>
                <TableCell css={cellStyles}>{renderDate(dataset.last_update_time)}</TableCell>
                <TableCell css={cellStyles}>{renderDate(dataset.create_time)}</TableCell>
                <TableRowAction>
                  <DropdownMenu.Root>
                    <DropdownMenu.Trigger asChild>
                      <Button
                        componentId="mlflow.eval-datasets-v2.list.row.actions"
                        size="small"
                        icon={<OverflowIcon />}
                        aria-label={intl.formatMessage(
                          {
                            defaultMessage: 'Actions for dataset {name}',
                            description:
                              'Aria label for the per-row actions dropdown trigger on the V2 evaluation datasets list',
                          },
                          { name: dataset.name ?? dataset.dataset_id },
                        )}
                        css={{ padding: '4px' }}
                      />
                    </DropdownMenu.Trigger>
                    <DropdownMenu.Content align="end">
                      <DropdownMenu.Item
                        componentId="mlflow.eval-datasets-v2.list.row.delete"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteDataset(dataset);
                        }}
                      >
                        <FormattedMessage
                          defaultMessage="Delete"
                          description="Dropdown item label to delete a V2 evaluation dataset row"
                        />
                      </DropdownMenu.Item>
                    </DropdownMenu.Content>
                  </DropdownMenu.Root>
                </TableRowAction>
              </TableRow>
            ))}
      </Table>
      {showPagination && (
        <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
          <CursorPagination
            componentId="mlflow.eval-datasets-v2.list.pagination"
            hasNextPage={hasNextPage}
            hasPreviousPage={hasPreviousPage}
            onNextPage={onNextPage}
            onPreviousPage={onPreviousPage}
          />
        </div>
      )}
    </div>
  );
};

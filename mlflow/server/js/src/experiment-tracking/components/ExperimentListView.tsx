import React, { Component, useMemo } from 'react';
import { Theme } from '@emotion/react';
import {
  Checkbox,
  WithDesignSystemThemeHoc,
  DesignSystemHocProps,
  Button,
  Tooltip,
  useDesignSystemTheme,
  Empty,
  NoIcon,
  Table,
  CursorPagination,
  TableRow,
  TableHeader,
  TableSkeletonRows,
  TableCell,
  TableFilterLayout,
  TableFilterInput,
  Spacer,
  Header,
  Popover,
  InfoIcon,
  Typography,
} from '@databricks/design-system';
import 'react-virtualized/styles.css';
import { Link } from '../../common/utils/RoutingUtils';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { withRouterNext, WithRouterNextProps } from '../../common/utils/withRouterNext';
import { ExperimentEntity } from '../types';
import { defaultContext, QueryClient } from '../../common/utils/reactQueryHooks';
import { ExperimentListQueryKeyHeader, useExperimentListQuery } from './experiment-page/hooks/useExperimentListQuery';
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  OnChangeFn,
  RowSelectionState,
  Updater,
  useReactTable,
} from '@tanstack/react-table';
import { isEmpty } from 'lodash';
import { defineMessage, FormattedMessage, useIntl } from 'react-intl';
import Utils from '../../common/utils/Utils';
import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { ExperimentSearchSyntaxDocUrl } from '../../common/constants';

type Props = {
  activeExperimentIds: string[];
  experiments: ExperimentEntity[];
  pagination: Pick<
    ReturnType<typeof useExperimentListQuery>,
    'hasNextPage' | 'hasPreviousPage' | 'onNextPage' | 'onPreviousPage' | 'isLoading' | 'error'
  >;
} & WithRouterNextProps &
  DesignSystemHocProps;

type State = {
  checkedKeys: string[];
  searchInput: string;
  showCreateExperimentModal: boolean;
};

export class ExperimentListView extends Component<Props, State> {
  static contextType = defaultContext;
  // declare context: QueryClient; // FIXME

  invalidateExperimentList = () => {
    this.context.invalidateQueries({ queryKey: [ExperimentListQueryKeyHeader] });
  };

  state = {
    checkedKeys: this.props.activeExperimentIds,
    searchInput: '',
    showCreateExperimentModal: false,
  };

  filterExperiments = (searchInput: string) => {
    const { experiments } = this.props;
    const lowerCasedSearchInput = searchInput.toLowerCase();
    return lowerCasedSearchInput === ''
      ? this.props.experiments
      : experiments.filter(({ name }) => name.toLowerCase().includes(lowerCasedSearchInput));
  };

  handleSearchInputChange: React.ChangeEventHandler<HTMLInputElement> = (event) => {
    this.setState({
      searchInput: event.target.value,
    });
  };

  handleCreateExperiment = () => {
    this.setState({
      showCreateExperimentModal: true,
    });
  };

  handleCloseCreateExperimentModal = () => {
    this.setState({
      showCreateExperimentModal: false,
    });
  };

  pushExperimentRoute = () => {
    if (this.state.checkedKeys.length > 0) {
      const route =
        this.state.checkedKeys.length === 1
          ? Routes.getExperimentPageRoute(this.state.checkedKeys[0])
          : Routes.getCompareExperimentsPageRoute(this.state.checkedKeys);
      this.props.navigate(route);
    }
  };

  getSelectedRows = () => {
    return Object.fromEntries(
      this.props.experiments.map((experiment) => [
        experiment.experimentId,
        this.state.checkedKeys.includes(experiment.experimentId),
      ]),
    );
  };

  setSelectedRows = (updater: Updater<RowSelectionState>) => {
    const rowSelection = this.getSelectedRows();
    const newRowSelection = typeof updater === 'function' ? updater(rowSelection) : updater;

    this.setState({
      checkedKeys: Object.entries(newRowSelection)
        .filter(([_, value]) => value)
        .map(([key, _]) => key),
    });
  };

  render() {
    const { pagination } = this.props;
    const { error, isLoading, onNextPage, onPreviousPage, hasNextPage, hasPreviousPage } = pagination;

    const { searchInput } = this.state;
    const filteredExperiments = this.filterExperiments(searchInput);

    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Spacer shrinks={false} />
        <Header
          title={<FormattedMessage defaultMessage="Experiments" description="Header title for the experiments page" />}
          buttons={
            <>
              <Button
                componentId="mlflow.experiment_list_view.new_experiment_button"
                type="primary"
                onClick={this.handleCreateExperiment}
                data-testid="create-experiment-button"
              >
                Create experiment
              </Button>
              <Tooltip
                componentId="mlflow.experiment_list_view.compare_experiments_button"
                content="Select at least two experiments from the table to compare them"
              >
                <Button
                  componentId="mlflow.experiment_list_view.new_experiment_button"
                  onClick={this.pushExperimentRoute}
                  data-testid="create-experiment-button"
                  disabled={this.state.checkedKeys.length < 2}
                >
                  Compare experiments
                </Button>
              </Tooltip>
            </>
          }
        />
        <Spacer shrinks={false} />
        <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <TableFilterLayout>
            <TableFilterInput
              placeholder="Search experiments by name"
              componentId="mlflow.experiment_list_view.search"
              value={searchInput}
              onChange={this.handleSearchInputChange}
              suffix={<ModelSearchInputHelpTooltip exampleEntityName="my-prompt-name" />}
            />
          </TableFilterLayout>
          <ExperimentListTable
            experiments={filteredExperiments}
            error={error}
            hasNextPage={hasNextPage}
            hasPreviousPage={hasPreviousPage}
            isLoading={isLoading}
            isFiltered={Boolean(searchInput)}
            onNextPage={onNextPage}
            onPreviousPage={onPreviousPage}
            rowSelection={this.getSelectedRows()}
            setRowSelection={this.setSelectedRows}
          />
        </div>
        <CreateExperimentModal
          isOpen={this.state.showCreateExperimentModal}
          onClose={this.handleCloseCreateExperimentModal}
          invalidate={this.invalidateExperimentList}
        />
      </ScrollablePageWrapper>
    );
  }
}

export default withRouterNext(WithDesignSystemThemeHoc(ExperimentListView));

const ExperimentListTableCell: ColumnDef<ExperimentEntity>['cell'] = ({ row: { original } }) => {
  // const dataTestId = isActive ? 'active-experiment-list-item' : 'experiment-list-item';
  return (
    <Link
      className="experiment-link"
      css={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 }}
      to={Routes.getExperimentPageRoute(original.experimentId)}
      title={original.name}
      // onClick={() => this.setState({ checkedKeys: [item.experimentId] })}
      // data-testid={`${dataTestId}-link`}
    >
      {original.name}
    </Link>
  );
};

const ExperimentListCheckbox: ColumnDef<ExperimentEntity>['cell'] = ({ row }) => {
  return (
    <Checkbox
      componentId="mlflow.experiment_list_view.check_box"
      id={row.original.experimentId}
      key={row.original.experimentId}
      data-testid={`experiment-list-item-check-box`}
      isChecked={row.getIsSelected()}
      disabled={!row.getCanSelect()}
      onChange={row.getToggleSelectedHandler()}
    />
  );
};

type ExperimentTableColumnDef = ColumnDef<ExperimentEntity>;

const useExperimentsTableColumns = () => {
  const intl = useIntl();
  return useMemo(() => {
    const resultColumns: ExperimentTableColumnDef[] = [
      {
        header: ({ table }) => (
          <Checkbox
            componentId="mlflow.experiment_list_view.check_all_box"
            isChecked={table.getIsAllRowsSelected()}
            onChange={(_, event) => table.getToggleAllRowsSelectedHandler()(event)}
            // indeterminate={table.getIsSomeRowsSelected()}
          />
        ),
        id: 'select',
        cell: ExperimentListCheckbox,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Name',
          description: 'Header for the name column in the experiments table',
        }),
        accessorKey: 'name',
        id: 'name',
        cell: ExperimentListTableCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Time created',
          description: 'Header for the time created column in the experiments table',
        }),
        id: 'timeCreated',
        accessorFn: ({ creationTime }) => Utils.formatTimestamp(creationTime, intl),
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Last modified',
          description: 'Header for the last modified column in the experiments table',
        }),
        id: 'lastModified',
        accessorFn: ({ lastUpdateTime }) => Utils.formatTimestamp(lastUpdateTime, intl),
      },
    ];

    return resultColumns;
  }, [intl]);
};

export const ExperimentListTable = ({
  experiments,
  hasNextPage,
  hasPreviousPage,
  isLoading,
  isFiltered,
  onNextPage,
  onPreviousPage,
  rowSelection,
  setRowSelection,
}: {
  experiments?: ExperimentEntity[];
  error?: Error;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  isLoading?: boolean;
  isFiltered?: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  rowSelection: RowSelectionState;
  setRowSelection: OnChangeFn<RowSelectionState>;
}) => {
  const { theme } = useDesignSystemTheme();
  const columns = useExperimentsTableColumns();

  const table = useReactTable({
    data: experiments ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.experimentId,
    enableRowSelection: true,
    enableMultiRowSelection: true,
    onRowSelectionChange: setRowSelection,
    state: { rowSelection },
  });

  const getEmptyState = () => {
    const isEmptyList = !isLoading && isEmpty(experiments);
    if (isEmptyList && isFiltered) {
      return (
        <Empty
          image={<NoIcon />}
          title={
            <FormattedMessage
              defaultMessage="No experiments found"
              description="Label for the empty state in the experiments table when no experiments are found"
            />
          }
          description={null}
        />
      );
    }
    if (isEmptyList) {
      return (
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No experiments created"
              description="A header for the empty state in the experiments table"
            />
          }
          description={
            <FormattedMessage
              defaultMessage='Use "Create experiment" button in order to create a new experiment'
              description="Guidelines for the user on how to create a new experiment in the experiments list page"
            />
          }
        />
      );
    }

    return null;
  };

  const selectColumnStyles = { flex: 'none' };

  return (
    <Table
      scrollable
      pagination={
        <CursorPagination
          hasNextPage={hasNextPage}
          hasPreviousPage={hasPreviousPage}
          onNextPage={onNextPage}
          onPreviousPage={onPreviousPage}
          componentId="mlflow.experiment_list_view.pagination"
        />
      }
      empty={getEmptyState()}
    >
      <TableRow isHeader>
        {table.getLeafHeaders().map((header) => (
          <TableHeader
            componentId="mlflow.experiment_list_view.table.header"
            key={header.id}
            css={header.column.id === 'select' ? selectColumnStyles : undefined}
          >
            {flexRender(header.column.columnDef.header, header.getContext())}
          </TableHeader>
        ))}
      </TableRow>
      {isLoading ? (
        <TableSkeletonRows table={table} />
      ) : (
        table.getRowModel().rows.map((row) => (
          <TableRow key={row.id} css={{ height: theme.general.buttonHeight }}>
            {row.getAllCells().map((cell) => (
              <TableCell
                key={cell.id}
                css={{ alignItems: 'center', ...(cell.column.id === 'select' ? selectColumnStyles : undefined) }}
              >
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </TableCell>
            ))}
          </TableRow>
        ))
      )}
    </Table>
  );
};

const ModelSearchInputHelpTooltip = () => {
  const { formatMessage } = useIntl();
  const tooltipIntroMessage = defineMessage({
    defaultMessage:
      'A filter expression over experiment attributes and tags that allows returning a subset of experiments.',
    description: 'Tooltip string to explain how to search experiments',
  });

  // Tooltips are not expected to contain links.
  const labelText = formatMessage(tooltipIntroMessage, { newline: ' ', whereBold: 'WHERE' });

  return (
    <Popover.Root componentId="codegen_mlflow_app_src_model-registry_components_model-list_modellistfilters.tsx_46">
      <Popover.Trigger
        aria-label={labelText}
        css={{ border: 0, background: 'none', padding: 0, lineHeight: 0, cursor: 'pointer' }}
      >
        <InfoIcon />
      </Popover.Trigger>
      <Popover.Content align="start">
        <div>
          <FormattedMessage {...tooltipIntroMessage} />
          <br /> The syntax is a subset of SQL that supports ANDing together binary operations between an attribute or
          tag, and a constant.
          <br />
          <FormattedMessage
            defaultMessage="<link>Learn more</link>"
            description="Learn more tooltip link to learn more on how to search models"
            values={{
              link: (chunks) => (
                <Typography.Link
                  componentId="codegen_mlflow_app_src_model-registry_components_model-list_modellistfilters.tsx_61"
                  href={ExperimentSearchSyntaxDocUrl + '#syntax'}
                  openInNewTab
                >
                  {chunks}
                </Typography.Link>
              ),
            }}
          />
          <br />
          <br />
          <FormattedMessage defaultMessage="Examples:" description="Text header for examples of mlflow search syntax" />
          <br />
          • "attributes.name = 'x'" # or "name = 'x'"
          <br />
          • "attributes.name LIKE 'x%'"
          <br />
          • "tags.group != 'x'"
          <br />
          • "tags.group ILIKE '%x%'"
          <br />• "attributes.name LIKE 'x%' AND tags.group = 'y'"
        </div>
        <Popover.Arrow />
      </Popover.Content>
    </Popover.Root>
  );
};

import React, { Component, useMemo } from 'react';
import { Theme } from '@emotion/react';
import {
  Checkbox,
  Input,
  PencilIcon,
  WithDesignSystemThemeHoc,
  DesignSystemHocProps,
  Button,
  TrashIcon,
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
} from '@databricks/design-system';
import { List as VList, ListRowRenderer } from 'react-virtualized';
import 'react-virtualized/styles.css';
import { Link } from '../../common/utils/RoutingUtils';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { DeleteExperimentModal } from './modals/DeleteExperimentModal';
import { RenameExperimentModal } from './modals/RenameExperimentModal';
import { withRouterNext, WithRouterNextProps } from '../../common/utils/withRouterNext';
import { ExperimentEntity } from '../types';
import { defaultContext, QueryClient } from '../../common/utils/reactQueryHooks';
import { ExperimentListQueryKeyHeader, useExperimentListQuery } from './experiment-page/hooks/useExperimentListQuery';
import { PageContainer } from '../../common/components/PageContainer';
import { PageHeader } from '../../shared/building_blocks/PageHeader';
import { ColumnDef, flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import { isEmpty } from 'lodash';
import { FormattedMessage, useIntl } from 'react-intl';
import Utils from '../../common/utils/Utils';
import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';

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
  showDeleteExperimentModal: boolean;
  showRenameExperimentModal: boolean;
  selectedExperimentId: string;
  selectedExperimentName: string;
};

export class ExperimentListView extends Component<Props, State> {
  static contextType = defaultContext;
  // declare context: QueryClient; // FIXME

  invalidateExperimentList = () => {
    this.context.invalidateQueries({ queryKey: [ExperimentListQueryKeyHeader] });
  };

  list?: VList = undefined;

  state = {
    checkedKeys: this.props.activeExperimentIds,
    searchInput: '',
    showCreateExperimentModal: false,
    showDeleteExperimentModal: false,
    showRenameExperimentModal: false,
    selectedExperimentId: '0',
    selectedExperimentName: '',
  };

  bindListRef = (ref: VList) => {
    this.list = ref;
  };

  componentDidUpdate = () => {
    // Ensure the filter is applied
    if (this.list) {
      this.list.forceUpdateGrid();
    }
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

  updateSelectedExperiment = (experimentId: string, experimentName: string) => {
    this.setState({
      selectedExperimentId: experimentId,
      selectedExperimentName: experimentName,
    });
  };

  handleCreateExperiment = () => {
    this.setState({
      showCreateExperimentModal: true,
    });
  };

  handleDeleteExperiment = (experimentId: string, experimentName: string) => () => {
    this.setState({
      showDeleteExperimentModal: true,
    });
    this.updateSelectedExperiment(experimentId, experimentName);
  };

  handleRenameExperiment = (experimentId: string, experimentName: string) => () => {
    this.setState({
      showRenameExperimentModal: true,
    });
    this.updateSelectedExperiment(experimentId, experimentName);
  };

  handleCloseCreateExperimentModal = () => {
    this.setState({
      showCreateExperimentModal: false,
    });
  };

  handleCloseDeleteExperimentModal = () => {
    this.setState({
      showDeleteExperimentModal: false,
    });
    // reset
    this.updateSelectedExperiment('0', '');
  };

  handleCloseRenameExperimentModal = () => {
    this.setState({
      showRenameExperimentModal: false,
    });
    // reset
    this.updateSelectedExperiment('0', '');
  };

  // Add a key if it does not exist, remove it if it does
  // Always keep at least one experiment checked if it is only the active one.
  handleCheck = (isChecked: boolean, key: string) => {
    this.setState((prevState, props) => {
      let { checkedKeys } = prevState;
      if (isChecked === true && !props.activeExperimentIds.includes(key)) {
        checkedKeys = [key, ...props.activeExperimentIds];
      }
      if (isChecked === false && props.activeExperimentIds.length !== 1) {
        checkedKeys = props.activeExperimentIds.filter((i) => i !== key);
      }
      return { checkedKeys: checkedKeys };
    }, this.pushExperimentRoute);
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

  renderListItem: ListRowRenderer = ({ index, key, style, parent }) => {
    // Use the parents props to index.
    const item = parent.props['data'][index];
    const { activeExperimentIds } = this.props;
    const isActive = activeExperimentIds.includes(item.experimentId);
    const dataTestId = isActive ? 'active-experiment-list-item' : 'experiment-list-item';
    const { theme } = this.props.designSystemThemeApi;
    // Clicking the link removes all checks and marks other experiments
    // as not active.
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          // gap: theme.spacing.xs,
          paddingLeft: theme.spacing.xs,
          paddingRight: theme.spacing.xs,
          borderLeft: isActive ? `solid ${theme.colors.primary}` : 'solid transparent',
          borderLeftWidth: 4,
          backgroundColor: isActive ? theme.colors.actionDefaultBackgroundPress : 'transparent',
          fontSize: theme.typography.fontSizeBase,
          svg: {
            width: 14,
            height: 14,
          },
        }}
        data-testid={dataTestId}
        key={key}
        style={style}
      >
        <Checkbox
          componentId="mlflow.experiment_list_view.check_box"
          id={item.experimentId}
          key={item.experimentId}
          onChange={(isChecked) => this.handleCheck(isChecked, item.experimentId)}
          isChecked={isActive}
          data-testid={`${dataTestId}-check-box`}
        />
        <Link
          className="experiment-link"
          css={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 }}
          to={Routes.getExperimentPageRoute(item.experimentId)}
          onClick={() => this.setState({ checkedKeys: [item.experimentId] })}
          title={item.name}
          data-testid={`${dataTestId}-link`}
        >
          {item.name}
        </Link>
        <Tooltip componentId="mlflow.experiment_list_view.rename_experiment_button.tooltip" content="Rename experiment">
          <Button
            type="link"
            componentId="mlflow.experiment_list_view.rename_experiment_button"
            icon={<PencilIcon />}
            onClick={this.handleRenameExperiment(item.experimentId, item.name)}
            data-testid="rename-experiment-button"
            size="small"
          />
        </Tooltip>
        <Tooltip componentId="mlflow.experiment_list_view.delete_experiment_button.tooltip" content="Delete experiment">
          <Button
            type="link"
            componentId="mlflow.experiment_list_view.delete_experiment_button"
            icon={<TrashIcon />}
            onClick={this.handleDeleteExperiment(item.experimentId, item.name)}
            data-testid="delete-experiment-button"
            size="small"
          />
        </Tooltip>
      </div>
    );
  };

  render() {
    const { activeExperimentIds, designSystemThemeApi, pagination } = this.props;
    const { error, isLoading, onNextPage, onPreviousPage, hasNextPage, hasPreviousPage } = pagination;
    const { theme } = designSystemThemeApi;

    const { searchInput } = this.state;
    const filteredExperiments = this.filterExperiments(searchInput);

    return (
      <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Spacer shrinks={false} />
        <Header
          title={<FormattedMessage defaultMessage="Experiments" description="Header title for the experiments page" />}
          buttons={
            <Button
              componentId="mlflow.experiment_list_view.new_experiment_button"
              type="primary"
              onClick={this.handleCreateExperiment}
              data-testid="create-experiment-button"
            >
              Create experiment
            </Button>
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
              // TODO: Add this back once we support searching with tags
              // suffix={<ModelSearchInputHelpTooltip exampleEntityName="my-prompt-name" />}
            />
          </TableFilterLayout>
          <ExperimentListTable
            experiments={filteredExperiments}
            // error={error}
            hasNextPage={hasNextPage}
            hasPreviousPage={hasPreviousPage}
            isLoading={isLoading}
            isFiltered={Boolean(searchInput)}
            onNextPage={onNextPage}
            onPreviousPage={onPreviousPage}
            // onEditTags={showEditPromptTagsModal}
          />
          {/* <AutoSizer>
            {({ width, height }) => (
              <VList
                rowRenderer={this.renderListItem}
                data={filteredExperiments}
                ref={this.bindListRef}
                rowHeight={32}
                overscanRowCount={10}
                height={height}
                width={width}
                rowCount={filteredExperiments.length}
              />
            )}
          </AutoSizer> */}
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

type ExperimentTableColumnDef = ColumnDef<ExperimentEntity>;

const useExperimentsTableColumns = () => {
  const intl = useIntl();
  return useMemo(() => {
    const resultColumns: ExperimentTableColumnDef[] = [
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
        id: 'lastModified',
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
}: // onEditTags,
{
  experiments?: ExperimentEntity[];
  error?: Error;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  isLoading?: boolean;
  isFiltered?: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  // onEditTags: (editedEntity: ExperimentEntity) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const columns = useExperimentsTableColumns();

  const table = useReactTable({
    data: experiments ?? [],
    columns,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row, index) => row.name ?? index.toString(),
    // meta: { onEditTags } satisfies PromptsTableMetadata,
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
          <TableHeader componentId="mlflow.experiment_list_view.table.header" key={header.id}>
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
              <TableCell key={cell.id} css={{ alignItems: 'center' }}>
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </TableCell>
            ))}
          </TableRow>
        ))
      )}
    </Table>
  );
};

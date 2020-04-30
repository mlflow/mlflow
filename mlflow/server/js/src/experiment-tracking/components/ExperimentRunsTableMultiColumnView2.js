/**
 * Ag-grid based implementation of multi-column view with a bunch of new interactive features
 */
import React from 'react';
import PropTypes from 'prop-types';
import { RunInfo } from '../sdk/MlflowMessages';
import { Link } from 'react-router-dom';
import Routes from '../routes';
import Utils from '../../common/utils/Utils';

import { AgGridReact } from '@ag-grid-community/react/main';
import { Grid } from '@ag-grid-community/core';
import { ClientSideRowModelModule } from '@ag-grid-community/client-side-row-model';
import { RunsTableCustomHeader } from '../../common/components/ag-grid/RunsTableCustomHeader';
import '@ag-grid-community/core/dist/styles/ag-grid.css';
import '@ag-grid-community/core/dist/styles/ag-theme-balham.css';
import ExperimentViewUtil from './ExperimentViewUtil';
import { LoadMoreBar } from './LoadMoreBar';
import _ from 'lodash';
import { Spinner } from '../../common/components/Spinner';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { AgGridPersistedState } from '../sdk/MlflowLocalStorageMessages';
import { ColumnTypes } from '../constants';

const PARAM_PREFIX = '$$$param$$$';
const METRIC_PREFIX = '$$$metric$$$';
const TAG_PREFIX = '$$$tag$$$';
const MAX_PARAMS_COLS = 3;
const MAX_METRICS_COLS = 3;
const MAX_TAG_COLS = 3;
const EMPTY_CELL_PLACEHOLDER = '-';

export class ExperimentRunsTableMultiColumnView2 extends React.Component {
  static propTypes = {
    experimentId: PropTypes.string,
    runInfos: PropTypes.arrayOf(RunInfo).isRequired,
    // List of list of params in all the visible runs
    paramsList: PropTypes.arrayOf(Array).isRequired,
    // List of list of metrics in all the visible runs
    metricsList: PropTypes.arrayOf(Array).isRequired,
    paramKeyList: PropTypes.arrayOf(PropTypes.string),
    metricKeyList: PropTypes.arrayOf(PropTypes.string),
    visibleTagKeyList: PropTypes.arrayOf(PropTypes.string),
    // List of tags dictionary in all the visible runs.
    tagsList: PropTypes.arrayOf(Object).isRequired,
    onSelectionChange: PropTypes.func.isRequired,
    onExpand: PropTypes.func.isRequired,
    onSortBy: PropTypes.func.isRequired,
    orderByKey: PropTypes.string,
    orderByAsc: PropTypes.bool,
    runsSelected: PropTypes.object.isRequired,
    runsExpanded: PropTypes.object.isRequired,
    nextPageToken: PropTypes.string,
    handleLoadMoreRuns: PropTypes.func.isRequired,
    loadingMore: PropTypes.bool.isRequired,
    isLoading: PropTypes.bool.isRequired,
    categorizedUncheckedKeys: PropTypes.object.isRequired,
  };

  static defaultColDef = {
    width: 100,
    headerComponentParams: { menuIcon: 'fa-bars' },
    resizable: true,
    filter: true,
    suppressMenu: true,
    suppressMovable: true,
  };

  // A map of framework(React) specific custom components.
  // Ideally we should minimize usage of framework cell renderer for better scrolling performance.
  // More scrolling performance related discussion is available here:
  // https://www.ag-grid.com/javascript-grid-performance/#3-create-fast-cell-renderers
  static frameworkComponents = {
    sourceCellRenderer: SourceCellRenderer,
    versionCellRenderer: VersionCellRenderer,
    dateCellRenderer: DateCellRenderer,
    agColumnHeader: RunsTableCustomHeader,
    loadingOverlayComponent: Spinner,
  };

  /**
   * Returns a { name: value } map from a list of parameters/metrics/tags list
   * @param list - all available parameter/metric/tag metadata objects
   * @param keys - selected parameter/metric/tag keys
   * @param prefix - category based string prefix to for the new key, we need this prefix to
   * prevent name collision among parameters/metrics/tags
   */
  static getNameValueMapFromList(list, keys, prefix) {
    const map = {};
    // populate with default placeholder '-'
    keys.forEach((key) => {
      map[`${prefix}-${key}`] = EMPTY_CELL_PLACEHOLDER;
    });
    // override with existing value
    list.forEach(({ key, value }) => {
      if (value || _.isNumber(value)) {
        map[`${prefix}-${key}`] = value;
      }
    });
    return map;
  }

  static isFullWidthCell(rowNode) {
    return rowNode.data.isFullWidth;
  }

  getLocalStore = () =>
    LocalStorageUtils.getStoreForComponent(
      'ExperimentRunsTableMultiColumnView2',
      this.props.experimentId,
    );

  applyingRowSelectionFromProps = false;

  getColumnDefs() {
    const {
      metricKeyList,
      paramKeyList,
      categorizedUncheckedKeys,
      visibleTagKeyList,
      orderByKey,
      orderByAsc,
      onSortBy,
    } = this.props;
    const commonSortOrderProps = { orderByKey, orderByAsc, onSortBy };

    return [
      ...[
        {
          checkboxSelection: true,
          headerCheckboxSelection: true,
          pinned: 'left',
          width: 50,
        },
        {
          headerName: 'Start Time',
          field: 'startTime',
          pinned: 'left',
          width: 216,
          cellRenderer: 'dateCellRenderer',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: 'attributes.start_time',
          },
        },
        {
          headerName: 'Run Name',
          pinned: 'left',
          field: 'runName',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: 'tags.`mlflow.runName`',
          },
        },
        {
          headerName: 'User',
          field: 'user',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: 'tags.`mlflow.user`',
          },
        },
        {
          headerName: 'Source',
          field: 'source',
          cellRenderer: 'sourceCellRenderer',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: 'tags.`mlflow.source.name`',
          },
        },
        {
          headerName: 'Version',
          field: 'version',
          cellRenderer: 'versionCellRenderer',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: 'tags.`mlflow.source.git.commit`',
          },
        },
      ].filter((c) => !categorizedUncheckedKeys[ColumnTypes.ATTRIBUTES].includes(c.headerName)),
      {
        headerName: 'Parameters',
        children: paramKeyList.map((paramKey, i) => {
          const columnKey = ExperimentViewUtil.makeCanonicalKey(ColumnTypes.PARAMS, paramKey);
          return {
            headerName: paramKey,
            headerTooltip: paramKey,
            field: `${PARAM_PREFIX}-${paramKey}`,
            // `columnGroupShow` controls whether to show the column when the group is open/closed.
            // Setting it to null means always show this column.
            // Here we want to show the first 3 columns plus the current orderByKey column if it
            // happens to be inside this column group.
            columnGroupShow: i >= MAX_PARAMS_COLS && columnKey !== orderByKey ? 'open' : null,
            sortable: true,
            headerComponentParams: {
              ...commonSortOrderProps,
              canonicalSortKey: columnKey,
            },
          };
        }),
      },
      {
        headerName: 'Metrics',
        children: metricKeyList.map((metricKey, i) => {
          const columnKey = ExperimentViewUtil.makeCanonicalKey(ColumnTypes.METRICS, metricKey);
          return {
            headerName: metricKey,
            headerTooltip: metricKey,
            field: `${METRIC_PREFIX}-${metricKey}`,
            // `columnGroupShow` controls whether to show the column when the group is open/closed.
            // Setting it to null means always show this column.
            // Here we want to show the first 3 columns plus the current orderByKey column if it
            // happens to be inside this column group.
            columnGroupShow: i >= MAX_METRICS_COLS && columnKey !== orderByKey ? 'open' : null,
            sortable: true,
            headerComponentParams: {
              ...commonSortOrderProps,
              canonicalSortKey: columnKey,
            },
          };
        }),
      },
      {
        headerName: 'Tags',
        children: visibleTagKeyList.map((tagKey, i) => ({
          headerName: tagKey,
          headerTooltip: tagKey,
          field: `${TAG_PREFIX}-${tagKey}`,
          ...(i >= MAX_TAG_COLS ? { columnGroupShow: 'open' } : null),
        })),
      },
    ];
  }

  // Only run based rows are selectable, other utility rows like "load more" row is not selectable
  isRowSelectable = (rowNode) => rowNode.data && rowNode.data.runInfo;

  getRowData() {
    const {
      runInfos,
      paramsList,
      metricsList,
      paramKeyList,
      metricKeyList,
      tagsList,
      runsExpanded,
      onExpand,
      nextPageToken,
      loadingMore,
      visibleTagKeyList,
    } = this.props;
    const { getNameValueMapFromList } = ExperimentRunsTableMultiColumnView2;
    const mergedRows = ExperimentViewUtil.getRowRenderMetadata({
      runInfos,
      tagsList,
      runsExpanded,
    });

    const runs = mergedRows.map(({ idx, isParent, hasExpander, expanderOpen, childrenIds }) => {
      const tags = tagsList[idx];
      const params = paramsList[idx];
      const metrics = metricsList[idx].map(({ key, value }) => ({
        key,
        value: Utils.formatMetric(value),
      }));
      const runInfo = runInfos[idx];

      const user = Utils.getUser(runInfo, tags);
      const queryParams = window.location && window.location.search ? window.location.search : '';
      const startTime = runInfo.start_time;
      const runName = Utils.getRunName(tags) || '-';
      const visibleTags = Utils.getVisibleTagValues(tags).map(([key, value]) => ({ key, value }));

      return {
        runInfo,
        startTime,
        user,
        runName,
        tags,
        queryParams,
        isParent,
        hasExpander,
        expanderOpen,
        childrenIds,
        onExpand,
        ...getNameValueMapFromList(params, paramKeyList, PARAM_PREFIX),
        ...getNameValueMapFromList(metrics, metricKeyList, METRIC_PREFIX),
        ...getNameValueMapFromList(visibleTags, visibleTagKeyList, TAG_PREFIX),
      };
    });

    // Handle "Load more" row
    if (nextPageToken || loadingMore) {
      runs.push({
        isFullWidth: true,
        loadingMore,
      });
    }

    return runs;
  }

  handleSelectionChange = (event) => {
    // Avoid triggering event handlers while we are applying row selections from props
    if (this.applyingRowSelectionFromProps) {
      return;
    }
    const { onSelectionChange } = this.props;
    if (onSelectionChange) {
      const selectedRunUuids = [];
      event.api.getSelectedRows().forEach(({ runInfo }) => {
        if (runInfo) {
          selectedRunUuids.push(runInfo.run_uuid);
        }
      });
      // Do not trigger callback if the selection is not changed. This check helps improving
      // rendering performance especially after applyRowSelectionFromProps where a large number of
      // run selection event gets triggered because there is no batch select API.
      if (!_.isEqual(selectedRunUuids, this.prevSelectRunUuids)) {
        onSelectionChange(selectedRunUuids);
        this.prevSelectRunUuids = selectedRunUuids;
      }
    }
  };

  handleGridReady = (params) => {
    this.gridApi = params.api;
    this.columnApi = params.columnApi;
    this.applyRowSelectionFromProps();
    this.handleColumnSizeRefit();
    this.handleLoadingOverlay();
  };

  // There is no way in ag-grid to declaratively specify row selections. Thus, we have to use grid
  // api to imperatively apply row selections per render. We don't want to trigger `selectionChange`
  // event in this method.
  applyRowSelectionFromProps() {
    if (!this.gridApi) return;
    const { runsSelected } = this.props;
    const selectedRunsSet = new Set(Object.keys(runsSelected));

    this.gridApi.deselectAll();
    // ag-grid has no existing component prop or batch select API to handle row selection so we have
    // to select rows with ag-grid API one by one. Currently, ag-grid batches selection value
    // changes but still fires events per setSelected call.
    // Like when we call:
    // row1.setSelected(true);
    // row2.setSelected(true);
    // row3.setSelected(true);
    // ag-grid will fire `selectionChange` event 3 times with the same selection [row1, row2, row3]
    // So, we need to be aware of this and not re-render when selection stays the same.
    this.gridApi.forEachNode((node) => {
      const { runInfo } = node.data;
      if (runInfo && selectedRunsSet.has(runInfo.getRunUuid())) {
        node.setSelected(true);
      }
    });
  }

  handleColumnSizeRefit() {
    if (!this.gridApi || !this.columnApi) return;
    // Only re-fit columns into current viewport when there is no open column group. We are doing
    // this because opened group can have arbitrary large number of child columns which will end
    // up creating a lot of columns with extremely small width.
    const columnGroupStates = this.columnApi.getColumnGroupState();
    if (columnGroupStates.every((group) => !group.open)) {
      this.gridApi.sizeColumnsToFit();
    }
  }

  handleLoadingOverlay() {
    if (!this.gridApi) return;
    if (this.props.isLoading) {
      this.gridApi.showLoadingOverlay();
    } else {
      this.gridApi.hideOverlay();
    }
  }

  persistGridState = () => {
    if (!this.columnApi) return;
    this.getLocalStore().saveComponentState(
      new AgGridPersistedState({
        columnGroupState: this.columnApi.getColumnGroupState(),
      }),
    );
  };

  restoreGridState() {
    if (!this.columnApi) return;
    const { columnGroupState } = this.getLocalStore().loadComponentState();
    if (columnGroupState) {
      this.columnApi.setColumnGroupState(columnGroupState);
    }
  }

  componentDidUpdate() {
    this.applyRowSelectionFromProps();
    this.handleColumnSizeRefit();
    this.handleLoadingOverlay();
    this.restoreGridState();
  }

  render() {
    const { handleLoadMoreRuns, loadingMore } = this.props;
    const columnDefs = this.getColumnDefs();
    const {
      defaultColDef,
      frameworkComponents,
      isFullWidthCell,
    } = ExperimentRunsTableMultiColumnView2;

    return (
      <div className='ag-theme-balham multi-column-view'>
        <AgGridReact
          defaultColDef={defaultColDef}
          columnDefs={columnDefs}
          rowData={this.getRowData()}
          modules={[Grid, ClientSideRowModelModule]}
          rowSelection='multiple'
          onGridReady={this.handleGridReady}
          onSelectionChanged={this.handleSelectionChange}
          onColumnGroupOpened={this.persistGridState}
          suppressRowClickSelection
          suppressScrollOnNewData // retain scroll position after nested run toggling operations
          suppressFieldDotNotation
          enableCellTextSelection
          frameworkComponents={frameworkComponents}
          fullWidthCellRendererFramework={FullWidthCellRenderer}
          fullWidthCellRendererParams={{ handleLoadMoreRuns, loadingMore }}
          loadingOverlayComponent='loadingOverlayComponent'
          loadingOverlayComponentParams={{ showImmediately: true }}
          isFullWidthCell={isFullWidthCell}
          isRowSelectable={this.isRowSelectable}
        />
      </div>
    );
  }
}

function FullWidthCellRenderer({ handleLoadMoreRuns, loadingMore }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <LoadMoreBar loadingMore={loadingMore} onLoadMore={handleLoadMoreRuns} />
    </div>
  );
}
FullWidthCellRenderer.propTypes = {
  handleLoadMoreRuns: PropTypes.func,
  loadingMore: PropTypes.bool,
};

function DateCellRenderer(props) {
  const {
    startTime,
    runInfo,
    isParent,
    hasExpander,
    expanderOpen,
    childrenIds,
    onExpand,
  } = props.data;
  return (
    <div>
      {hasExpander ? (
        <div
          onClick={() => {
            onExpand(runInfo.run_uuid, childrenIds);
          }}
          key={'Expander-' + runInfo.run_uuid}
          style={{ paddingRight: 8, display: 'inline' }}
        >
          <i
            className={`ExperimentView-expander far fa-${expanderOpen ? 'minus' : 'plus'}-square`}
          />
        </div>
      ) : (
        <span style={{ paddingLeft: 18 }} />
      )}
      <Link
        to={Routes.getRunPageRoute(runInfo.experiment_id, runInfo.run_uuid)}
        style={{ paddingLeft: isParent ? 0 : 16 }}
      >
        {ExperimentViewUtil.getRunStatusIcon(runInfo.status)} {Utils.formatTimestamp(startTime)}
      </Link>
    </div>
  );
}
DateCellRenderer.propTypes = { data: PropTypes.object };

function SourceCellRenderer(props) {
  const { tags, queryParams } = props.data;
  const sourceType = Utils.renderSource(tags, queryParams);
  return sourceType ? (
    <React.Fragment>
      {Utils.renderSourceTypeIcon(Utils.getSourceType(tags))}
      {sourceType}
    </React.Fragment>
  ) : (
    <React.Fragment>{EMPTY_CELL_PLACEHOLDER}</React.Fragment>
  );
}
SourceCellRenderer.propTypes = { data: PropTypes.object };

function VersionCellRenderer(props) {
  const { tags } = props.data;
  return Utils.renderVersion(tags) || EMPTY_CELL_PLACEHOLDER;
}

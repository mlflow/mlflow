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
import registeredModelSvg from '../../common/static/registered-model.svg';
import loggedModelSvg from '../../common/static/logged-model.svg';
import ExperimentViewUtil from './ExperimentViewUtil';
import { LoadMoreBar } from './LoadMoreBar';
import _ from 'lodash';
import { Spinner } from '../../common/components/Spinner';
import { ExperimentRunsTableEmptyOverlay } from '../../common/components/ExperimentRunsTableEmptyOverlay';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { AgGridPersistedState } from '../sdk/MlflowLocalStorageMessages';
import { ColumnTypes } from '../constants';
import { TrimmedText } from '../../common/components/TrimmedText';
import { getModelVersionPageRoute } from '../../model-registry/routes';
import { css } from 'emotion';

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
    runInfos: PropTypes.arrayOf(PropTypes.instanceOf(RunInfo)).isRequired,
    modelVersionsByRunUuid: PropTypes.object.isRequired,
    // List of list of params in all the visible runs
    paramsList: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    // List of list of metrics in all the visible runs
    metricsList: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    paramKeyList: PropTypes.arrayOf(PropTypes.string),
    metricKeyList: PropTypes.arrayOf(PropTypes.string),
    visibleTagKeyList: PropTypes.arrayOf(PropTypes.string),
    // List of tags dictionary in all the visible runs.
    tagsList: PropTypes.arrayOf(PropTypes.object).isRequired,
    onSelectionChange: PropTypes.func.isRequired,
    onExpand: PropTypes.func.isRequired,
    onSortBy: PropTypes.func.isRequired,
    orderByKey: PropTypes.string,
    orderByAsc: PropTypes.bool,
    runsSelected: PropTypes.object.isRequired,
    runsExpanded: PropTypes.object.isRequired,
    numRunsFromLatestSearch: PropTypes.number,
    handleLoadMoreRuns: PropTypes.func.isRequired,
    loadingMore: PropTypes.bool.isRequired,
    isLoading: PropTypes.bool.isRequired,
    categorizedUncheckedKeys: PropTypes.object.isRequired,
    nestChildren: PropTypes.bool,
  };

  static defaultColDef = {
    initialWidth: 100,
    // eslint-disable-next-line max-len
    // autoSizePadding property is set to 0 so that the size of the columns don't change for sort icon or anything and remains stable
    autoSizePadding: 0,
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
    modelsCellRenderer: ModelsCellRenderer,
    dateCellRenderer: DateCellRenderer,
    agColumnHeader: RunsTableCustomHeader,
    loadingOverlayComponent: Spinner,
    noRowsOverlayComponent: ExperimentRunsTableEmptyOverlay,
  };

  constructor(props) {
    super(props);
    this.getColumnDefs = this.getColumnDefs.bind(this);
    // In some cases, the API request to fetch run info resolves
    // before this component is constructed and rendered. We need to get
    // column defs here to handle that case, as well as the one already
    // handled in componentDidUpdate for when the request resolves after the
    // fact.
    this.columnDefs = this.getColumnDefs();
  }

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
    const getStyle = (key) => (key === this.props.orderByKey ? { backgroundColor: '#e6f7ff' } : {});
    const headerStyle = (key) => getStyle(key);
    const cellStyle = (params) => getStyle(params.colDef.headerComponentParams.canonicalSortKey);

    return [
      ...[
        {
          field: '',
          checkboxSelection: true,
          headerCheckboxSelection: true,
          pinned: 'left',
          initialWidth: 50,
        },
        {
          headerName: ExperimentViewUtil.AttributeColumnLabels.DATE,
          field: 'startTime',
          pinned: 'left',
          initialWidth: 150,
          cellRenderer: 'dateCellRenderer',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: ExperimentViewUtil.AttributeColumnSortKey.DATE,
            computedStylesOnSortKey: headerStyle,
          },
          cellStyle,
        },
        {
          headerName: ExperimentViewUtil.AttributeColumnLabels.RUN_NAME,
          pinned: 'left',
          field: 'runName',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: ExperimentViewUtil.AttributeColumnSortKey.RUN_NAME,
            computedStylesOnSortKey: headerStyle,
          },
          cellStyle,
        },
        {
          headerName: ExperimentViewUtil.AttributeColumnLabels.USER,
          field: 'user',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: ExperimentViewUtil.AttributeColumnSortKey.USER,
            computedStylesOnSortKey: headerStyle,
          },
          cellStyle,
        },
        {
          headerName: ExperimentViewUtil.AttributeColumnLabels.SOURCE,
          field: 'source',
          cellRenderer: 'sourceCellRenderer',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: ExperimentViewUtil.AttributeColumnSortKey.SOURCE,
            computedStylesOnSortKey: headerStyle,
          },
          cellStyle,
        },
        {
          headerName: ExperimentViewUtil.AttributeColumnLabels.VERSION,
          field: 'version',
          cellRenderer: 'versionCellRenderer',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: ExperimentViewUtil.AttributeColumnSortKey.VERSION,
            computedStylesOnSortKey: headerStyle,
          },
          cellStyle,
        },
        {
          headerName: ExperimentViewUtil.AttributeColumnLabels.MODELS,
          field: 'models',
          cellRenderer: 'modelsCellRenderer',
          initialWidth: 200,
        },
      ].filter((c) => !categorizedUncheckedKeys[ColumnTypes.ATTRIBUTES].includes(c.headerName)),
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
              computedStylesOnSortKey: headerStyle,
            },
            cellStyle,
          };
        }),
      },
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
              computedStylesOnSortKey: headerStyle,
            },
            cellStyle,
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
      modelVersionsByRunUuid,
      tagsList,
      numRunsFromLatestSearch,
      runsExpanded,
      onExpand,
      loadingMore,
      visibleTagKeyList,
      nestChildren,
    } = this.props;
    const { getNameValueMapFromList } = ExperimentRunsTableMultiColumnView2;
    const mergedRows = ExperimentViewUtil.getRowRenderMetadata({
      runInfos,
      tagsList,
      runsExpanded,
      nestChildren,
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
      const visibleTags = Utils.getVisibleTagValues(tags).map(([key, value]) => ({
        key,
        value,
      }));

      return {
        runInfo,
        startTime,
        user,
        runName,
        tags,
        queryParams,
        modelVersionsByRunUuid,
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

    // don't show LoadMoreBar if there are no runs at all
    if (runs.length) {
      runs.push({
        isFullWidth: true,
        loadingMore,
        numRunsFromLatestSearch,
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

  // Please do not call handleLoadingOverlay here. It results in the component state duplicating the
  // overlay, as a new overlay was added in https://github.com/databricks/universe/pull/66242.
  handleGridReady = (params) => {
    this.gridApi = params.api;
    this.columnApi = params.columnApi;
    this.applyRowSelectionFromProps();
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

  handleLoadingOverlay() {
    if (!this.gridApi) return;
    if (this.props.isLoading) {
      this.gridApi.showLoadingOverlay();
    } else if (this.props.runInfos.length === 0) {
      this.gridApi.showNoRowsOverlay();
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

  componentDidUpdate(prevProps) {
    this.applyRowSelectionFromProps();
    this.handleLoadingOverlay();
    this.restoreGridState();
    // The following block checks if any columnDefs parameters have changed to
    // update the columnDefs to prevent resizing and other column property issues.
    if (
      prevProps.metricKeyList.length !== this.props.metricKeyList.length ||
      prevProps.paramKeyList.length !== this.props.paramKeyList.length ||
      prevProps.categorizedUncheckedKeys.length !== this.props.categorizedUncheckedKeys.length ||
      prevProps.visibleTagKeyList.length !== this.props.visibleTagKeyList.length ||
      prevProps.orderByKey !== this.props.orderByKey ||
      prevProps.orderByAsc !== this.props.orderByAsc ||
      prevProps.onSortBy !== this.props.onSortBy
    ) {
      this.columnDefs = this.getColumnDefs();
    }
  }

  render() {
    const { handleLoadMoreRuns, loadingMore, numRunsFromLatestSearch, nestChildren } = this.props;
    const {
      defaultColDef,
      frameworkComponents,
      isFullWidthCell,
    } = ExperimentRunsTableMultiColumnView2;
    return (
      <div className='ag-theme-balham multi-column-view' data-test-id='detailed-runs-table-view'>
        <AgGridReact
          defaultColDef={defaultColDef}
          columnDefs={this.columnDefs}
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
          fullWidthCellRendererParams={{
            handleLoadMoreRuns,
            loadingMore,
            numRunsFromLatestSearch,
            nestChildren,
          }}
          loadingOverlayComponent='loadingOverlayComponent'
          loadingOverlayComponentParams={{ showImmediately: true }}
          isFullWidthCell={isFullWidthCell}
          isRowSelectable={this.isRowSelectable}
          noRowsOverlayComponent='noRowsOverlayComponent'
        />
      </div>
    );
  }
}

function FullWidthCellRenderer({
  handleLoadMoreRuns,
  loadingMore,
  numRunsFromLatestSearch,
  nestChildren,
}) {
  return (
    <div style={{ textAlign: 'center' }}>
      <LoadMoreBar
        loadingMore={loadingMore}
        onLoadMore={handleLoadMoreRuns}
        disableButton={ExperimentViewUtil.disableLoadMoreButton({
          numRunsFromLatestSearch,
        })}
        nestChildren={nestChildren}
      />
    </div>
  );
}

FullWidthCellRenderer.propTypes = {
  handleLoadMoreRuns: PropTypes.func,
  loadingMore: PropTypes.bool,
  nestChildren: PropTypes.bool,
  numRunsFromLatestSearch: PropTypes.number,
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
        title={Utils.formatTimestamp(startTime)}
      >
        {ExperimentViewUtil.getRunStatusIcon(runInfo.status)} {Utils.timeSinceStr(startTime)}
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
      {Utils.renderSourceTypeIcon(tags)}
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

export function ModelsCellRenderer(props) {
  const { runInfo, tags, modelVersionsByRunUuid } = props.data;
  const registeredModels = modelVersionsByRunUuid[runInfo.run_uuid] || [];
  const loggedModels = Utils.getLoggedModelsFromTags(tags);
  const models = Utils.mergeLoggedAndRegisteredModels(loggedModels, registeredModels);
  const imageStyle = {
    wrapper: css({
      img: {
        height: '15px',
        position: 'relative',
        marginRight: '4px',
      },
    }),
  };
  if (models && models.length) {
    const modelToRender = models[0];
    let modelDiv;
    if (modelToRender.registeredModelName) {
      const { registeredModelName, registeredModelVersion } = modelToRender;
      modelDiv = (
        <>
          <img
            data-test-id='registered-model-icon'
            alt=''
            title='Registered Model'
            src={registeredModelSvg}
          />
          <a
            href={Utils.getIframeCorrectedRoute(
              getModelVersionPageRoute(registeredModelName, registeredModelVersion),
            )}
            className='registered-model-link'
            target='_blank'
          >
            <TrimmedText text={registeredModelName} maxSize={10} className={'model-name'} />
            {`/${registeredModelVersion}`}
          </a>
        </>
      );
    } else if (modelToRender.flavors) {
      const loggedModelFlavorText = modelToRender.flavors ? modelToRender.flavors[0] : 'Model';
      const loggedModelLink = Utils.getIframeCorrectedRoute(
        `${Routes.getRunPageRoute(runInfo.experiment_id, runInfo.run_uuid)}/artifactPath/${
          modelToRender.artifactPath
        }`,
      );
      modelDiv = (
        <>
          <img data-test-id='logged-model-icon' alt='' title='Logged Model' src={loggedModelSvg} />
          <a href={loggedModelLink} target='_blank' className='logged-model-link'>
            {loggedModelFlavorText}
          </a>
        </>
      );
    }

    return (
      <div className={`logged-model-cell ${imageStyle.wrapper}`}>
        {modelDiv}
        {loggedModels.length > 1 ? `, ${loggedModels.length - 1} more` : ''}
      </div>
    );
  }
  return EMPTY_CELL_PLACEHOLDER;
}

ModelsCellRenderer.propTypes = { data: PropTypes.object };

/**
 * Ag-grid based implementation of multi-column view with a bunch of new interactive features
 */
import React from 'react';
import PropTypes from 'prop-types';
import { RunInfo, Experiment } from '../sdk/MlflowMessages';
import { Link } from 'react-router-dom';
import { WithDesignSystemThemeHoc } from '@databricks/design-system';
import Routes from '../routes';
import Utils from '../../common/utils/Utils';

import { RunsTableCustomHeader } from '../../common/components/ag-grid/RunsTableCustomHeader';
import { MLFlowAgGridLoader } from '../../common/components/ag-grid/AgGridLoader';

import registeredModelSvg from '../../common/static/registered-model.svg';
import loggedModelSvg from '../../common/static/logged-model.svg';
import ExperimentViewUtil from './ExperimentViewUtil';
import { LoadMoreBar } from './LoadMoreBar';
import _ from 'lodash';
import { Spinner } from '../../common/components/Spinner';
import { ExperimentRunsTableEmptyOverlay } from '../../common/components/ExperimentRunsTableEmptyOverlay';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { AgGridPersistedState } from '../sdk/MlflowLocalStorageMessages';
import { TrimmedText } from '../../common/components/TrimmedText';
import { getModelVersionPageRoute } from '../../model-registry/routes';
import { COLUMN_TYPES, ATTRIBUTE_COLUMN_LABELS, ATTRIBUTE_COLUMN_SORT_KEY } from '../constants';

const PARAM_PREFIX = '$$$param$$$';
const METRIC_PREFIX = '$$$metric$$$';
const TAG_PREFIX = '$$$tag$$$';
const MAX_PARAMS_COLS = 3;
const MAX_METRICS_COLS = 3;
const MAX_TAG_COLS = 3;
const EMPTY_CELL_PLACEHOLDER = '-';

export class ExperimentRunsTableMultiColumnView2Impl extends React.Component {
  static propTypes = {
    experiments: PropTypes.arrayOf(PropTypes.instanceOf(Experiment)),
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
    nextPageToken: PropTypes.string,
    numRunsFromLatestSearch: PropTypes.number,
    handleLoadMoreRuns: PropTypes.func.isRequired,
    loadingMore: PropTypes.bool.isRequired,
    isLoading: PropTypes.bool.isRequired,
    categorizedUncheckedKeys: PropTypes.object.isRequired,
    nestChildren: PropTypes.bool,
    compareExperiments: PropTypes.bool,
    designSystemThemeApi: PropTypes.shape({ theme: PropTypes.object }).isRequired,
  };

  static defaultProps = {
    compareExperiments: false,
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
    experimentNameRenderer: ExperimentNameRenderer,
    versionCellRenderer: VersionCellRenderer,
    modelsCellRenderer: ModelsCellRenderer,
    dateCellRenderer: DateCellRenderer,
    tagCellRenderer: TagCellRenderer,
    agColumnHeader: RunsTableCustomHeader,
    loadingOverlayComponent: Spinner,
    noRowsOverlayComponent: ExperimentRunsTableEmptyOverlay,
  };

  constructor(props) {
    super(props);
    this.getColumnDefs = this.getColumnDefs.bind(this);
    this.state = {
      columnDefs: [],
    };
  }

  componentDidMount() {
    // In some cases, the API request to fetch run info resolves
    // before this component is constructed and mounted. We need to get
    // column defs here to handle that case, as well as the one already
    // handled in componentDidUpdate for when the request resolves after the
    // fact.
    this.setColumnDefs();
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

  getLocalStore = () =>
    LocalStorageUtils.getStoreForComponent(
      'ExperimentRunsTableMultiColumnView2',
      JSON.stringify(this.props.experiments.map(({ experiment_id }) => experiment_id).sort()),
    );

  applyingRowSelectionFromProps = false;

  hasMultipleExperiments() {
    return this.props.experiments.length > 1;
  }

  getColumnDefs() {
    const {
      metricKeyList,
      paramKeyList,
      categorizedUncheckedKeys,
      visibleTagKeyList,
      orderByKey,
      orderByAsc,
      onSortBy,
      onExpand,
      designSystemThemeApi,
    } = this.props;
    const commonSortOrderProps = { orderByKey, orderByAsc, onSortBy };
    const { theme } = designSystemThemeApi;
    const getStyle = (key) =>
      key === this.props.orderByKey ? { backgroundColor: theme.colors.blue100 } : {};
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
          headerName: ATTRIBUTE_COLUMN_LABELS.DATE,
          field: 'runDateInfo',
          pinned: 'left',
          initialWidth: 150,
          cellRenderer: 'dateCellRenderer',
          cellRendererParams: {
            onExpand: onExpand,
          },
          equals: (dateInfo1, dateInfo2) => _.isEqual(dateInfo1, dateInfo2),
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: ATTRIBUTE_COLUMN_SORT_KEY.DATE,
            computedStylesOnSortKey: headerStyle,
          },
          cellStyle,
        },
        ...(this.props.compareExperiments
          ? [
              {
                headerName: ATTRIBUTE_COLUMN_LABELS.EXPERIMENT_NAME,
                field: 'experimentName',
                cellRenderer: 'experimentNameRenderer',
                equals: (experimentName1, experimentName2) =>
                  _.isEqual(experimentName1, experimentName2),
                pinned: 'left',
                initialWidth: 140,
                cellStyle,
              },
            ]
          : []),
        {
          headerName: ATTRIBUTE_COLUMN_LABELS.DURATION,
          field: 'duration',
          pinned: 'left',
          initialWidth: 80,
          cellStyle,
        },
        {
          headerName: ATTRIBUTE_COLUMN_LABELS.RUN_NAME,
          pinned: 'left',
          field: 'runName',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: ATTRIBUTE_COLUMN_SORT_KEY.RUN_NAME,
            computedStylesOnSortKey: headerStyle,
          },
          cellStyle,
        },
        {
          headerName: ATTRIBUTE_COLUMN_LABELS.USER,
          field: 'user',
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: ATTRIBUTE_COLUMN_SORT_KEY.USER,
            computedStylesOnSortKey: headerStyle,
          },
          cellStyle,
        },
        {
          headerName: ATTRIBUTE_COLUMN_LABELS.SOURCE,
          field: 'tags',
          cellRenderer: 'sourceCellRenderer',
          equals: (tags1, tags2) => Utils.getSourceName(tags1) === Utils.getSourceName(tags2),
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: ATTRIBUTE_COLUMN_SORT_KEY.SOURCE,
            computedStylesOnSortKey: headerStyle,
          },
          cellStyle,
        },
        {
          headerName: ATTRIBUTE_COLUMN_LABELS.VERSION,
          field: 'version',
          cellRenderer: 'versionCellRenderer',
          equals: (version1, version2) => _.isEqual(version1, version2),
          sortable: true,
          headerComponentParams: {
            ...commonSortOrderProps,
            canonicalSortKey: ATTRIBUTE_COLUMN_SORT_KEY.VERSION,
            computedStylesOnSortKey: headerStyle,
          },
          cellStyle,
        },
        {
          headerName: ATTRIBUTE_COLUMN_LABELS.MODELS,
          field: 'models',
          cellRenderer: 'modelsCellRenderer',
          initialWidth: 200,
          equals: (models1, models2) => _.isEqual(models1, models2),
        },
      ].filter((c) => !categorizedUncheckedKeys[COLUMN_TYPES.ATTRIBUTES].includes(c.headerName)),
      {
        headerName: 'Metrics',
        children: metricKeyList.map((metricKey, i) => {
          const columnKey = ExperimentViewUtil.makeCanonicalKey(COLUMN_TYPES.METRICS, metricKey);
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
          const columnKey = ExperimentViewUtil.makeCanonicalKey(COLUMN_TYPES.PARAMS, paramKey);
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
          cellRenderer: 'tagCellRenderer',
          ...(i >= MAX_TAG_COLS ? { columnGroupShow: 'open' } : null),
        })),
      },
    ];
  }

  // Only run based rows are selectable, other utility rows like "load more" row is not selectable
  isRowSelectable = (rowNode) => rowNode.data && rowNode.data.runInfo;

  getRowData() {
    const {
      experiments,
      runInfos,
      paramsList,
      metricsList,
      paramKeyList,
      metricKeyList,
      modelVersionsByRunUuid,
      tagsList,
      runsExpanded,
      visibleTagKeyList,
      nestChildren,
    } = this.props;
    const { getNameValueMapFromList } = ExperimentRunsTableMultiColumnView2Impl;
    const mergedRows = ExperimentViewUtil.getRowRenderMetadata({
      runInfos,
      tagsList,
      runsExpanded,
      nestChildren,
    });

    const experimentNameMap = Utils.getExperimentNameMap(Utils.sortExperimentsById(experiments));
    const referenceTime = new Date();
    // Round reference time down to the nearest second, to avoid unnecessary re-renders
    referenceTime.setMilliseconds(0);
    const runs = mergedRows.map(
      ({ idx, isParent, hasExpander, expanderOpen, childrenIds, level }) => {
        const tags = tagsList[idx];
        const params = paramsList[idx];
        const metrics = metricsList[idx].map(({ key, value }) => ({
          key,
          value: Utils.formatMetric(value),
        }));
        const runInfo = runInfos[idx];

        const runUuid = runInfo.getRunUuid();
        const { experiment_id: experimentId } = runInfo;
        const experimentName = experimentNameMap[experimentId];
        const user = Utils.getUser(runInfo, tags);
        const duration = Utils.getDuration(runInfo.start_time, runInfo.end_time);
        const runName = Utils.getRunName(tags) || '-';
        const visibleTags = Utils.getVisibleTagValues(tags).map(([key, value]) => ({
          key,
          value,
        }));

        const runDateInfo = {
          startTime: runInfo.start_time,
          referenceTime,
          experimentId,
          runUuid,
          runStatus: runInfo.status,
          isParent,
          hasExpander,
          expanderOpen,
          childrenIds,
          level,
        };

        const models = {
          registeredModels: modelVersionsByRunUuid[runInfo.run_uuid] || [],
          loggedModels: Utils.getLoggedModelsFromTags(tags),
          experimentId: runInfo.experiment_id,
          runUuid: runInfo.run_uuid,
        };

        const version = {
          version: Utils.getSourceVersion(tags),
          name: Utils.getSourceName(tags),
          type: Utils.getSourceType(tags),
        };

        return {
          runUuid,
          runDateInfo,
          runInfo,
          experimentName,
          experimentId,
          duration,
          user,
          runName,
          tags,
          models,
          version,
          ...getNameValueMapFromList(params, paramKeyList, PARAM_PREFIX),
          ...getNameValueMapFromList(metrics, metricKeyList, METRIC_PREFIX),
          ...getNameValueMapFromList(visibleTags, visibleTagKeyList, TAG_PREFIX),
        };
      },
    );

    return runs;
  }

  /**
   * A onRowSelected event handler that runs before onSelectionChanged.
   * It checks if the currently (de)selected row contains any children
   * and if true, (de)select them as well.
   */
  handleRowSelected = (event) => {
    const selectedRows = event.api.getSelectedRows();

    // Let's check if the actual number of selected rows have changed
    // to avoid empty runs
    if (this.prevSelectRunUuids && selectedRows.length !== this.prevSelectRunUuids.length) {
      const isSelected = event.node.isSelected();

      // We will continue only if the selected row has properly set runDateInfo
      const { runDateInfo } = event.data;
      if (!runDateInfo) {
        return;
      }
      const { isParent, expanderOpen, childrenIds } = runDateInfo;

      // We will continue only if the selected row is a parent containing
      // children and is actually expanded
      if (isParent && expanderOpen && childrenIds) {
        const childrenIdsToSelect = childrenIds;

        event.api.forEachNode((node) => {
          const { runInfo, runDateInfo: childRunDateInfo } = node.data;

          const childrenRunUuid = runInfo.run_uuid;
          if (childrenIdsToSelect.includes(childrenRunUuid)) {
            // If we found children being parents, mark their children
            // to be selected as well.
            if (childRunDateInfo?.childrenIds) {
              childrenIdsToSelect.push(...childRunDateInfo.childrenIds);
            }

            node.setSelected(isSelected, false, true);
          }
        });
      }
    }
  };

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
      prevProps.visibleTagKeyList.length !== this.props.visibleTagKeyList.length ||
      !_.isEqual(
        prevProps.categorizedUncheckedKeys[COLUMN_TYPES.ATTRIBUTES],
        this.props.categorizedUncheckedKeys[COLUMN_TYPES.ATTRIBUTES],
      ) ||
      prevProps.orderByKey !== this.props.orderByKey ||
      prevProps.orderByAsc !== this.props.orderByAsc ||
      prevProps.onSortBy !== this.props.onSortBy
    ) {
      this.setColumnDefs();
    }
  }

  setColumnDefs = () => {
    this.setState(() => ({
      columnDefs: this.getColumnDefs(),
    }));
  };

  render() {
    const {
      runInfos,
      handleLoadMoreRuns,
      loadingMore,
      nextPageToken,
      numRunsFromLatestSearch,
      nestChildren,
      designSystemThemeApi,
    } = this.props;
    const { theme } = designSystemThemeApi;
    const { defaultColDef, frameworkComponents } = ExperimentRunsTableMultiColumnView2Impl;
    const agGridOverrides = {
      '--ag-border-color': 'rgba(0, 0, 0, 0.06)',
      '--ag-header-foreground-color': '#20272e',
      '--ag-header-background-color': `${theme.colors.grey100}`,
      '--ag-row-hover-color': `${theme.colors.grey200}`,
      '&.ag-grid-sticky .ag-header': {
        position: 'sticky',
        top: 0,
        zIndex: 1,
      },
      '&.ag-grid-sticky .ag-root': {
        overflow: 'visible',
      },
      '&.ag-grid-sticky .ag-root-wrapper': {
        border: '0',
        borderRadius: '4px',
        overflow: 'visible',
      },
    };

    return (
      <div
        className='ag-theme-balham multi-column-view ag-grid-sticky'
        css={agGridOverrides}
        data-test-id='detailed-runs-table-view'
      >
        <MLFlowAgGridLoader
          defaultColDef={defaultColDef}
          columnDefs={this.state.columnDefs}
          rowData={this.getRowData()}
          domLayout='autoHeight'
          rowSelection='multiple'
          onGridReady={this.handleGridReady}
          onSelectionChanged={this.handleSelectionChange}
          onColumnGroupOpened={this.persistGridState}
          onRowSelected={this.handleRowSelected}
          suppressRowClickSelection
          suppressScrollOnNewData // retain scroll position after nested run toggling operations
          suppressFieldDotNotation
          enableCellTextSelection
          components={frameworkComponents}
          loadingOverlayComponent='loadingOverlayComponent'
          loadingOverlayComponentParams={{ showImmediately: true }}
          isRowSelectable={this.isRowSelectable}
          noRowsOverlayComponent='noRowsOverlayComponent'
          getRowId={getRowId}
        />
        <div style={{ textAlign: 'center' }}>
          {
            // don't show LoadMoreBar if there are no runs at all
            runInfos.length ? (
              <LoadMoreBar
                loadingMore={loadingMore}
                onLoadMore={handleLoadMoreRuns}
                disableButton={ExperimentViewUtil.disableLoadMoreButton({
                  numRunsFromLatestSearch,
                  nextPageToken,
                })}
                nestChildren={nestChildren}
              />
            ) : null
          }
        </div>
      </div>
    );
  }
}

export const ExperimentRunsTableMultiColumnView2 = WithDesignSystemThemeHoc(
  ExperimentRunsTableMultiColumnView2Impl,
);

function getRowId(params) {
  return params.data.runUuid;
}

function DateCellRenderer(props) {
  const {
    startTime,
    referenceTime,
    experimentId,
    runUuid,
    runStatus,
    isParent,
    hasExpander,
    expanderOpen,
    childrenIds,
    level,
  } = props.value;
  const { onExpand } = props;
  return (
    <div>
      {hasExpander ? (
        <div
          onClick={() => {
            onExpand(runUuid, childrenIds);
          }}
          key={'Expander-' + runUuid}
          style={{ paddingRight: 8, display: 'inline' }}
        >
          <span style={{ paddingLeft: 18 * level }} />
          <i
            className={`ExperimentView-expander far fa-${expanderOpen ? 'minus' : 'plus'}-square`}
          />
        </div>
      ) : (
        <span style={{ paddingLeft: level === 0 ? 12 : 18 * level + 12 }} />
      )}
      <Link
        to={Routes.getRunPageRoute(experimentId, runUuid)}
        style={{ paddingLeft: isParent ? 0 : 8 }}
        title={Utils.formatTimestamp(startTime)}
      >
        {ExperimentViewUtil.getRunStatusIcon(runStatus)}{' '}
        {Utils.timeSinceStr(startTime, referenceTime)}
      </Link>
    </div>
  );
}

DateCellRenderer.propTypes = {
  value: PropTypes.object,
  onExpand: PropTypes.func,
};

function SourceCellRenderer(props) {
  const tags = props.value;
  const sourceType = Utils.renderSource(tags);
  return sourceType ? (
    <React.Fragment>
      {Utils.renderSourceTypeIcon(tags)}
      {sourceType}
    </React.Fragment>
  ) : (
    <React.Fragment>{EMPTY_CELL_PLACEHOLDER}</React.Fragment>
  );
}

SourceCellRenderer.propTypes = { value: PropTypes.object };

function VersionCellRenderer(props) {
  // prettier-ignore
  const {
    version,
    name,
    type,
  } = props.value;
  return (
    // prettier-ignore
    Utils.renderSourceVersion(
      version,
      name,
      type,
    ) || EMPTY_CELL_PLACEHOLDER
  );
}

VersionCellRenderer.propTypes = { value: PropTypes.object };

function ExperimentNameRenderer(props) {
  // We can get the experiment id from the row data rather than needing to
  // include it in the column value, as the experiment id for a row won't change.
  // (If it could, we would need to include it in the value so that we would
  // re-render this cell when it changed).
  const { experimentId } = props.data;
  const { name, basename } = props.value;
  return (
    <Link to={Routes.getExperimentPageRoute(experimentId)} title={name}>
      {basename}
    </Link>
  );
}

ExperimentNameRenderer.propTypes = {
  value: PropTypes.object,
  data: PropTypes.object,
};

export function ModelsCellRenderer(props) {
  const { registeredModels, loggedModels, experimentId, runUuid } = props.value;
  const models = Utils.mergeLoggedAndRegisteredModels(loggedModels, registeredModels);

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
          {/* Reported during ESLint upgrade */}
          {/* eslint-disable-next-line react/jsx-no-target-blank */}
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
        `${Routes.getRunPageRoute(experimentId, runUuid)}/artifactPath/${
          modelToRender.artifactPath
        }`,
      );
      modelDiv = (
        <>
          <img data-test-id='logged-model-icon' alt='' title='Logged Model' src={loggedModelSvg} />
          {/* Reported during ESLint upgrade */}
          {/* eslint-disable-next-line react/jsx-no-target-blank */}
          <a href={loggedModelLink} target='_blank' className='logged-model-link'>
            {loggedModelFlavorText}
          </a>
        </>
      );
    }

    return (
      <div className='logged-model-cell' css={styles.imageWrapper}>
        {modelDiv}
        {loggedModels.length > 1 ? `, ${loggedModels.length - 1} more` : ''}
      </div>
    );
  }
  return EMPTY_CELL_PLACEHOLDER;
}

const styles = {
  imageWrapper: {
    img: {
      height: '15px',
      position: 'relative',
      marginRight: '4px',
    },
  },
};

ModelsCellRenderer.propTypes = { value: PropTypes.object };

export function TagCellRenderer(props) {
  const tagValue = props.value;

  return Utils.isValidHttpUrl(tagValue) ? (
    <a href={tagValue} target='_blank' rel='noreferrer'>
      {tagValue}
    </a>
  ) : (
    tagValue
  );
}

TagCellRenderer.propTypes = { value: PropTypes.string };

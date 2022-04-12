import classNames from 'classnames';
import React from 'react';
import Utils from '../../common/utils/Utils';
import { Link } from 'react-router-dom';
import Routes from '../routes';
import { getModelVersionPageRoute } from '../../model-registry/routes';
import { CollapsibleTagsCell } from '../../common/components/CollapsibleTagsCell';
import _ from 'lodash';
import ExpandableList from '../../common/components/ExpandableList';
import registryIcon from '../../common/static/registryIcon.svg';
import { TrimmedText } from '../../common/components/TrimmedText';
import {
  ATTRIBUTE_COLUMN_LABELS,
  ATTRIBUTE_COLUMN_SORT_KEY,
  DEFAULT_EXPANDED_VALUE,
  COLUMN_TYPES,
} from '../constants';
import {
  CheckCircleBorderIcon,
  Clock1Icon,
  XCircleBorderIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { css } from 'emotion';

function ErrorIcon() {
  const { theme } = useDesignSystemTheme();
  return (
    <XCircleBorderIcon
      className={css({
        color: theme.colors.textValidationDanger,
      })}
    />
  );
}

function FinishedIcon() {
  const { theme } = useDesignSystemTheme();
  return (
    <CheckCircleBorderIcon
      className={css({
        color: theme.colors.textValidationSuccess,
      })}
    />
  );
}

export default class ExperimentViewUtil {
  /** Returns checkbox cell for a row. */
  static getCheckboxForRow(selected, checkboxHandler, cellType) {
    const CellComponent = `${cellType}`;
    return (
      <CellComponent key='meta-check' className='run-table-container'>
        <div>
          <input type='checkbox' checked={selected} onChange={checkboxHandler} />
        </div>
      </CellComponent>
    );
  }

  static styles = {
    sortIconStyle: {
      verticalAlign: 'middle',
      fontSize: 20,
    },
    sortIconContainer: {
      marginLeft: 2,
      minWidth: 12.5,
      display: 'inline-block',
    },
    expander: {
      pointer: 'cursor',
    },
    runInfoCell: {
      maxWidth: 250,
    },
  };

  /**
   * Returns an icon depending on run status.
   */
  static getRunStatusIcon(status) {
    switch (status) {
      case 'FAILED':
      case 'KILLED':
        return <ErrorIcon />;
      case 'FINISHED':
        return <FinishedIcon />;
      case 'SCHEDULED':
        return <Clock1Icon />; // This one is the same color as the link
      default:
        return <i />;
    }
  }

  /**
   * Returns table cells describing run metadata (i.e. not params/metrics) comprising part of
   * the display row for a run.
   */
  static getRunInfoCellsForRow(runInfo, tags, isParent, cellType, handleCellToggle, excludedKeys) {
    const CellComponent = `${cellType}`;
    const user = Utils.getUser(runInfo, tags);
    const sourceType = Utils.renderSource(tags);
    const { status, start_time: startTime, end_time: endTime } = runInfo;
    const duration = Utils.getDuration(startTime, endTime);
    const runName = Utils.getRunName(tags);
    const childLeftMargin = isParent ? {} : { paddingLeft: 16 };
    const columnProps = [
      {
        key: 'status',
        className: 'run-table-container',
        title: status,
        children: ExperimentViewUtil.getRunStatusIcon(status),
      },
      {
        key: ATTRIBUTE_COLUMN_LABELS.DATE,
        className: 'run-table-container',
        style: { whiteSpace: 'inherit' },
        children: (
          <div style={childLeftMargin}>
            <Link to={Routes.getRunPageRoute(runInfo.experiment_id, runInfo.run_uuid)}>
              {Utils.formatTimestamp(startTime)}
            </Link>
          </div>
        ),
      },
      {
        key: ATTRIBUTE_COLUMN_LABELS.DURATION,
        className: 'run-table-container',
        title: duration,
        children: (
          <div className='truncate-text single-line' style={ExperimentViewUtil.styles.runInfoCell}>
            {duration}
          </div>
        ),
      },
      {
        key: ATTRIBUTE_COLUMN_LABELS.USER,
        className: 'run-table-container',
        title: user,
        children: (
          <div className='truncate-text single-line' style={ExperimentViewUtil.styles.runInfoCell}>
            {user}
          </div>
        ),
      },
      {
        key: ATTRIBUTE_COLUMN_LABELS.RUN_NAME,
        className: 'run-table-container',
        title: runName,
        children: (
          <div className='truncate-text single-line' style={ExperimentViewUtil.styles.runInfoCell}>
            {runName}
          </div>
        ),
      },
      {
        key: ATTRIBUTE_COLUMN_LABELS.SOURCE,
        className: 'run-table-container',
        title: sourceType,
        children: (
          <div className='truncate-text single-line' style={ExperimentViewUtil.styles.runInfoCell}>
            {Utils.renderSourceTypeIcon(tags)}
            {sourceType}
          </div>
        ),
      },
      {
        key: ATTRIBUTE_COLUMN_LABELS.VERSION,
        className: 'run-table-container',
        children: (
          <div className='truncate-text single-line' style={ExperimentViewUtil.styles.runInfoCell}>
            {Utils.renderVersion(tags)}
          </div>
        ),
      },
      {
        key: 'Tags',
        className: 'run-table-container',
        children: (
          <div style={ExperimentViewUtil.styles.runInfoCell}>
            <CollapsibleTagsCell tags={tags} onToggle={handleCellToggle} />
          </div>
        ),
      },
    ];
    const excludedKeysSet = new Set(excludedKeys);
    return columnProps
      .filter((column) => !excludedKeysSet.has(column.key))
      .map((props) => <CellComponent {...props} />);
  }

  /**
   * Format a string for insertion into a CSV file.
   */
  static csvEscape(str) {
    if (str === undefined) {
      return '';
    }
    if (/[,"\r\n]/.test(str)) {
      return '"' + str.replace(/"/g, '""') + '"';
    }
    return str;
  }

  /**
   * Convert a table to a CSV string.
   *
   * @param columns Names of columns
   * @param data Array of rows, each of which are an array of field values
   */
  static tableToCsv(columns, data) {
    let csv = '';
    let i;

    for (i = 0; i < columns.length; i++) {
      csv += ExperimentViewUtil.csvEscape(columns[i]);
      if (i < columns.length - 1) {
        csv += ',';
      }
    }
    csv += '\n';

    for (i = 0; i < data.length; i++) {
      for (let j = 0; j < data[i].length; j++) {
        csv += ExperimentViewUtil.csvEscape(data[i][j]);
        if (j < data[i].length - 1) {
          csv += ',';
        }
      }
      csv += '\n';
    }

    return csv;
  }

  /**
   * Convert an array of run infos to a CSV string, extracting the params and metrics in the
   * provided lists.
   */
  static runInfosToCsv(
    runInfos,
    paramKeyList,
    metricKeyList,
    tagKeyList,
    paramsList,
    metricsList,
    tagsList,
  ) {
    const columns = [
      'Start Time',
      'Duration',
      'Run ID',
      'Name',
      'Source Type',
      'Source Name',
      'User',
      'Status',
      ...paramKeyList,
      ...metricKeyList,
      ...tagKeyList,
    ];

    const data = runInfos.map((runInfo, index) => {
      const row = [
        Utils.formatTimestamp(runInfo.start_time),
        Utils.getDuration(runInfo.start_time, runInfo.end_time) || '',
        runInfo.run_uuid,
        Utils.getRunName(tagsList[index]), // add run name to csv export row
        Utils.getSourceType(tagsList[index]),
        Utils.getSourceName(tagsList[index]),
        Utils.getUser(runInfo, tagsList[index]),
        runInfo.status,
      ];
      const paramsMap = ExperimentViewUtil.toParamsMap(paramsList[index]);
      const metricsMap = ExperimentViewUtil.toMetricsMap(metricsList[index]);
      const tagsMap = tagsList[index];

      paramKeyList.forEach((paramKey) => {
        if (paramsMap[paramKey]) {
          row.push(paramsMap[paramKey].getValue());
        } else {
          row.push('');
        }
      });
      metricKeyList.forEach((metricKey) => {
        if (metricsMap[metricKey]) {
          row.push(metricsMap[metricKey].getValue());
        } else {
          row.push('');
        }
      });
      tagKeyList.forEach((tagKey) => {
        if (tagsMap[tagKey]) {
          row.push(tagsMap[tagKey].getValue());
        } else {
          row.push('');
        }
      });
      return row;
    });

    return ExperimentViewUtil.tableToCsv(columns, data);
  }

  /**
   * Returns an icon for sorting the metric or param column with the specified key. The icon
   * is visible if we're currently sorting by the corresponding column. Otherwise, the icon is
   * invisible but takes up space.
   */
  static getSortIcon(curOrderByKey, curOrderByAsc, canonicalKey) {
    if (curOrderByKey === canonicalKey) {
      return (
        <span>
          <i
            className={curOrderByAsc ? 'fas fa-caret-up' : 'fas fa-caret-down'}
            style={ExperimentViewUtil.styles.sortIconStyle}
          />
        </span>
      );
    }
    return undefined;
  }

  /** Returns checkbox element for selecting all runs */
  static getSelectAllCheckbox(onCheckAll, isAllCheckedBool, cellType) {
    const CellComponent = `${cellType}`;
    return (
      <CellComponent key='meta-check' className='bottom-row run-table-container'>
        <input type='checkbox' onChange={onCheckAll} checked={isAllCheckedBool} />
      </CellComponent>
    );
  }

  /**
   * Returns header-row table cells for columns containing run metadata.
   */
  static getRunMetadataHeaderCells(onSortBy, curOrderByKey, curOrderByAsc, cellType, excludedCols) {
    const CellComponent = `${cellType}`;
    const getHeaderCell = (key, text, canonicalSortKey) => {
      const sortIcon = ExperimentViewUtil.getSortIcon(
        curOrderByKey,
        curOrderByAsc,
        canonicalSortKey,
      );
      const isSortable = canonicalSortKey !== null;
      const cellClassName = classNames('run-table-container', {
        sortable: isSortable,
      });
      return (
        <CellComponent
          key={'meta-' + key}
          className={cellClassName}
          onClick={() => (isSortable ? onSortBy(canonicalSortKey, !curOrderByAsc) : null)}
        >
          <span>{text}</span>
          {isSortable && (
            <span style={ExperimentViewUtil.styles.sortIconContainer}>{sortIcon}</span>
          )}
        </CellComponent>
      );
    };
    const excludedColsSet = new Set(excludedCols);
    return [
      {
        key: 'status',
        canonicalSortKey: null,
      },
      {
        key: 'start_time',
        displayName: ATTRIBUTE_COLUMN_LABELS.DATE,
        canonicalSortKey: ATTRIBUTE_COLUMN_SORT_KEY.DATE,
      },
      {
        key: 'duration',
        displayName: ATTRIBUTE_COLUMN_LABELS.DURATION,
        canonicalSortKey: null,
      },
      {
        key: 'user_id',
        displayName: ATTRIBUTE_COLUMN_LABELS.USER,
        canonicalSortKey: ATTRIBUTE_COLUMN_SORT_KEY.USER,
      },
      {
        key: 'run_name',
        displayName: ATTRIBUTE_COLUMN_LABELS.RUN_NAME,
        canonicalSortKey: ATTRIBUTE_COLUMN_SORT_KEY.RUN_NAME,
      },
      {
        key: 'source',
        displayName: ATTRIBUTE_COLUMN_LABELS.SOURCE,
        canonicalSortKey: ATTRIBUTE_COLUMN_SORT_KEY.SOURCE,
      },
      {
        key: 'source_version',
        displayName: ATTRIBUTE_COLUMN_LABELS.VERSION,
        canonicalSortKey: ATTRIBUTE_COLUMN_SORT_KEY.VERSION,
      },
      {
        key: 'tags',
        displayName: 'Tags',
        canonicalSortKey: null,
      },
      {
        key: 'linked-models',
        displayName: 'Linked Models',
        canonicalSortKey: null,
      },
    ]
      .filter((column) => !excludedColsSet.has(column.displayName))
      .map((h) => getHeaderCell(h.key, <span>{h.displayName}</span>, h.canonicalSortKey));
  }

  static makeCanonicalKey(keyType, keyName) {
    return keyType + '.`' + keyName + '`';
  }

  static getExpanderHeader(cellType) {
    const CellComponent = `${cellType}`;
    return (
      <CellComponent
        key={'meta-expander'}
        className={'bottom-row run-table-container'}
        style={{ width: '5px' }}
      />
    );
  }

  /**
   * Returns a table cell corresponding to a single metric value. The metric is assumed to be
   * unbagged (marked to be displayed in its own column).
   * @param metricKey The key of the desired metric
   * @param metricsMap Object mapping metric keys to their latest values for a single run
   * @param metricRanges Object mapping metric keys to objects of the form {min: ..., max: ...}
   *                     containing min and max values of the metric across all visible runs.
   * @param cellType Tag type (string like "div", "td", etc) of containing cell.
   */
  static getUnbaggedMetricCell(metricKey, metricsMap, metricRanges, cellType) {
    const className = 'left-border run-table-container';
    const keyName = 'metric-' + metricKey;
    const CellComponent = `${cellType}`;
    if (metricsMap[metricKey]) {
      const metric = metricsMap[metricKey].getValue();
      const range = metricRanges[metricKey];
      let fraction = 1.0;
      if (range.max > range.min) {
        fraction = (metric - range.min) / (range.max - range.min);
      }
      const percent = fraction * 100 + '%';
      return (
        <CellComponent className={className} key={keyName}>
          {/* We need the extra div because metric-filler-bg is inline-block */}
          <div>
            <div className='metric-filler-bg'>
              <div className='metric-filler-fg' style={{ width: percent }} />
              <div className='metric-text'>{Utils.formatMetric(metric)}</div>
            </div>
          </div>
        </CellComponent>
      );
    }
    return <CellComponent className={className} key={keyName} />;
  }

  static getUnbaggedParamCell(paramKey, paramsMap, cellType) {
    const CellComponent = `${cellType}`;
    const className = 'left-border run-table-container';
    const keyName = 'param-' + paramKey;
    if (paramsMap[paramKey]) {
      return (
        <CellComponent className={className} key={keyName}>
          <div>{paramsMap[paramKey].getValue()}</div>
        </CellComponent>
      );
    } else {
      return <CellComponent className={className} key={keyName} />;
    }
  }

  static renderLinkedModelCell(modelVersion) {
    const { name, version } = modelVersion;
    return (
      <div className='version-link'>
        <img src={registryIcon} alt='MLflow Model Registry Icon' />
        <span className='model-link-text'>
          {/* Reported during ESLint upgrade */}
          {/* eslint-disable-next-line react/jsx-no-target-blank */}
          <a
            href={Utils.getIframeCorrectedRoute(getModelVersionPageRoute(name, version))}
            className='model-version-link'
            title={`${name}, v${version}`}
            style={{ verticalAlign: 'middle' }}
            target='_blank'
          >
            <TrimmedText text={name} maxSize={10} className={'model-name'} />
            <span>/{version}&nbsp;</span>
          </a>
        </span>
      </div>
    );
  }

  static getLinkedModelCell(associatedModelVersions, handleCellToggle) {
    const className = 'run-table-container';
    if (associatedModelVersions && associatedModelVersions.length > 0) {
      return (
        <div className={className} key='linked=models'>
          <ExpandableList
            children={associatedModelVersions.map((version) => this.renderLinkedModelCell(version))}
            showLines={1}
            onToggle={handleCellToggle}
          />
        </div>
      );
    } else {
      return null;
    }
  }

  static computeMetricRanges(metricsByRun) {
    const ret = {};
    metricsByRun.forEach((metrics) => {
      metrics.forEach((metric) => {
        if (!ret.hasOwnProperty(metric.key)) {
          ret[metric.key] = {
            min: Math.min(metric.value, metric.value * 0.7),
            max: metric.value,
          };
        } else {
          if (metric.value < ret[metric.key].min) {
            ret[metric.key].min = Math.min(metric.value, metric.value * 0.7);
          }
          if (metric.value > ret[metric.key].max) {
            ret[metric.key].max = metric.value;
          }
        }
      });
    });
    return ret;
  }

  /**
   * Turn a list of metrics to a map of metric key to metric.
   */
  static toMetricsMap(metrics) {
    const ret = {};
    metrics.forEach((metric) => {
      ret[metric.key] = metric;
    });
    return ret;
  }

  /**
   * Turn a list of param to a map of metric key to metric.
   */
  static toParamsMap(params) {
    const ret = {};
    params.forEach((param) => {
      ret[param.key] = param;
    });
    return ret;
  }

  static isExpanderOpen(runsExpanded, runId) {
    let expanderOpen = DEFAULT_EXPANDED_VALUE;
    if (runsExpanded[runId] !== undefined) expanderOpen = runsExpanded[runId];
    return expanderOpen;
  }

  static getExpander(hasExpander, expanderOpen, onExpandBound, runUuid, cellType) {
    const CellComponent = `${cellType}`;
    if (!hasExpander) {
      return <CellComponent key={'Expander-' + runUuid} style={{ padding: 8 }}></CellComponent>;
    }
    if (expanderOpen) {
      return (
        <CellComponent onClick={onExpandBound} key={'Expander-' + runUuid} style={{ padding: 8 }}>
          <i className='ExperimentView-expander far fa-minus-square' />
        </CellComponent>
      );
    } else {
      return (
        <CellComponent onClick={onExpandBound} key={'Expander-' + runUuid} style={{ padding: 8 }}>
          <i className='ExperimentView-expander far fa-plus-square' />
        </CellComponent>
      );
    }
  }

  static getNestedRowRenderMetadata({ runInfos, tagsList, runsExpanded }) {
    const runIdToIdx = {};
    runInfos.forEach((r, idx) => {
      runIdToIdx[r.run_uuid] = idx;
    });
    const treeNodes = runInfos.map((r) => new TreeNode(r.run_uuid));
    tagsList.forEach((tags, idx) => {
      const parentRunId = tags['mlflow.parentRunId'];
      if (parentRunId) {
        const parentRunIdx = runIdToIdx[parentRunId.value];
        if (parentRunIdx !== undefined) {
          treeNodes[idx].parent = treeNodes[parentRunIdx];
        }
      }
    });
    // Map of parentRunIds to list of children runs (idx)
    const parentIdToChildren = {};
    treeNodes.forEach((t, idx) => {
      const root = t.findRoot();
      if (root !== undefined && root.value !== t.value) {
        const old = parentIdToChildren[root.value];
        let newList;
        if (old) {
          old.push(idx);
          newList = old;
        } else {
          newList = [idx];
        }
        parentIdToChildren[root.value] = newList;
      }
    });

    const parentRows = _.flatMap([...Array(runInfos.length).keys()], (idx) => {
      if (treeNodes[idx].isCycle() || !treeNodes[idx].isRoot()) return [];
      const runId = runInfos[idx].run_uuid;
      let hasExpander = false;
      let childrenIds = undefined;
      if (parentIdToChildren[runId]) {
        hasExpander = true;
        childrenIds = parentIdToChildren[runId].map((cIdx) => runInfos[cIdx].run_uuid);
      }
      return [
        {
          idx,
          isParent: true,
          hasExpander,
          expanderOpen: ExperimentViewUtil.isExpanderOpen(runsExpanded, runId),
          childrenIds,
          runId,
        },
      ];
    });
    const mergedRows = [];
    parentRows.forEach((r) => {
      const { runId } = r;
      mergedRows.push(r);
      const childrenIdxs = parentIdToChildren[runId];
      if (childrenIdxs) {
        if (ExperimentViewUtil.isExpanderOpen(runsExpanded, runId)) {
          const childrenRows = childrenIdxs.map((idx) => {
            return { idx, isParent: false, hasExpander: false };
          });
          mergedRows.push(...childrenRows);
        }
      }
    });
    return mergedRows.slice(0);
  }

  static getRowRenderMetadata({ runInfos, tagsList, runsExpanded, nestChildren }) {
    if (nestChildren) {
      return this.getNestedRowRenderMetadata({ runInfos, tagsList, runsExpanded });
    } else {
      return [...Array(runInfos.length).keys()].map((idx) => ({
        idx,
        isParent: true,
        hasExpander: false,
        runId: runInfos[idx].run_uuid,
      }));
    }
  }

  static getRows({ runInfos, tagsList, runsExpanded, getRow }) {
    const mergedRows = ExperimentViewUtil.getRowRenderMetadata({
      runInfos,
      tagsList,
      runsExpanded,
    });
    return mergedRows.map((rowMetadata) => getRow(rowMetadata));
  }

  static renderRows(rows) {
    return rows.map((row) => {
      const style = row.isChild ? { backgroundColor: '#fafafa' } : {};
      return (
        <tr key={row.key} style={style} className='ExperimentView-row'>
          {row.contents}
        </tr>
      );
    });
  }

  static disableLoadMoreButton({ numRunsFromLatestSearch, nextPageToken }) {
    if (numRunsFromLatestSearch === null) {
      // numRunsFromLatestSearch is null by default, so we should not disable the button
      return false;
    }
    return nextPageToken === null;
  }

  /**
   * Obtain the categorized columns (params, metrics & tags) for which the values
   * in them have only a single value (or are undefined). For attribute columns,
   * obtain the columns for which every value is undefined.
   */
  static getCategorizedUncheckedKeysDiffView({
    categorizedUncheckedKeys,
    paramKeyList,
    metricKeyList,
    runInfos,
    paramsList,
    metricsList,
    tagsList,
  }) {
    const attributeColumnsToTags = {
      // Leave the User and Source columns out of consideration because they normally have values.
      [ATTRIBUTE_COLUMN_LABELS.RUN_NAME]: Utils.runNameTag,
      [ATTRIBUTE_COLUMN_LABELS.VERSION]: Utils.gitCommitTag,
      [ATTRIBUTE_COLUMN_LABELS.MODELS]: Utils.loggedModelsTag,
    };
    const attributeKeyList = Object.keys(attributeColumnsToTags);
    const tagKeyList = Utils.getVisibleTagKeyList(tagsList);
    let attributeColumnsToUncheck = _.difference(
      attributeKeyList,
      categorizedUncheckedKeys[COLUMN_TYPES.ATTRIBUTES],
    );
    let paramColumnsToUncheck = _.difference(
      paramKeyList,
      categorizedUncheckedKeys[COLUMN_TYPES.PARAMS],
    );
    let metricColumnsToUncheck = _.difference(
      metricKeyList,
      categorizedUncheckedKeys[COLUMN_TYPES.METRICS],
    );
    let tagColumnsToUncheck = _.difference(tagKeyList, categorizedUncheckedKeys[COLUMN_TYPES.TAGS]);

    const dropDiffColumns = (columns, prevRow, currRow) => {
      // What each argument represents:
      // | a   | b   | c   | d   | e   | <- columns
      // | --- | --- | --- | --- | --- |
      // | -   | 1   | -   | 1   | 1   | <- prevRow
      // | -   | -   | 1   | 1   | 2   | <- currRow
      // | ?   | ?   | ?   | ?   | ?   |
      //
      // a, d: may be a diff column, we need to check the next row
      // b, c, e: is a diff column, we don't need to check the next row

      return columns.filter((col) => {
        const prevValue = prevRow[col];
        const currValue = currRow[col];
        if (!prevValue && !currValue) {
          // Case a
          return true;
        } else if (!prevValue || !currValue) {
          // Case b & c
          return false;
        } else if (prevValue.getValue() === currValue.getValue()) {
          // Case d
          return true;
        } else {
          // Case e
          return false;
        }
      });
    };

    for (const [index] of runInfos.entries()) {
      // Drop non-empty attribute columns
      attributeColumnsToUncheck = attributeColumnsToUncheck.filter(
        (col) => !(attributeColumnsToTags[col] in tagsList[index]),
      );

      if (index === 0) {
        continue;
      }

      // The following operations need to be skipped in the first iteration.

      paramColumnsToUncheck = dropDiffColumns(
        paramColumnsToUncheck,
        ExperimentViewUtil.toParamsMap(paramsList[index - 1]),
        ExperimentViewUtil.toParamsMap(paramsList[index]),
      );

      metricColumnsToUncheck = dropDiffColumns(
        metricColumnsToUncheck,
        ExperimentViewUtil.toMetricsMap(metricsList[index - 1]),
        ExperimentViewUtil.toMetricsMap(metricsList[index]),
      );

      tagColumnsToUncheck = dropDiffColumns(
        tagColumnsToUncheck,
        tagsList[index - 1],
        tagsList[index],
      );

      // Short-circuit loop if there are no more columns to take a look at
      if (
        attributeColumnsToUncheck.length === 0 &&
        paramColumnsToUncheck.length === 0 &&
        metricColumnsToUncheck.length === 0 &&
        tagColumnsToUncheck.length === 0
      ) {
        break;
      }
    }

    return {
      [COLUMN_TYPES.ATTRIBUTES]: _.concat(
        categorizedUncheckedKeys[COLUMN_TYPES.ATTRIBUTES],
        attributeColumnsToUncheck,
      ),
      [COLUMN_TYPES.PARAMS]: _.concat(
        categorizedUncheckedKeys[COLUMN_TYPES.PARAMS],
        paramColumnsToUncheck,
      ),
      [COLUMN_TYPES.METRICS]: _.concat(
        categorizedUncheckedKeys[COLUMN_TYPES.METRICS],
        metricColumnsToUncheck,
      ),
      [COLUMN_TYPES.TAGS]: _.concat(
        categorizedUncheckedKeys[COLUMN_TYPES.TAGS],
        tagColumnsToUncheck,
      ),
    };
  }

  /**
   * Get the categorized unchecked keys that were in place before hitting the diff switch
   * with state changes in between also reflected
   * @param preSwitchCategorizedUncheckedKeys the keys that were unchecked before diff-view
   *  switch was turned on
   * @param postSwitchCategorizedUncheckedKeys the keys that were unchecked by turning the
   *  diff-view switch on
   * @param currCategorizedUncheckedKeys currently unchecked keys (possibly includes keys
   * that were checked or unchecked while being in the diff view)
   */
  static getRestoredCategorizedUncheckedKeys({
    preSwitchCategorizedUncheckedKeys,
    postSwitchCategorizedUncheckedKeys,
    currCategorizedUncheckedKeys,
  }) {
    const restoredUncheckedKeys = (column_type) => {
      // keys that the user checked while being in diff view
      const userCheckedKeys = _.difference(
        postSwitchCategorizedUncheckedKeys[column_type],
        currCategorizedUncheckedKeys[column_type],
      );
      // keys that the user unchecked while being in diff view
      const userUncheckedKeys = _.difference(
        currCategorizedUncheckedKeys[column_type],
        postSwitchCategorizedUncheckedKeys[column_type],
      );
      return _.uniq(
        _.without(
          _.concat(preSwitchCategorizedUncheckedKeys[column_type], userUncheckedKeys),
          ...userCheckedKeys,
        ),
      );
    };
    return {
      [COLUMN_TYPES.ATTRIBUTES]: restoredUncheckedKeys(COLUMN_TYPES.ATTRIBUTES),
      [COLUMN_TYPES.PARAMS]: restoredUncheckedKeys(COLUMN_TYPES.PARAMS),
      [COLUMN_TYPES.METRICS]: restoredUncheckedKeys(COLUMN_TYPES.METRICS),
      [COLUMN_TYPES.TAGS]: restoredUncheckedKeys(COLUMN_TYPES.TAGS),
    };
  }
}

export class TreeNode {
  constructor(value) {
    this.value = value;
    this.parent = undefined;
  }
  /**
   * Returns the root node. If there is a cycle it will return undefined;
   */
  findRoot() {
    const visited = new Set([this.value]);
    let current = this;
    while (current.parent !== undefined) {
      if (visited.has(current.parent.value)) {
        return undefined;
      }
      visited.add(current.value);
      current = current.parent;
    }
    return current;
  }
  isRoot() {
    return this.findRoot().value === this.value;
  }
  isCycle() {
    return this.findRoot() === undefined;
  }
}

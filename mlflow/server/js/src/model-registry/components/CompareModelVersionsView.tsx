/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Link } from '../../common/utils/RoutingUtils';
import _ from 'lodash';
import { FormattedMessage, injectIntl, IntlShape } from 'react-intl';
import {
  Switch,
  LegacyTabs,
  useDesignSystemTheme,
  TableRow,
  TableCell,
  Table,
  Spacer,
} from '@databricks/design-system';

import { getParams, getRunInfo } from '../../experiment-tracking/reducers/Reducers';
import './CompareModelVersionsView.css';
import { CompareRunScatter } from '../../experiment-tracking/components/CompareRunScatter';
import { CompareRunBox } from '../../experiment-tracking/components/CompareRunBox';
import CompareRunContour from '../../experiment-tracking/components/CompareRunContour';
import Routes from '../../experiment-tracking/routes';
import { getLatestMetrics } from '../../experiment-tracking/reducers/MetricReducer';
import CompareRunUtil from '../../experiment-tracking/components/CompareRunUtil';
import Utils from '../../common/utils/Utils';
import ParallelCoordinatesPlotPanel from '../../experiment-tracking/components/ParallelCoordinatesPlotPanel';
import { ModelRegistryRoutes } from '../routes';
import { getModelVersionSchemas } from '../reducers';
import { PageHeader } from '../../shared/building_blocks/PageHeader';
import type { RunInfoEntity } from '../../experiment-tracking/types';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';

const { TabPane } = LegacyTabs;

const DEFAULT_TABLE_COLUMN_WIDTH = 200;

function CenteredText(props: any) {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        textAlign: 'center',
        color: theme.colors.textSecondary,
      }}
      {...props}
    />
  );
}

type CompareModelVersionsViewImplProps = {
  runInfos: RunInfoEntity[];
  runInfosValid: boolean[];
  runUuids: string[];
  metricLists: any[][];
  paramLists: any[][];
  runNames: string[];
  runDisplayNames: string[];
  modelName: string;
  versionsToRuns: any;
  // @ts-expect-error TS(2314): Generic type 'Array<T>' requires 1 type argument(s... Remove this comment to see the full error message
  inputsListByName: Array[];
  // @ts-expect-error TS(2314): Generic type 'Array<T>' requires 1 type argument(s... Remove this comment to see the full error message
  inputsListByIndex: Array[];
  // @ts-expect-error TS(2314): Generic type 'Array<T>' requires 1 type argument(s... Remove this comment to see the full error message
  outputsListByName: Array[];
  // @ts-expect-error TS(2314): Generic type 'Array<T>' requires 1 type argument(s... Remove this comment to see the full error message
  outputsListByIndex: Array[];
  intl: IntlShape;
};

type CompareModelVersionsViewImplState = any;

export class CompareModelVersionsViewImpl extends Component<
  CompareModelVersionsViewImplProps,
  CompareModelVersionsViewImplState
> {
  compareModelVersionViewRef: any;

  constructor(props: CompareModelVersionsViewImplProps) {
    super(props);
    this.compareModelVersionViewRef = React.createRef();
    this.onCompareModelTableScrollHandler = this.onCompareModelTableScrollHandler.bind(this);
  }

  state = {
    inputActive: true,
    outputActive: true,
    onlyShowParameterDiff: true,
    onlyShowSchemaDiff: true,
    onlyShowMetricDiff: true,
    compareByColumnNameToggle: false,
  };

  icons = {
    plusIcon: <i className="far fa-plus-square-o" />,
    minusIcon: <i className="far fa-minus-square-o" />,
    downIcon: <i className="fas fa-caret-down" />,
    rightIcon: <i className="fas fa-caret-right" />,
    chartIcon: <i className="fas fa-line-chart padding-left-text" />,
  };

  onToggleClick = (active: any) => {
    this.setState((state: any) => ({
      [active]: !state[active],
    }));
  };

  onCompareModelTableScrollHandler(e: React.ChangeEvent<HTMLDivElement>) {
    const blocks = this.compareModelVersionViewRef.current?.querySelectorAll('.compare-model-table');
    blocks.forEach((_: any, index: any) => {
      const block = blocks[index];
      if (block !== e.target) {
        block.scrollLeft = e.target.scrollLeft;
      }
    });
  }

  getTableHeaderColumnWidth() {
    return {
      width: `${DEFAULT_TABLE_COLUMN_WIDTH}px`,
      minWidth: `${DEFAULT_TABLE_COLUMN_WIDTH}px`,
      maxWidth: `${DEFAULT_TABLE_COLUMN_WIDTH}px`,
    };
  }

  getTableBodyColumnWidth() {
    const runInfoLength = this.props.runInfos.length;
    const widthStyle = `max(${DEFAULT_TABLE_COLUMN_WIDTH}px, calc(100vw / ${runInfoLength}))`;
    return { width: widthStyle, minWidth: widthStyle, maxWidth: widthStyle };
  }

  render() {
    const {
      inputsListByIndex,
      inputsListByName,
      modelName,
      outputsListByIndex,
      outputsListByName,
      runInfos,
      runUuids,
      runDisplayNames,
      paramLists,
      metricLists,
      intl,
    } = this.props;
    const title = (
      <FormattedMessage
        defaultMessage="Comparing {numVersions} Versions"
        description="Text for main title for the model comparison page"
        values={{ numVersions: this.props.runInfos.length }}
      />
    );
    const breadcrumbs = [
      <Link to={ModelRegistryRoutes.modelListPageRoute}>
        <FormattedMessage
          defaultMessage="Registered Models"
          description="Text for registered model link in the title for model comparison page"
        />
      </Link>,
      <Link to={ModelRegistryRoutes.getModelPageRoute(modelName)}>{modelName}</Link>,
    ];

    const showdiffIntlMessage = intl.formatMessage({
      defaultMessage: 'Show diff only',
      description: 'Toggle text that determines whether to show diff only in the model comparison page',
    });

    return (
      <div>
        <PageHeader title={title} breadcrumbs={breadcrumbs} />
        <div ref={this.compareModelVersionViewRef}>
          {this.renderTable(
            <>
              {this.renderTableHeader()}
              {this.renderModelVersionInfo()}
            </>,
          )}
          <CollapsibleSection
            title={intl.formatMessage({
              defaultMessage: 'Parameters',
              description: 'Table title text for parameters table in the model comparison page',
            })}
          >
            <Switch
              componentId="mlflow.model-registry.compare-model-versions-parameters-diff-switch"
              label={showdiffIntlMessage}
              checked={this.state.onlyShowParameterDiff}
              onChange={(checked, e) => this.setState({ onlyShowParameterDiff: checked })}
            />
            <Spacer size="sm" />
            {this.renderTable(this.renderParams())}
          </CollapsibleSection>
          <CollapsibleSection
            title={intl.formatMessage({
              defaultMessage: 'Schema',
              description: 'Table title text for schema table in the model comparison page',
            })}
          >
            <Switch
              componentId="mlflow.model-registry.compare-model-versions-schema-ignore-column-order-switch"
              label={intl.formatMessage({
                defaultMessage: 'Ignore column ordering',
                description:
                  'Toggle text that determines whether to ignore column order in the\n                      model comparison page',
              })}
              checked={this.state.compareByColumnNameToggle}
              onChange={(checked, e) => this.setState({ compareByColumnNameToggle: checked })}
            />
            <Spacer size="sm" />
            <Switch
              componentId="mlflow.model-registry.compare-model-versions-schema-diff-switch"
              label={showdiffIntlMessage}
              checked={this.state.onlyShowSchemaDiff}
              onChange={(checked, e) => this.setState({ onlyShowSchemaDiff: checked })}
            />
            <Spacer size="sm" />
            <div>
              {this.renderSchemaSectionHeader(
                'inputActive',
                <FormattedMessage
                  defaultMessage="Inputs"
                  description="Table subtitle for schema inputs in the model comparison page"
                />,
              )}
              {this.state.inputActive && (
                <>
                  {this.renderTable(
                    this.renderSchema(
                      'inputActive',
                      <FormattedMessage
                        defaultMessage="Inputs"
                        description="Table section name for schema inputs in the model comparison page"
                      />,
                      inputsListByIndex,
                      inputsListByName,
                    ),
                  )}
                </>
              )}
            </div>
            <Spacer size="sm" />
            <div>
              {this.renderSchemaSectionHeader(
                'outputActive',
                <FormattedMessage
                  defaultMessage="Outputs"
                  description="Table subtitle for schema outputs in the model comparison page"
                />,
              )}
              {this.state.outputActive && (
                <>
                  {this.renderTable(
                    this.renderSchema(
                      'outputActive',
                      <FormattedMessage
                        defaultMessage="Outputs"
                        description="Table section name for schema outputs in the model comparison page"
                      />,
                      outputsListByIndex,
                      outputsListByName,
                    ),
                  )}
                </>
              )}
            </div>
          </CollapsibleSection>
          <CollapsibleSection
            title={intl.formatMessage({
              defaultMessage: 'Metrics',
              description: 'Table title text for metrics table in the model comparison page',
            })}
          >
            <Switch
              componentId="mlflow.model-registry.compare-model-versions-metrics-diff-switch"
              label={showdiffIntlMessage}
              checked={this.state.onlyShowMetricDiff}
              onChange={(checked, e) => this.setState({ onlyShowMetricDiff: checked })}
            />
            <Spacer size="sm" />
            {this.renderTable(this.renderMetrics())}
          </CollapsibleSection>
        </div>
        <LegacyTabs>
          <TabPane
            tab={
              <FormattedMessage
                defaultMessage="Parallel Coordinates Plot"
                description="Tab text for parallel coordinates plot on the model comparison page"
              />
            }
            key="parallel-coordinates-plot"
          >
            <ParallelCoordinatesPlotPanel runUuids={runUuids} />
          </TabPane>
          <TabPane
            tab={
              <FormattedMessage
                defaultMessage="Scatter Plot"
                description="Tab text for scatter plot on the model comparison page"
              />
            }
            key="scatter-plot"
          >
            <CompareRunScatter runUuids={runUuids} runDisplayNames={runDisplayNames} />
          </TabPane>
          <TabPane
            tab={
              <FormattedMessage
                defaultMessage="Box Plot"
                description="Tab pane title for box plot on the compare runs page"
              />
            }
            key="box-plot"
          >
            <CompareRunBox runUuids={runUuids} runInfos={runInfos} paramLists={paramLists} metricLists={metricLists} />
          </TabPane>
          <TabPane
            tab={
              <FormattedMessage
                defaultMessage="Contour Plot"
                description="Tab text for contour plot on the model comparison page"
              />
            }
            key="contour-plot"
          >
            <CompareRunContour runUuids={runUuids} runDisplayNames={runDisplayNames} />
          </TabPane>
        </LegacyTabs>
      </div>
    );
  }

  renderTable(children: React.ReactNode): React.ReactNode {
    return (
      // @ts-expect-error TS(2322): Property 'onScroll' does not exist... Remove this comment to see the full error message
      <Table className="table compare-table compare-model-table" onScroll={this.onCompareModelTableScrollHandler}>
        {children}
      </Table>
    );
  }

  renderTableHeader() {
    const { runInfos, runInfosValid } = this.props;
    return (
      <TableRow className="compare-table-row">
        <TableCell
          className="head-value sticky-header"
          css={{
            backgroundColor: 'var(--table-header-background-color)',
            ...this.getTableHeaderColumnWidth(),
          }}
        >
          <FormattedMessage
            defaultMessage="Run ID:"
            description="Text for run ID header in the main table in the model comparison page"
          />
        </TableCell>
        {runInfos.map((r, idx) => (
          <TableCell className="data-value" css={{ ...this.getTableBodyColumnWidth() }} key={r.runUuid}>
            {/* Do not show links for invalid run IDs */}
            {runInfosValid[idx] ? (
              <Link to={Routes.getRunPageRoute(r.experimentId ?? '0', r.runUuid ?? '')}>{r.runUuid}</Link>
            ) : (
              r.runUuid
            )}
          </TableCell>
        ))}
      </TableRow>
    );
  }

  renderModelVersionInfo() {
    const { runInfos, runInfosValid, versionsToRuns, runNames, modelName } = this.props;
    return (
      <>
        <TableRow className="compare-table-row">
          <TableCell
            className="head-value sticky-header"
            css={{
              backgroundColor: 'var(--table-header-background-color)',
              ...this.getTableHeaderColumnWidth(),
            }}
          >
            <FormattedMessage
              defaultMessage="Model Version:"
              description="Text for model version row header in the main table in the model
                comparison page"
            />
          </TableCell>
          {Object.keys(versionsToRuns).map((modelVersion) => {
            const run = versionsToRuns[modelVersion];
            return (
              <TableCell className="data-value" key={run} css={{ ...this.getTableBodyColumnWidth() }}>
                <Link to={ModelRegistryRoutes.getModelVersionPageRoute(modelName, modelVersion)}>{modelVersion}</Link>
              </TableCell>
            );
          })}
        </TableRow>
        <TableRow className="compare-table-row">
          <TableCell
            className="head-value sticky-header"
            css={{
              backgroundColor: 'var(--table-header-background-color)',
              ...this.getTableHeaderColumnWidth(),
            }}
          >
            <FormattedMessage
              defaultMessage="Run Name:"
              description="Text for run name row header in the main table in the model comparison
                page"
            />
          </TableCell>
          {runNames.map((runName, i) => {
            return (
              <TableCell className="data-value" key={runInfos[i].runUuid} css={{ ...this.getTableBodyColumnWidth() }}>
                <div className="truncate-text single-line cell-content">{runName}</div>
              </TableCell>
            );
          })}
        </TableRow>
        <TableRow className="compare-table-row">
          <TableCell
            className="head-value sticky-header"
            css={{
              backgroundColor: 'var(--table-header-background-color)',
              ...this.getTableHeaderColumnWidth(),
            }}
          >
            <FormattedMessage
              defaultMessage="Start Time:"
              description="Text for start time row header in the main table in the model comparison
                page"
            />
          </TableCell>
          {runInfos.map((run, idx) => {
            /* Do not attempt to get timestamps for invalid run IDs */
            const startTime =
              run.startTime && runInfosValid[idx] ? Utils.formatTimestamp(run.startTime, this.props.intl) : '(unknown)';
            return (
              <TableCell className="data-value" key={run.runUuid} css={{ ...this.getTableBodyColumnWidth() }}>
                {startTime}
              </TableCell>
            );
          })}
        </TableRow>
      </>
    );
  }

  renderParams() {
    return (
      <>
        {this.renderDataRows(
          this.props.paramLists,
          <FormattedMessage
            defaultMessage="Parameters"
            description="Field name text for parameters table in the model comparison page"
          />,
          true,
          this.state.onlyShowParameterDiff,
        )}
      </>
    );
  }

  renderSchemaSectionHeader(activeSection: string, sectionName: React.ReactNode) {
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    const isActive = this.state[activeSection];
    const { minusIcon, plusIcon } = this.icons;
    return (
      <button onClick={() => this.onToggleClick(activeSection)}>
        {isActive ? minusIcon : plusIcon}
        <strong style={{ paddingLeft: 4 }}>{sectionName}</strong>
      </button>
    );
  }

  renderSchema(activeSection: any, sectionName: any, listByIndex: any, listByName: any) {
    const { compareByColumnNameToggle } = this.state;
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    const isActive = this.state[activeSection];
    const showSchemaSection = isActive;
    const showListByIndex = !compareByColumnNameToggle && !_.isEmpty(listByIndex);
    const showListByName = compareByColumnNameToggle && !_.isEmpty(listByName);
    const listByIndexHeaderMap = (key: any, data: any) => (
      <>
        {sectionName} [{key}]
      </>
    );
    const listByNameHeaderMap = (key: any, data: any) => key;
    const schemaFormatter = (value: any) => value;
    const schemaFieldName = (
      <FormattedMessage
        defaultMessage="Schema {sectionName}"
        description="Field name text for schema table in the model comparison page"
        values={{ sectionName: sectionName }}
      />
    );
    return (
      <>
        {this.renderDataRows(
          listByIndex,
          schemaFieldName,
          showSchemaSection && showListByIndex,
          this.state.onlyShowSchemaDiff,
          listByIndexHeaderMap,
          schemaFormatter,
        )}
        {this.renderDataRows(
          listByName,
          schemaFieldName,
          showSchemaSection && showListByName,
          this.state.onlyShowSchemaDiff,
          listByNameHeaderMap,
          schemaFormatter,
        )}
      </>
    );
  }

  renderMetrics() {
    const { runInfos, metricLists } = this.props;
    const { chartIcon } = this.icons;
    const metricsHeaderMap = (key: any, data: any) => {
      return (
        <Link
          to={Routes.getMetricPageRoute(
            runInfos.map((info) => info.runUuid).filter((uuid, idx) => data[idx] !== undefined),
            key,
            // TODO: Refactor so that the breadcrumb
            // on the linked page is for model registry
            [runInfos[0].experimentId],
          )}
          target="_blank"
          title="Plot chart"
        >
          {key}
          {chartIcon}
        </Link>
      );
    };
    return (
      <>
        {this.renderDataRows(
          metricLists,
          <FormattedMessage
            defaultMessage="Metrics"
            description="Field name text for metrics table in the model comparison page"
          />,
          true,
          this.state.onlyShowMetricDiff,
          metricsHeaderMap,
          Utils.formatMetric,
        )}
      </>
    );
  }

  renderDataRows(
    list: any,
    fieldName: any,
    show = true,
    onlyShowDiff = false,
    headerMap = (key: any, data: any) => key,
    formatter = (value: any) => (isNaN(value) ? `"${value}"` : value),
  ): React.ReactNode {
    // @ts-expect-error TS(2554): Expected 2 arguments, but got 1.
    const keys = CompareRunUtil.getKeys(list);
    const data = {};
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    keys.forEach((k) => (data[k] = []));
    list.forEach((records: any, i: any) => {
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      keys.forEach((k) => data[k].push(undefined));
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      records.forEach((r: any) => (data[r.key][i] = r.value));
    });
    if (_.isEmpty(keys) || _.isEmpty(list)) {
      return (
        <TableRow className="compare-table-row">
          <TableCell
            className="head-value sticky-header"
            css={{
              backgroundColor: 'var(--table-header-background-color)',
              ...this.getTableHeaderColumnWidth(),
            }}
          >
            <CenteredText>
              <FormattedMessage
                defaultMessage="{fieldName} are empty"
                description="Default text in data table where items are empty in the model
                  comparison page"
                values={{ fieldName: fieldName }}
              />
            </CenteredText>
          </TableCell>
        </TableRow>
      );
    }
    // @ts-expect-error TS(2345): Argument of type 'string' is not assignable to par... Remove this comment to see the full error message
    const isAllNumeric = _.every(keys, (key) => !isNaN(key));
    if (isAllNumeric) {
      keys.sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
    } else {
      keys.sort();
    }
    let identical = true;
    const resultRows = keys.map((k) => {
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      const isDifferent = data[k].length > 1 && _.uniq(data[k]).length > 1;
      if (onlyShowDiff && !isDifferent) {
        return null;
      }
      identical = !isDifferent && identical;
      return (
        <TableRow
          key={k}
          style={{ display: `${(onlyShowDiff && !isDifferent) || !show ? 'none' : ''}` }}
          className={`compare-table-row ${isDifferent ? 'diff-row' : ''}`}
        >
          <TableCell
            className="head-value sticky-header"
            css={{
              backgroundColor: 'var(--table-header-background-color)',
              ...this.getTableHeaderColumnWidth(),
            }}
          >
            {/* @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message */}
            {headerMap(k, data[k])}
          </TableCell>
          {/* @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message */}
          {data[k].map((value: any, i: any) => (
            <TableCell
              className={`data-value`}
              css={{ ...this.getTableBodyColumnWidth() }}
              key={this.props.runInfos[i].runUuid}
            >
              <span className="truncate-text single-line cell-content">
                {value === undefined ? '-' : formatter(value)}
              </span>
            </TableCell>
          ))}
        </TableRow>
      );
    });
    if (identical && onlyShowDiff) {
      return (
        <TableRow className={`compare-table-row`} style={{ display: `${show ? '' : 'none'}` }}>
          <TableCell className="data-value" css={{ ...this.getTableBodyColumnWidth() }}>
            <CenteredText>
              <FormattedMessage
                defaultMessage="{fieldName} are identical"
                description="Default text in data table where items are identical in the model comparison page"
                values={{ fieldName: fieldName }}
              />
            </CenteredText>
          </TableCell>
        </TableRow>
      );
    }
    return resultRows;
  }
}

const getModelVersionSchemaColumnsByIndex = (columns: any) => {
  const columnsByIndex = {};
  columns.forEach((column: any, index: any) => {
    const name = column.name ? column.name : '';
    const type = column.type ? column.type : '';
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    columnsByIndex[index] = {
      key: index,
      value: name !== '' && type !== '' ? `${name}: ${type}` : `${name}${type}`,
    };
  });
  return columnsByIndex;
};

const getModelVersionSchemaColumnsByName = (columns: any) => {
  const columnsByName: Record<string, { key: string; value: string }> = {};
  columns.forEach((column: any) => {
    const name = column.name ? column.name : '-';
    const type = column.type ? column.type : '-';
    columnsByName[name] = {
      key: name,
      value: type,
    };
  });
  return columnsByName;
};

const mapStateToProps = (state: any, ownProps: any) => {
  const runInfos = [];
  const runInfosValid = [];
  const metricLists = [];
  const paramLists = [];
  const runNames = [];
  const runDisplayNames = [];
  const runUuids = [];
  const { modelName, versionsToRuns } = ownProps;
  const inputsListByName = [];
  const inputsListByIndex = [];
  const outputsListByName = [];
  const outputsListByIndex = [];
  for (const modelVersion in versionsToRuns) {
    if (versionsToRuns && modelVersion in versionsToRuns) {
      const runUuid = versionsToRuns[modelVersion];
      const runInfo = getRunInfo(runUuid, state);
      if (runInfo) {
        runInfos.push(runInfo);
        runInfosValid.push(true);
        metricLists.push(Object.values(getLatestMetrics(runUuid, state)));
        paramLists.push(Object.values(getParams(runUuid, state)));
        runNames.push(Utils.getRunName(runInfo));
        // the following are used to render plots - we only include valid run IDs here
        runDisplayNames.push(Utils.getRunDisplayName(runInfo, runUuid));
        runUuids.push(runUuid);
      } else {
        if (runUuid) {
          runInfos.push({ runUuid });
        } else {
          runInfos.push({ runUuid: 'None' });
        }
        runInfosValid.push(false);
        metricLists.push([]);
        paramLists.push([]);
        runNames.push('Invalid Run');
      }
      const schema = getModelVersionSchemas(state, modelName, modelVersion);
      inputsListByIndex.push(Object.values(getModelVersionSchemaColumnsByIndex((schema as any).inputs)));
      inputsListByName.push(Object.values(getModelVersionSchemaColumnsByName((schema as any).inputs)));
      outputsListByIndex.push(Object.values(getModelVersionSchemaColumnsByIndex((schema as any).outputs)));
      outputsListByName.push(Object.values(getModelVersionSchemaColumnsByName((schema as any).outputs)));
    }
  }

  return {
    runInfos,
    runInfosValid,
    metricLists,
    paramLists,
    runNames,
    runDisplayNames,
    runUuids,
    modelName,
    inputsListByName,
    inputsListByIndex,
    outputsListByName,
    outputsListByIndex,
  };
};

export const CompareModelVersionsView = connect(mapStateToProps)(injectIntl(CompareModelVersionsViewImpl));

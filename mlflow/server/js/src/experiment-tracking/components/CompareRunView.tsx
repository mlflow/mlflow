import React, { useState, useEffect, useRef } from 'react';
import { connect } from 'react-redux';
import { injectIntl, FormattedMessage, IntlShape } from 'react-intl';
import { Spacer, Switch, Tabs, LegacyTooltip } from '@databricks/design-system';

import { getExperiment, getParams, getRunInfo, getRunTags } from '../reducers/Reducers';
import './CompareRunView.css';
import { CompareRunScatter } from './CompareRunScatter';
import { CompareRunBox } from './CompareRunBox';
import CompareRunContour from './CompareRunContour';
import Routes from '../routes';
import { Link } from '../../common/utils/RoutingUtils';
import { getLatestMetrics } from '../reducers/MetricReducer';
import CompareRunUtil from './CompareRunUtil';
import Utils from '../../common/utils/Utils';
import ParallelCoordinatesPlotPanel from './ParallelCoordinatesPlotPanel';
import { PageHeader } from '../../shared/building_blocks/PageHeader';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';
import { shouldDisableLegacyRunCompareCharts } from '../../common/utils/FeatureUtils';
import { RunInfoEntity } from '../types';
import { CompareRunArtifactView } from './CompareRunArtifactView';

const { TabPane } = Tabs;

type CompareRunViewProps = {
  experiments: any[]; // TODO: PropTypes.instanceOf(Experiment)
  experimentIds: string[];
  comparedExperimentIds?: string[];
  hasComparedExperimentsBefore?: boolean;
  runInfos: RunInfoEntity[];
  runUuids: string[];
  metricLists: any[][];
  paramLists: any[][];
  tagLists: any[][];
  runNames: string[];
  runDisplayNames: string[];
  intl: IntlShape;
};

const CompareRunView: React.FC<CompareRunViewProps> = ({
    runInfos,
    experimentIds,
    metricLists,
    paramLists,
    tagLists,
    runNames,
    runDisplayNames,
    intl,
    runUuids,
    hasComparedExperimentsBefore,
  }) => {

  const [tableWidth, setTableWidth] = useState<number | null>(null);
  const [onlyShowParamDiff, setOnlyShowParamDiff] = useState(false);
  const [onlyShowTagDiff, setOnlyShowTagDiff] = useState(false);
  const [onlyShowMetricDiff, setOnlyShowMetricDiff] = useState(false);

  const runDetailsTableRef = useRef<HTMLDivElement>(null);
  const compareRunViewRef = useRef<HTMLDivElement>(null);

  const paramsLabel = intl.formatMessage({
    defaultMessage: 'Parameters',
    description: 'Row group title for parameters of runs on the experiment compare runs page',
  });

  const metricsLabel = intl.formatMessage({
    defaultMessage: 'Metrics',
    description: 'Row group title for metrics of runs on the experiment compare runs page',
  });

  const artifactsLabel = intl.formatMessage({
    defaultMessage: 'Artifacts',
    description: 'Row group title for artifacts of runs on the experiment compare runs page',
  });

  const tagsLabel = intl.formatMessage({
    defaultMessage: 'Tags',
    description: 'Row group title for tags of runs on the experiment compare runs page',
  });

  const diffOnlyLabel = intl.formatMessage({
    defaultMessage: 'Show diff only',
    description:
      'Label next to the switch that controls displaying only differing values in comparison tables on the compare runs page',
  });

  useEffect(() => {
    const pageTitle = intl.formatMessage(
      {
        description: 'Page title for the compare runs page',
        defaultMessage: 'Comparing {runs} MLflow Runs',
      },
      {
        runs: runInfos.length,
      },
    );
    Utils.updatePageTitle(pageTitle);

    const handleResize = () => {
      const table = runDetailsTableRef.current;
      if (table !== null) {
        setTableWidth(table.clientWidth);
      }
    };

    window.addEventListener('resize', handleResize, true);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize, true);
    };
  }, [runInfos.length, intl]);

  const onCompareRunTableScrollHandler = (e: React.UIEvent) => {
    const blocks = compareRunViewRef.current?.querySelectorAll('.compare-run-table') || [];
    blocks.forEach((block, index) => {
      if (block !== e.target) {
        block.scrollLeft = (e.target as HTMLDivElement).scrollLeft;
      }
    });
  };

  const genWidthStyle = (width: any) => {
    return {
      width: `${width}px`,
      minWidth: `${width}px`,
      maxWidth: `${width}px`,
    };
  };

  const getTableColumnWidth = () => {
    const minColWidth = 200;
    let colWidth = minColWidth;

    if (tableWidth !== null) {
      colWidth = Math.round(tableWidth / (runInfos.length + 1));
      if (colWidth < minColWidth) {
        colWidth = minColWidth;
      }
    }
    return colWidth;
  };

  const getExperimentPageLink = (experimentId: any, experimentName: any) => {
    return <Link to={Routes.getExperimentPageRoute(experimentId)}>{experimentName}</Link>;
  };

  const getCompareExperimentsPageLinkText = (numExperiments: any) => {
    return (
      <FormattedMessage
        defaultMessage="Displaying Runs from {numExperiments} Experiments"
        // eslint-disable-next-line max-len
        description="Breadcrumb nav item to link to compare-experiments page on compare runs page"
        values={{ numExperiments }}
      />
    );
  };

  const getCompareExperimentsPageLink = (experimentIds: any) => {
    return (
      <Link to={Routes.getCompareExperimentsPageRoute(experimentIds)}>
        {getCompareExperimentsPageLinkText(experimentIds.length)}
      </Link>
    );
  };

  const getExperimentLink = () => {
    const { comparedExperimentIds, hasComparedExperimentsBefore, experimentIds, experiments } = props;

    if (hasComparedExperimentsBefore) {
      return getCompareExperimentsPageLink(comparedExperimentIds);
    }

    if (hasMultipleExperiments()) {
      return getCompareExperimentsPageLink(experimentIds);
    }

    return getExperimentPageLink(experimentIds[0], experiments[0].name);
  };

  const getTitle = () => {
    return hasMultipleExperiments() ? (
      <FormattedMessage
        defaultMessage="Comparing {numRuns} Runs from {numExperiments} Experiments"
        // eslint-disable-next-line max-len
        description="Breadcrumb title for compare runs page with multiple experiments"
        values={{
          numRuns: runInfos.length,
          numExperiments: experimentIds.length,
        }}
      />
    ) : (
      <FormattedMessage
        defaultMessage="Comparing {numRuns} Runs from 1 Experiment"
        description="Breadcrumb title for compare runs page with single experiment"
        values={{
          numRuns: runInfos.length,
        }}
      />
    );
  };

  const renderExperimentNameRowItems = () => {
    const { experiments } = props;
    const experimentNameMap: Record<string, { name: string; basename: string }> = Utils.getExperimentNameMap(
      Utils.sortExperimentsById(experiments),
    );
    return runInfos.map(({ experimentId, runUuid }) => {
      const { name, basename } = experimentNameMap[experimentId];
      return (
        <td className="meta-info" key={runUuid}>
          <Link to={Routes.getExperimentPageRoute(experimentId)} title={name}>
            {basename}
          </Link>
        </td>
      );
    });
  };

  const hasMultipleExperiments = () => experimentIds.length > 1;

  const shouldShowExperimentNameRow = () => hasComparedExperimentsBefore || hasMultipleExperiments();

  const renderDataRows = (
    dataList: any[],
    colWidth: number,
    diffOnly: boolean,
    highlightDiff = false,
    headerMap = (key: any, data: any) => key,
    formatter = (x: any) => x,
  ) => {
    // @ts-expect-error TS(2554): Expected 2 arguments, but got 1.
    const keys = CompareRunUtil.getKeys(dataList);
    const data = {};
    const checkHasDiff = (values: any) => values.some((x: any) => x !== values[0]);
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    keys.forEach((k) => (data[k] = { values: Array(dataList.length).fill(undefined) }));
    dataList.forEach((records: any, i: any) => {
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      records.forEach((r: any) => (data[r.key].values[i] = r.value));
    });
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    keys.forEach((k) => (data[k].hasDiff = checkHasDiff(data[k].values)));

    const colWidthStyle = genWidthStyle(colWidth);

    return (
      keys
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        .filter((k) => !diffOnly || data[k].hasDiff)
        .map((k) => {
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          const { values, hasDiff } = data[k];
          const rowClass = highlightDiff && hasDiff ? 'diff-row' : undefined;
          return (
            <tr key={k} className={rowClass}>
              <th scope="row" className="head-value sticky-header" style={colWidthStyle}>
                {headerMap(k, values)}
              </th>
              {values.map((value: any, i: any) => {
                const cellText = value === undefined ? '' : formatter(value);
                return (
                  <td className="data-value" key={runInfos[i].runUuid} style={colWidthStyle}>
                    <LegacyTooltip
                      title={cellText}
                      // @ts-expect-error TS(2322): Type '{ children: Element; title: any; color: stri... Remove this comment to see the full error message
                      color="gray"
                      placement="topLeft"
                      overlayStyle={{ maxWidth: '400px' }}
                      mouseEnterDelay={1.0}
                    >
                      <span className="truncate-text single-line">{cellText}</span>
                    </LegacyTooltip>
                  </td>
                );
              })}
            </tr>
          );
        })
    );
  };

  const renderParamTable = (colWidth: number) => {
    const dataRows = renderDataRows(
      paramLists,
      colWidth,
      onlyShowParamDiff,
      true,
      (key: any, data: any) => key,
      (value: any) => {
        try {
          const jsonValue = parsePythonDictString(value);

          // Pretty print if parsed value is an object or array
          if (typeof jsonValue === 'object' && jsonValue !== null) {
            return renderPrettyJson(jsonValue);
          } else {
            return value;
          }
        } catch (e) {
          return value; // Fallback if JSON parsing fails
        }
      },
    );

    return dataRows.length === 0 ? (
      <h2>
        <FormattedMessage defaultMessage="No parameters to display." description="No parameters to display" />
      </h2>
    ) : (
      <table
        className="table compare-table compare-run-table"
        style={{ maxHeight: '500px' }}
        onScroll={onCompareRunTableScrollHandler}
      >
        <tbody>{dataRows}</tbody>
      </table>
    );
  };

  const renderPrettyJson = (jsonValue: any) => {
    return <pre>{JSON.stringify(jsonValue, null, 2)}</pre>;
  };

  const renderMetricTable = (colWidth: number, experimentIds: string[]) => {
    const dataRows = renderDataRows(
      metricLists,
      colWidth,
      onlyShowMetricDiff,
      false,
      (key, data) => (
        <Link
          to={Routes.getMetricPageRoute(
            runInfos.map((info) => info.runUuid).filter((uuid, idx) => data[idx] !== undefined),
            key,
            experimentIds,
          )}
          title="Plot chart"
        >
          {key}
          <i className="fas fa-chart-line" style={{ paddingLeft: '6px' }} />
        </Link>
      ),
      Utils.formatMetric,
    );

    return dataRows.length === 0 ? (
      <h2>
        <FormattedMessage defaultMessage="No metrics to display." description="No metrics to display" />
      </h2>
    ) : (
      <table
        className="table compare-table compare-run-table"
        style={{ maxHeight: '300px' }}
        onScroll={onCompareRunTableScrollHandler}
      >
        <tbody>{dataRows}</tbody>
      </table>
    );
  };

  const renderArtifactTable = (colWidth: any) => {
    return <CompareRunArtifactView runUuids={runUuids} runInfos={runInfos} colWidth={colWidth} />;
  };

  const renderTagTable = (colWidth: number) => {
    const dataRows = renderDataRows(tagLists, colWidth, onlyShowTagDiff, true);
    return dataRows.length === 0 ? (
      <h2>
        <FormattedMessage defaultMessage="No tags to display." description="No tags to display" />
      </h2>
    ) : (
      <table
        className="table compare-table compare-run-table"
        style={{ maxHeight: '500px' }}
        onScroll={onCompareRunTableScrollHandler}
      >
        <tbody>{dataRows}</tbody>
      </table>
    );
  };

  const renderTimeRows = (colWidthStyle: any) => {
    const unknown = <FormattedMessage defaultMessage="(unknown)" description="Unknown time" />;
    const timeAttributes = runInfos.map(({ runUuid, startTime, endTime }) => ({
      runUuid,
      startTime: startTime ? Utils.formatTimestamp(startTime) : unknown,
      endTime: endTime ? Utils.formatTimestamp(endTime) : unknown,
      duration: startTime && endTime ? Utils.getDuration(startTime, endTime) : unknown,
    }));

    const rows = [
      {
        key: 'startTime',
        title: <FormattedMessage defaultMessage="Start Time:" />,
        data: timeAttributes.map(({ runUuid, startTime }) => [runUuid, startTime]),
      },
      {
        key: 'endTime',
        title: <FormattedMessage defaultMessage="End Time:" />,
        data: timeAttributes.map(({ runUuid, endTime }) => [runUuid, endTime]),
      },
      {
        key: 'duration',
        title: <FormattedMessage defaultMessage="Duration:" />,
        data: timeAttributes.map(({ runUuid, duration }) => [runUuid, duration]),
      },
    ];

    return rows.map(({ key, title, data }) => (
      <tr key={key}>
        <th scope="row" className="head-value sticky-header" style={colWidthStyle}>
          {title}
        </th>
        {data.map(([runUuid, value]) => (
          <td className="data-value" key={runUuid} style={colWidthStyle}>
            <LegacyTooltip title={value} color="gray" placement="topLeft" overlayStyle={{ maxWidth: '400px' }}>
              {value}
            </LegacyTooltip>
          </td>
        ))}
      </tr>
    ));
  };

  const displayChartSection = !shouldDisableLegacyRunCompareCharts();

  const colWidth = getTableColumnWidth();
  const colWidthStyle = genWidthStyle(colWidth);

  const title = getTitle();
  const breadcrumbs = [getExperimentLink()];

  return (
    <div className="CompareRunView" ref={compareRunViewRef}>
      <PageHeader title={title} breadcrumbs={breadcrumbs} />
      {displayChartSection && (
        <CollapsibleSection
          title={intl.formatMessage({
            defaultMessage: 'Visualizations',
            description: 'Tabs title for plots on the compare runs page',
          })}
        >
          <Tabs>
            <TabPane
              tab={
                <FormattedMessage
                  defaultMessage="Parallel Coordinates Plot"
                  description="Tab pane title for parallel coordinate plots on the compare runs page"
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
                  description="Tab pane title for scatterplots on the compare runs page"
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
              <CompareRunBox
                runUuids={runUuids}
                runInfos={runInfos}
                paramLists={paramLists}
                metricLists={metricLists}
              />
            </TabPane>
            <TabPane
              tab={
                <FormattedMessage
                  defaultMessage="Contour Plot"
                  description="Tab pane title for contour plots on the compare runs page"
                />
              }
              key="contour-plot"
            >
              <CompareRunContour runUuids={runUuids} runDisplayNames={runDisplayNames} />
            </TabPane>
          </Tabs>
        </CollapsibleSection>
      )}
      <CollapsibleSection
        title={intl.formatMessage({
          defaultMessage: 'Run details',
          description: 'Compare table title on the compare runs page',
        })}
      >
        <table
          className="table compare-table compare-run-table"
          ref={runDetailsTableRef}
          onScroll={onCompareRunTableScrollHandler}
        >
          <thead>
            <tr>
              <th scope="row" className="head-value sticky-header" css={colWidthStyle}>
                <FormattedMessage
                  defaultMessage="Run ID:"
                  description="Row title for the run id on the experiment compare runs page"
                />
              </th>
              {runInfos.map((r) => (
                <th scope="row" className="data-value" key={r.runUuid} css={colWidthStyle}>
                  <LegacyTooltip
                    title={r.runUuid}
                    color="gray"
                    placement="topLeft"
                    overlayStyle={{ maxWidth: '400px' }}
                    mouseEnterDelay={1.0}
                  >
                    <Link to={Routes.getRunPageRoute(r.experimentId ?? '0', r.runUuid ?? '')}>{r.runUuid}</Link>
                  </LegacyTooltip>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row" className="head-value sticky-header" css={colWidthStyle}>
                <FormattedMessage
                  defaultMessage="Run Name:"
                  description="Row title for the run name on the experiment compare runs page"
                />
              </th>
              {runNames.map((runName, i) => (
                <td className="data-value" key={runInfos[i].runUuid} css={colWidthStyle}>
                  <div className="truncate-text single-line">
                    <LegacyTooltip
                      title={runName}
                      color="gray"
                      placement="topLeft"
                      overlayStyle={{ maxWidth: '400px' }}
                      mouseEnterDelay={1.0}
                    >
                      {runName}
                    </LegacyTooltip>
                  </div>
                </td>
              ))}
            </tr>
            {renderTimeRows(colWidthStyle)}
            {shouldShowExperimentNameRow() && (
              <tr>
                <th scope="row" className="data-value">
                  <FormattedMessage
                    defaultMessage="Experiment Name:"
                    description="Row title for the experiment IDs of runs on the experiment compare runs page"
                  />
                </th>
                {renderExperimentNameRowItems()}
              </tr>
            )}
          </tbody>
        </table>
      </CollapsibleSection>
      <CollapsibleSection title={paramsLabel}>
        <Switch
          label={diffOnlyLabel}
          aria-label={[paramsLabel, diffOnlyLabel].join(' - ')}
          checked={onlyShowParamDiff}
          onChange={(checked) => setOnlyShowParamDiff(checked)}
        />
        <Spacer size="lg" />
        {renderParamTable(colWidth)}
      </CollapsibleSection>
      <CollapsibleSection title={metricsLabel}>
        <Switch
          label={diffOnlyLabel}
          aria-label={[metricsLabel, diffOnlyLabel].join(' - ')}
          checked={onlyShowMetricDiff}
          onChange={(checked) => setOnlyShowMetricDiff(checked)}
        />
        <Spacer size="lg" />
        {renderMetricTable(colWidth, experimentIds)}
      </CollapsibleSection>
      <CollapsibleSection title={artifactsLabel}>{renderArtifactTable(colWidth)}</CollapsibleSection>
      <CollapsibleSection title={tagsLabel}>
        <Switch
          label={diffOnlyLabel}
          aria-label={[tagsLabel, diffOnlyLabel].join(' - ')}
          checked={onlyShowTagDiff}
          onChange={(checked) => setOnlyShowTagDiff(checked)}
        />
        <Spacer size="lg" />
        {renderTagTable(colWidth)}
      </CollapsibleSection>
    </div>
  );
};

const mapStateToProps = (state: any, ownProps: any) => {
  const { comparedExperimentIds, hasComparedExperimentsBefore } = state.compareExperiments;
  const runInfos: any = [];
  const metricLists: any = [];
  const paramLists: any = [];
  const tagLists: any = [];
  const runNames: any = [];
  const runDisplayNames: any = [];
  const { experimentIds, runUuids } = ownProps;
  const experiments = experimentIds.map((experimentId: any) => getExperiment(experimentId, state));
  runUuids.forEach((runUuid: any) => {
    const runInfo = getRunInfo(runUuid, state);
    // Skip processing data if run info is not available yet
    if (!runInfo) {
      return;
    }
    runInfos.push(runInfo);
    metricLists.push(Object.values(getLatestMetrics(runUuid, state)));
    paramLists.push(Object.values(getParams(runUuid, state)));
    const runTags = getRunTags(runUuid, state);
    const visibleTags = Utils.getVisibleTagValues(runTags).map(([key, value]) => ({
      key,
      value,
    }));
    tagLists.push(visibleTags);
    runDisplayNames.push(Utils.getRunDisplayName(runInfo, runUuid));
    runNames.push(Utils.getRunName(runInfo));
  });
  return {
    experiments,
    runInfos,
    metricLists,
    paramLists,
    tagLists,
    runNames,
    runDisplayNames,
    comparedExperimentIds,
    hasComparedExperimentsBefore,
  };
};

/**
 * Parse a Python dictionary in string format into a JSON object.
 * @param value The Python dictionary string to parse
 * @returns The parsed JSON object, or null if parsing fails
 */
const parsePythonDictString = (value: string) => {
  try {
    const jsonString = value.replace(/'/g, '"');
    return JSON.parse(jsonString);
  } catch (e) {
    console.error('Failed to parse string to JSON:', e);
    return null;
  }
};

export default connect(mapStateToProps)(injectIntl(CompareRunView));

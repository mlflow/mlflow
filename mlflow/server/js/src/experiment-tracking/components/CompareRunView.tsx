/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { injectIntl, FormattedMessage } from 'react-intl';
import { Alert, Button, Spacer, Switch, Tabs, Tooltip } from '@databricks/design-system';

import { getExperiment, getParams, getRunInfo, getRunTags } from '../reducers/Reducers';
import './CompareRunView.css';
import { Experiment, RunInfo } from '../sdk/MlflowMessages';
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
import { useExperimentPageFeedbackUrl } from './experiment-page/hooks/useExperimentPageFeedbackUrl';
import { withRouterNext } from '../../common/utils/withRouterNext';

const { TabPane } = Tabs;

const LegacyCompareRunsPageCallout = (props: any) => {
  (LegacyCompareRunsPageCallout as any).propTypes = {
    experimentIds: PropTypes.arrayOf(PropTypes.string).isRequired,
  };
  const feedbackFormUrl = useExperimentPageFeedbackUrl();
  return (
    <Alert
      type='info'
      closable={false}
      message={
        <>
          <FormattedMessage
            defaultMessage='We’ve made several improvements to the new runs comparison experience.'
            // eslint-disable-next-line max-len
            description='Callout message to tell user about new compare runs feature'
          />{' '}
          <Link to={Routes.getExperimentPageRoute(props.experimentIds[0], true)}>
            <FormattedMessage
              defaultMessage='Give it a try.'
              description='Link to new runs compare page'
            />
          </Link>{' '}
          {feedbackFormUrl && (
            <>
              <FormattedMessage
                defaultMessage="If you prefer to use this experience, we'd love to know more."
                // eslint-disable-next-line max-len
                description='Callout message to ask user for feedback about using legacy compare runs feature'
              />{' '}
              <a href={feedbackFormUrl} target='_blank' rel='noreferrer'>
                <Button css={{ marginLeft: 16 }} type='link' size='small'>
                  <FormattedMessage
                    defaultMessage='Provide Feedback'
                    description='Link to a survey for users to give feedback'
                  />
                </Button>
              </a>
            </>
          )}
        </>
      }
    />
  );
};

type CompareRunViewProps = {
  experiments: any[]; // TODO: PropTypes.instanceOf(Experiment)
  experimentIds: string[];
  comparedExperimentIds?: string[];
  hasComparedExperimentsBefore?: boolean;
  runInfos: any[]; // TODO: PropTypes.instanceOf(RunInfo)
  runUuids: string[];
  metricLists: any[][];
  paramLists: any[][];
  tagLists: any[][];
  runNames: string[];
  runDisplayNames: string[];
  intl: {
    formatMessage: (...args: any[]) => any;
  };
};

type CompareRunViewState = any;
export class CompareRunView extends Component<CompareRunViewProps, CompareRunViewState> {
  compareRunViewRef: any;
  runDetailsTableRef: any;

  constructor(props: CompareRunViewProps) {
    super(props);
    this.state = {
      tableWidth: null,
      onlyShowParamDiff: false,
      onlyShowTagDiff: false,
      onlyShowMetricDiff: false,
    };
    this.onResizeHandler = this.onResizeHandler.bind(this);
    this.onCompareRunTableScrollHandler = this.onCompareRunTableScrollHandler.bind(this);

    this.runDetailsTableRef = React.createRef();
    this.compareRunViewRef = React.createRef();
  }

  onResizeHandler(e: any) {
    const table = this.runDetailsTableRef.current;
    if (table !== null) {
      const containerWidth = table.clientWidth;
      this.setState({ tableWidth: containerWidth });
    }
  }

  onCompareRunTableScrollHandler(e: any) {
    const blocks = this.compareRunViewRef.current.querySelectorAll('.compare-run-table');
    blocks.forEach((_: any, index: any) => {
      const block = blocks[index];
      if (block !== e.target) {
        block.scrollLeft = e.target.scrollLeft;
      }
    });
  }

  componentDidMount() {
    const pageTitle = this.props.intl.formatMessage(
      {
        description: 'Page title for the compare runs page',
        defaultMessage: 'Comparing {runs} MLflow Runs',
      },
      {
        runs: this.props.runInfos.length,
      },
    );
    Utils.updatePageTitle(pageTitle);

    window.addEventListener('resize', this.onResizeHandler, true);
    window.dispatchEvent(new Event('resize'));
  }

  componentWillUnmount() {
    // Avoid registering `onResizeHandler` every time this component mounts
    window.removeEventListener('resize', this.onResizeHandler, true);
  }

  getTableColumnWidth() {
    const minColWidth = 200;
    let colWidth = minColWidth;

    // @ts-expect-error TS(4111): Property 'tableWidth' comes from an index signatur... Remove this comment to see the full error message
    if (this.state.tableWidth !== null) {
      // @ts-expect-error TS(4111): Property 'tableWidth' comes from an index signatur... Remove this comment to see the full error message
      colWidth = Math.round(this.state.tableWidth / (this.props.runInfos.length + 1));
      if (colWidth < minColWidth) {
        colWidth = minColWidth;
      }
    }
    return colWidth;
  }

  renderExperimentNameRowItems() {
    const { experiments } = this.props;
    const experimentNameMap = Utils.getExperimentNameMap(Utils.sortExperimentsById(experiments));
    return this.props.runInfos.map(({ experiment_id, run_uuid }) => {
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      const { name, basename } = experimentNameMap[experiment_id];
      return (
        <td className='meta-info' key={run_uuid}>
          <Link to={Routes.getExperimentPageRoute(experiment_id)} title={name}>
            {basename}
          </Link>
        </td>
      );
    });
  }

  hasMultipleExperiments() {
    return this.props.experimentIds.length > 1;
  }

  shouldShowExperimentNameRow() {
    return this.props.hasComparedExperimentsBefore || this.hasMultipleExperiments();
  }

  getExperimentPageLink(experimentId: any, experimentName: any) {
    return <Link to={Routes.getExperimentPageRoute(experimentId)}>{experimentName}</Link>;
  }

  getCompareExperimentsPageLinkText(numExperiments: any) {
    return (
      <FormattedMessage
        defaultMessage='Displaying Runs from {numExperiments} Experiments'
        // eslint-disable-next-line max-len
        description='Breadcrumb nav item to link to compare-experiments page on compare runs page'
        values={{ numExperiments }}
      />
    );
  }

  getCompareExperimentsPageLink(experimentIds: any) {
    return (
      <Link to={Routes.getCompareExperimentsPageRoute(experimentIds)}>
        {this.getCompareExperimentsPageLinkText(experimentIds.length)}
      </Link>
    );
  }

  getExperimentLink() {
    const { comparedExperimentIds, hasComparedExperimentsBefore, experimentIds, experiments } =
      this.props;

    if (hasComparedExperimentsBefore) {
      return this.getCompareExperimentsPageLink(comparedExperimentIds);
    }

    if (this.hasMultipleExperiments()) {
      return this.getCompareExperimentsPageLink(experimentIds);
    }

    return this.getExperimentPageLink(experimentIds[0], experiments[0].name);
  }

  getTitle() {
    return this.hasMultipleExperiments() ? (
      <FormattedMessage
        defaultMessage='Comparing {numRuns} Runs from {numExperiments} Experiments'
        // eslint-disable-next-line max-len
        description='Breadcrumb title for compare runs page with multiple experiments'
        values={{
          numRuns: this.props.runInfos.length,
          numExperiments: this.props.experimentIds.length,
        }}
      />
    ) : (
      <FormattedMessage
        defaultMessage='Comparing {numRuns} Runs from 1 Experiment'
        description='Breadcrumb title for compare runs page with single experiment'
        values={{
          numRuns: this.props.runInfos.length,
        }}
      />
    );
  }

  renderParamTable(colWidth: any) {
    const dataRows = this.renderDataRows(
      this.props.paramLists,
      colWidth,
      // @ts-expect-error TS(4111): Property 'onlyShowParamDiff' comes from an index s... Remove this comment to see the full error message
      this.state.onlyShowParamDiff,
      true,
    );
    if (dataRows.length === 0) {
      return (
        <h2>
          <FormattedMessage
            defaultMessage='No parameters to display.'
            description='Text shown when there are no parameters to display'
          />
        </h2>
      );
    }
    return (
      <table
        className='table compare-table compare-run-table'
        css={{ maxHeight: '500px' }}
        onScroll={this.onCompareRunTableScrollHandler}
      >
        <tbody>{dataRows}</tbody>
      </table>
    );
  }

  renderMetricTable(colWidth: any, experimentIds: any) {
    const dataRows = this.renderDataRows(
      this.props.metricLists,
      colWidth,
      // @ts-expect-error TS(4111): Property 'onlyShowMetricDiff' comes from an index ... Remove this comment to see the full error message
      this.state.onlyShowMetricDiff,
      false,
      (key, data) => {
        return (
          <Link
            to={Routes.getMetricPageRoute(
              this.props.runInfos
                .map((info) => info.run_uuid)
                .filter((uuid, idx) => data[idx] !== undefined),
              key,
              experimentIds,
            )}
            title='Plot chart'
          >
            {key}
            <i className='fas fa-chart-line' css={{ paddingLeft: '6px' }} />
          </Link>
        );
      },
      Utils.formatMetric,
    );
    if (dataRows.length === 0) {
      return (
        <h2>
          <FormattedMessage
            defaultMessage='No metrics to display.'
            description='Text shown when there are no metrics to display'
          />
        </h2>
      );
    }
    return (
      <table
        className='table compare-table compare-run-table'
        css={{ maxHeight: '300px' }}
        onScroll={this.onCompareRunTableScrollHandler}
      >
        <tbody>{dataRows}</tbody>
      </table>
    );
  }

  renderTagTable(colWidth: any) {
    const dataRows = this.renderDataRows(
      this.props.tagLists,
      colWidth,
      // @ts-expect-error TS(4111): Property 'onlyShowTagDiff' comes from an index sig... Remove this comment to see the full error message
      this.state.onlyShowTagDiff,
      true,
    );
    if (dataRows.length === 0) {
      return (
        <h2>
          <FormattedMessage
            defaultMessage='No tags to display.'
            description='Text shown when there are no tags to display'
          />
        </h2>
      );
    }
    return (
      <table
        className='table compare-table compare-run-table'
        css={{ maxHeight: '500px' }}
        onScroll={this.onCompareRunTableScrollHandler}
      >
        <tbody>{dataRows}</tbody>
      </table>
    );
  }

  renderTimeRows(colWidthStyle: any) {
    const unknown = (
      <FormattedMessage
        defaultMessage='(unknown)'
        description="Filler text when run's time information is unavailable"
      />
    );
    const getTimeAttributes = (runInfo: any) => {
      const startTime = runInfo.getStartTime();
      const endTime = runInfo.getEndTime();
      return {
        runUuid: runInfo.run_uuid,
        startTime: startTime ? Utils.formatTimestamp(startTime) : unknown,
        endTime: endTime ? Utils.formatTimestamp(endTime) : unknown,
        duration: startTime && endTime ? Utils.getDuration(startTime, endTime) : unknown,
      };
    };
    const timeAttributes = this.props.runInfos.map(getTimeAttributes);
    const rows = [
      {
        key: 'startTime',
        title: (
          <FormattedMessage
            defaultMessage='Start Time:'
            description='Row title for the start time of runs on the experiment compare runs page'
          />
        ),
        data: timeAttributes.map(({ runUuid, startTime }) => [runUuid, startTime]),
      },
      {
        key: 'endTime',
        title: (
          <FormattedMessage
            defaultMessage='End Time:'
            description='Row title for the end time of runs on the experiment compare runs page'
          />
        ),
        data: timeAttributes.map(({ runUuid, endTime }) => [runUuid, endTime]),
      },
      {
        key: 'duration',
        title: (
          <FormattedMessage
            defaultMessage='Duration:'
            description='Row title for the duration of runs on the experiment compare runs page'
          />
        ),
        data: timeAttributes.map(({ runUuid, duration }) => [runUuid, duration]),
      },
    ];
    return rows.map(({ key, title, data }) => (
      <tr key={key}>
        <th scope='row' className='head-value sticky-header' css={colWidthStyle}>
          {title}
        </th>
        {data.map(([runUuid, value]) => (
          <td className='data-value' key={runUuid} css={colWidthStyle}>
            <Tooltip
              title={value}
              // @ts-expect-error TS(2322): Type '{ children: any; title: any; color: string; ... Remove this comment to see the full error message
              color='gray'
              placement='topLeft'
              overlayStyle={{ maxWidth: '400px' }}
              // mouseEnterDelay prop is not available in DuBois design system (yet)
              dangerouslySetAntdProps={{ mouseEnterDelay: 1 }}
            >
              {value}
            </Tooltip>
          </td>
        ))}
      </tr>
    ));
  }

  render() {
    const { experimentIds } = this.props;
    const { runInfos, runNames, paramLists, metricLists, runUuids } = this.props;

    const colWidth = this.getTableColumnWidth();
    const colWidthStyle = this.genWidthStyle(colWidth);

    const title = this.getTitle();
    /* eslint-disable-next-line prefer-const */
    let breadcrumbs = [this.getExperimentLink()];

    const paramsLabel = this.props.intl.formatMessage({
      defaultMessage: 'Parameters',
      description: 'Row group title for parameters of runs on the experiment compare runs page',
    });

    const metricsLabel = this.props.intl.formatMessage({
      defaultMessage: 'Metrics',
      description: 'Row group title for metrics of runs on the experiment compare runs page',
    });

    const tagsLabel = this.props.intl.formatMessage({
      defaultMessage: 'Tags',
      description: 'Row group title for tags of runs on the experiment compare runs page',
    });
    const diffOnlyLabel = this.props.intl.formatMessage({
      defaultMessage: 'Show diff only',
      description:
        // eslint-disable-next-line max-len
        'Label next to the switch that controls displaying only differing values in comparision tables on the compare runs page',
    });

    const displayChartSection = !shouldDisableLegacyRunCompareCharts();

    return (
      <div className='CompareRunView' ref={this.compareRunViewRef}>
        <PageHeader title={title} breadcrumbs={breadcrumbs} />
        <LegacyCompareRunsPageCallout experimentIds={experimentIds} />
        {displayChartSection && (
          <CollapsibleSection
            title={this.props.intl.formatMessage({
              defaultMessage: 'Visualizations',
              description: 'Tabs title for plots on the compare runs page',
            })}
          >
            <Tabs>
              <TabPane
                tab={
                  <FormattedMessage
                    defaultMessage='Parallel Coordinates Plot'
                    // eslint-disable-next-line max-len
                    description='Tab pane title for parallel coordinate plots on the compare runs page'
                  />
                }
                key='parallel-coordinates-plot'
              >
                <ParallelCoordinatesPlotPanel runUuids={this.props.runUuids} />
              </TabPane>
              <TabPane
                tab={
                  <FormattedMessage
                    defaultMessage='Scatter Plot'
                    description='Tab pane title for scatterplots on the compare runs page'
                  />
                }
                key='scatter-plot'
              >
                <CompareRunScatter
                  runUuids={this.props.runUuids}
                  runDisplayNames={this.props.runDisplayNames}
                />
              </TabPane>
              <TabPane
                tab={
                  <FormattedMessage
                    defaultMessage='Box Plot'
                    description='Tab pane title for box plot on the compare runs page'
                  />
                }
                key='box-plot'
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
                    defaultMessage='Contour Plot'
                    description='Tab pane title for contour plots on the compare runs page'
                  />
                }
                key='contour-plot'
              >
                <CompareRunContour
                  runUuids={this.props.runUuids}
                  runDisplayNames={this.props.runDisplayNames}
                />
              </TabPane>
            </Tabs>
          </CollapsibleSection>
        )}
        <CollapsibleSection
          title={this.props.intl.formatMessage({
            defaultMessage: 'Run details',
            description: 'Compare table title on the compare runs page',
          })}
        >
          <table
            className='table compare-table compare-run-table'
            ref={this.runDetailsTableRef}
            onScroll={this.onCompareRunTableScrollHandler}
          >
            <thead>
              <tr>
                <th scope='row' className='head-value sticky-header' css={colWidthStyle}>
                  <FormattedMessage
                    defaultMessage='Run ID:'
                    description='Row title for the run id on the experiment compare runs page'
                  />
                </th>
                {this.props.runInfos.map((r) => (
                  <th scope='row' className='data-value' key={r.run_uuid} css={colWidthStyle}>
                    <Tooltip
                      title={r.getRunUuid()}
                      // @ts-expect-error TS(2322): Type '{ children: Element; title: any; color: stri... Remove this comment to see the full error message
                      color='gray'
                      placement='topLeft'
                      overlayStyle={{ maxWidth: '400px' }}
                      mouseEnterDelay={1.0}
                    >
                      <Link to={Routes.getRunPageRoute(r.getExperimentId(), r.getRunUuid())}>
                        {r.getRunUuid()}
                      </Link>
                    </Tooltip>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope='row' className='head-value sticky-header' css={colWidthStyle}>
                  <FormattedMessage
                    defaultMessage='Run Name:'
                    description='Row title for the run name on the experiment compare runs page'
                  />
                </th>
                {runNames.map((runName, i) => {
                  return (
                    <td className='data-value' key={runInfos[i].run_uuid} css={colWidthStyle}>
                      <div className='truncate-text single-line'>
                        <Tooltip
                          title={runName}
                          // @ts-expect-error TS(2322): Type '{ children: string; title: string; color: st... Remove this comment to see the full error message
                          color='gray'
                          placement='topLeft'
                          overlayStyle={{ maxWidth: '400px' }}
                          mouseEnterDelay={1.0}
                        >
                          {runName}
                        </Tooltip>
                      </div>
                    </td>
                  );
                })}
              </tr>
              {this.renderTimeRows(colWidthStyle)}
              {this.shouldShowExperimentNameRow() && (
                <tr>
                  <th scope='row' className='data-value'>
                    <FormattedMessage
                      defaultMessage='Experiment Name:'
                      // eslint-disable-next-line max-len
                      description='Row title for the experiment IDs of runs on the experiment compare runs page'
                    />
                  </th>
                  {this.renderExperimentNameRowItems()}
                </tr>
              )}
            </tbody>
          </table>
        </CollapsibleSection>
        <CollapsibleSection title={paramsLabel}>
          <Switch
            label={diffOnlyLabel}
            aria-label={[paramsLabel, diffOnlyLabel].join(' - ')}
            // @ts-expect-error TS(4111): Property 'onlyShowParamDiff' comes from an index s... Remove this comment to see the full error message
            checked={this.state.onlyShowParamDiff}
            onChange={(checked, e) => this.setState({ onlyShowParamDiff: checked })}
          />
          <Spacer size='lg' />
          {this.renderParamTable(colWidth)}
        </CollapsibleSection>
        <CollapsibleSection title={metricsLabel}>
          <Switch
            label={diffOnlyLabel}
            aria-label={[metricsLabel, diffOnlyLabel].join(' - ')}
            // @ts-expect-error TS(4111): Property 'onlyShowMetricDiff' comes from an index ... Remove this comment to see the full error message
            checked={this.state.onlyShowMetricDiff}
            onChange={(checked, e) => this.setState({ onlyShowMetricDiff: checked })}
          />
          <Spacer size='lg' />
          {this.renderMetricTable(colWidth, experimentIds)}
        </CollapsibleSection>
        <CollapsibleSection title={tagsLabel}>
          <Switch
            label={diffOnlyLabel}
            aria-label={[tagsLabel, diffOnlyLabel].join(' - ')}
            // @ts-expect-error TS(4111): Property 'onlyShowTagDiff' comes from an index sig... Remove this comment to see the full error message
            checked={this.state.onlyShowTagDiff}
            onChange={(checked, e) => this.setState({ onlyShowTagDiff: checked })}
          />
          <Spacer size='lg' />
          {this.renderTagTable(colWidth)}
        </CollapsibleSection>
      </div>
    );
  }

  genWidthStyle(width: any) {
    return {
      width: `${width}px`,
      minWidth: `${width}px`,
      maxWidth: `${width}px`,
    };
  }

  // eslint-disable-next-line no-unused-vars
  renderDataRows(
    list: any,
    colWidth: any,
    onlyShowDiff: any,
    highlightDiff = false,
    headerMap = (key: any, data: any) => key,
    formatter = (value: any) => value,
  ) {
    // @ts-expect-error TS(2554): Expected 2 arguments, but got 1.
    const keys = CompareRunUtil.getKeys(list);
    const data = {};
    const checkHasDiff = (values: any) => values.some((x: any) => x !== values[0]);
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    keys.forEach((k) => (data[k] = { values: Array(list.length).fill(undefined) }));
    list.forEach((records: any, i: any) => {
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      records.forEach((r: any) => (data[r.key].values[i] = r.value));
    });
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    keys.forEach((k) => (data[k].hasDiff = checkHasDiff(data[k].values)));

    const colWidthStyle = this.genWidthStyle(colWidth);

    return (
      keys
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        .filter((k) => !onlyShowDiff || data[k].hasDiff)
        .map((k) => {
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          const { values, hasDiff } = data[k];
          const rowClass = highlightDiff && hasDiff ? 'diff-row' : undefined;
          return (
            <tr key={k} className={rowClass}>
              <th scope='row' className='head-value sticky-header' css={colWidthStyle}>
                {headerMap(k, values)}
              </th>
              {values.map((value: any, i: any) => {
                const cellText = value === undefined ? '' : formatter(value);
                return (
                  <td
                    className='data-value'
                    key={this.props.runInfos[i].run_uuid}
                    css={colWidthStyle}
                  >
                    <Tooltip
                      title={cellText}
                      // @ts-expect-error TS(2322): Type '{ children: Element; title: any; color: stri... Remove this comment to see the full error message
                      color='gray'
                      placement='topLeft'
                      overlayStyle={{ maxWidth: '400px' }}
                      mouseEnterDelay={1.0}
                    >
                      <span className='truncate-text single-line'>{cellText}</span>
                    </Tooltip>
                  </td>
                );
              })}
            </tr>
          );
        })
    );
  }
}

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

// @ts-expect-error TS(2769): No overload matches this call.
export default connect(mapStateToProps)(injectIntl(CompareRunView));

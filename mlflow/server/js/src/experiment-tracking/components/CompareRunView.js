import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getExperiment, getParams, getRunInfo, getRunTags } from '../reducers/Reducers';
import { connect } from 'react-redux';
import { injectIntl, FormattedMessage } from 'react-intl';
import './CompareRunView.css';
import { Experiment, RunInfo } from '../sdk/MlflowMessages';
import { CompareRunScatter } from './CompareRunScatter';
import CompareRunContour from './CompareRunContour';
import Routes from '../routes';
import { Link } from 'react-router-dom';
import { getLatestMetrics } from '../reducers/MetricReducer';
import CompareRunUtil from './CompareRunUtil';
import Utils from '../../common/utils/Utils';
import { Tabs, Tooltip, Switch } from 'antd';
import ParallelCoordinatesPlotPanel from './ParallelCoordinatesPlotPanel';
import { PageHeader } from '../../shared/building_blocks/PageHeader';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';

const { TabPane } = Tabs;

export class CompareRunView extends Component {
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    experimentId: PropTypes.string.isRequired,
    runInfos: PropTypes.arrayOf(PropTypes.instanceOf(RunInfo)).isRequired,
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricLists: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    paramLists: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    // Array of user-specified run names. Elements may be falsy (e.g. empty string or undefined) if
    // a run was never given a name.
    runNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    // Array of names to use when displaying runs. No element in this array should be falsy;
    // we expect this array to contain user-specified run names, or default display names
    // ("Run <uuid>") for runs without names.
    runDisplayNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  constructor(props) {
    super(props);
    this.state = {
      tableContainerWidth: null,
      collapseParamBlock: false,
      collapseMetricBlock: false,
      onlyShowParamDiff: false,
      onlyShowMetricDiff: false,
    };
    // onResizeHandler need access `this.state`, so bind `this`.
    this.onResizeHandler = this.onResizeHandler.bind(this);
    this.onTableBlockScrollHandler = this.onTableBlockScrollHandler.bind(this);

    this.tableContainerRef = React.createRef()
  }

  onResizeHandler(e) {
    const container = this.tableContainerRef.current;
    if (container != null) {
      const containerWidth = container.clientWidth;
      this.setState({tableContainerWidth: containerWidth});
    }
  }

  onTableBlockScrollHandler(e) {
    const blocks = this.tableContainerRef.current.querySelectorAll('.compare-table .table-block');
    for (let index = 0; index < blocks.length; ++index) {
      const block = blocks[index];
      if (block !== e.target) {
        block.scrollLeft = e.target.scrollLeft;
      }
    }
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

    window.addEventListener(
      'resize', this.onResizeHandler, true,
    );
    window.dispatchEvent(new Event('resize'));
  }

  componentWillUnmount() {
    // Because onResizeHandler is registered in `componentDidMount`,
    // We need unregister it here otherwise every time mount will add a new listener.
    window.removeEventListener(
      'resize', this.onResizeHandler, true,
    );
  }

  getTableColumnWidth() {
    const minColWidth = 200;
    var colWidth = minColWidth;

    if (this.state.tableContainerWidth != null) {
      colWidth = Math.round(this.state.tableContainerWidth / (this.props.runInfos.length + 1));
      if (colWidth < minColWidth) {
        colWidth = minColWidth;
      }
    }
    return colWidth;
  }

  render() {
    const { experiment } = this.props;
    const experimentId = experiment.getExperimentId();
    const { runInfos, runNames } = this.props;

    const colWidth = this.getTableColumnWidth();
    const tableBlockWidth = this.state.tableContainerWidth;
    const colWidthStyle = {width: `${colWidth}px`, minWidth: `${colWidth}px`, maxWidth: `${colWidth}px`};
    const tableBlockWidthStyle = {width: `${tableBlockWidth}px`, minWidth: `${tableBlockWidth}px`, maxWidth: `${tableBlockWidth}px`};

    const title = (
      <FormattedMessage
        defaultMessage='Comparing {runs} Runs'
        description='Breadcrumb title for compare runs page'
        values={{
          runs: this.props.runInfos.length,
        }}
      />
    );
    /* eslint-disable-next-line prefer-const */
    let breadcrumbs = [
      <Link to={Routes.getExperimentPageRoute(experimentId)}>{experiment.getName()}</Link>,
      title,
    ];

    return (
      <div className='CompareRunView'>
        <PageHeader title={title} breadcrumbs={breadcrumbs} />
        <CollapsibleSection
          title={
            <h1>
              <FormattedMessage
                defaultMessage='Visualizations'
                description='Tabs title for plots on the compare runs page'
              />
            </h1>
          }
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
              key='1'
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
              key='2'
            >
              <CompareRunScatter
                runUuids={this.props.runUuids}
                runDisplayNames={this.props.runDisplayNames}
              />
            </TabPane>
            <TabPane
              tab={
                <FormattedMessage
                  defaultMessage='Contour Plot'
                  description='Tab pane title for contour plots on the compare runs page'
                />
              }
              key='3'
            >
              <CompareRunContour
                runUuids={this.props.runUuids}
                runDisplayNames={this.props.runDisplayNames}
              />
            </TabPane>
          </Tabs>
        </CollapsibleSection>
        <CollapsibleSection
          title={
            <h1>
              <FormattedMessage
                defaultMessage='Run details'
                // eslint-disable-next-line max-len
                description='Compare table title on the compare runs page'
              />
            </h1>
          }
        >
          <div className='responsive-table-container' ref={this.tableContainerRef}>
            <table className='compare-table table'>
              <thead className='table-block no-scrollbar' style={tableBlockWidthStyle}
               onScroll={this.onTableBlockScrollHandler}
              >
                <tr>
                  <th scope='row' className='head-value sticky_header' style={colWidthStyle}>
                    <FormattedMessage
                      defaultMessage='Run ID:'
                      description='Row title for the run id on the experiment compare runs page'
                    />
                  </th>
                  {this.props.runInfos.map((r) => (
                    <th scope='row' className='data-value' key={r.run_uuid} style={colWidthStyle}>
                      <Tooltip
                        title={r.getRunUuid()}
                        color='blue'
                        overlayStyle={{ maxWidth: '400px' }}
                      >
                        <Link to={Routes.getRunPageRoute(r.getExperimentId(), r.getRunUuid())}>
                          {r.getRunUuid()}
                        </Link>
                      </Tooltip>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className='table-block no-scrollbar' style={tableBlockWidthStyle}
               onScroll={this.onTableBlockScrollHandler}
              >
                <tr>
                  <th scope='row' className='head-value sticky_header' style={colWidthStyle}>
                    <FormattedMessage
                      defaultMessage='Run Name:'
                      description='Row title for the run name on the experiment compare runs page'
                    />
                  </th>
                  {runNames.map((runName, i) => {
                    return (
                      <td className='data-value' key={runInfos[i].run_uuid} style={colWidthStyle}>
                        <div className='truncate-text single-line'>
                          <Tooltip
                            title={runName}
                            color='blue'
                            overlayStyle={{ 'max-width': '400px' }}
                          >
                            {runName}
                          </Tooltip>
                        </div>
                      </td>
                    );
                  })}
                </tr>
                <tr>
                  <th scope='row' className='head-value sticky_header' style={colWidthStyle}>
                    <FormattedMessage
                      defaultMessage='Start Time:'
                      // eslint-disable-next-line max-len
                      description='Row title for the start time of runs on the experiment compare runs page'
                    />
                  </th>
                  {this.props.runInfos.map((run) => {
                    const startTime = run.getStartTime()
                      ? Utils.formatTimestamp(run.getStartTime())
                      : '(unknown)';
                    return (
                      <td className='data-value' key={run.run_uuid} style={colWidthStyle}>
                        <Tooltip
                          title={startTime}
                          color='blue'
                          overlayStyle={{ 'max-width': '400px' }}
                        >
                          {startTime}
                        </Tooltip>
                      </td>
                    );
                  })}
                </tr>
              </tbody>
              <tbody className='table-block' style={tableBlockWidthStyle}>
                <tr>
                  <th
                    scope='rowgroup'
                    className='inter-title sticky_header'
                    colSpan={this.props.runInfos.length + 1}
                    style={tableBlockWidthStyle}
                  >
                    <CollapsibleSection
                      title={
                        <FormattedMessage
                          defaultMessage='Parameters'
                          // eslint-disable-next-line max-len
                          description='Row group title for parameters of runs on the experiment compare runs page'
                        />
                      }
                      onChange={(activeKeys) =>
                        this.setState({collapseParamBlock: activeKeys.length === 0})
                      }
                    >
                      <Switch
                        checkedChildren='Show diff only'
                        unCheckedChildren='Show diff only'
                        onChange={(isShowDiffOnly, e) =>
                          this.setState({onlyShowParamDiff: isShowDiffOnly})
                        }
                      />
                    </CollapsibleSection>
                  </th>
                </tr>
              </tbody>
              <tbody className='table-block param-block' style={Object.assign({}, tableBlockWidthStyle, {maxHeight: '500px'})}
               onScroll={this.onTableBlockScrollHandler}
              >
                { this.state.collapseParamBlock ? undefined : this.renderDataRows(this.props.paramLists, colWidth, "onlyShowParamDiff", true)}
              </tbody>
              <tbody className='table-block' style={tableBlockWidthStyle}>
                <tr>
                  <th
                    scope='rowgroup'
                    className='inter-title sticky_header'
                    colSpan={this.props.runInfos.length + 1}
                    style={tableBlockWidthStyle}
                  >
                    <CollapsibleSection
                      title={
                        <FormattedMessage
                          defaultMessage='Metrics'
                          // eslint-disable-next-line max-len
                          description='Row group title for metrics of runs on the experiment compare runs page'
                        />
                      }
                      onChange={(activeKeys) =>
                        this.setState({collapseMetricBlock: activeKeys.length === 0})
                      }
                    >
                      <Switch
                        checkedChildren='Show diff only'
                        unCheckedChildren='Show diff only'
                        onChange={(isShowDiffOnly, e) =>
                          this.setState({onlyShowMetricDiff: isShowDiffOnly})
                        }
                      />
                    </CollapsibleSection>
                  </th>
                </tr>
              </tbody>
              <tbody className='table-block metric-block' style={Object.assign({}, tableBlockWidthStyle, {maxHeight: '300px'})}
               onScroll={this.onTableBlockScrollHandler}
              >
                {this.state.collapseMetricBlock ? undefined : this.renderDataRows(
                  this.props.metricLists,
                  colWidth,
                  "onlyShowMetricDiff",
                  false,
                  (key, data) => {
                    return (
                      <Link
                        to={Routes.getMetricPageRoute(
                          this.props.runInfos
                            .map((info) => info.run_uuid)
                            .filter((uuid, idx) => data[idx] !== undefined),
                          key,
                          experimentId,
                        )}
                        title='Plot chart'
                      >
                        {key}
                        <i className='fas fa-chart-line' style={{ paddingLeft: '6px' }} />
                      </Link>
                    );
                  },
                  Utils.formatMetric,
                )}
              </tbody>
            </table>
          </div>
        </CollapsibleSection>
      </div>
    );
  }

  // eslint-disable-next-line no-unused-vars
  renderDataRows(
    list,
    colWidth,
    onlyShowDiffStateKey,
    highlightChanges = false,
    headerMap = (key, data) => key,
    formatter = (value) => value,
  ) {
    var keys = CompareRunUtil.getKeys(list);
    const data = {};
    keys.forEach((k) => (data[k] = []));
    list.forEach((records, i) => {
      keys.forEach((k) => data[k].push(undefined));
      records.forEach((r) => (data[r.key][i] = r.value));
    });

    const colWidthStyle = {width: `${colWidth}px`, minWidth: `${colWidth}px`, maxWidth: `${colWidth}px`}

    function isAllEqual(k) {
      return data[k].every((x) => x === data[k][0])
    }

    if (this.state[onlyShowDiffStateKey]) {
      keys = keys.filter(k => !isAllEqual(k))
    }
    return keys.map((k) => {
      const allEqual = isAllEqual(k);

      let rowClass = undefined
      if (highlightChanges && !allEqual) {
        rowClass = 'diff-row';
      }

      return (
        <tr key={k} className={rowClass}>
          <th scope='row' className='head-value sticky_header' style={colWidthStyle}>
            {headerMap(k, data[k])}
          </th>
          {data[k].map((value, i) => {
            const cellText = value === undefined ? '' : formatter(value);
            return (
              <td className='data-value' key={this.props.runInfos[i].run_uuid} style={colWidthStyle}>
                <Tooltip title={cellText} color='blue' overlayStyle={{ maxWidth: '400px' }}>
                  <span className='truncate-text single-line'>{cellText}</span>
                </Tooltip>
              </td>
            );
          })}
        </tr>
      );
    });
  }
}

const mapStateToProps = (state, ownProps) => {
  const runInfos = [];
  const metricLists = [];
  const paramLists = [];
  const runNames = [];
  const runDisplayNames = [];
  const { experimentId, runUuids } = ownProps;
  const experiment = getExperiment(experimentId, state);
  runUuids.forEach((runUuid) => {
    runInfos.push(getRunInfo(runUuid, state));
    metricLists.push(Object.values(getLatestMetrics(runUuid, state)));
    paramLists.push(Object.values(getParams(runUuid, state)));
    const runTags = getRunTags(runUuid, state);
    runDisplayNames.push(Utils.getRunDisplayName(runTags, runUuid));
    runNames.push(Utils.getRunName(runTags));
  });
  return { experiment, runInfos, metricLists, paramLists, runNames, runDisplayNames };
};

export default connect(mapStateToProps)(injectIntl(CompareRunView));

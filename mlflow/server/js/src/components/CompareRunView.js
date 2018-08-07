import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getParams, getRunInfo } from '../reducers/Reducers';
import { connect } from 'react-redux';
import './CompareRunView.css';
import { RunInfo } from '../sdk/MlflowMessages';
import HtmlTableView from './HtmlTableView';
import Routes from '../Routes';
import { Link } from 'react-router-dom';
import Utils from '../utils/Utils';
import { getLatestMetrics } from '../reducers/MetricReducer';
import {ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer} from 'recharts';

class CompareRunView extends Component {
  static propTypes = {
    runInfos: PropTypes.arrayOf(RunInfo).required,
    metricLists: PropTypes.arrayOf(Array).required,
    paramLists: PropTypes.arrayOf(Array).required,
  };

  state = {
    scatter: {
      x: null,
      y: null,
    }
  }

  // Get the value of the metric/param described by {key, isMetric}, in run i
  getValue(i, {key, isMetric}) {
    const value = Private.findInList(
      (isMetric?this.props.metricLists:this.props.paramLists)[i],
      key);
    return value === undefined ? value : value.value;
  }

  render() {
    const tableStyles = {
      'tr': {
        display: 'flex',
        justifyContent: 'flex-start',
      },
      'td': {
        flex: '1',
      },
      'th': {
        flex: '1',
      },
      'td-first': {
        width: '500px',
      },
      'th-first': {
        width: '500px',
      },
    };

    const metricKeys = Private.getKeys(this.props.metricLists);
    const paramKeys = Private.getKeys(this.props.paramLists);
    const scatter = this.state.scatter;
    console.log(this.state);
    const showScatter = metricKeys.length + paramKeys.length >= 2;
    const scatterData = [];
    const renderSelectOptions = (axis) => {
      const options = [];
      options.push(<option disabled>Parameter</option>);
      paramKeys.forEach((p) => options.push(
        <option value={"0"+p}>{p}</option>));
      options.push(<option disabled>Metric</option>);
      metricKeys.forEach((p) => options.push(
        <option value={"1"+p}>{p}</option>));
      return options;
    };
    const onChangeSelect = (axis) => (e) => {
      const isMetric = !!+e.target.value.slice(0,1);
      const key = e.target.value.slice(1);
      scatter[axis] = {isMetric, key};
      this.setState({scatter});
    };
    if(showScatter) {
      if(scatter.y === null) {
        if(metricKeys.length > 0)
          scatter.y = {
            key: metricKeys[0],
            isMetric: true
          };
        else
          scatter.y = {
            key: paramKeys[1],
            isMetric: false
          };
      }
      if(scatter.x === null) {
        if(paramKeys.length > 0)
          scatter.x = {
            key: paramKeys[0],
            isMetric: false
          };
        else
          scatter.x = {
            key: metricKeys[1],
            isMetric: true
          };
      }

      this.props.runInfos.forEach((_, i) => {
        const x = this.getValue(i, scatter.x);
        const y = this.getValue(i, scatter.y);
        if(x === undefined || y === undefined)
          return;
        scatterData.push({i, x: +x, y: +y});
      });
    }
    // Private.findInList(this.props.metricLists, '');

    return (
      <div className="CompareRunView">
        <h1>Comparing {this.props.runInfos.length} Runs</h1>
        <div className="run-metadata-container">
          <div className="run-metadata-label">Run UUID:</div>
          <div className="run-metadata-row">
            {this.props.runInfos.map((r) => <div className="run-metadata-item">{r.getRunUuid()}</div>)}
          </div>
        </div>
        <div className="run-metadata-container last-run-metadata-container">
          <div className="run-metadata-label">Start Time:</div>
          <div className="run-metadata-row">
            {this.props.runInfos.map((run) => {
               const startTime = run.getStartTime() ? Utils.formatTimestamp(run.getStartTime()) : '(unknown)';
               return <div className="run-metadata-item">{startTime}</div>;
             }
            )}
          </div>
        </div>
        <h2>Parameters</h2>
        <HtmlTableView
          columns={["Name", "", ""]}
          values={Private.getParamRows(this.props.runInfos, this.props.paramLists)}
          styles={tableStyles}
        />
        <h2>Metrics</h2>
        <HtmlTableView
          columns={["Name", "", ""]}
          values={Private.getLatestMetricRows(this.props.runInfos, this.props.metricLists)}
          styles={tableStyles}
        />
        {showScatter?[
          <h2>Scatter Plot</h2>,
          <div>
            <div style={{width: "25%", float: "right"}}>
              <p>
                <label for="selectXAxis">X-axis</label> 
                <select id="selectXAxis" onChange={onChangeSelect("x")}
                  value={(scatter.x.isMetric?"1":"0")+scatter.x.key}>
                  {renderSelectOptions("x")}
                </select>
              </p>
              <p>
                <label for="selectYAxis">Y-axis</label> 
                <select id="selectYAxis" onChange={onChangeSelect("y")}
                  value={(scatter.y.isMetric?"1":"0")+scatter.y.key}>
                  {renderSelectOptions("y")}
                </select>
              </p>
            </div>
            <ResponsiveContainer width="70%" aspect={1.55}>
              <ScatterChart>
                <XAxis type="number" dataKey={'x'} name={'x'}/>
                <YAxis type="number" dataKey={'y'} name={'y'}/>
                <CartesianGrid />
                <Tooltip cursor={{strokeDasharray: '3 3'}}
                  content={this.renderTooltip.bind(this)}/>
                <Scatter data={scatterData} fill='#8884d8' />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        ]:""}
      </div>
    );
  }

  renderTooltip(item) {
    if(item.payload.length > 0) {
      const i = item.payload[0].payload.i;
      return <div>
        {this.props.runInfos[i].run_uuid}
        <h4>Parameters</h4>
        <ul>{
          this.props.paramLists[i].map((p) =>
            <li>{p.key}: {p.value}</li>
          )
        }</ul>
        <h4>Metrics</h4>
        <ul>{
          this.props.metricLists[i].map((p) =>
            <li>{p.key}: {Utils.formatMetric(p.value)}</li>
          )
        }</ul>
      </div>
    }
  }
}

const mapStateToProps = (state, ownProps) => {
  const runInfos = [];
  const metricLists = [];
  const paramLists = [];
  const { runUuids } = ownProps;
  runUuids.forEach((runUuid) => {
    runInfos.push(getRunInfo(runUuid, state));
    metricLists.push(Object.values(getLatestMetrics(runUuid, state)));
    paramLists.push(Object.values(getParams(runUuid, state)));
  });
  return { runInfos, metricLists, paramLists };
};

export default connect(mapStateToProps)(CompareRunView);

class Private {
  static getKeys(lists) {
    let keys = new Set();
    lists.forEach((list) => 
      list.forEach((item) => 
        keys.add(item.key)
    ));
    return [...keys].sort();
  }

  /**
   * Find in a list of metrics/params a certain key
   */
  static findInList(data, key) {
    let found = undefined;
    data.forEach((value) => {
      if(value.key == key)
        found = value;
    });
    return found;
  }

  static getParamRows(runInfos, paramLists) {
    const rows = [];
    // Map of parameter key to a map of (runUuid -> value)
    const paramKeyValueList = [];
    paramLists.forEach((paramList) => {
      const curKeyValueObj = {};
      paramList.forEach((param) => {
        curKeyValueObj[param.key] = param.value;
      });
      paramKeyValueList.push(curKeyValueObj);
    });

    const mergedParams = Utils.mergeRuns(runInfos.map((r) => r.run_uuid), paramKeyValueList);

    Object.keys(mergedParams).sort().forEach((paramKey) => {
      const curRow = [];
      curRow.push(paramKey);
      runInfos.forEach((r) => {
        const curUuid = r.run_uuid;
        curRow.push(mergedParams[paramKey][curUuid]);
      });
      rows.push(curRow)
    });
    return rows;
  }

  static getLatestMetricRows(runInfos, metricLists) {
    const rows = [];
    // Map of parameter key to a map of (runUuid -> value)
    const metricKeyValueList = [];
    metricLists.forEach((metricList) => {
      const curKeyValueObj = {};
      metricList.forEach((metric) => {
        curKeyValueObj[metric.key] = Utils.formatMetric(metric.value);
      });
      metricKeyValueList.push(curKeyValueObj);
    });

    const mergedMetrics = Utils.mergeRuns(runInfos.map((r) => r.run_uuid), metricKeyValueList);

    const runUuids = runInfos.map((r) => r.run_uuid);
    Object.keys(mergedMetrics).sort().forEach((metricKey) => {
      // Figure out which runUuids actually have this metric.
      const runUuidsWithMetric = Object.keys(mergedMetrics[metricKey]);
      const curRow = [];
      curRow.push(<Link to={Routes.getMetricPageRoute(runUuidsWithMetric, metricKey)}>{metricKey}</Link>);
      runInfos.forEach((r) => {
        const curUuid = r.run_uuid;
        curRow.push(Math.round(mergedMetrics[metricKey][curUuid] * 1e4)/1e4);
      });
      rows.push(curRow)
    });
    return rows;
  }
}

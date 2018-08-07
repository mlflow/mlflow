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


class CompareRunScatter extends Component {
  static propTypes = {
    runInfos: PropTypes.arrayOf(RunInfo).isRequired,
    metricLists: PropTypes.arrayOf(Array).isRequired,
    paramLists: PropTypes.arrayOf(Array).isRequired,
  };

  state = {
    scatter: {
      x: null,
      y: null,
    }
  }

  constructor(props) {
    super(props);

    this.metricKeys = CompareRunScatter.getKeys(this.props.metricLists);
    this.paramKeys = CompareRunScatter.getKeys(this.props.paramLists);

    if(this.paramKeys.length + this.metricKeys.length < 2)
      this.state = {disabled: true};
    else
      this.state = {
        disabled: false,
        x: this.paramKeys.length > 0 ?
          {
            key: this.paramKeys[0],
            isMetric: false
          } : {
            key: this.metricKeys[1],
            isMetric: true
          },
        y: this.metricKeys.length > 0 ?
          {
            key: this.metricKeys[0],
            isMetric: true
          } : {
            key: this.paramKeys[1],
            isMetric: false
          }
      }
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

  static getKeys(lists) {
    let keys = {};
    lists.forEach((list) => 
      list.forEach((item) => {
        if(!(item.key in keys))
          keys[item.key] = true;
        if(isNaN(parseFloat(item.value)))
          keys[item.key] = false;
      }
    ));
    return Object.keys(keys).filter(k => keys[k]).sort();
  }

  // Get the value of the metric/param described by {key, isMetric}, in run i
  getValue(i, {key, isMetric}) {
    const value = CompareRunScatter.findInList(
      (isMetric?this.props.metricLists:this.props.paramLists)[i],
      key);
    return value === undefined ? value : value.value;
  }

  render() {
    if(this.state.disabled)
      return <div></div>

    const scatterData = [];

    this.props.runInfos.forEach((_, index) => {
      const x = this.getValue(index, this.state.x);
      const y = this.getValue(index, this.state.y);
      if(x === undefined || y === undefined)
        return;
      scatterData.push({index, x: +x, y: +y});
    });

    return (
      <div>
        <h2>Scatter Plot</h2>
        <div style={{width: "25%", float: "right"}}>
          <p>
            <label htmlFor="selectXAxis">X-axis</label> 
            {this.renderSelect("selectXAxis", "x")}
          </p>
          <p>
            <label htmlFor="selectYAxis">Y-axis</label> 
            {this.renderSelect("selectYAxis", "y")}
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
      </div>);
  }

  renderSelect(id, axis) {
    const onChangeSelect = (axis) => (e) => {
      const isMetric = !!+e.target.value.slice(0,1);
      const key = e.target.value.slice(1);
      this.setState({[axis]: {isMetric, key}});
    };

    return (
      <select id={id}
              onChange={onChangeSelect(axis)}
              value={(this.state[axis].isMetric?"1":"0")
                +this.state[axis].key}>
        <optgroup label="Parameter">
          {this.paramKeys.map((p) =>
            <option key={p} value={"0"+p}>{p}</option>
          )}
        </optgroup>
        <optgroup label="Metric">
          {this.metricKeys.map((m) =>
            <option key={m} value={"1"+m}>{m}</option>
          )}
        </optgroup>
      </select>);
  }


  renderTooltip(item) {
    if(item.payload.length > 0) {
      const i = item.payload[0].payload.index;
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

export default connect(mapStateToProps)(CompareRunScatter);

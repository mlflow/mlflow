import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getParams, getRunInfo } from '../reducers/Reducers';
import { connect } from 'react-redux';
import './CompareRunView.css';
import { RunInfo } from '../sdk/MlflowMessages';
import Utils from '../utils/Utils';
import { getLatestMetrics } from '../reducers/MetricReducer';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Label,
} from 'recharts';
import './CompareRunScatter.css';
import CompareRunUtil from './CompareRunUtil';

class CompareRunScatter extends Component {
  static propTypes = {
    runInfos: PropTypes.arrayOf(RunInfo).isRequired,
    metricLists: PropTypes.arrayOf(Array).isRequired,
    paramLists: PropTypes.arrayOf(Array).isRequired,
  };

  constructor(props) {
    super(props);

    this.renderTooltip = this.renderTooltip.bind(this);

    this.metricKeys = CompareRunUtil.getKeys(this.props.metricLists, true);
    this.paramKeys = CompareRunUtil.getKeys(this.props.paramLists, true);

    if (this.paramKeys.length + this.metricKeys.length < 2) {
      this.state = {disabled: true};
    } else {
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
      };
    }
  }

  /**
   * Get the value of the metric/param described by {key, isMetric}, in run i
   */
  getValue(i, {key, isMetric}) {
    const value = CompareRunUtil.findInList(
      (isMetric ? this.props.metricLists : this.props.paramLists)[i], key);
    return value === undefined ? value : value.value;
  }

  render() {
    if (this.state.disabled) {
      return <div></div>;
    }

    const scatterData = [];

    this.props.runInfos.forEach((_, index) => {
      const x = this.getValue(index, this.state.x);
      const y = this.getValue(index, this.state.y);
      if (x === undefined || y === undefined) {
        return;
      }
      scatterData.push({index, x: +x, y: +y});
    });

    return (<div>
      <h2>Scatter Plot</h2>
      <div className="container-fluid">
        <div className="row">
          <form className="col-xs-3">
            <div className="form-group">
              <label htmlFor="y-axis-selector">X-axis:</label>
              {this.renderSelect("x")}
            </div>
            <div className="form-group">
              <label htmlFor="y-axis-selector">Y-axis:</label>
              {this.renderSelect("y")}
            </div>
          </form>
          <div className="col-xs-9">
            <ResponsiveContainer width="100%" aspect={1.55}>
              <ScatterChart>
                <XAxis type="number" dataKey='x' name='x'>
                  {this.renderAxisLabel('x')}
                </XAxis>
                <YAxis type="number" dataKey='y' name='y'>
                  {this.renderAxisLabel('y')}
                </YAxis>
                <CartesianGrid/>
                <Tooltip
                  isAnimationActive={false}
                  cursor={{strokeDasharray: '3 3'}}
                  content={this.renderTooltip}
                />
                <Scatter
                  data={scatterData}
                  fill='#AE76A6'
                  isAnimationActive={false}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>);
  }

  renderAxisLabel(axis) {
    const key = this.state[axis];
    return (<Label
      angle={axis === "x" ? 0 : -90}
      offset={axis === "x" ? -5 : 5}
      value={(key.isMetric ? "Metric" : "Parameter") + ": " + key.key}
      position={axis === "x" ? "insideBottom" : "insideLeft"}
    />);
  }

  renderSelect(axis) {
    return (
      <select
        className="form-control"
        id={axis + "-axis-selector"}
        onChange={(e) => {
          const [prefix, ...keyParts] = e.target.value.split("-");
          const key = keyParts.join("-");
          const isMetric = prefix === "metric";
          this.setState({[axis]: {isMetric, key}});
        }}
        value={(this.state[axis].isMetric ? "metric-" : "param-") + this.state[axis].key}
      >
        <optgroup label="Parameter">
          {this.paramKeys.map((p) =>
            <option key={p} value={"param-" + p}>{p}</option>
          )}
        </optgroup>
        <optgroup label="Metric">
          {this.metricKeys.map((m) =>
            <option key={m} value={"metric-" + m}>{m}</option>
          )}
        </optgroup>
      </select>);
  }

  renderTooltip(item) {
    if (item.payload.length > 0) {
      const i = item.payload[0].payload.index;
      return (
        <div className="panel panel-default scatter-tooltip">
          <div className="panel-heading">
            <h3 className="panel-title">{this.props.runInfos[i].run_uuid}</h3>
          </div>
          <div className="panel-body">
            <div className="row">
              <div className="col-xs-6">
                <h4>Parameters</h4>
                <ul>{
                  this.props.paramLists[i].map((p) =>
                    <li key={p.key}>{p.key}: <span className="value">{p.value}</span></li>
                  )
                }</ul>
              </div>
              <div className="col-xs-6">
                <h4>Metrics</h4>
                <ul>
                  {this.props.metricLists[i].map((p) =>
                    <li key={p.key}>{p.key}: {Utils.formatMetric(p.value)}</li>
                  )}
                </ul>
              </div>
            </div>
          </div>
        </div>
      );
    }
    return null;
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

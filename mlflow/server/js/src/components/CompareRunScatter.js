import React, { Component } from 'react';
import { AllHtmlEntities } from 'html-entities';
import Plot from 'react-plotly.js';
import PropTypes from 'prop-types';
import { getParams, getRunInfo } from '../reducers/Reducers';
import { connect } from 'react-redux';
import './CompareRunView.css';
import { RunInfo } from '../sdk/MlflowMessages';
import Utils from '../utils/Utils';
import { getLatestMetrics } from '../reducers/MetricReducer';
import './CompareRunScatter.css';
import CompareRunUtil from './CompareRunUtil';

class CompareRunScatter extends Component {
  static propTypes = {
    runInfos: PropTypes.arrayOf(RunInfo).isRequired,
    metricLists: PropTypes.arrayOf(Array).isRequired,
    paramLists: PropTypes.arrayOf(Array).isRequired,
    runDisplayNames: PropTypes.arrayOf(String).isRequired,
  };

  constructor(props) {
    super(props);

    this.entities = new AllHtmlEntities();

    this.metricKeys = CompareRunUtil.getKeys(this.props.metricLists, false);
    this.paramKeys = CompareRunUtil.getKeys(this.props.paramLists, false);

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

  /**
   * Encode HTML entities in a string (since Plotly's tooltips take HTML)
   */
  encodeHtml(str) {
    return this.entities.encode(str);
  }

  render() {
    if (this.state.disabled) {
      return <div/>;
    }

    const xs = [];
    const ys = [];
    const tooltips = [];

    this.props.runInfos.forEach((_, index) => {
      const x = this.getValue(index, this.state.x);
      const y = this.getValue(index, this.state.y);
      if (x === undefined || y === undefined) {
        return;
      }
      xs.push(x);
      ys.push(y);
      tooltips.push(this.getPlotlyTooltip(index));
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
            <Plot
              data={[
                {
                  x: xs,
                  y: ys,
                  text: tooltips,
                  hoverinfo: "text",
                  type: 'scatter',
                  mode: 'markers',
                  marker: {
                    size: 10,
                    color: "rgba(200, 50, 100, .75)"
                  },
                },
              ]}
              layout={{
                margin: {
                  l: 40,
                  r: 40,
                  b: 30,
                  t: 30
                },
                hovermode: "closest",
                xaxis: {
                  title: this.encodeHtml(this.state["x"].key)
                },
                yaxis: {
                  title: this.encodeHtml(this.state["y"].key)
                }
              }}
              className={"scatter-plotly"}
              config={{
                responsive: true,
                displaylogo: false,
                modeBarButtonsToRemove: [
                  "sendDataToCloud",
                  "select2d",
                  "lasso2d",
                  "resetScale2d",
                  "hoverClosestCartesian",
                  "hoverCompareCartesian"
                ]
              }}
              useResizeHandler
            />
          </div>
        </div>
      </div>
    </div>);
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
            <option value={"param-" + p}>{p}</option>
          )}
        </optgroup>
        <optgroup label="Metric">
          {this.metricKeys.map((m) =>
            <option value={"metric-" + m}>{m}</option>
          )}
        </optgroup>
      </select>);
  }

  getPlotlyTooltip(index) {
    const runName = this.props.runDisplayNames[index];
    let result = `<b>${this.encodeHtml(runName)}</b><br>`;
    const paramList = this.props.paramLists[index];
    paramList.forEach(p => {
      result += this.encodeHtml(p.key) + ': ' + this.encodeHtml(p.value) + '<br>';
    });
    const metricList = this.props.metricLists[index];
    if (metricList.length > 0) {
      result += (paramList.length > 0) ? '<br>' : '';
      metricList.forEach(m => {
        result += this.encodeHtml(m.key) + ': ' + Utils.formatMetric(m.value) + '<br>';
      });
    }
    return result;
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

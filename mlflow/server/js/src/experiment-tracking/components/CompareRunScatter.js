import React, { Component } from 'react';
import { AllHtmlEntities } from 'html-entities';
import PropTypes from 'prop-types';
import { getParams, getRunInfo } from '../reducers/Reducers';
import { connect } from 'react-redux';
import './CompareRunView.css';
import { RunInfo } from '../sdk/MlflowMessages';
import Utils from '../../common/utils/Utils';
import { getLatestMetrics } from '../reducers/MetricReducer';
import './CompareRunScatter.css';
import CompareRunUtil from './CompareRunUtil';
import { FormattedMessage } from 'react-intl';
import { LazyPlot } from './LazyPlot';

export class CompareRunScatterImpl extends Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    runInfos: PropTypes.arrayOf(PropTypes.instanceOf(RunInfo)).isRequired,
    metricLists: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    paramLists: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    runDisplayNames: PropTypes.arrayOf(PropTypes.string).isRequired,
  };

  // Size limits for displaying keys and values in our plot axes and tooltips
  static MAX_PLOT_KEY_LENGTH = 40;
  static MAX_PLOT_VALUE_LENGTH = 60;

  constructor(props) {
    super(props);

    this.entities = new AllHtmlEntities();

    this.metricKeys = CompareRunUtil.getKeys(this.props.metricLists, false);
    this.paramKeys = CompareRunUtil.getKeys(this.props.paramLists, false);

    if (this.paramKeys.length + this.metricKeys.length < 2) {
      this.state = { disabled: true };
    } else {
      this.state = {
        disabled: false,
        x:
          this.paramKeys.length > 0
            ? {
                key: this.paramKeys[0],
                isMetric: false,
              }
            : {
                key: this.metricKeys[1],
                isMetric: true,
              },
        y:
          this.metricKeys.length > 0
            ? {
                key: this.metricKeys[0],
                isMetric: true,
              }
            : {
                key: this.paramKeys[1],
                isMetric: false,
              },
      };
    }
  }

  /**
   * Get the value of the metric/param described by {key, isMetric}, in run i
   */
  getValue(i, { key, isMetric }) {
    const value = CompareRunUtil.findInList(
      (isMetric ? this.props.metricLists : this.props.paramLists)[i],
      key,
    );
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
      return <div />;
    }

    const keyLength = CompareRunScatterImpl.MAX_PLOT_KEY_LENGTH;

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

    return (
      <div className='responsive-table-container'>
        <div className='container-fluid'>
          <div className='row'>
            <form className='col-xs-3'>
              <div className='form-group'>
                <label htmlFor='y-axis-selector'>
                  <FormattedMessage
                    defaultMessage='X-axis:'
                    description='Label text for x-axis in scatter plot comparison in MLflow'
                  />
                </label>
                {this.renderSelect('x')}
              </div>
              <div className='form-group'>
                <label htmlFor='y-axis-selector'>
                  <FormattedMessage
                    defaultMessage='Y-axis:'
                    description='Label text for y-axis in scatter plot comparison in MLflow'
                  />
                </label>
                {this.renderSelect('y')}
              </div>
            </form>
            <div className='col-xs-9'>
              <LazyPlot
                data={[
                  {
                    x: xs,
                    y: ys,
                    text: tooltips,
                    hoverinfo: 'text',
                    type: 'scattergl',
                    mode: 'markers',
                    marker: {
                      size: 10,
                      color: 'rgba(200, 50, 100, .75)',
                    },
                  },
                ]}
                layout={{
                  margin: {
                    t: 30,
                  },
                  hovermode: 'closest',
                  xaxis: {
                    title: this.encodeHtml(Utils.truncateString(this.state['x'].key, keyLength)),
                  },
                  yaxis: {
                    title: this.encodeHtml(Utils.truncateString(this.state['y'].key, keyLength)),
                  },
                }}
                className={'scatter-plotly'}
                config={{
                  responsive: true,
                  displaylogo: false,
                  scrollZoom: true,
                  modeBarButtonsToRemove: [
                    'sendDataToCloud',
                    'select2d',
                    'lasso2d',
                    'resetScale2d',
                    'hoverClosestCartesian',
                    'hoverCompareCartesian',
                  ],
                }}
                useResizeHandler
              />
            </div>
          </div>
        </div>
      </div>
    );
  }

  renderSelect(axis) {
    return (
      <select
        className='form-control'
        id={axis + '-axis-selector'}
        aria-label={`${axis} axis`}
        onChange={(e) => {
          const [prefix, ...keyParts] = e.target.value.split('-');
          const key = keyParts.join('-');
          const isMetric = prefix === 'metric';
          this.setState({ [axis]: { isMetric, key } });
        }}
        value={(this.state[axis].isMetric ? 'metric-' : 'param-') + this.state[axis].key}
      >
        <optgroup label='Parameter'>
          {this.paramKeys.map((p) => (
            <option key={'param-' + p} value={'param-' + p}>
              {p}
            </option>
          ))}
        </optgroup>
        <optgroup label='Metric'>
          {this.metricKeys.map((m) => (
            <option key={'metric-' + m} value={'metric-' + m}>
              {m}
            </option>
          ))}
        </optgroup>
      </select>
    );
  }

  getPlotlyTooltip(index) {
    const keyLength = CompareRunScatterImpl.MAX_PLOT_KEY_LENGTH;
    const valueLength = CompareRunScatterImpl.MAX_PLOT_VALUE_LENGTH;
    const runName = this.props.runDisplayNames[index];
    let result = `<b>${this.encodeHtml(runName)}</b><br>`;
    const paramList = this.props.paramLists[index];
    paramList.forEach((p) => {
      result +=
        this.encodeHtml(Utils.truncateString(p.key, keyLength)) +
        ': ' +
        this.encodeHtml(Utils.truncateString(p.value, valueLength)) +
        '<br>';
    });
    const metricList = this.props.metricLists[index];
    if (metricList.length > 0) {
      result += paramList.length > 0 ? '<br>' : '';
      metricList.forEach((m) => {
        result +=
          this.encodeHtml(Utils.truncateString(m.key, keyLength)) +
          ': ' +
          Utils.formatMetric(m.value) +
          '<br>';
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

export const CompareRunScatter = connect(mapStateToProps)(CompareRunScatterImpl);

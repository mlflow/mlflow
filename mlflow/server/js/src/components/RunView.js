import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getExperiment, getParams, getRunInfo, getRunTags } from '../reducers/Reducers';
import { connect } from 'react-redux';
import './RunView.css';
import HtmlTableView from './HtmlTableView';
import { Link } from 'react-router-dom';
import ArtifactPage from './ArtifactPage';
import { getLatestMetrics } from '../reducers/MetricReducer';
import { Experiment } from '../sdk/MlflowMessages';
import Routes from '../Routes';
import Utils from '../utils/Utils';

const PARAMATERS_KEY = 'parameters';
const METRICS_KEY = 'metrics';
const ARTIFACTS_KEY = 'artifacts';
const TAGS_KEY = 'tags';

class RunView extends Component {
  constructor(props) {
    super(props);
    this.onClickExpander = this.onClickExpander.bind(this);
    this.getExpanderClassName = this.getExpanderClassName.bind(this);
    this.state.showTags = getTagValues(props.tags).length > 0;
  }

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    run: PropTypes.object.isRequired,
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    params: PropTypes.object.isRequired,
    tags: PropTypes.object.isRequired,
    latestMetrics: PropTypes.object.isRequired,
    getMetricPagePath: PropTypes.func.isRequired,
  };

  state = {
    showParameters: true,
    showMetrics: true,
    showArtifacts: true,
    showTags: true,
  };

  onClickExpander(key) {
    switch (key) {
      case PARAMATERS_KEY: {
        this.setState({ showParameters: !this.state.showParameters });
        return;
      }
      case METRICS_KEY: {
        this.setState({ showMetrics: !this.state.showMetrics });
        return;
      }
      case TAGS_KEY: {
        this.setState({ showTags: !this.state.showTags });
        return;
      }
      case ARTIFACTS_KEY: {
        this.setState({ showArtifacts: !this.state.showArtifacts });
        return;
      }
    }
  }

  getExpanderClassName(key) {
    switch (key) {
      case PARAMATERS_KEY: {
        return this.state.showParameters ? 'fa-caret-down': 'fa-caret-right';
      }
      case METRICS_KEY: {
        return this.state.showMetrics ? 'fa-caret-down': 'fa-caret-right';
      }
      case TAGS_KEY: {
        return this.state.showTags? 'fa-caret-down': 'fa-caret-right';
      }
      case ARTIFACTS_KEY: {
        return this.state.showArtifacts ? 'fa-caret-down': 'fa-caret-right';
      }
    }

  }

  render() {
    const { run, experiment, params, tags, latestMetrics, getMetricPagePath } = this.props;
    const startTime = run.getStartTime() ? Utils.formatTimestamp(run.getStartTime()) : '(unknown)';
    const duration = run.getStartTime() && run.getEndTime() ? run.getEndTime() - run.getStartTime() : '(unknown)';
    const experimentId = experiment.getExperimentId();
    const tableStyles = {
      table: {
        width: 'auto',
        minWidth: '400px',
      },
      th: {
        width: 'auto',
        minWidth: '200px',
        marginRight: '80px',
      }
    };

    let runCommand = null;
    if (run.source_type === "PROJECT") {
      runCommand = 'mlflow run ' + shellEscape(run.source_name);
      if (run.source_version && run.source_version !== "latest") {
        runCommand += ' -v ' + shellEscape(run.source_version);
      }
      if (run.entry_point_name && run.entry_point_name !== "main") {
        runCommand += ' -e ' + shellEscape(run.entry_point_name);
      }
      Object.values(params).sort().forEach(p => {
        runCommand += ' -P ' + shellEscape(p.key + '=' + p.value);
      });
    }

    return (
      <div className="RunView">
        <div className="header-container">
          <h1 className="run-uuid">Run {run.getRunUuid()}</h1>
        </div>
        <div className="run-info-container">
          <div className="run-info">
            <span className="metadata-header">Experiment Name: </span>
            <Link to={Routes.getExperimentPageRoute(experimentId)}>
              <span className="metadata-info">{experiment.getName()}</span>
            </Link>
          </div>
          <div className="run-info">
            <span className="metadata-header">Start Time: </span>
            <span className="metadata-info">{startTime}</span>
          </div>
          <div className="run-info">
            <span className="metadata-header">Source: </span>
            <span className="metadata-info">{Utils.renderSource(run)}</span>
          </div>
          {run.source_version ?
            <div className="run-info">
              <span className="metadata-header">Git Commit: </span>
              <span className="metadata-info">{run.source_version.substring(0, 20)}</span>
            </div>
            : null
          }
          {run.source_type === "PROJECT" ?
            <div className="run-info">
              <span className="metadata-header">Entry Point: </span>
              <span className="metadata-info">{run.entry_point_name || "main"}</span>
            </div>
            : null
          }
          <div className="run-info">
            <span className="metadata-header">User: </span>
            <span className="metadata-info">{run.getUserId()}</span>
          </div>
          <div className="run-info">
            <span className="metadata-header">Duration: </span>
            <span className="metadata-info">{Utils.formatDuration(duration)}</span>
          </div>
        </div>
        {runCommand ?
          <div className="RunView-info">
            <h2>Run Command</h2>
            <textarea className="run-command text-area" readOnly={true}>{runCommand}</textarea>
          </div>
          : null
        }
        <div className="RunView-info">
          <h2 onClick={() => this.onClickExpander(PARAMATERS_KEY)} className="table-name">
            <span ><i className={`fa ${this.getExpanderClassName(PARAMATERS_KEY)}`}/></span>
            {' '}Parameters
          </h2>
          {this.state.showParameters ?
            <HtmlTableView
              columns={[ "Name", "Value" ]}
              values={getParamValues(params)}
              styles={tableStyles}
            /> :
            null
          }
          <h2 onClick={() => this.onClickExpander(METRICS_KEY)} className="table-name">
            <span><i className={`fa ${this.getExpanderClassName(METRICS_KEY)}`}/></span>
            {' '}Metrics
          </h2>
          {this.state.showMetrics ?
            <HtmlTableView
              columns={[ "Name", "Value" ]}
              values={getMetricValues(latestMetrics, getMetricPagePath)}
              styles={tableStyles}
            /> :
            null
          }
          <h2 onClick={() => this.onClickExpander(TAGS_KEY)} className="table-name">
            <span><i className={`fa ${this.getExpanderClassName(TAGS_KEY)}`}/></span>
            {' '}Tags
          </h2>
          {this.state.showTags ?
            <HtmlTableView
              columns={[ "Name", "Value" ]}
              values={getTagValues(tags)}
              styles={tableStyles}
            /> :
            null
          }
        </div>
          <div>
            <h2 onClick={() => this.onClickExpander(ARTIFACTS_KEY)} className="table-name">
              <span><i className={`fa ${this.getExpanderClassName(ARTIFACTS_KEY)}`}/></span>
              {' '}Artifacts
            </h2>
            {this.state.showArtifacts ?
              <ArtifactPage runUuid={this.props.runUuid} isHydrated={true}/> :
              null
            }
          </div>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { runUuid, metricPageRoute, getMetricPagePath, experimentId } = ownProps;
  const run = getRunInfo(runUuid, state);
  const experiment = getExperiment(experimentId, state);
  const params = getParams(runUuid, state);
  const tags = getRunTags(runUuid, state);
  const latestMetrics = getLatestMetrics(runUuid, state);
  return { run, experiment, params, tags, latestMetrics, metricPageRoute, getMetricPagePath };
};

export default connect(mapStateToProps)(RunView);

// Private helper functions.

const getParamValues = (params) => {
  return Object.values(params).sort().map((p) =>
    [p.getKey(), p.getValue()]
  );
};

const getTagValues = (tags) => {
  return Object.values(tags).map((t) =>
    [t.getKey(), t.getValue()]
  );
};

const getMetricValues = (latestMetrics, getMetricPagePath) => {
  return Object.values(latestMetrics).sort().map((m) => {
    const key = m.key;
    return [<Link to={getMetricPagePath(key)}>{key}</Link>, Utils.formatMetric(m.value)]
  });
};

const shellEscape = (str) => {
  if (/["\r\n\t ]/.test(str)) {
    return '"' + str.replace(/"/g, '\\"') + '"';
  }
  return str;
};


import dateFormat from 'dateformat';
import React from 'react';
import notebookSvg from '../static/notebook.svg';
import revisionSvg from '../static/revision.svg';
import emptySvg from '../static/empty.svg';
import laptopSvg from '../static/laptop.svg';
import projectSvg from '../static/project.svg';
import jobSvg from '../static/job.svg';
import qs from 'qs';
import { MLFLOW_INTERNAL_PREFIX } from './TagUtils';
import { message } from 'antd';
import _ from 'lodash';
import { ErrorCodes, SupportPageUrl } from '../constants';

message.config({
  maxCount: 1,
  duration: 5,
});

class Utils {
  /**
   * Merge a runs parameters / metrics.
   * @param runUuids - A list of Run UUIDs.
   * @param keyValueList - A list of objects. One object for each run.
   * @retuns A key to a map of (runUuid -> value)
   */
  static mergeRuns(runUuids, keyValueList) {
    const ret = {};
    keyValueList.forEach((keyValueObj, i) => {
      const curRunUuid = runUuids[i];
      Object.keys(keyValueObj).forEach((key) => {
        const cur = ret[key] || {};
        ret[key] = {
          ...cur,
          [curRunUuid]: keyValueObj[key],
        };
      });
    });
    return ret;
  }

  static runNameTag = 'mlflow.runName';
  static sourceNameTag = 'mlflow.source.name';
  static sourceTypeTag = 'mlflow.source.type';
  static gitCommitTag = 'mlflow.source.git.commit';
  static entryPointTag = 'mlflow.project.entryPoint';
  static backendTag = 'mlflow.project.backend';
  static userTag = 'mlflow.user';
  static loggedModelsTag = 'mlflow.log-model.history';

  static formatMetric(value) {
    if (value === 0) {
      return '0';
    } else if (Math.abs(value) < 1e-3) {
      return value.toExponential(3).toString();
    } else if (Math.abs(value) < 10) {
      return (Math.round(value * 1000) / 1000).toString();
    } else if (Math.abs(value) < 100) {
      return (Math.round(value * 100) / 100).toString();
    } else {
      return (Math.round(value * 10) / 10).toString();
    }
  }

  /**
   * Helper method for that returns a truncated version of the passed-in string (with trailing
   * ellipsis) if the string is longer than maxLength. Otherwise, just returns the passed-in string.
   */
  static truncateString(string, maxLength) {
    if (string.length > maxLength) {
      return string.slice(0, maxLength - 3) + '...';
    }
    return string;
  }

  /**
   * We need to cast all of the timestamps back to numbers since keys of JS objects are auto casted
   * to strings.
   *
   * @param metrics - List of { timestamp: "1", [run1.uuid]: 7, ... }
   * @returns Same list but all of the timestamps casted to numbers.
   */
  static convertTimestampToInt(metrics) {
    return metrics.map((metric) => {
      return {
        ...metric,
        timestamp: Number.parseFloat(metric.timestamp),
      };
    });
  }

  /**
   * Format timestamps from millisecond epoch time.
   */
  static formatTimestamp(timestamp, format = 'yyyy-mm-dd HH:MM:ss') {
    if (timestamp === undefined) {
      return '(unknown)';
    }
    const d = new Date(0);
    d.setUTCMilliseconds(timestamp);
    return dateFormat(d, format);
  }

  static timeSince(date) {
    const seconds = Math.max(0, Math.floor((new Date() - date) / 1000));

    let interval = Math.floor(seconds / 31536000);

    if (interval >= 1) {
      return interval + ' year' + (interval === 1 ? '' : 's');
    }
    interval = Math.floor(seconds / 2592000);
    if (interval >= 1) {
      return interval + ' month' + (interval === 1 ? '' : 's');
    }
    interval = Math.floor(seconds / 86400);
    if (interval >= 1) {
      return interval + ' day' + (interval === 1 ? '' : 's');
    }
    interval = Math.floor(seconds / 3600);
    if (interval >= 1) {
      return interval + ' hour' + (interval === 1 ? '' : 's');
    }
    interval = Math.floor(seconds / 60);
    if (interval >= 1) {
      return interval + ' minute' + (interval === 1 ? '' : 's');
    }
    return Math.floor(seconds) + ' seconds';
  }

  /**
   * Format a duration given in milliseconds.
   *
   * @param duration in milliseconds
   */
  static formatDuration(duration) {
    if (duration < 500) {
      return duration + 'ms';
    } else if (duration < 1000 * 60) {
      return (duration / 1000).toFixed(1) + 's';
    } else if (duration < 1000 * 60 * 60) {
      return (duration / 1000 / 60).toFixed(1) + 'min';
    } else if (duration < 1000 * 60 * 60 * 24) {
      return (duration / 1000 / 60 / 60).toFixed(1) + 'h';
    } else {
      return (duration / 1000 / 60 / 60 / 24).toFixed(1) + 'd';
    }
  }

  static baseName(path) {
    const pieces = path.split('/');
    return pieces[pieces.length - 1];
  }

  static dropExtension(path) {
    return path.replace(/(.*[^/])\.[^/.]+$/, '$1');
  }

  /**
   * Normalizes a URI, removing redundant slashes and trailing slashes
   * For example, normalize("foo://bar///baz/") === "foo://bar/baz"
   */
  static normalize(uri) {
    // Remove empty authority component (e.g., "foo:///" becomes "foo:/")
    const withNormalizedAuthority = uri.replace(/[:]\/\/\/+/, ':/');
    // Remove redundant slashes while ensuring that double slashes immediately following
    // the scheme component are preserved
    const withoutRedundantSlashes = withNormalizedAuthority.replace(/(^\/|[^:]\/)\/+/g, '$1');
    const withoutTrailingSlash = withoutRedundantSlashes.replace(/\/$/, '');
    return withoutTrailingSlash;
  }

  static getGitHubRegex() {
    return /[@/]github.com[:/]([^/.]+)\/([^/#]+)#?(.*)/;
  }

  static getGitLabRegex() {
    return /[@/]gitlab.com[:/]([^/.]+)\/([^/#]+)#?(.*)/;
  }

  static getBitbucketRegex() {
    return /[@/]bitbucket.org[:/]([^/.]+)\/([^/#]+)#?(.*)/;
  }

  static getGitRepoUrl(sourceName) {
    const gitHubMatch = sourceName.match(Utils.getGitHubRegex());
    const gitLabMatch = sourceName.match(Utils.getGitLabRegex());
    const bitbucketMatch = sourceName.match(Utils.getBitbucketRegex());
    let url = null;
    if (gitHubMatch || gitLabMatch) {
      const baseUrl = gitHubMatch ? 'https://github.com/' : 'https://gitlab.com/';
      const match = gitHubMatch || gitLabMatch;
      url = baseUrl + match[1] + '/' + match[2].replace(/.git/, '');
      if (match[3]) {
        url = url + '/tree/master/' + match[3];
      }
    } else if (bitbucketMatch) {
      const baseUrl = 'https://bitbucket.org/';
      url = baseUrl + bitbucketMatch[1] + '/' + bitbucketMatch[2].replace(/.git/, '');
      if (bitbucketMatch[3]) {
        url = url + '/src/master/' + bitbucketMatch[3];
      }
    }
    return url;
  }

  static getGitCommitUrl(sourceName, sourceVersion) {
    const gitHubMatch = sourceName.match(Utils.getGitHubRegex());
    const gitLabMatch = sourceName.match(Utils.getGitLabRegex());
    const bitbucketMatch = sourceName.match(Utils.getBitbucketRegex());
    let url = null;
    if (gitHubMatch || gitLabMatch) {
      const baseUrl = gitHubMatch ? 'https://github.com/' : 'https://gitlab.com/';
      const match = gitHubMatch || gitLabMatch;
      url =
        baseUrl +
        match[1] +
        '/' +
        match[2].replace(/.git/, '') +
        '/tree/' +
        sourceVersion +
        '/' +
        match[3];
    } else if (bitbucketMatch) {
      const baseUrl = 'https://bitbucket.org/';
      url =
        baseUrl +
        bitbucketMatch[1] +
        '/' +
        bitbucketMatch[2].replace(/.git/, '') +
        '/src/' +
        sourceVersion +
        '/' +
        bitbucketMatch[3];
    }
    return url;
  }

  /**
   * Returns a copy of the provided URL with its query parameters set to `queryParams`.
   * @param url URL string like "http://my-mlflow-server.com/#/experiments/9.
   * @param queryParams Optional query parameter string like "?param=12345". Query params provided
   *        via this string will override existing query param values in `url`
   */
  static setQueryParams(url, queryParams) {
    const urlObj = new URL(url);
    urlObj.search = queryParams || '';
    return urlObj.toString();
  }

  static getNotebookId(tags) {
    const notebookIdTag = 'mlflow.databricks.notebookID';
    return tags && tags[notebookIdTag] && tags[notebookIdTag].value;
  }

  static getNotebookRevisionId(tags) {
    const revisionIdTag = 'mlflow.databricks.notebookRevisionID';
    return tags && tags[revisionIdTag] && tags[revisionIdTag].value;
  }

  /**
   * Renders the source name and entry point into an HTML element. Used for display.
   * @param tags Object containing tag key value pairs.
   * @param queryParams Query params to add to certain source type links.
   * @param runUuid ID of the MLflow run to add to certain source (revision) links.
   */
  static renderSource(tags, queryParams, runUuid) {
    const sourceName = Utils.getSourceName(tags);
    const sourceType = Utils.getSourceType(tags);
    let res = Utils.formatSource(tags);
    if (sourceType === 'PROJECT') {
      const url = Utils.getGitRepoUrl(sourceName);
      if (url) {
        res = (
          <a target='_top' href={url}>
            {res}
          </a>
        );
      }
      return res;
    } else if (sourceType === 'NOTEBOOK') {
      const revisionId = Utils.getNotebookRevisionId(tags);
      const notebookId = Utils.getNotebookId(tags);
      if (notebookId) {
        let url = Utils.setQueryParams(window.location.origin, queryParams);
        url += `#notebook/${notebookId}`;
        if (revisionId) {
          url += `/revision/${revisionId}`;
          if (runUuid) {
            url += `/mlflow/run/${runUuid}`;
          }
        }
        res = (
          <a title={sourceName} href={url} target='_top'>
            {Utils.baseName(sourceName)}
          </a>
        );
      }
      return res;
    } else if (sourceType === 'JOB') {
      const jobIdTag = 'mlflow.databricks.jobID';
      const jobRunIdTag = 'mlflow.databricks.jobRunID';
      const jobId = tags && tags[jobIdTag] && tags[jobIdTag].value;
      const jobRunId = tags && tags[jobRunIdTag] && tags[jobRunIdTag].value;
      if (jobId && jobRunId) {
        let url = Utils.setQueryParams(window.location.origin, queryParams);
        url += `#job/${jobId}/run/${jobRunId}`;
        res = (
          <a title={res} href={url} target='_top'>
            {res}
          </a>
        );
      }
      return res;
    } else {
      return res;
    }
  }

  /**
   * Returns an svg with some styling applied.
   */
  static renderSourceTypeIcon(tags) {
    const imageStyle = {
      height: '20px',
      position: 'relative',
      top: '-3px',
      marginRight: '4px',
    };
    const sourceType = this.getSourceType(tags);
    if (sourceType === 'NOTEBOOK') {
      if (Utils.getNotebookRevisionId(tags)) {
        return (
          <img
            alt=''
            title='Notebook Revision'
            style={{ ...imageStyle, top: '0px' }}
            src={revisionSvg}
          />
        );
      } else {
        return <img alt='' title='Notebook' style={imageStyle} src={notebookSvg} />;
      }
    } else if (sourceType === 'LOCAL') {
      return <img alt='' title='Local Source' style={imageStyle} src={laptopSvg} />;
    } else if (sourceType === 'PROJECT') {
      return <img alt='' title='Project' style={imageStyle} src={projectSvg} />;
    } else if (sourceType === 'JOB') {
      return <img alt='' title='Job' style={imageStyle} src={jobSvg} />;
    }
    return <img alt='' style={imageStyle} src={emptySvg} />;
  }

  /**
   * Renders the source name and entry point into a string. Used for sorting.
   * @param run MlflowMessages.RunInfo
   */
  static formatSource(tags) {
    const sourceName = Utils.getSourceName(tags);
    const sourceType = Utils.getSourceType(tags);
    const entryPointName = Utils.getEntryPointName(tags);
    if (sourceType === 'PROJECT') {
      let res = Utils.dropExtension(Utils.baseName(sourceName));
      if (entryPointName && entryPointName !== 'main') {
        res += ':' + entryPointName;
      }
      return res;
    } else if (sourceType === 'JOB') {
      const jobIdTag = 'mlflow.databricks.jobID';
      const jobRunIdTag = 'mlflow.databricks.jobRunID';
      const jobId = tags && tags[jobIdTag] && tags[jobIdTag].value;
      const jobRunId = tags && tags[jobRunIdTag] && tags[jobRunIdTag].value;
      if (jobId && jobRunId) {
        return `run ${jobRunId} of job ${jobId}`;
      }
      return sourceName;
    } else {
      return Utils.baseName(sourceName);
    }
  }

  /**
   * Renders the run name into a string.
   * @param runTags Object of tag name to MlflowMessages.RunTag instance
   */
  static getRunDisplayName(runTags, runUuid) {
    return Utils.getRunName(runTags) || 'Run ' + runUuid;
  }

  static getRunName(runTags) {
    const runNameTag = runTags[Utils.runNameTag];
    if (runNameTag) {
      return runNameTag.value;
    }
    return '';
  }

  static getSourceName(runTags) {
    const sourceNameTag = runTags[Utils.sourceNameTag];
    if (sourceNameTag) {
      return sourceNameTag.value;
    }
    return '';
  }

  static getSourceType(runTags) {
    const sourceTypeTag = runTags[Utils.sourceTypeTag];
    if (sourceTypeTag) {
      return sourceTypeTag.value;
    }
    return '';
  }

  static getSourceVersion(runTags) {
    const gitCommitTag = runTags[Utils.gitCommitTag];
    if (gitCommitTag) {
      return gitCommitTag.value;
    }
    return '';
  }

  static getEntryPointName(runTags) {
    const entryPointTag = runTags[Utils.entryPointTag];
    if (entryPointTag) {
      return entryPointTag.value;
    }
    return '';
  }

  static getBackend(runTags) {
    const backendTag = runTags[Utils.backendTag];
    if (backendTag) {
      return backendTag.value;
    }
    return '';
  }

  // TODO(aaron) Remove runInfo when user_id deprecation is complete.
  static getUser(runInfo, runTags) {
    const userTag = runTags[Utils.userTag];
    if (userTag) {
      return userTag.value;
    }
    return runInfo.user_id;
  }

  static renderVersion(tags, shortVersion = true) {
    const sourceVersion = Utils.getSourceVersion(tags);
    const sourceName = Utils.getSourceName(tags);
    const sourceType = Utils.getSourceType(tags);
    if (sourceVersion) {
      const versionString = shortVersion ? sourceVersion.substring(0, 6) : sourceVersion;
      if (sourceType === 'PROJECT') {
        const url = Utils.getGitCommitUrl(sourceName, sourceVersion);
        if (url) {
          return (
            <a href={url} target='_top'>
              {versionString}
            </a>
          );
        }
        return versionString;
      } else {
        return versionString;
      }
    }
    return null;
  }

  static pluralize(word, quantity) {
    if (quantity > 1) {
      return word + 's';
    } else {
      return word;
    }
  }

  static getRequestWithId(requests, requestId) {
    return requests.find((r) => r.id === requestId);
  }

  static getCurveKey(runId, metricName) {
    return `${runId}-${metricName}`;
  }

  static getCurveInfoFromKey(curvePair) {
    const splitPair = curvePair.split('-');
    return { runId: splitPair[0], metricName: splitPair.slice(1, splitPair.length).join('-') };
  }

  /**
   * Return metric plot state from the current URL
   *
   * The reverse transformation (from metric plot component state to URL) is exposed as a component
   * method, as it only needs to be called within the MetricsPlotPanel component
   *
   * See documentation in Routes.getMetricPageRoute for descriptions of the individual fields
   * within the returned state object.
   *
   * @param search - window.location.search component of the URL - in particular, the query string
   *   from the URL.
   */
  static getMetricPlotStateFromUrl(search) {
    const defaultState = {
      selectedXAxis: 'relative',
      selectedMetricKeys: [],
      showPoint: false,
      yAxisLogScale: false,
      lineSmoothness: 1,
      layout: {},
    };
    const params = qs.parse(search.slice(1, search.length));
    if (!params) {
      return defaultState;
    }

    const selectedXAxis = params['x_axis'] || 'relative';
    const selectedMetricKeys =
      JSON.parse(params['plot_metric_keys']) || defaultState.selectedMetricKeys;
    const showPoint = params['show_point'] === 'true';
    const yAxisLogScale = params['y_axis_scale'] === 'log';
    const lineSmoothness = params['line_smoothness'] ? parseFloat(params['line_smoothness']) : 0;
    const layout = params['plot_layout'] ? JSON.parse(params['plot_layout']) : { autosize: true };
    // Default to displaying all runs, i.e. to deselectedCurves being empty
    const deselectedCurves = params['deselected_curves']
      ? JSON.parse(params['deselected_curves'])
      : [];
    const lastLinearYAxisRange = params['last_linear_y_axis_range']
      ? JSON.parse(params['last_linear_y_axis_range'])
      : [];
    return {
      selectedXAxis,
      selectedMetricKeys,
      showPoint,
      yAxisLogScale,
      lineSmoothness,
      layout,
      deselectedCurves,
      lastLinearYAxisRange,
    };
  }

  static getPlotLayoutFromUrl(search) {
    const params = qs.parse(search);
    const layout = params['plot_layout'];
    return layout ? JSON.parse(layout) : {};
  }

  static getSearchParamsFromUrl(search) {
    const params = qs.parse(search, { ignoreQueryPrefix: true });
    const str = JSON.stringify(params, function replaceUndefined(key, value) {
      return value === undefined ? '' : value;
    });

    return params ? JSON.parse(str) : [];
  }

  static getSearchUrlFromState(state) {
    const replaced = {};
    for (const key in state) {
      if (state[key] === undefined) {
        replaced[key] = '';
      } else {
        replaced[key] = state[key];
      }
    }
    return qs.stringify(replaced);
  }

  static compareByTimestamp(history1, history2) {
    return history1.timestamp - history2.timestamp;
  }

  static compareByStepAndTimestamp(history1, history2) {
    const stepResult = history1.step - history2.step;
    return stepResult === 0 ? history1.timestamp - history2.timestamp : stepResult;
  }

  static getVisibleTagValues(tags) {
    // Collate tag objects into list of [key, value] lists and filter MLflow-internal tags
    return Object.values(tags)
      .map((t) => [t.getKey(), t.getValue()])
      .filter((t) => !t[0].startsWith(MLFLOW_INTERNAL_PREFIX));
  }

  static getVisibleTagKeyList(tagsList) {
    return _.uniq(
      _.flatMap(tagsList, (tags) => Utils.getVisibleTagValues(tags).map(([key]) => key)),
    );
  }

  static getLoggedModelsFromTags(tags) {
    const modelsTag = tags[Utils.loggedModelsTag];
    if (modelsTag) {
      const models = JSON.parse(modelsTag.value);
      if (models) {
        // extract artifact path, flavors and creation time from tag.
        // 'python_function' should be interpreted as pyfunc flavor
        const filtered = models.map((model) => {
          const removeFunc = Object.keys(_.omit(model.flavors, 'python_function'));
          const flavors = removeFunc.length ? removeFunc : ['pyfunc'];
          return {
            artifact_path: model.artifact_path,
            flavors: flavors,
            utc_time_created: new Date(model.utc_time_created).getTime() / 1000,
          };
        });
        // sort in descending order of creation time
        const sorted = filtered.sort(
          (a, b) => parseFloat(b.utc_time_created) - parseFloat(a.utc_time_created),
        );
        return _.uniqWith(sorted, (a, b) => a.artifact_path === b.artifact_path);
      }
    }
    return [];
  }

  static getAjaxUrl(relativeUrl) {
    if (process.env.USE_ABSOLUTE_AJAX_URLS === 'true') {
      return '/' + relativeUrl;
    }
    return relativeUrl;
  }

  static logErrorAndNotifyUser(e) {
    console.error(e);
    // not all error is wrapped by ErrorWrapper
    if (e.renderHttpError) {
      message.error(e.renderHttpError());
    }
  }

  static isModelRegistryEnabled() {
    return true;
  }

  static updatePageTitle(title) {
    window.parent.postMessage(
      {
        // Please keep this type name in sync with PostMessage.js
        type: 'UPDATE_TITLE',
        title,
      },
      window.parent.location.origin,
    );
  }

  /**
   * Check if current browser tab is the visible tab.
   * More info about document.visibilityState:
   * https://developer.mozilla.org/en-US/docs/Web/API/Document/visibilityState
   * @returns {boolean}
   */
  static isBrowserTabVisible() {
    return document.visibilityState !== 'hidden';
  }

  static shouldRender404(requests, requestIdsToCheck) {
    const requestsToCheck = requests.filter((request) => requestIdsToCheck.includes(request.id));
    return requestsToCheck.some((request) => {
      const { error } = request;
      return error && error.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST;
    });
  }

  static compareExperiments(a, b) {
    const aId = typeof a.getExperimentId === 'function' ? a.getExperimentId() : a.experiment_id;
    const bId = typeof b.getExperimentId === 'function' ? b.getExperimentId() : b.experiment_id;

    const aIntId = parseInt(aId, 10);
    const bIntId = parseInt(bId, 10);

    if (Number.isNaN(aIntId)) {
      if (!Number.isNaN(bIntId)) {
        // Int IDs before anything else
        return 1;
      }
    } else if (Number.isNaN(bIntId)) {
      // Int IDs before anything else
      return -1;
    } else {
      return aIntId - bIntId;
    }

    return aId.localeCompare(bId);
  }

  static getSupportPageUrl = () => SupportPageUrl;
}

export default Utils;

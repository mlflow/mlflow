/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

// @ts-expect-error TS(7016): Could not find a declaration file for module 'date... Remove this comment to see the full error message
import dateFormat from 'dateformat';
import React from 'react';
import notebookSvg from '../static/notebook.svg';
import revisionSvg from '../static/revision.svg';
import emptySvg from '../static/empty.svg';
import laptopSvg from '../static/laptop.svg';
import projectSvg from '../static/project.svg';
import workflowsIconSvg from '../static/WorkflowsIcon.svg';
import qs from 'qs';
import { MLFLOW_INTERNAL_PREFIX } from './TagUtils';
import _ from 'lodash';
import { ErrorCodes, SupportPageUrl } from '../constants';
import { FormattedMessage } from 'react-intl';
import { ErrorWrapper } from './ErrorWrapper';
import { shouldUsePathRouting } from './FeatureUtils';

class Utils {
  /**
   * Merge a runs parameters / metrics.
   * @param runUuids - A list of Run UUIDs.
   * @param keyValueList - A list of objects. One object for each run.
   * @retuns A key to a map of (runUuid -> value)
   */
  static mergeRuns(runUuids: any, keyValueList: any) {
    const ret = {};
    keyValueList.forEach((keyValueObj: any, i: any) => {
      const curRunUuid = runUuids[i];
      Object.keys(keyValueObj).forEach((key) => {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        const cur = ret[key] || {};
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        ret[key] = {
          ...cur,
          [curRunUuid]: keyValueObj[key],
        };
      });
    });
    return ret;
  }

  /**
   * Notifications API object compatible with @databricks/design-system's Notifications.
   * Used to display global errors.
   */
  static #notificationsApi = null;

  /**
   * Method used to register notifications API instance
   */
  static registerNotificationsApi(api: any) {
    Utils.#notificationsApi = api;
  }

  /**
   * Displays the error notification in the UI.
   */
  static displayGlobalErrorNotification(content: any, duration: any) {
    if (!Utils.#notificationsApi) {
      return;
    }
    (Utils.#notificationsApi as any).error({ message: content, duration: duration });
  }

  static runNameTag = 'mlflow.runName';
  static sourceNameTag = 'mlflow.source.name';
  static sourceTypeTag = 'mlflow.source.type';
  static gitCommitTag = 'mlflow.source.git.commit';
  static entryPointTag = 'mlflow.project.entryPoint';
  static backendTag = 'mlflow.project.backend';
  static userTag = 'mlflow.user';
  static loggedModelsTag = 'mlflow.log-model.history';
  static pipelineProfileNameTag = 'mlflow.pipeline.profile.name';
  static pipeLineStepNameTag = 'mlflow.pipeline.step.name';

  static formatMetric(value: any) {
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
  static truncateString(string: any, maxLength: any) {
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
  static convertTimestampToInt(metrics: any) {
    return metrics.map((metric: any) => {
      return {
        ...metric,
        timestamp: Number.parseFloat(metric.timestamp),
      };
    });
  }

  /**
   * Format timestamps from millisecond epoch time.
   */
  static formatTimestamp(timestamp: any, format = 'yyyy-mm-dd HH:MM:ss') {
    if (timestamp === undefined) {
      return '(unknown)';
    }
    const d = new Date(0);
    d.setUTCMilliseconds(timestamp);
    return dateFormat(d, format);
  }

  static timeSinceStr(date: any, referenceDate = new Date()) {
    // @ts-expect-error TS(2362): The left-hand side of an arithmetic operation must... Remove this comment to see the full error message
    const seconds = Math.max(0, Math.floor((referenceDate - date) / 1000));
    let interval = Math.floor(seconds / 31536000);

    if (interval >= 1) {
      return (
        <FormattedMessage
          defaultMessage='{timeSince, plural, =1 {1 year} other {# years}} ago'
          description='Text for time in years since given date for MLflow views'
          values={{ timeSince: interval }}
        />
      );
    }
    interval = Math.floor(seconds / 2592000);
    if (interval >= 1) {
      return (
        <FormattedMessage
          defaultMessage='{timeSince, plural, =1 {1 month} other {# months}} ago'
          description='Text for time in months since given date for MLflow views'
          values={{ timeSince: interval }}
        />
      );
    }
    interval = Math.floor(seconds / 86400);
    if (interval >= 1) {
      return (
        <FormattedMessage
          defaultMessage='{timeSince, plural, =1 {1 day} other {# days}} ago'
          description='Text for time in days since given date for MLflow views'
          values={{ timeSince: interval }}
        />
      );
    }
    interval = Math.floor(seconds / 3600);
    if (interval >= 1) {
      return (
        <FormattedMessage
          defaultMessage='{timeSince, plural, =1 {1 hour} other {# hours}} ago'
          description='Text for time in hours since given date for MLflow views'
          values={{ timeSince: interval }}
        />
      );
    }
    interval = Math.floor(seconds / 60);
    if (interval >= 1) {
      return (
        <FormattedMessage
          defaultMessage='{timeSince, plural, =1 {1 minute} other {# minutes}} ago'
          description='Text for time in minutes since given date for MLflow views'
          values={{ timeSince: interval }}
        />
      );
    }
    return (
      <FormattedMessage
        defaultMessage='{timeSince, plural, =1 {1 second} other {# seconds}} ago'
        description='Text for time in seconds since given date for MLflow views'
        values={{ timeSince: seconds }}
      />
    );
  }

  /**
   * Format a duration given in milliseconds.
   *
   * @param duration in milliseconds
   */
  static formatDuration(duration: any) {
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

  /**
   * Get the duration of a run given start- and end time.
   *
   * @param startTime in milliseconds
   * @param endTime in milliseconds
   */
  static getDuration(startTime: any, endTime: any) {
    return startTime && endTime ? Utils.formatDuration(endTime - startTime) : null;
  }

  static baseName(path: any) {
    const pieces = path.split('/');
    return pieces[pieces.length - 1];
  }

  static dropExtension(path: any) {
    return path.replace(/(.*[^/])\.[^/.]+$/, '$1');
  }

  /**
   * Normalizes a URI, removing redundant slashes and trailing slashes
   * For example, normalize("foo://bar///baz/") === "foo://bar/baz"
   */
  static normalize(uri: any) {
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

  /**
   * Regular expression for URLs containing the string 'git'.
   * It can be a custom git domain (e.g. https://git.custom.in/repo/dir#file/dir).
   * Excluding the first overall match, there are three groups:
   *    git url, repo directory, and file directory.
   * (e.g. group1: https://custom.git.domain, group2: repo/directory, group3: project/directory)
   */
  static getGitRegex() {
    return /(.*?[@/][^?]*git.*?)[:/]([^#]+)(?:#(.*))?/;
  }

  static getGitRepoUrl(sourceName: any) {
    const gitHubMatch = sourceName.match(Utils.getGitHubRegex());
    const gitLabMatch = sourceName.match(Utils.getGitLabRegex());
    const bitbucketMatch = sourceName.match(Utils.getBitbucketRegex());
    const gitMatch = sourceName.match(Utils.getGitRegex());
    let url = null;
    if (gitHubMatch) {
      url = `https://github.com/${gitHubMatch[1]}/${gitHubMatch[2].replace(/.git/, '')}`;
      if (gitHubMatch[3]) {
        url += `/tree/master/${gitHubMatch[3]}`;
      }
    } else if (gitLabMatch) {
      url = `https://gitlab.com/${gitLabMatch[1]}/${gitLabMatch[2].replace(/.git/, '')}`;
      if (gitLabMatch[3]) {
        url += `/-/tree/master/${gitLabMatch[3]}`;
      }
    } else if (bitbucketMatch) {
      url = `https://bitbucket.org/${bitbucketMatch[1]}/${bitbucketMatch[2].replace(/.git/, '')}`;
      if (bitbucketMatch[3]) {
        url += `/src/master/${bitbucketMatch[3]}`;
      }
    } else if (gitMatch) {
      const [, baseUrl, repoDir, fileDir] = gitMatch;
      url = baseUrl.replace(/git@/, 'https://') + '/' + repoDir.replace(/.git/, '');
      if (fileDir) {
        url += `/tree/master/${fileDir}`;
      }
    }
    return url;
  }

  static getGitCommitUrl(sourceName: any, sourceVersion: any) {
    const gitHubMatch = sourceName.match(Utils.getGitHubRegex());
    const gitLabMatch = sourceName.match(Utils.getGitLabRegex());
    const bitbucketMatch = sourceName.match(Utils.getBitbucketRegex());
    const gitMatch = sourceName.match(Utils.getGitRegex());
    let url = null;
    if (gitHubMatch) {
      url = `https://github.com/${gitHubMatch[1]}/${gitHubMatch[2].replace(/.git/, '')}/tree/${sourceVersion}/${gitHubMatch[3]}`;
    } else if (gitLabMatch) {
      url = `https://gitlab.com/${gitLabMatch[1]}/${gitLabMatch[2].replace(/.git/, '')}/-/tree/${sourceVersion}/${gitLabMatch[3]}`;
    } else if (bitbucketMatch) {
      url = `https://bitbucket.org/${bitbucketMatch[1]}/${bitbucketMatch[2].replace(/.git/, '')}/src/${sourceVersion}/${bitbucketMatch[3]}`;
    } else if (gitMatch) {
      const [, baseUrl, repoDir, fileDir] = gitMatch;
      url = `${baseUrl.replace(/git@/, 'https://')}/${repoDir.replace(/.git/, '')}/tree/${sourceVersion}/${fileDir}`;
    }
    return url;
  }

  static getQueryParams = () => {
    return window.location && window.location.search ? window.location.search : '';
  };

  /**
   * Makes sure that the URL begins with correct scheme according
   * to RFC3986 [https://datatracker.ietf.org/doc/html/rfc3986#section-3.1]
   * It does not support slash-less schemes (e.g. news:abc, urn:anc).
   * @param url URL string like "my-mlflow-server.com/#/experiments/9" or
   *        "https://my-mlflow-server.com/#/experiments/9"
   * @param defaultScheme scheme to add if missing in the provided URL, defaults to "https"
   * @returns {string} the URL string with ensured default scheme
   */
  static ensureUrlScheme(url: any, defaultScheme = 'https') {
    // Falsy values should yield itself
    if (!url) return url;

    // Scheme-less URL with colon and dashes
    if (url.match(/^:\/\//i)) {
      return `${defaultScheme}${url}`;
    }

    // URL without scheme, colon nor dashes
    if (!url.match(/^[a-z1-9+-.]+:\/\//i)) {
      return `${defaultScheme}://${url}`;
    }

    // Pass-through for "correct" entries
    return url;
  }

  /**
   * Returns a copy of the provided URL with its query parameters set to `queryParams`.
   * @param url URL string like "http://my-mlflow-server.com/#/experiments/9.
   * @param queryParams Optional query parameter string like "?param=12345". Query params provided
   *        via this string will override existing query param values in `url`
   */
  static setQueryParams(url: any, queryParams: any) {
    // Using new URL() is the preferred way of constructing the URL object,
    // however according to [https://url.spec.whatwg.org/#constructors] it requires
    // providing the protocol. We're gracefully ensuring that the scheme exists here.
    const urlObj = new URL(Utils.ensureUrlScheme(url));
    urlObj.search = queryParams || '';
    return urlObj.toString();
  }

  /**
   * Set query params and returns the updated query params.
   * @returns {string} updated query params
   */
  static addQueryParams(currentQueryParams: any, newQueryParams: any) {
    if (!newQueryParams || Object.keys(newQueryParams).length === 0) {
      return currentQueryParams;
    }
    const urlSearchParams = new URLSearchParams(currentQueryParams);
    Object.entries(newQueryParams).forEach(
      // @ts-expect-error TS(2345): Argument of type 'unknown' is not assignable to pa... Remove this comment to see the full error message
      ([key, value]) => !!key && !!value && urlSearchParams.set(key, value),
    );
    const queryParams = urlSearchParams.toString();
    if (queryParams !== '' && !queryParams.includes('?')) {
      return `?${queryParams}`;
    }
    return queryParams;
  }

  static getDefaultJobRunName(jobId: any, runId: any, workspaceId = null) {
    if (!jobId) {
      return '-';
    }
    let name = `job ${jobId}`;
    if (runId) {
      name = `run ${runId} of ` + name;
    }
    if (workspaceId) {
      name = `workspace ${workspaceId}: ` + name;
    }
    return name;
  }

  static getDefaultNotebookRevisionName(notebookId: any, revisionId: any, workspaceId = null) {
    if (!notebookId) {
      return '-';
    }
    let name = `notebook ${notebookId}`;
    if (revisionId) {
      name = `revision ${revisionId} of ` + name;
    }
    if (workspaceId) {
      name = `workspace ${workspaceId}: ` + name;
    }
    return name;
  }

  static getNotebookId(tags: any) {
    const notebookIdTag = 'mlflow.databricks.notebookID';
    return tags && tags[notebookIdTag] && tags[notebookIdTag].value;
  }

  static getClusterSpecJson(tags: any) {
    const clusterSpecJsonTag = 'mlflow.databricks.cluster.info';
    return tags && tags[clusterSpecJsonTag] && tags[clusterSpecJsonTag].value;
  }

  static getClusterLibrariesJson(tags: any) {
    const clusterLibrariesJsonTag = 'mlflow.databricks.cluster.libraries';
    return tags && tags[clusterLibrariesJsonTag] && tags[clusterLibrariesJsonTag].value;
  }

  static getClusterId(tags: any) {
    const clusterIdTag = 'mlflow.databricks.cluster.id';
    return tags && tags[clusterIdTag] && tags[clusterIdTag].value;
  }

  static getNotebookRevisionId(tags: any) {
    const revisionIdTag = 'mlflow.databricks.notebookRevisionID';
    return tags && tags[revisionIdTag] && tags[revisionIdTag].value;
  }

  /**
   * Renders the source name and entry point into an HTML element. Used for display.
   * @param tags Object containing tag key value pairs.
   * @param queryParams Query params to add to certain source type links.
   * @param runUuid ID of the MLflow run to add to certain source (revision) links.
   */
  static renderSource(tags: any, queryParams: any, runUuid: any) {
    const sourceName = Utils.getSourceName(tags);
    let res = Utils.formatSource(tags);
    const gitRepoUrlOrNull = Utils.getGitRepoUrl(sourceName);
    if (gitRepoUrlOrNull) {
      res = (
        <a target='_top' href={gitRepoUrlOrNull}>
          {res}
        </a>
      );
    }
    return res;
  }

  /**
   * Renders the notebook source name and entry point into an HTML element. Used for display.
   */
  static renderNotebookSource(
    queryParams: any,
    notebookId: any,
    revisionId: any,
    runUuid: any,
    sourceName: any,
    workspaceUrl = null,
    nameOverride = null,
  ) {
    // sourceName may not be present when rendering feature table notebook consumers from remote
    // workspaces or when notebook fetcher failed to fetch the sourceName. Always provide a default
    // notebook name in such case.
    const baseName = sourceName
      ? Utils.baseName(sourceName)
      : Utils.getDefaultNotebookRevisionName(notebookId, revisionId);
    const name = nameOverride || baseName;

    if (notebookId) {
      const url = Utils.getNotebookSourceUrl(
        queryParams,
        notebookId,
        revisionId,
        runUuid,
        workspaceUrl,
      );
      return (
        <a
          title={sourceName || Utils.getDefaultNotebookRevisionName(notebookId, revisionId)}
          href={url}
          target='_top'
        >
          {name}
        </a>
      );
    } else {
      return name;
    }
  }

  /**
   * Returns the URL for the notebook source.
   */
  static getNotebookSourceUrl(
    queryParams: any,
    notebookId: any,
    revisionId: any,
    runUuid: any,
    workspaceUrl = null,
  ) {
    let url = Utils.setQueryParams(workspaceUrl || window.location.origin, queryParams);
    url += `#notebook/${notebookId}`;
    if (revisionId) {
      url += `/revision/${revisionId}`;
      if (runUuid) {
        url += `/mlflow/run/${runUuid}`;
      }
    }
    return url;
  }

  /**
   * Renders the job source name and entry point into an HTML element. Used for display.
   */
  static renderJobSource(
    queryParams: any,
    jobId: any,
    jobRunId: any,
    jobName: any,
    workspaceUrl = null,
    nameOverride = null,
  ) {
    // jobName may not be present when rendering feature table job consumers from remote
    // workspaces or when getJob API failed to fetch the jobName. Always provide a default
    // job name in such case.
    const reformatJobName = jobName || Utils.getDefaultJobRunName(jobId, jobRunId);
    const name = nameOverride || reformatJobName;

    if (jobId) {
      const url = Utils.getJobSourceUrl(queryParams, jobId, jobRunId, workspaceUrl);
      return (
        <a title={reformatJobName} href={url} target='_top'>
          {name}
        </a>
      );
    } else {
      return name;
    }
  }

  /**
   * Returns the URL for the job source.
   */
  static getJobSourceUrl(queryParams: any, jobId: any, jobRunId: any, workspaceUrl = null) {
    let url = Utils.setQueryParams(workspaceUrl || window.location.origin, queryParams);
    url += `#job/${jobId}`;
    if (jobRunId) {
      url += `/run/${jobRunId}`;
    }
    return url;
  }

  /**
   * Returns an svg with some styling applied.
   */
  static renderSourceTypeIcon(tags: any) {
    const imageStyle = {
      height: '20px',
      marginRight: '4px',
    };

    const sourceType = Utils.getSourceType(tags);
    if (sourceType === 'NOTEBOOK') {
      if (Utils.getNotebookRevisionId(tags)) {
        return (
          <img
            alt='Notebook Revision Icon'
            title='Notebook Revision'
            style={imageStyle}
            src={revisionSvg}
          />
        );
      } else {
        return <img alt='Notebook Icon' title='Notebook' style={imageStyle} src={notebookSvg} />;
      }
    } else if (sourceType === 'LOCAL') {
      return (
        <img alt='Local Source Icon' title='Local Source' style={imageStyle} src={laptopSvg} />
      );
    } else if (sourceType === 'PROJECT') {
      return <img alt='Project Icon' title='Project' style={imageStyle} src={projectSvg} />;
    } else if (sourceType === 'JOB') {
      return <img alt='Job Icon' title='Job' style={imageStyle} src={workflowsIconSvg} />;
    }
    return <img alt='No icon' style={imageStyle} src={emptySvg} />;
  }

  /**
   * Renders the source name and entry point into a string. Used for sorting.
   * @param run MlflowMessages.RunInfo
   */
  static formatSource(tags: any) {
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
        return Utils.getDefaultJobRunName(jobId, jobRunId);
      }
      return sourceName;
    } else {
      return Utils.baseName(sourceName);
    }
  }

  /**
   * Returns the absolute path to a notebook given a notebook id
   * @param notebookId Notebook object id
   * @returns
   */
  static getNotebookLink(notebookId: any) {
    return window.location.origin + '/#notebook/' + notebookId;
  }

  /**
   * Renders the run name into a string.
   * @param runTags Object of tag name to MlflowMessages.RunTag instance
   */
  static getRunDisplayName(runInfo: any, runUuid: any) {
    return Utils.getRunName(runInfo) || 'Run ' + runUuid;
  }

  static getRunName(runInfo: any) {
    return runInfo.run_name || '';
  }

  static getRunNameFromTags(runTags: any) {
    const runNameTag = runTags[Utils.runNameTag];
    if (runNameTag) {
      return runNameTag.value;
    }
    return '';
  }

  static getSourceName(runTags: any) {
    const sourceNameTag = runTags[Utils.sourceNameTag];
    if (sourceNameTag) {
      return sourceNameTag.value;
    }
    return '';
  }

  static getSourceType(runTags: any) {
    const sourceTypeTag = runTags[Utils.sourceTypeTag];
    if (sourceTypeTag) {
      return sourceTypeTag.value;
    }
    return '';
  }

  static getSourceVersion(runTags: any) {
    const gitCommitTag = runTags[Utils.gitCommitTag];
    if (gitCommitTag) {
      return gitCommitTag.value;
    }
    return '';
  }

  static getPipelineProfileName(runTags: any) {
    const tag = runTags[Utils.pipelineProfileNameTag];
    if (tag) {
      return tag.value;
    }
    return '';
  }

  static getPipelineStepName(runTags: any) {
    const tag = runTags[Utils.pipeLineStepNameTag];
    if (tag) {
      return tag.value;
    }
    return '';
  }

  static getEntryPointName(runTags: any) {
    const entryPointTag = runTags[Utils.entryPointTag];
    if (entryPointTag) {
      return entryPointTag.value;
    }
    return '';
  }

  static getBackend(runTags: any) {
    const backendTag = runTags[Utils.backendTag];
    if (backendTag) {
      return backendTag.value;
    }
    return '';
  }

  // TODO(aaron) Remove runInfo when user_id deprecation is complete.
  static getUser(runInfo: any, runTags: any) {
    const userTag = runTags[Utils.userTag];
    if (userTag) {
      return userTag.value;
    }
    return runInfo.user_id;
  }

  static renderVersion(tags: any, shortVersion = true) {
    const sourceVersion = Utils.getSourceVersion(tags);
    const sourceName = Utils.getSourceName(tags);
    const sourceType = Utils.getSourceType(tags);
    // prettier-ignore
    return Utils.renderSourceVersion(
      sourceVersion,
      sourceName,
      sourceType,
      shortVersion,
    );
  }

  // prettier-ignore
  static renderSourceVersion(
    sourceVersion: any,
    sourceName: any,
    sourceType: any,
    shortVersion = true,
  ) {
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

  static pluralize(word: any, quantity: any) {
    if (quantity > 1) {
      return word + 's';
    } else {
      return word;
    }
  }

  static getRequestWithId(requests: any, requestId: any) {
    return requests.find((r: any) => r.id === requestId);
  }

  static getCurveKey(runId: any, metricName: any) {
    return `${runId}-${metricName}`;
  }

  static getCurveInfoFromKey(curvePair: any) {
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
  static getMetricPlotStateFromUrl(search: any) {
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
      // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
      JSON.parse(params['plot_metric_keys']) || defaultState.selectedMetricKeys;
    const showPoint = params['show_point'] === 'true';
    const yAxisLogScale = params['y_axis_scale'] === 'log';
    // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
    const lineSmoothness = params['line_smoothness'] ? parseFloat(params['line_smoothness']) : 0;
    // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
    const layout = params['plot_layout'] ? JSON.parse(params['plot_layout']) : { autosize: true };
    // Default to displaying all runs, i.e. to deselectedCurves being empty
    const deselectedCurves = params['deselected_curves']
      ? // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
        JSON.parse(params['deselected_curves'])
      : [];
    const lastLinearYAxisRange = params['last_linear_y_axis_range']
      ? // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
        JSON.parse(params['last_linear_y_axis_range'])
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

  static getPlotLayoutFromUrl(search: any) {
    const params = qs.parse(search);
    const layout = params['plot_layout'];
    // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
    return layout ? JSON.parse(layout) : {};
  }

  static getSearchParamsFromUrl(search: any) {
    return qs.parse(search, {
      ignoreQueryPrefix: true,
      comma: true,
      arrayLimit: 500,
      decoder(str, defaultDecoder, charset, type) {
        if (type === 'value') {
          if (str === 'true') {
            return true;
          } else if (str === 'false') {
            return false;
          } else if (str === undefined) {
            return '';
          }
          return defaultDecoder(str);
        }
        return defaultDecoder(str);
      },
    });
  }

  static getSearchUrlFromState(state: any) {
    const replaced = {};
    for (const key in state) {
      if (state[key] === undefined) {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        replaced[key] = '';
      } else {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        replaced[key] = state[key];
      }
    }
    return qs.stringify(replaced, { arrayFormat: 'comma', encodeValuesOnly: true });
  }

  static compareByTimestamp(history1: any, history2: any) {
    return history1.timestamp - history2.timestamp;
  }

  static compareByStepAndTimestamp(history1: any, history2: any) {
    const stepResult = history1.step - history2.step;
    return stepResult === 0 ? history1.timestamp - history2.timestamp : stepResult;
  }

  static getVisibleTagValues(tags: any) {
    // Collate tag objects into list of [key, value] lists and filter MLflow-internal tags
    return Object.values(tags)
      .map((t) => [
        (t as any).key || (t as any).getKey(),
        (t as any).value || (t as any).getValue(),
      ])
      .filter((t) => !t[0].startsWith(MLFLOW_INTERNAL_PREFIX));
  }

  static getVisibleTagKeyList(tagsList: any) {
    return _.uniq(
      _.flatMap(tagsList, (tags) => Utils.getVisibleTagValues(tags).map(([key]) => key)),
    );
  }

  /**
   * Concat array with arrayToConcat and group by specified key 'id'.
   * if array==[{'theId': 123, 'a': 2}, {'theId': 456, 'b': 3}]
   * and arrayToConcat==[{'theId': 123, 'c': 3}, {'theId': 456, 'd': 4}]
   * then concatAndGroupArraysById(array, arrayToConcat, 'theId')
   * == [{'theId': 123, 'a': 2, 'c': 3}, {'theId': 456, 'b': 3, 'd': 4}].
   * From https://stackoverflow.com/a/38506572/13837474
   */
  static concatAndGroupArraysById(array: any, arrayToConcat: any, id: any) {
    return (
      _(array)
        .concat(arrayToConcat)
        .groupBy(id)
        // complication of _.merge necessary to avoid mutating arguments
        .map(_.spread((obj, source) => _.merge({}, obj, source)))
        .value()
    );
  }

  /**
   * Parses the mlflow.log-model.history tag and returns a list of logged models,
   * with duplicates (as defined by two logged models with the same path) removed by
   * keeping the logged model with the most recent creation date.
   * Each logged model will be of the form:
   * { artifactPath: string, flavors: string[], utcTimeCreated: number }
   */
  static getLoggedModelsFromTags(tags: any) {
    const modelsTag = tags[Utils.loggedModelsTag];
    if (modelsTag) {
      const models = JSON.parse(modelsTag.value);
      if (models) {
        // extract artifact path, flavors and creation time from tag.
        // 'python_function' should be interpreted as pyfunc flavor
        const filtered = models.map((model: any) => {
          const removeFunc = Object.keys(_.omit(model.flavors, 'python_function'));
          const flavors = removeFunc.length ? removeFunc : ['pyfunc'];
          return {
            artifactPath: model.artifact_path,
            flavors: flavors,
            utcTimeCreated: new Date(model.utc_time_created).getTime() / 1000,
          };
        });
        // sort in descending order of creation time
        const sorted = filtered.sort(
          (a: any, b: any) => parseFloat(b.utcTimeCreated) - parseFloat(a.utcTimeCreated),
        );
        return _.uniqWith(sorted, (a, b) => (a as any).artifactPath === (b as any).artifactPath);
      }
    }
    return [];
  }

  /**
   * Returns a list of models formed by merging the given logged models and registered models.
   * Sort such that models that are logged and registered come first, followed by
   * only registered models, followed by only logged models. Ties broken in favor of newer creation
   * time.
   * @param loggedModels
   * @param registeredModels Model versions by run uuid, from redux state.
   */
  static mergeLoggedAndRegisteredModels(loggedModels: any, registeredModels: any) {
    // use artifactPath for grouping while merging lists
    const registeredModelsWithNormalizedPath = registeredModels.map((model: any) => {
      const registeredModel: { [key: string]: any } = {
        registeredModelName: model.name,
        artifactPath: Utils.normalize(model.source).split('/artifacts/')[1],
        registeredModelVersion: model.version,
        registeredModelCreationTimestamp: model.creation_timestamp,
      };
      return registeredModel;
    });
    const loggedModelsWithNormalizedPath = loggedModels.flatMap((model: any) => {
      return model.artifactPath
        ? [{ ...model, artifactPath: Utils.normalize(model.artifactPath) }]
        : [];
    });
    const models = Utils.concatAndGroupArraysById(
      loggedModelsWithNormalizedPath,
      registeredModelsWithNormalizedPath,
      'artifactPath',
    );
    return models.sort((a, b) => {
      // @ts-expect-error TODO: fix this
      if (a.registeredModelVersion && b.registeredModelVersion) {
        // @ts-expect-error TODO: fix this
        if (a.flavors && !b.flavors) {
          return -1;
          // @ts-expect-error TODO: fix this
        } else if (!a.flavors && b.flavors) {
          return 1;
        } else {
          return (
            // @ts-expect-error TODO: fix this
            parseInt(b.registeredModelCreationTimestamp, 10) -
            // @ts-expect-error TODO: fix this
            parseInt(a.registeredModelCreationTimestamp, 10)
          );
        }
        // @ts-expect-error TODO: fix this
      } else if (a.registeredModelVersion && !b.registeredModelVersion) {
        return -1;
        // @ts-expect-error TODO: fix this
      } else if (!a.registeredModelVersion && b.registeredModelVersion) {
        return 1;
      }
      // @ts-expect-error TODO: fix this
      return b.utcTimeCreated - a.utcTimeCreated;
    });
  }

  static logErrorAndNotifyUser(
    // Prevent formatting after edge block removal
    // prettier-ignore
    e: any,
    duration = 3,
    passErrorToParentFrame = false,
  ) {
    console.error(e);
    if (typeof e === 'string') {
      Utils.displayGlobalErrorNotification(e, duration);
    } else if (e instanceof ErrorWrapper) {
      // not all error is wrapped by ErrorWrapper
      Utils.displayGlobalErrorNotification(e.renderHttpError(), duration);
      // eslint-disable-next-line no-empty
    } else {
    }
  }

  static logGenericUserFriendlyError(e: any, intl: any) {
    const errorMessages = {
      404: intl.formatMessage({
        defaultMessage: '404: Resource not found',
        description: 'Generic 404 user-friendly error for the MLflow UI',
      }),
      500: intl.formatMessage({
        defaultMessage: '500: Internal server error',
        description: 'Generic 500 user-friendly error for the MLflow UI',
      }),
    };

    if (
      e instanceof ErrorWrapper &&
      typeof intl === 'object' &&
      Object.keys(errorMessages).includes(e.getStatus().toString())
    ) {
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      return Utils.logErrorAndNotifyUser(errorMessages[e.getStatus()]);
    }

    return Utils.logErrorAndNotifyUser(e);
  }

  static sortExperimentsById = (experiments: any) => {
    return _.sortBy(experiments, [({ experiment_id }) => experiment_id]);
  };

  static getExperimentNameMap = (experiments: any) => {
    // Input:
    // [
    //  { experiment_id: 1, name: '/1/bar' },
    //  { experiment_id: 2, name: '/2/foo' },
    //  { experiment_id: 3, name: '/3/bar' },
    // ]
    //
    // Output:
    // {
    //   1: {name: '/1/bar', basename: 'bar (1)'},
    //   2: {name: '/2/foo', basename: 'foo'},
    //   3: {name: '/3/bar', basename: 'bar (2)'},
    // }
    const experimentsByBasename = {};
    experiments.forEach((experiment: any) => {
      const { name } = experiment;
      const basename = name.split('/').pop();
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      experimentsByBasename[basename] = [...(experimentsByBasename[basename] || []), experiment];
    });

    const idToNames = {};
    Object.entries(experimentsByBasename).forEach(([basename, exps]) => {
      const isUnique = (exps as any).length === 1;
      (exps as any).forEach(({ experiment_id, name }: any, index: any) => {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        idToNames[experiment_id] = {
          name,
          basename: isUnique ? basename : `${basename} (${index + 1})`,
        };
      });
    });

    return idToNames;
  };

  static isModelRegistryEnabled() {
    return true;
  }

  static updatePageTitle(title: any) {
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

  static shouldRender404(requests: any, requestIdsToCheck: any) {
    const requestsToCheck = requests.filter((request: any) =>
      requestIdsToCheck.includes(request.id),
    );
    return requestsToCheck.some((request: any) => {
      const { error } = request;
      return error && error.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST;
    });
  }

  static getResourceConflictError(requests: any, requestIdsToCheck: any) {
    const result = requests.filter((request: any) => {
      if (requestIdsToCheck.includes(request.id)) {
        const { error } = request;
        return error && error.getErrorCode() === ErrorCodes.RESOURCE_CONFLICT;
      }
      return false;
    });
    return result[0];
  }

  static compareExperiments(a: any, b: any) {
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

  static isUsingExternalRouter() {
    // Running inside the iFrame indicates that we're using externally managed routing.
    if (window.isTestingIframe) {
      return true;
    }

    return false;
  }

  static getIframeCorrectedRoute(route: any) {
    if (shouldUsePathRouting()) {
      // After enabling path routing, we don't need any hash splitting etc.
      return route;
    }
    if (Utils.isUsingExternalRouter()) {
      // If using external routing, include the parent params and assume mlflow served at #
      const parentHref = window.parent.location.href;
      const parentHrefBeforeMlflowHash = parentHref.split('#')[0];
      return `${parentHrefBeforeMlflowHash}#mlflow${route}`;
    }
    return `./#${route}`; // issue-2213 use relative path in case there is a url prefix
  }

  static isValidHttpUrl(str: any) {
    // The URL() constructor will throw on invalid URL
    try {
      const url = new URL(str);
      return url.protocol === 'http:' || url.protocol === 'https:';
    } catch (err) {
      return false;
    }
  }
}

export default Utils;

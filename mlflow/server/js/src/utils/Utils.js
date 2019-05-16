import dateFormat from 'dateformat';
import React from 'react';
import notebookSvg from '../static/notebook.svg';
import emptySvg from '../static/empty.svg';
import laptopSvg from '../static/laptop.svg';
import projectSvg from '../static/project.svg';
import qs from 'qs';

class Utils {
  /**
   * Merge a runs parameters / metrics.
   * @param runsUuids - A list of Run UUIDs.
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
          [curRunUuid]: keyValueObj[key]
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
  static userTag = 'mlflow.user';

  static formatMetric(value) {
    if (Math.abs(value) < 10) {
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
      return string.slice(0, maxLength - 3) + "...";
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

  /**
   * Format a duration given in milliseconds.
   *
   * @param duration in milliseconds
   */
  static formatDuration(duration) {
    if (duration < 500) {
      return duration + "ms";
    } else if (duration < 1000 * 60) {
      return (duration / 1000).toFixed(1) + "s";
    } else if (duration < 1000 * 60 * 60) {
      return (duration / 1000 / 60).toFixed(1) + "min";
    } else if (duration < 1000 * 60 * 60 * 24) {
      return (duration / 1000 / 60 / 60).toFixed(1) + "h";
    } else {
      return (duration / 1000 / 60 / 60 / 24).toFixed(1) + "d";
    }
  }

  static formatUser(userId) {
    return userId.replace(/@.*/, "");
  }

  static baseName(path) {
    const pieces = path.split("/");
    return pieces[pieces.length - 1];
  }

  static dropExtension(path) {
    return path.replace(/(.*[^/])\.[^/.]+$/, "$1");
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
      const baseUrl = gitHubMatch ? "https://github.com/" : "https://gitlab.com/";
      const match = gitHubMatch || gitLabMatch;
      url = baseUrl + match[1] + "/" + match[2].replace(/.git/, '');
      if (match[3]) {
        url = url + "/tree/master/" + match[3];
      }
    } else if (bitbucketMatch) {
      const baseUrl = "https://bitbucket.org/";
      url = baseUrl + bitbucketMatch[1] + "/" + bitbucketMatch[2].replace(/.git/, '');
      if (bitbucketMatch[3]) {
        url = url + "/src/master/" + bitbucketMatch[3];
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
      const baseUrl = gitHubMatch ? "https://github.com/" : "https://gitlab.com/";
      const match = gitHubMatch || gitLabMatch;
      url = (baseUrl + match[1] + "/" + match[2].replace(/.git/, '') +
            "/tree/" + sourceVersion) + "/" + match[3];
    } else if (bitbucketMatch) {
      const baseUrl = "https://bitbucket.org/";
      url = (baseUrl + bitbucketMatch[1] + "/" + bitbucketMatch[2].replace(/.git/, '') +
            "/src/" + sourceVersion) + "/" + bitbucketMatch[3];
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
    urlObj.search = queryParams || "";
    return urlObj.toString();
  }

  /**
   * Renders the source name and entry point into an HTML element. Used for display.
   * @param tags Object containing tag key value pairs.
   * @param queryParams Query params to add to certain source type links.
   */
  static renderSource(tags, queryParams) {
    const sourceName = Utils.getSourceName(tags);
    const sourceType = Utils.getSourceType(tags);
    let res = Utils.formatSource(tags);
    if (sourceType === "PROJECT") {
      const url = Utils.getGitRepoUrl(sourceName);
      if (url) {
        res = <a target="_top" href={url}>{res}</a>;
      }
      return res;
    } else if (sourceType === "NOTEBOOK") {
      const revisionIdTag = 'mlflow.databricks.notebookRevisionID';
      const notebookIdTag = 'mlflow.databricks.notebookID';
      const revisionId = tags && tags[revisionIdTag] && tags[revisionIdTag].value;
      const notebookId = tags && tags[notebookIdTag] && tags[notebookIdTag].value;
      if (notebookId) {
        let url = Utils.setQueryParams(window.location.origin, queryParams);
        url += `#notebook/${notebookId}`;
        if (revisionId) {
          url += `/revision/${revisionId}`;
        }
        res = (<a title={sourceName} href={url} target='_top'>
          {Utils.baseName(sourceName)}
        </a>);
      }
      return res;
    } else {
      return res;
    }
  }

  /**
   * Returns an svg with some styling applied.
   */
  static renderSourceTypeIcon(sourceType) {
    const imageStyle = {
      height: '20px',
      position: 'relative',
      top: '-1px',
      marginRight: '2px',
    };
    if (sourceType === "NOTEBOOK") {
      return <img title="Notebook" style={imageStyle} src={notebookSvg} />;
    } else if (sourceType === "LOCAL") {
      return <img title="Local Source" style={imageStyle} src={laptopSvg} />;
    } else if (sourceType === "PROJECT") {
      return <img title="Project" style={imageStyle} src={projectSvg} />;
    }
    return <img style={imageStyle} src={emptySvg} />;
  }

  /**
   * Renders the source name and entry point into a string. Used for sorting.
   * @param run MlflowMessages.RunInfo
   */
  static formatSource(tags) {
    const sourceName = Utils.getSourceName(tags);
    const entryPointName = Utils.getEntryPointName(tags);
    if (Utils.getSourceType(tags) === "PROJECT") {
      let res = Utils.dropExtension(Utils.baseName(sourceName));
      if (entryPointName && entryPointName !== "main") {
        res += ":" + entryPointName;
      }
      return res;
    } else {
      return Utils.baseName(sourceName);
    }
  }

  /**
   * Renders the run name into a string.
   * @param runTags Object of tag name to MlflowMessages.RunTag instance
   */
  static getRunDisplayName(runTags, runUuid) {
    return Utils.getRunName(runTags) || "Run " + runUuid;
  }

  static getRunName(runTags) {
    const runNameTag = runTags[Utils.runNameTag];
    if (runNameTag) {
      return runNameTag.value;
    }
    return "";
  }

  static getSourceName(runTags) {
    const sourceNameTag = runTags[Utils.sourceNameTag];
    if (sourceNameTag) {
      return sourceNameTag.value;
    }
    return "";
  }

  static getSourceType(runTags) {
    const sourceTypeTag = runTags[Utils.sourceTypeTag];
    if (sourceTypeTag) {
      return sourceTypeTag.value;
    }
    return "";
  }

  static getSourceVersion(runTags) {
    const gitCommitTag = runTags[Utils.gitCommitTag];
    if (gitCommitTag) {
      return gitCommitTag.value;
    }
    return "";
  }

  static getEntryPointName(runTags) {
    const entryPointTag = runTags[Utils.entryPointTag];
    if (entryPointTag) {
      return entryPointTag.value;
    }
    return "";
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
      if (sourceType === "PROJECT") {
        const url = Utils.getGitCommitUrl(sourceName, sourceVersion);
        if (url) {
          return <a href={url} target='_top'>{versionString}</a>;
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

  static getPlotMetricKeysFromUrl(search) {
    const params = qs.parse(search);
    const plotMetricKeysStr = params && params['plot_metric_keys'];
    return plotMetricKeysStr ? JSON.parse(plotMetricKeysStr) : [];
  }

  static compareByTimestamp(history1, history2) {
    return history1.timestamp - history2.timestamp;
  }

  static compareByStepAndTimestamp(history1, history2) {
    const stepResult = history1.step - history2.step;
    return stepResult === 0 ? (history1.timestamp - history2.timestamp) : stepResult;
  }
}

export default Utils;

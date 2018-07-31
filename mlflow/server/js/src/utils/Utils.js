import dateFormat from 'dateformat';
import React from 'react';

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
      }
    })
  }

  /**
   * Format timestamps from millisecond epoch time.
   */
  static formatTimestamp(timestamp) {
    const d = new Date(0);
    d.setUTCMilliseconds(timestamp);
    return dateFormat(d, "yyyy-mm-dd HH:MM:ss");
  }

  /**
   * Format a duration given in milliseconds.
   *
   * @param duration in milliseconds
   */
  static formatDuration(duration) {
    if (duration < 500) {
      return duration + "ms"
    } else if (duration < 1000 * 60) {
      return (duration / 1000).toFixed(1) + "s"
    } else if (duration < 1000 * 60 * 60) {
      return (duration / 1000 / 60).toFixed(1) + "min"
    } else if (duration < 1000 * 60 * 60 * 24) {
      return (duration / 1000 / 60 / 60).toFixed(1) + "h"
    } else {
      return (duration / 1000 / 60 / 60 / 24).toFixed(1) + "d"
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
    return path.replace(/(.*[^/])\.[^/.]+$/, "$1")
  }

  static renderSource(run) {
    if (run.source_type === "PROJECT") {
      let res = Utils.dropExtension(Utils.baseName(run.source_name));
      if (run.entry_point && run.entry_point !== "main") {
        res += ":" + run.entry_point;
      }
      const GITHUB_RE = /[:@]github.com[:/]([^/.]+)\/([^/.]+)/;
      const match = run.source_name.match(GITHUB_RE);
      if (match) {
        const url = "https://github.com/" + match[1] + "/" + match[2];
        res = <a href={url}>{res}</a>;
      }
      return res;
    } else {
      return Utils.baseName(run.source_name);
    }
  }

  static renderVersion(run) {
    if (run.source_version) {
      const shortVersion = run.source_version.substring(0, 6);
      if (run.source_type === "PROJECT") {
        const GITHUB_RE = /[:@]github.com[:/]([^/.]+)\/([^/.]+)/;
        const match = run.source_name.match(GITHUB_RE);
        if (match) {
          const url = "https://github.com/" + match[1] + "/" + match[2] + "/tree/" + run.source_version;
          return <a href={url}>{shortVersion}</a>;
        }
      } else {
        return shortVersion;
      }
    }
    return null;
  }
}

export default Utils;
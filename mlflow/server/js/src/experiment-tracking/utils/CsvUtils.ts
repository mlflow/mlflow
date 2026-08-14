import { compact, uniq } from 'lodash';
import Utils from '../../common/utils/Utils';
import type { RunsChartsRunData } from '../components/runs-charts/components/RunsCharts.common';
import type { MetricEntity, RunInfoEntity } from '../types';
import type { KeyValueEntity } from '../../common/types';
import moment from 'moment';

const { getDuration, getRunNameFromTags, getSourceType, getSourceName, getUser } = Utils;

/**
 * Turn a list of params/metrics to a map of metric key to metric.
 */
const toMap = <T extends MetricEntity | KeyValueEntity>(params: T[]) =>
  params.reduce((result, entity) => ({ ...result, [entity.key]: entity }), {} as Record<string, T>);

/**
 * Escapes a string for safe inclusion in a CSV file to prevent CSV injection attacks.
 * See https://owasp.org/www-community/attacks/CSV_Injection for more information.
 */
const csvEscape = (x: any) => {
  if (x === null || x === undefined) {
    return '';
  }
  let sanitized = typeof x === 'string' ? x : x.toString();

  // Escape double quotes by doubling them
  sanitized = sanitized.replace(/"/g, '""');

  // If the string starts with a character that could be interpreted as a formula, escape it
  // by prepending a single quote
  if (/^[=+\-@\r\t]/.test(sanitized)) {
    sanitized = `'${sanitized}`;
  }

  return `"${sanitized}"`;
};

/**
 * Convert a table to a CSV string.
 *
 * @param columns Names of columns
 * @param data Array of rows, each of which are an array of field values
 */
const tableToCsv = (columns: any /* TODO */, data: any /* TODO */) => {
  let csv = '';
  let i;

  for (i = 0; i < columns.length; i++) {
    csv += csvEscape(columns[i]);
    if (i < columns.length - 1) {
      csv += ',';
    }
  }
  csv += '\n';

  for (i = 0; i < data.length; i++) {
    for (let j = 0; j < data[i].length; j++) {
      csv += csvEscape(data[i][j]);
      if (j < data[i].length - 1) {
        csv += ',';
      }
    }
    csv += '\n';
  }

  return csv;
};

/**
 * Convert an array of run infos to a CSV string, extracting the params and metrics in the
 * provided lists.
 */
export const runInfosToCsv = (params: {
  runInfos: RunInfoEntity[];
  paramKeyList: string[];
  metricKeyList: string[];
  tagKeyList: string[];
  paramsList: KeyValueEntity[][];
  metricsList: MetricEntity[][];
  tagsList: Record<string, KeyValueEntity>[];
}) => {
  const { runInfos, paramKeyList, metricKeyList, tagKeyList, paramsList, metricsList, tagsList } = params;

  const columns = [
    'Start Time',
    'Duration',
    'Run ID',
    'Name',
    'Source Type',
    'Source Name',
    'User',
    'Status',
    ...paramKeyList,
    ...metricKeyList,
    ...tagKeyList,
  ];

  const data = runInfos.map((runInfo, index) => {
    // To avoid including a comma in the timestamp string, use manual formatting instead of one from intl
    const startTime = moment(new Date(runInfo.startTime)).format('YYYY-MM-DD HH:mm:ss');
    const row = [
      startTime,
      getDuration(runInfo.startTime, runInfo.endTime) || '',
      runInfo.runUuid,
      runInfo.runName || getRunNameFromTags(tagsList[index]), // add run name to csv export row
      getSourceType(tagsList[index]),
      getSourceName(tagsList[index]),
      getUser(runInfo, tagsList[index]),
      runInfo.status,
    ];
    const paramsMap = toMap(paramsList[index]);
    const metricsMap = toMap(metricsList[index]);
    const tagsMap = tagsList[index];

    paramKeyList.forEach((paramKey) => {
      if (paramsMap[paramKey]) {
        row.push(paramsMap[paramKey].value);
      } else {
        row.push('');
      }
    });
    metricKeyList.forEach((metricKey) => {
      if (metricsMap[metricKey]) {
        row.push(metricsMap[metricKey].value);
      } else {
        row.push('');
      }
    });
    tagKeyList.forEach((tagKey) => {
      if (tagsMap[tagKey]) {
        row.push(tagsMap[tagKey].value);
      } else {
        row.push('');
      }
    });
    return row;
  });

  return tableToCsv(columns, data);
};

export const chartMetricHistoryToCsv = (traces: RunsChartsRunData[], metricKeys: string[]) => {
  const isGrouped = traces.some((trace) => trace.groupParentInfo);

  const headerColumn = isGrouped ? 'Group' : 'Run';

  const columns = [headerColumn, 'Run ID', 'metric', 'step', 'timestamp', 'value'];

  const data = metricKeys.flatMap((metricKey) => {
    const perDataTrace = traces.flatMap((trace) => {
      const perMetricEntry = trace.metricsHistory?.[metricKey]?.map((value) => [
        trace.displayName,
        trace.runInfo?.runUuid || '',
        value.key,
        value.step.toString(),
        value.timestamp.toString(),
        value.value.toString(),
      ]);
      return perMetricEntry || [];
    });
    return perDataTrace;
  });

  return tableToCsv(columns, data);
};

export const chartDataToCsv = (traces: RunsChartsRunData[], metricKeys: string[], paramKeys: string[]) => {
  const isGrouped = traces.some((trace) => trace.groupParentInfo);

  const headerColumn = isGrouped ? 'Group' : 'Run';

  const columns = [headerColumn, 'Run ID', ...metricKeys, ...paramKeys];

  const data = traces.map((trace) => {
    const row = [trace.displayName, trace.runInfo?.runUuid || ''];

    metricKeys.forEach((metricKey) => {
      const metricValue = trace.metrics?.[metricKey];
      row.push(metricValue?.value.toString() || '');
    });

    paramKeys.forEach((paramKey) => {
      const paramValue = trace.params?.[paramKey];
      row.push(paramValue?.value.toString() || '');
    });

    return row;
  });

  return tableToCsv(columns, data);
};

import Utils from '../../common/utils/Utils';
import { KeyValueEntity, MetricEntity, RunInfoEntity } from '../types';

const { formatTimestamp, getDuration, getRunNameFromTags, getSourceType, getSourceName, getUser } = Utils;

/**
 * Turn a list of params/metrics to a map of metric key to metric.
 */
const toMap = <T extends MetricEntity | KeyValueEntity>(params: T[]) =>
  params.reduce((result, entity) => ({ ...result, [entity.key]: entity }), {} as Record<string, T>);

/**
 * Format a string for insertion into a CSV file.
 */
const csvEscape = (str: string) => {
  if (str === undefined) {
    return '';
  }
  if (/[,"\r\n]/.test(str)) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
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
    const row = [
      formatTimestamp(runInfo.start_time),
      getDuration(runInfo.start_time, runInfo.end_time) || '',
      runInfo.run_uuid,
      runInfo.run_name || getRunNameFromTags(tagsList[index]), // add run name to csv export row
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

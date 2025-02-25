/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Fragment } from 'react';
import { connect } from 'react-redux';
import { injectIntl } from 'react-intl';
import { HtmlTableView } from './HtmlTableView';
import { getRunInfo } from '../reducers/Reducers';
import { getLatestMetrics, getMinMetrics, getMaxMetrics } from '../reducers/MetricReducer';
import Utils from '../../common/utils/Utils';
import { Link } from '../../common/utils/RoutingUtils';
import Routes from '../routes';
import type { RunInfoEntity } from '../types';

const maxTableHeight = 300;
// Because we make the table body scrollable, column widths must be fixed
// so that the header widths match the table body column widths.
const headerColWidth = 350;
const dataColWidth = 200;

type MetricsSummaryTableProps = {
  runUuids: string[];
  runExperimentIds: any;
  runDisplayNames: string[];
  metricKeys: string[];
  latestMetrics: any;
  minMetrics: any;
  maxMetrics: any;
  intl: {
    formatMessage: (...args: any[]) => any;
  };
};

class MetricsSummaryTable extends React.Component<MetricsSummaryTableProps> {
  render() {
    const { runUuids } = this.props;
    return (
      <div className="metrics-summary">
        {runUuids.length > 1 ? this.renderMetricTables() : this.renderRunTable(runUuids[0])}
      </div>
    );
  }

  renderRunTable(runUuid: any) {
    const { metricKeys, latestMetrics, minMetrics, maxMetrics, intl } = this.props;
    const columns = [
      {
        title: intl.formatMessage({
          defaultMessage: 'Metric',
          description:
            // eslint-disable-next-line max-len
            'Column title for the column displaying the metric names for a run',
        }),
        dataIndex: 'metricKey',
        sorter: (a: any, b: any) => (a.metricKey < b.metricKey ? -1 : a.metricKey > b.metricKey ? 1 : 0),
        width: headerColWidth,
      },
      ...this.dataColumns(),
    ];
    return metricKeys.length === 0 ? null : (
      <HtmlTableView
        columns={columns}
        values={getRunValuesByMetric(runUuid, metricKeys, latestMetrics, minMetrics, maxMetrics, intl)}
        scroll={{ y: maxTableHeight }}
      />
    );
  }

  renderMetricTables() {
    const { runExperimentIds, runUuids, runDisplayNames, metricKeys, latestMetrics, minMetrics, maxMetrics, intl } =
      this.props;
    const columns = [
      {
        title: intl.formatMessage({
          defaultMessage: 'Run',
          description:
            // eslint-disable-next-line max-len
            'Column title for the column displaying the run names for a metric',
        }),
        dataIndex: 'runLink',
        sorter: (a: any, b: any) => (a.runName < b.runName ? -1 : a.runName > b.runName ? 1 : 0),
        width: headerColWidth,
      },
      ...this.dataColumns(),
    ];
    return metricKeys.map((metricKey) => {
      return (
        <Fragment key={metricKey}>
          <h1>{metricKey}</h1>
          <HtmlTableView
            columns={columns}
            values={getMetricValuesByRun(
              metricKey,
              runExperimentIds,
              runUuids,
              runDisplayNames,
              latestMetrics,
              minMetrics,
              maxMetrics,
              intl,
            )}
            scroll={{ y: maxTableHeight }}
          />
        </Fragment>
      );
    });
  }

  dataColumns() {
    return [
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Latest',
          description: 'Column title for the column displaying the latest metric values for a metric',
        }),
        dataIndex: 'latestFormatted',
        sorter: (a: any, b: any) => a.latestValue - b.latestValue,
        width: dataColWidth,
        ellipsis: true,
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Min',
          description: 'Column title for the column displaying the minimum metric values for a metric',
        }),
        dataIndex: 'minFormatted',
        sorter: (a: any, b: any) => a.minValue - b.minValue,
        width: dataColWidth,
        ellipsis: true,
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Max',
          description: 'Column title for the column displaying the maximum metric values for a metric',
        }),
        dataIndex: 'maxFormatted',
        sorter: (a: any, b: any) => a.maxValue - b.maxValue,
        width: dataColWidth,
        ellipsis: true,
      },
    ];
  }
}

const getMetricValuesByRun = (
  metricKey: any,
  runExperimentIds: any,
  runUuids: any,
  runDisplayNames: any,
  latestMetrics: any,
  minMetrics: any,
  maxMetrics: any,
  intl: any,
) => {
  return runUuids.map((runUuid: any, runIdx: any) => {
    const runName = runDisplayNames[runIdx];
    return {
      runName: runName,
      runLink: <Link to={Routes.getRunPageRoute(runExperimentIds[runUuid] || '', runUuid)}>{runName}</Link>,
      key: runUuid,
      ...rowData(runUuid, metricKey, latestMetrics, minMetrics, maxMetrics, intl),
    };
  });
};

const getRunValuesByMetric = (
  runUuid: any,
  metricKeys: any,
  latestMetrics: any,
  minMetrics: any,
  maxMetrics: any,
  intl: any,
) => {
  return metricKeys.map((metricKey: any) => {
    return {
      metricKey,
      key: metricKey,
      ...rowData(runUuid, metricKey, latestMetrics, minMetrics, maxMetrics, intl),
    };
  });
};

const rowData = (runUuid: any, metricKey: any, latestMetrics: any, minMetrics: any, maxMetrics: any, intl: any) => {
  const latestMetric = getMetric(latestMetrics, runUuid, metricKey);
  const minMetric = getMetric(minMetrics, runUuid, metricKey);
  const maxMetric = getMetric(maxMetrics, runUuid, metricKey);
  const latestValue = getValue(latestMetric);
  const minValue = getValue(minMetric);
  const maxValue = getValue(maxMetric);
  return {
    latestFormatted: (
      <span title={latestValue} css={{ marginRight: 10 }}>
        {formatMetric(latestMetric, intl)}
      </span>
    ),
    minFormatted: (
      <span title={minValue} css={{ marginRight: 10 }}>
        {formatMetric(minMetric, intl)}
      </span>
    ),
    maxFormatted: (
      <span title={maxValue} css={{ marginRight: 10 }}>
        {formatMetric(maxMetric, intl)}
      </span>
    ),
    latestValue,
    minValue,
    maxValue,
  };
};

const getMetric = (valuesMap: any, runUuid: any, metricKey: any) => valuesMap[runUuid] && valuesMap[runUuid][metricKey];

const getValue = (metric: any) => metric && metric.value;

const formatMetric = (metric: any, intl: any) =>
  metric === undefined
    ? ''
    : intl.formatMessage(
        {
          defaultMessage: '{value} (step={step})',
          description: 'Formats a metric value along with the step number it corresponds to',
        },
        {
          value: metric.value,
          step: metric.step,
        },
      );

const mapStateToProps = (state: any, ownProps: any) => {
  const { runUuids } = ownProps;
  const runExperimentIds = {};
  const latestMetrics = {};
  const minMetrics = {};
  const maxMetrics = {};
  runUuids.forEach((runUuid: any) => {
    const runInfo = getRunInfo(runUuid, state) as RunInfoEntity;
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    runExperimentIds[runUuid] = runInfo && runInfo.experimentId;
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    latestMetrics[runUuid] = getLatestMetrics(runUuid, state);
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    minMetrics[runUuid] = getMinMetrics(runUuid, state);
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    maxMetrics[runUuid] = getMaxMetrics(runUuid, state);
  });
  return { runExperimentIds, latestMetrics, minMetrics, maxMetrics };
};

// @ts-expect-error TS(2769): No overload matches this call.
const MetricsSummaryTableWithIntl = injectIntl(MetricsSummaryTable);

export default connect(mapStateToProps)(MetricsSummaryTableWithIntl);

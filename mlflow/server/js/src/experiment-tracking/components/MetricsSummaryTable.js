import React, { Fragment } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { injectIntl } from 'react-intl';
import { HtmlTableView } from './HtmlTableView';
import { getLatestMetrics, getMinMetrics, getMaxMetrics } from '../reducers/MetricReducer';
import Utils from '../../common/utils/Utils';

class MetricsSummaryTable extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    runDisplayNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricKeys: PropTypes.arrayOf(PropTypes.string).isRequired,
    latestMetrics: PropTypes.object.isRequired,
    minMetrics: PropTypes.object.isRequired,
    maxMetrics: PropTypes.object.isRequired,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  render() {
    const { runUuids } = this.props;
    return (
      <div className='metrics-summary'>
        {runUuids.length > 1 ? this.renderMetricTables() : this.renderRunTable(runUuids[0])}
      </div>
    );
  }

  renderRunTable(runUuid) {
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
      },
      ...this.dataColumns(),
    ];
    return metricKeys.length === 0 ? null : (
      <HtmlTableView
        columns={columns}
        values={getRunValuesByMetric(
          runUuid,
          metricKeys,
          latestMetrics,
          minMetrics,
          maxMetrics,
          intl,
        )}
      />
    );
  }

  renderMetricTables() {
    const {
      runUuids,
      runDisplayNames,
      metricKeys,
      latestMetrics,
      minMetrics,
      maxMetrics,
      intl,
    } = this.props;
    const columns = [
      {
        title: intl.formatMessage({
          defaultMessage: 'Run',
          description:
            // eslint-disable-next-line max-len
            'Column title for the column displaying the run names for a metric',
        }),
        dataIndex: 'runName',
      },
      ...this.dataColumns(),
    ];
    return metricKeys.map((metricKey) => {
      return (
        <Fragment key={metricKey}>
          {metricKeys.length > 1 ? <h1>{metricKey}</h1> : null}
          <HtmlTableView
            columns={columns}
            values={getMetricValuesByRun(
              metricKey,
              runUuids,
              runDisplayNames,
              latestMetrics,
              minMetrics,
              maxMetrics,
              intl,
            )}
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
          description:
            'Column title for the column displaying the latest metric values for a metric',
        }),
        dataIndex: 'latest',
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Min',
          description:
            'Column title for the column displaying the minimum metric values for a metric',
        }),
        dataIndex: 'min',
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Max',
          description:
            'Column title for the column displaying the maximum metric values for a metric',
        }),
        dataIndex: 'max',
      },
    ];
  }
}

const getMetricValuesByRun = (
  metricKey,
  runUuids,
  runDisplayNames,
  latestMetrics,
  minMetrics,
  maxMetrics,
  intl,
) => {
  return runUuids.map((runUuid, runIdx) => {
    const runName = runDisplayNames[runIdx];
    return {
      runName,
      key: runUuid,
      ...rowData(runUuid, metricKey, latestMetrics, minMetrics, maxMetrics, intl),
    };
  });
};

const getRunValuesByMetric = (runUuid, metricKeys, latestMetrics, minMetrics, maxMetrics, intl) => {
  return metricKeys.map((metricKey) => {
    return {
      metricKey,
      key: metricKey,
      ...rowData(runUuid, metricKey, latestMetrics, minMetrics, maxMetrics, intl),
    };
  });
};

const rowData = (runUuid, metricKey, latestMetrics, minMetrics, maxMetrics, intl) => {
  const latestValue = getMetric(latestMetrics, runUuid, metricKey);
  const minValue = getMetric(minMetrics, runUuid, metricKey);
  const maxValue = getMetric(maxMetrics, runUuid, metricKey);
  return {
    latest: <span title={getValue(latestValue)}>{formatMetric(latestValue, intl)}</span>,
    min: <span title={getValue(minValue)}>{formatMetric(minValue, intl)}</span>,
    max: <span title={getValue(maxValue)}>{formatMetric(maxValue, intl)}</span>,
  };
};

const getMetric = (valuesMap, runUuid, metricKey) =>
  valuesMap[runUuid] && valuesMap[runUuid][metricKey];

const getValue = (metric) => metric && metric.value;

const formatMetric = (metric, intl) =>
  metric === undefined
    ? ''
    : intl.formatMessage(
        {
          defaultMessage: '{value} (step={step})',
          description: 'Formats a metric value along with the step number it corresponds to',
        },
        {
          value: Utils.formatMetric(metric.value),
          step: metric.step,
        },
      );

const mapStateToProps = (state, ownProps) => {
  const { runUuids } = ownProps;
  const latestMetrics = {};
  const minMetrics = {};
  const maxMetrics = {};
  runUuids.forEach((runUuid) => {
    latestMetrics[runUuid] = getLatestMetrics(runUuid, state);
    minMetrics[runUuid] = getMinMetrics(runUuid, state);
    maxMetrics[runUuid] = getMaxMetrics(runUuid, state);
  });
  return { latestMetrics, minMetrics, maxMetrics };
};

export const MetricsSummaryTableWithIntl = injectIntl(MetricsSummaryTable);

export default connect(mapStateToProps)(MetricsSummaryTableWithIntl);

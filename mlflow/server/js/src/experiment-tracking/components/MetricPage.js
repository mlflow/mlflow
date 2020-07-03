import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import qs from 'qs';
import { getExperimentApi, getMetricHistoryApi, getRunApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import NotFoundPage from './NotFoundPage';
import { MetricView } from './MetricView';
import { getUUID } from '../../common/utils/ActionUtils';

export class MetricPageImpl extends Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricKey: PropTypes.string.isRequired,
    experimentId: PropTypes.string,
    dispatch: PropTypes.func.isRequired,
  };

  componentWillMount() {
    this.requestIds = [];
    if (this.props.experimentId !== null) {
      const experimentRequestId = getUUID();
      this.props.dispatch(getExperimentApi(this.props.experimentId, experimentRequestId));
      this.requestIds.push(experimentRequestId);
    }
    this.props.runUuids.forEach((runUuid) => {
      const getMetricHistoryReqId = getUUID();
      this.requestIds.push(getMetricHistoryReqId);
      this.props.dispatch(
        getMetricHistoryApi(runUuid, this.props.metricKey, getMetricHistoryReqId),
      );
      // Fetch tags for each run. TODO: it'd be nice if we could just fetch the tags directly
      const getRunRequestId = getUUID();
      this.requestIds.push(getRunRequestId);
      this.props.dispatch(getRunApi(runUuid, getRunRequestId));
    });
  }

  renderPageContent() {
    const { runUuids } = this.props;
    return runUuids.length >= 1 ? (
      <MetricView
        runUuids={this.props.runUuids}
        metricKey={this.props.metricKey}
        experimentId={this.props.experimentId}
      />
    ) : (
      <NotFoundPage />
    );
  }

  render() {
    return (
      <div className='App-content'>
        <RequestStateWrapper requestIds={this.requestIds}>
          {this.renderPageContent()}
        </RequestStateWrapper>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { match, location } = ownProps;
  const searchValues = qs.parse(location.search);
  const runUuids = JSON.parse(searchValues['?runs']);
  let experimentId = null;
  if (searchValues.hasOwnProperty('experiment')) {
    experimentId = searchValues['experiment'];
  }
  const { metricKey } = match.params;
  return {
    runUuids,
    metricKey,
    experimentId,
  };
};

export const MetricPage = connect(mapStateToProps)(MetricPageImpl);

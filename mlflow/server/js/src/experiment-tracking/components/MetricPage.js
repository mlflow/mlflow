import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import qs from 'qs';
import { getExperimentApi, getMetricHistoryApi, getRunApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import NotFoundPage from './NotFoundPage';
import { MetricView } from './MetricView';
import { getUUID } from '../../common/utils/ActionUtils';
import { PageContainer } from '../../common/components/PageContainer';

export class MetricPageImpl extends Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricKey: PropTypes.string.isRequired,
    experimentIds: PropTypes.arrayOf(PropTypes.string),
    dispatch: PropTypes.func.isRequired,
  };

  constructor(props) {
    super(props);
    this.requestIds = [];
  }

  fetchExperiments() {
    return this.props.experimentIds.map((experimentId) => {
      const experimentRequestId = getUUID();
      this.props.dispatch(getExperimentApi(experimentId, experimentRequestId));
      return experimentRequestId;
    });
  }

  componentDidMount() {
    if (this.props.experimentIds !== null) {
      const getExperimentsRequestIds = this.fetchExperiments();
      this.requestIds.push(...getExperimentsRequestIds);
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
        experimentIds={this.props.experimentIds}
      />
    ) : (
      <NotFoundPage />
    );
  }

  render() {
    return (
      <PageContainer>
        <RequestStateWrapper requestIds={this.requestIds}>
          {this.renderPageContent()}
        </RequestStateWrapper>
      </PageContainer>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { match, location } = ownProps;
  const searchValues = qs.parse(location.search);
  const runUuids = JSON.parse(searchValues['?runs']);
  let experimentIds = null;
  if (searchValues.hasOwnProperty('experiments')) {
    experimentIds = JSON.parse(searchValues['experiments']);
  }
  const { metricKey } = match.params;
  return {
    runUuids,
    metricKey,
    experimentIds,
  };
};

export const MetricPage = connect(mapStateToProps)(MetricPageImpl);

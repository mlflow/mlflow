import React, { Component } from 'react';
import PropTypes from 'prop-types';
import qs from 'qs';
import { connect } from 'react-redux';
import { getRunApi, getExperimentApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import CompareRunView from './CompareRunView';
import { getUUID } from '../../common/utils/ActionUtils';
import { PageContainer } from '../../common/components/PageContainer';

class CompareRunPage extends Component {
  static propTypes = {
    experimentIds: PropTypes.arrayOf(PropTypes.string).isRequired,
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
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
    this.requestIds.push(...this.fetchExperiments());
    this.props.runUuids.forEach((runUuid) => {
      const requestId = getUUID();
      this.requestIds.push(requestId);
      this.props.dispatch(getRunApi(runUuid, requestId));
    });
  }

  render() {
    return (
      <PageContainer>
        <RequestStateWrapper requestIds={this.requestIds}>
          <CompareRunView runUuids={this.props.runUuids} experimentIds={this.props.experimentIds} />
        </RequestStateWrapper>
      </PageContainer>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { location } = ownProps;
  const searchValues = qs.parse(location.search);
  const runUuids = JSON.parse(searchValues['?runs']);
  const experimentIds = JSON.parse(searchValues['experiments']);
  return { experimentIds, runUuids };
};

export default connect(mapStateToProps)(CompareRunPage);

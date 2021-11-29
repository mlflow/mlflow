import React, { Component } from 'react';
import PropTypes from 'prop-types';
import qs from 'qs';
import { connect } from 'react-redux';
import { getExperimentApi, getRunApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import CompareRunView from './CompareRunView';
import { getUUID } from '../../common/utils/ActionUtils';
import { PageContainer } from '../../common/components/PageContainer';

class CompareRunPage extends Component {
  static propTypes = {
    experimentId: PropTypes.string.isRequired,
    runUuids: PropTypes.arrayOf(PropTypes.string).isRequired,
    dispatch: PropTypes.func.isRequired,
  };

  constructor(props) {
    super(props);
    this.requestIds = [];
  }

  componentDidMount() {
    const experimentRequestId = getUUID();
    this.props.dispatch(getExperimentApi(this.props.experimentId, experimentRequestId));
    this.requestIds.push(experimentRequestId);
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
          <CompareRunView runUuids={this.props.runUuids} experimentId={this.props.experimentId} />
        </RequestStateWrapper>
      </PageContainer>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { location } = ownProps;
  const searchValues = qs.parse(location.search);
  const runUuids = JSON.parse(searchValues['?runs']);
  const experimentId = searchValues['experiment'];
  return { experimentId, runUuids };
};

export default connect(mapStateToProps)(CompareRunPage);

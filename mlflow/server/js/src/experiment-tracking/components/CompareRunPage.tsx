import React, { Component } from 'react';
import qs from 'qs';
import { connect } from 'react-redux';
import { getRunApi, getExperimentApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import CompareRunView from './CompareRunView';
import { getUUID } from '../../common/utils/ActionUtils';
import { PageContainer } from '../../common/components/PageContainer';

type CompareRunPageProps = {
  experimentIds: string[];
  runUuids: string[];
  dispatch: (...args: any[]) => any;
};

class CompareRunPage extends Component<CompareRunPageProps> {
  requestIds: any;

  constructor(props: CompareRunPageProps) {
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
        <RequestStateWrapper
          requestIds={this.requestIds}
          // eslint-disable-next-line no-trailing-spaces
        >
          <CompareRunView runUuids={this.props.runUuids} experimentIds={this.props.experimentIds} />
        </RequestStateWrapper>
      </PageContainer>
    );
  }
}

const mapStateToProps = (state: any, ownProps: any) => {
  const { location } = ownProps;
  const searchValues = qs.parse(location.search);
  // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
  const runUuids = JSON.parse(searchValues['?runs']);
  // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
  const experimentIds = JSON.parse(searchValues['experiments']);
  return { experimentIds, runUuids };
};

export default connect(mapStateToProps)(CompareRunPage);

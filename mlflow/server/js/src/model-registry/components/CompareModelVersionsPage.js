import React, { Component } from 'react';
import PropTypes from 'prop-types';
import qs from 'qs';
import { connect } from 'react-redux';
import { getExperimentApi, getRunApi, getUUID } from '../../Actions';
import { getRegisteredModelDetailsApi, getModelVersionDetailsApi } from "../actions";
import RequestStateWrapper from '../../components/RequestStateWrapper';
import CompareModelVersionsView from './CompareModelVersionsView';

class CompareRunPage extends Component {
    static propTypes = {
        modelName: PropTypes.string.isRequired,
        experimentId: PropTypes.number.isRequired,
        runsToVersions: PropTypes.object.isRequired,
        dispatch: PropTypes.func.isRequired,
    };

    componentWillMount() {
        this.requestIds = [];
        const experimentRequestId = getUUID();
        this.props.dispatch(getExperimentApi(this.props.experimentId, experimentRequestId));
        this.requestIds.push(experimentRequestId);
        const registeredModelRequestId = getUUID();
        this.requestIds.push(registeredModelRequestId);
        this.props.dispatch(getRegisteredModelDetailsApi(this.props.modelName, registeredModelRequestId));
        for (const runUuid in this.props.runsToVersions) {
            const runRequestId = getUUID();
            this.requestIds.push(runRequestId);
            this.props.dispatch(getRunApi(runUuid, runRequestId));
            const versionRequestId = getUUID();
            this.requestIds.push(versionRequestId);
            this.props.dispatch(getModelVersionDetailsApi(this.props.runsToVersions[runUuid], versionRequestId))
        }
    }

    render() {
        return (
            <div className='App-content'>
                <RequestStateWrapper requestIds={this.requestIds}>
                    <CompareModelVersionsView modelName={this.props.modelName} runsToVersions={this.props.runsToVersions} experimentId={this.props.experimentId}/>
                </RequestStateWrapper>
            </div>
        );
    }
}

const mapStateToProps = (state, ownProps) => {
    const { location } = ownProps;
    const searchValues = qs.parse(location.search);
    const modelName = JSON.parse(searchValues["?name"]);
    const runsToVersions = JSON.parse(searchValues["?runs"]);
    const experimentId = parseInt(searchValues["experiment"], 10);
    return { modelName, experimentId, runsToVersions };
};

export default connect(mapStateToProps)(CompareRunPage);

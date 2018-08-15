import React, { Component } from 'react';
import PropTypes from 'prop-types';
import ExperimentPage from './ExperimentPage';
import { connect } from 'react-redux';
import { getUUID, listExperimentsApi } from '../Actions';
import RequestStateWrapper from './RequestStateWrapper';
import './HomePage.css';
import ExperimentListView from './ExperimentListView';

class HomePage extends Component {
  constructor(props) {
    super(props);
    this.onClickListExperiments = this.onClickListExperiments.bind(this);
  }

  static propTypes = {
    match: PropTypes.object.isRequired,
    dispatchListExperimentsApi: PropTypes.func.isRequired,
    experimentId: PropTypes.number.isRequired,
  };

  state = {
    listExperimentsExpanded: true,
    listExperimentsRequestId: getUUID(),
  };

  componentWillMount() {
    this.props.dispatchListExperimentsApi(this.state.listExperimentsRequestId);
  }

  onClickListExperiments() {
    this.setState({ listExperimentsExpanded: !this.state.listExperimentsExpanded });
  }

  render() {
    if (this.state.listExperimentsExpanded) {
      return (
        <div className="outer-container">
          <div className="HomePage-experiment-list-container">
            <RequestStateWrapper requestIds={[this.state.listExperimentsRequestId]}>
              <div className="collapsed-expander-container">
                <ExperimentListView
                  activeExperimentId={this.props.experimentId}
                  onClickListExperiments={this.onClickListExperiments}
                />
              </div>
            </RequestStateWrapper>
          </div>
          <div className="experiment-view-container">
            <ExperimentPage match={this.props.match}/>
          </div>
          <div className="experiment-view-right"/>
        </div>
      );
    } else {
      return (
        <div>
          <div className="collapsed-expander-container">
            <i onClick={this.onClickListExperiments}
               title="Show experiment list"
               className="expander fa fa-chevron-right login-icon"/>
          </div>
          <div className="experiment-page-container">
            <ExperimentPage match={this.props.match}/>
          </div>
        </div>
      );
    }
  }
}

const mapStateToProps = (state, ownProps) => {
  const { match } = ownProps;
  if (match.url === "/") {
    return { experimentId: 0 };
  }
  return { experimentId: parseInt(match.params.experimentId, 10) };
};

const mapDispatchToProps = (dispatch) => {
  return {
    dispatchListExperimentsApi: (requestId) => {
      dispatch(listExperimentsApi(requestId));
    }
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(HomePage);

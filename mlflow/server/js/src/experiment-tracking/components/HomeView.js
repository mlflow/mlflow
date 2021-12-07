import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import ExperimentListView from './ExperimentListView';
import ExperimentPage from './ExperimentPage';
import { getExperiments } from '../reducers/Reducers';
import { NoExperimentView } from './NoExperimentView';
import Utils from '../../common/utils/Utils';
import { PageContainer } from '../../common/components/PageContainer';

export const getFirstActiveExperiment = (experiments) => {
  const sorted = experiments.concat().sort(Utils.compareExperiments);
  return sorted.find((e) => e.lifecycle_stage === 'active');
};

class HomeView extends Component {
  constructor(props) {
    super(props);
    this.onClickListExperiments = this.onClickListExperiments.bind(this);
  }

  static propTypes = {
    experimentId: PropTypes.string,
  };

  state = {
    listExperimentsExpanded: true,
  };

  onClickListExperiments() {
    this.setState({ listExperimentsExpanded: !this.state.listExperimentsExpanded });
  }

  render() {
    const headerHeight = process.env.HIDE_HEADER === 'true' ? 0 : 60;
    const containerHeight = 'calc(100% - ' + headerHeight + 'px)';
    if (process.env.HIDE_EXPERIMENT_LIST === 'true') {
      return (
        <div style={{ height: containerHeight }}>
          {this.props.experimentId !== undefined ? (
            <PageContainer>
              <ExperimentPage experimentId={this.props.experimentId} />
            </PageContainer>
          ) : (
            <NoExperimentView />
          )}
        </div>
      );
    }
    return (
      <div className='outer-container' style={{ height: containerHeight }}>
        <div>
          {this.state.listExperimentsExpanded ? (
            <ExperimentListView
              activeExperimentId={this.props.experimentId}
              onClickListExperiments={this.onClickListExperiments}
            />
          ) : (
            <i
              onClick={this.onClickListExperiments}
              title='Show experiment list'
              style={styles.showExperimentListExpander}
              className='expander fa fa-chevron-right login-icon'
            />
          )}
        </div>
        <PageContainer>
          {this.props.experimentId !== undefined ? (
            <ExperimentPage experimentId={this.props.experimentId} />
          ) : (
            <NoExperimentView />
          )}
        </PageContainer>
      </div>
    );
  }
}

const styles = {
  showExperimentListExpander: {
    marginTop: 24,
  },
};

const mapStateToProps = (state, ownProps) => {
  if (ownProps.experimentId === undefined) {
    const firstExp = getFirstActiveExperiment(getExperiments(state));
    if (firstExp) {
      return { experimentId: firstExp.experiment_id };
    }
  }
  return {};
};

export default connect(mapStateToProps)(HomeView);

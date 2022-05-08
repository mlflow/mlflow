import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Spacer } from '@databricks/design-system';
import ExperimentListView from './ExperimentListView';
import ExperimentPage from './ExperimentPage';
import { getExperiments } from '../reducers/Reducers';
import { NoExperimentView } from './NoExperimentView';
import Utils from '../../common/utils/Utils';
import { PageContainer } from '../../common/components/PageContainer';
import Routes from '../routes';

export const getFirstActiveExperiment = (experiments) => {
  const sorted = experiments.concat().sort(Utils.compareExperiments);
  return sorted.find((e) => e.lifecycle_stage === 'active');
};

class HomeView extends Component {
  static propTypes = {
    history: PropTypes.shape({}),
    experiments: PropTypes.shape({}),
    experimentIds: PropTypes.arrayOf(PropTypes.string),
    compareExperiments: PropTypes.bool,
  };

  componentDidMount() {
    const { history, experiments, experimentIds } = this.props;
    if (experimentIds === undefined) {
      const firstExp = getFirstActiveExperiment(experiments);
      if (firstExp) {
        // Reported during ESLint upgrade
        // eslint-disable-next-line react/prop-types
        history.push(Routes.getExperimentPageRoute(firstExp.experiment_id));
      }
    }
  }

  render() {
    const { experimentIds, experiments, compareExperiments } = this.props;
    const headerHeight = process.env.HIDE_HEADER === 'true' ? 0 : 60;
    const containerHeight = 'calc(100% - ' + headerHeight + 'px)';

    if (experimentIds === undefined) {
      return <NoExperimentView />;
    }

    return (
      <div className='outer-container' style={{ height: containerHeight, display: 'flex' }}>
        {process.env.HIDE_EXPERIMENT_LIST !== 'true' && (
          <div>
            <Spacer />
            <ExperimentListView activeExperimentId={experimentIds[0]} experiments={experiments} />
          </div>
        )}
        <PageContainer>
          <ExperimentPage experimentIds={experimentIds} compareExperiments={compareExperiments} />
        </PageContainer>
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  const experiments = getExperiments(state);
  return { experiments };
};

export default connect(mapStateToProps)(HomeView);

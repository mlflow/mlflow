import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import './ExperimentListView.css';
import { getExperiments } from '../reducers/Reducers';
import { Experiment } from '../sdk/MlflowMessages';
import Routes from '../Routes';
import { Link } from 'react-router-dom';

class ExperimentListView extends Component {
  static propTypes = {
    onClickListExperiments: PropTypes.func.isRequired,
    activeExperimentId: PropTypes.number.isRequired,
    experiments: PropTypes.arrayOf(Experiment).isRequired,
  };
  state = {
    height: undefined,
  };

  componentDidMount() {
    window.addEventListener('resize', () => { this.setState({height: window.innerHeight })});
  }

  render() {
    const height = this.state.height || window.innerHeight;
    // 60 pixels for the height of the top bar.
    // 100 for the experiments header and some for bottom padding.
    const experimentListHeight = height - 60 - 100;
    return (
      <div className="experiment-list-outer-container">
        <div>
          <h1 className="experiments-header">Experiments</h1>
          <div className="collapser-container">
            <i onClick={this.props.onClickListExperiments} className="collapser fa fa-chevron-left login-icon"/>
          </div>
          <div className="experiment-list-container" style={{ height: experimentListHeight }}>
            {this.props.experiments.map((e) => {

              let className = "experiment-list-item";
              if (parseInt(e.getExperimentId(), 10) === this.props.activeExperimentId) {
                className = `${className} active-experiment-list-item`
              }
              return (
                <div
                  className={className}
                  key={e.getExperimentId()}
                  title={e.getName()}
                >
                  <Link
                    style={{ textDecoration: 'none', color: 'unset' }}
                    to={Routes.getExperimentPageRoute(e.getExperimentId())}
                  >
                    {e.getName()}
                  </Link>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  const experiments = getExperiments(state);
  experiments.sort((a, b) => {
    return parseInt(a.getExperimentId(), 10) - parseInt(b.getExperimentId(), 10);
  });
  return { experiments };
};

export default connect(mapStateToProps)(ExperimentListView);

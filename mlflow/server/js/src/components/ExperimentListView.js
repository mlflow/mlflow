import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import './ExperimentListView.css';
import { getExperiments } from '../reducers/Reducers';
import { Experiment } from '../sdk/MlflowMessages';
import Routes from '../Routes';
import { Link } from 'react-router-dom';

export class ExperimentListView extends Component {
  static propTypes = {
    onClickListExperiments: PropTypes.func.isRequired,
    // If activeExperimentId is undefined, then the active experiment is the first one.
    activeExperimentId: PropTypes.number,
    experiments: PropTypes.arrayOf(Experiment).isRequired,
  };

  state = {
    height: undefined,
  };

  componentDidMount() {
    this.resizeListener = () => {
      this.setState({height: window.innerHeight });
    };
    window.addEventListener('resize', this.resizeListener);
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.resizeListener);
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
            <i onClick={this.props.onClickListExperiments}
               title="Hide experiment list"
               className="collapser fa fa-chevron-left login-icon"/>
          </div>
          <div className="experiment-list-container" style={{ height: experimentListHeight }}>
            {this.props.experiments.map((e, idx) => {
              let active;
              if (this.props.activeExperimentId) {
                active = parseInt(e.getExperimentId(), 10) === this.props.activeExperimentId;
              } else {
                active = idx === 0;
              }
              let className = "experiment-list-item";
              if (active) {
                className = `${className} active-experiment-list-item`;
              }
              return (
                <Link
                  style={{ textDecoration: 'none', color: 'unset' }}
                  key={e.getExperimentId()}
                  to={Routes.getExperimentPageRoute(e.getExperimentId())}
                  onClick={active ? ev => ev.preventDefault() : ev => ev}
                >
                  <div
                    className={className}
                    title={e.getName()}
                  >
                    {e.getName()}
                  </div>
                </Link>
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

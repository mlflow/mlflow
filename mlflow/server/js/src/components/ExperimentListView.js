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
    searchInput: '',
  };

  componentDidMount() {
    this.resizeListener = () => {
      this.setState({ height: window.innerHeight });
    };
    window.addEventListener('resize', this.resizeListener);
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.resizeListener);
  }

  handleSearchInputChange = (event) => {
    this.setState({ searchInput: event.target.value });
  };

  preventDefault = (ev) => ev.preventDefault();

  render() {
    const height = this.state.height || window.innerHeight;
    // 60 pixels for the height of the top bar.
    // 100 for the experiments header and some for bottom padding.
    const experimentListHeight = height - 60 - 100;
    // get searchInput from state
    const { searchInput } = this.state;
    return (
      <div className='experiment-list-outer-container'>
        <div>
          <h1 className='experiments-header'>Experiments</h1>
          <div className='collapser-container'>
            <i
              onClick={this.props.onClickListExperiments}
              title='Hide experiment list'
              className='collapser fa fa-chevron-left login-icon'
            />
          </div>
          <input
            className='experiment-list-search-input'
            type='text'
            placeholder='Search Experiments'
            value={searchInput}
            onChange={this.handleSearchInputChange}
          />
          <div className='experiment-list-container' style={{ height: experimentListHeight }}>
            {this.props.experiments
              // filter experiments based on searchInput
              .filter((exp) => exp.getName().toLowerCase().includes(searchInput.toLowerCase()))
              .map((exp, idx) => {
                const { name, experiment_id } = exp;
                const parsedExperimentId = parseInt(experiment_id, 10);
                const active = this.props.activeExperimentId !== undefined
                  ? parsedExperimentId === this.props.activeExperimentId
                  : idx === 0;
                const className =
                  `experiment-list-item ${active ? 'active-experiment-list-item' : ''}`;
                return (
                  <Link
                    style={{ textDecoration: 'none', color: 'unset' }}
                    key={name}
                    to={Routes.getExperimentPageRoute(experiment_id)}
                    onClick={active ? this.preventDefault : undefined}
                  >
                    <div className={className} title={name}>{name}</div>
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

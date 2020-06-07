import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Icon } from 'antd';
import './ExperimentListView.css';
import { getExperiments } from '../reducers/Reducers';
import { Experiment } from '../sdk/MlflowMessages';
import Routes from '../routes';
import { Link } from 'react-router-dom';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { DeleteExperimentModal } from './modals/DeleteExperimentModal';
import { RenameExperimentModal } from './modals/RenameExperimentModal';
import { IconButton } from '../../common/components/IconButton';
import Utils from '../../common/utils/Utils';

export class ExperimentListView extends Component {
  static propTypes = {
    onClickListExperiments: PropTypes.func.isRequired,
    // If activeExperimentId is undefined, then the active experiment is the first one.
    activeExperimentId: PropTypes.string,
    experiments: PropTypes.arrayOf(Experiment).isRequired,
  };

  state = {
    height: undefined,
    searchInput: '',
    showCreateExperimentModal: false,
    showDeleteExperimentModal: false,
    showRenameExperimentModal: false,
    selectedExperimentId: '0',
    selectedExperimentName: '',
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

  updateSelectedExperiment = (experimentId, experimentName) => {
    this.setState({
      selectedExperimentId: experimentId,
      selectedExperimentName: experimentName,
    });
  };

  handleCreateExperiment = () => {
    this.setState({
      showCreateExperimentModal: true,
    });
  };

  handleDeleteExperiment = (ev) => {
    this.setState({
      showDeleteExperimentModal: true,
    });

    const data = ev.currentTarget.dataset;
    this.updateSelectedExperiment(data.experimentid, data.experimentname);
  };

  handleRenameExperiment = (ev) => {
    this.setState({
      showRenameExperimentModal: true,
    });

    const data = ev.currentTarget.dataset;
    this.updateSelectedExperiment(data.experimentid, data.experimentname);
  };

  handleCloseCreateExperimentModal = () => {
    this.setState({
      showCreateExperimentModal: false,
    });
  };

  handleCloseDeleteExperimentModal = () => {
    this.setState({
      showDeleteExperimentModal: false,
    });
    // reset
    this.updateSelectedExperiment('0', '');
  };

  handleCloseRenameExperimentModal = () => {
    this.setState({
      showRenameExperimentModal: false,
    });
    // reset
    this.updateSelectedExperiment('0', '');
  };

  render() {
    const height = this.state.height || window.innerHeight;
    // 60 pixels for the height of the top bar.
    // 100 for the experiments header and some for bottom padding.
    const experimentListHeight = height - 60 - 100;
    // get searchInput from state
    const { searchInput } = this.state;
    return (
      <div className='experiment-list-outer-container'>
        <CreateExperimentModal
          isOpen={this.state.showCreateExperimentModal}
          onClose={this.handleCloseCreateExperimentModal}
        />
        <DeleteExperimentModal
          isOpen={this.state.showDeleteExperimentModal}
          onClose={this.handleCloseDeleteExperimentModal}
          activeExperimentId={this.props.activeExperimentId}
          experimentId={this.state.selectedExperimentId}
          experimentName={this.state.selectedExperimentName}
        />
        <RenameExperimentModal
          isOpen={this.state.showRenameExperimentModal}
          onClose={this.handleCloseRenameExperimentModal}
          experimentId={this.state.selectedExperimentId}
          experimentName={this.state.selectedExperimentName}
        />
        <div>
          <h1 className='experiments-header'>Experiments</h1>
          <div className='experiment-list-create-btn-container'>
            <i
              onClick={this.handleCreateExperiment}
              title='New Experiment'
              className='fas fa-plus fa-border experiment-list-create-btn'
            />
          </div>
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
            aria-label='search experiments'
            value={searchInput}
            onChange={this.handleSearchInputChange}
          />
          <div className='experiment-list-container' style={{ height: experimentListHeight }}>
            {this.props.experiments
              // filter experiments based on searchInput
              .filter((exp) =>
                exp
                  .getName()
                  .toLowerCase()
                  .includes(searchInput.toLowerCase()),
              )
              .map((exp, idx) => {
                const { name, experiment_id } = exp;
                const active =
                  this.props.activeExperimentId !== undefined
                    ? experiment_id === this.props.activeExperimentId
                    : idx === 0;
                const className = `experiment-list-item ${
                  active ? 'active-experiment-list-item' : ''
                }`;
                return (
                  <div key={experiment_id} title={name} className={`header-container ${className}`}>
                    <Link
                      style={{ textDecoration: 'none', color: 'unset', width: '80%' }}
                      to={Routes.getExperimentPageRoute(experiment_id)}
                      onClick={active ? (ev) => ev.preventDefault() : (ev) => ev}
                    >
                      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{name}</div>
                    </Link>
                    {/* Edit/Rename Experiment Option */}
                    <IconButton
                      icon={<Icon type='edit' />}
                      onClick={this.handleRenameExperiment}
                      data-experimentid={experiment_id}
                      data-experimentname={name}
                      style={{ marginRight: 10 }}
                    />
                    {/* Delete Experiment option */}
                    <IconButton
                      icon={<i className='far fa-trash-alt' />}
                      onClick={this.handleDeleteExperiment}
                      data-experimentid={experiment_id}
                      data-experimentname={name}
                    />
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
  experiments.sort(Utils.compareExperiments);
  return { experiments };
};

export default connect(mapStateToProps)(ExperimentListView);

import React, { Component } from 'react';
import PropTypes from 'prop-types';
import {
  EditOutlined,
  LeftSquareFilled,
  RightSquareFilled,
  PlusSquareFilled,
} from '@ant-design/icons';
import { Tree, Input, Typography } from '@databricks/design-system';
import { Link } from 'react-router-dom';
import './ExperimentListView.css';
import { Experiment } from '../sdk/MlflowMessages';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { DeleteExperimentModal } from './modals/DeleteExperimentModal';
import { RenameExperimentModal } from './modals/RenameExperimentModal';
import { IconButton } from '../../common/components/IconButton';

export class ExperimentListView extends Component {
  static propTypes = {
    history: PropTypes.object,
    activeExperimentId: PropTypes.string,
    experiments: PropTypes.arrayOf(Experiment).isRequired,
  };

  static defaultProps = {
    activeExperimentId: '0',
  };

  state = {
    hidden: false,
    searchInput: '',
    showCreateExperimentModal: false,
    showDeleteExperimentModal: false,
    showRenameExperimentModal: false,
    selectedExperimentId: '0',
    selectedExperimentName: '',
  };

  handleSearchInputChange = (event) => {
    this.setState({ searchInput: event.target.value });
  };

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

  handleDeleteExperiment = (experimentId, experimentName) => () => {
    this.setState({
      showDeleteExperimentModal: true,
    });
    this.updateSelectedExperiment(experimentId, experimentName);
  };

  handleRenameExperiment = (experimentId, experimentName) => () => {
    this.setState({
      showRenameExperimentModal: true,
    });
    this.updateSelectedExperiment(experimentId, experimentName);
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

  renderListItem = ({ title, key }) => {
    const { activeExperimentId } = this.props;
    const dataTestId =
      activeExperimentId === key ? 'active-experiment-list-item' : 'experiment-list-item';
    return (
      <div style={{ display: 'flex', marginLeft: '8px' }} data-test-id={dataTestId}>
        <Link
          to={Routes.getExperimentPageRoute(key)}
          style={{
            width: '180px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {title}
        </Link>
        <IconButton
          icon={<EditOutlined />}
          onClick={this.handleRenameExperiment(key, title)}
          style={{ marginRight: 5 }}
          data-test-id='rename-experiment-button'
        />
        <IconButton
          icon={<i className='far fa-trash-alt' />}
          onClick={this.handleDeleteExperiment(key, title)}
          // Use a larger margin to avoid overlapping the vertical scrollbar
          style={{ marginRight: 15 }}
          data-test-id='delete-experiment-button'
        />
      </div>
    );
  };

  render() {
    const { searchInput, hidden } = this.state;
    const { experiments, activeExperimentId } = this.props;
    const lowerCasedSearchInput = searchInput.toLowerCase();
    const filteredExperiments = experiments.filter(({ name }) =>
      name.toLowerCase().includes(lowerCasedSearchInput),
    );
    const treeData = filteredExperiments.map(({ name, experiment_id }) => ({
      title: name,
      key: experiment_id,
    }));

    if (hidden) {
      return (
        <RightSquareFilled
          onClick={() => this.setState({ hidden: false })}
          style={{ fontSize: '24px' }}
          title='Show experiment list'
        />
      );
    }

    return (
      <div className='experiment-list-outer-container'>
        <CreateExperimentModal
          isOpen={this.state.showCreateExperimentModal}
          onClose={this.handleCloseCreateExperimentModal}
        />
        <DeleteExperimentModal
          isOpen={this.state.showDeleteExperimentModal}
          onClose={this.handleCloseDeleteExperimentModal}
          activeExperimentId={activeExperimentId}
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
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '8px',
            }}
          >
            <Typography.Title level={2} style={{ margin: 0 }}>
              Experiments
            </Typography.Title>
            <PlusSquareFilled
              onClick={this.handleCreateExperiment}
              style={{
                fontSize: '24px',
                marginLeft: 'auto',
              }}
              title='New Experiment'
              data-test-id='create-experiment-button'
            />
            <LeftSquareFilled
              onClick={() => this.setState({ hidden: true })}
              style={{ fontSize: '24px' }}
              title='Hide experiment list'
            />
          </div>
          <Input
            placeholder='Search Experiments'
            aria-label='search experiments'
            value={searchInput}
            onChange={this.handleSearchInputChange}
            data-test-id='search-experiment-input'
          />
          <div className='experiment-list-container'>
            <Tree
              treeData={treeData}
              dangerouslySetAntdProps={{
                selectable: true,
                multiple: true,
                selectedKeys: [activeExperimentId],
                titleRender: this.renderListItem,
              }}
            />
          </div>
        </div>
      </div>
    );
  }
}

export default ExperimentListView;

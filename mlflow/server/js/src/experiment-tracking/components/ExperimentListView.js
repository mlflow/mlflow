import React, { Component } from 'react';
import PropTypes from 'prop-types';
import {
  EditOutlined,
  LeftSquareFilled,
  RightSquareFilled,
  PlusSquareFilled,
} from '@ant-design/icons';
import { Tree, Input, Typography } from '@databricks/design-system';
import { withRouter } from 'react-router-dom';
import _ from 'lodash';
import './ExperimentListView.css';
import { Experiment } from '../sdk/MlflowMessages';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { DeleteExperimentModal } from './modals/DeleteExperimentModal';
import { RenameExperimentModal } from './modals/RenameExperimentModal';
import { IconButton } from '../../common/components/IconButton';

export class ExperimentListView extends Component {
  static propTypes = {
    history: PropTypes.object.isRequired,
    activeExperimentId: PropTypes.string.isRequired,
    experiments: PropTypes.arrayOf(Experiment).isRequired,
  };

  state = {
    expanded: true,
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

  onSelect = (experimentId) => () => {
    const { history, activeExperimentId } = this.props;
    if (experimentId === activeExperimentId) {
      return;
    }
    history.push(Routes.getExperimentPageRoute(experimentId));
  };

  renderListItem = ({ title, key }) => {
    return (
      <div style={{ display: 'flex', marginLeft: '8px' }}>
        <div
          style={{
            width: '180px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          onClick={this.onSelect(key)}
        >
          {title}
        </div>
        <IconButton
          icon={<EditOutlined />}
          onClick={this.handleRenameExperiment(key, title)}
          style={{ marginRight: 5 }}
        />
        <IconButton
          icon={<i className='far fa-trash-alt' />}
          onClick={this.handleDeleteExperiment(key, title)}
          style={{ marginRight: 5 }}
        />
      </div>
    );
  };

  render() {
    const { searchInput, expanded } = this.state;
    const { experiments, activeExperimentId } = this.props;
    const lowerCasedSearchInput = searchInput.toLowerCase();
    const filteredExperiments = experiments.filter(({ name }) =>
      name.toLowerCase().includes(lowerCasedSearchInput),
    );
    const treeData = filteredExperiments.map(({ name, experiment_id }) => ({
      title: name,
      key: experiment_id,
    }));

    if (!expanded) {
      return (
        <RightSquareFilled
          onClick={() => this.setState({ expanded: true })}
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
                // Align the icon to the right
                marginLeft: 'auto',
              }}
              title='New Experiment'
            />
            <LeftSquareFilled
              onClick={() => this.setState({ expanded: false })}
              style={{ fontSize: '24px' }}
              title='Hide experiment list'
            />
          </div>
          <Input
            placeholder='Search Experiments'
            aria-label='search experiments'
            value={searchInput}
            onChange={this.handleSearchInputChange}
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

export default withRouter(ExperimentListView);

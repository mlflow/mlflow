import React, { Component } from 'react';
import PropTypes from 'prop-types';
import {
  Tree,
  Input,
  Typography,
  CaretDownSquareIcon,
  PlusCircleBorderIcon,
  PencilIcon,
} from '@databricks/design-system';
import { Link, withRouter } from 'react-router-dom';
import { Experiment } from '../sdk/MlflowMessages';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { DeleteExperimentModal } from './modals/DeleteExperimentModal';
import { RenameExperimentModal } from './modals/RenameExperimentModal';
import { IconButton } from '../../common/components/IconButton';

export class ExperimentListView extends Component {
  static propTypes = {
    activeExperimentIds: PropTypes.arrayOf(PropTypes.string).isRequired,
    experiments: PropTypes.arrayOf(Experiment).isRequired,
    history: PropTypes.object.isRequired,
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

  handleCheck = (checkedKeys) => {
    if (checkedKeys.length > 0) {
      const route =
        checkedKeys.length === 1
          ? Routes.getExperimentPageRoute(checkedKeys[0])
          : Routes.getCompareExperimentsPageRoute(checkedKeys);
      this.props.history.push(route);
    }
  };

  renderListItem = ({ title, key }) => {
    const { activeExperimentIds } = this.props;
    const isActive = activeExperimentIds.includes(key);
    const dataTestId = isActive ? 'active-experiment-list-item' : 'experiment-list-item';
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
          icon={<PencilIcon />}
          onClick={this.handleRenameExperiment(key, title)}
          style={{ marginRight: 5 }}
          data-test-id='rename-experiment-button'
        />
        <IconButton
          icon={<i className='far fa-trash-o' />}
          onClick={this.handleDeleteExperiment(key, title)}
          // Use a larger margin to avoid overlapping the vertical scrollbar
          style={{ marginRight: 15 }}
          data-test-id='delete-experiment-button'
        />
      </div>
    );
  };

  render() {
    const { hidden } = this.state;
    if (hidden) {
      return (
        <CaretDownSquareIcon
          rotate={-90}
          onClick={() => this.setState({ hidden: false })}
          css={{ fontSize: '24px' }}
          title='Show experiment list'
        />
      );
    }

    const { searchInput } = this.state;
    const { experiments, activeExperimentIds } = this.props;
    const lowerCasedSearchInput = searchInput.toLowerCase();
    const filteredExperiments = experiments.filter(({ name }) =>
      name.toLowerCase().includes(lowerCasedSearchInput),
    );
    const treeData = filteredExperiments.map(({ name, experiment_id }) => ({
      title: name,
      key: experiment_id,
    }));

    return (
      <div css={classNames.experimentListOuterContainer}>
        <CreateExperimentModal
          isOpen={this.state.showCreateExperimentModal}
          onClose={this.handleCloseCreateExperimentModal}
        />
        <DeleteExperimentModal
          isOpen={this.state.showDeleteExperimentModal}
          onClose={this.handleCloseDeleteExperimentModal}
          activeExperimentIds={activeExperimentIds}
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
            <PlusCircleBorderIcon
              onClick={this.handleCreateExperiment}
              css={{
                fontSize: '24px',
                marginLeft: 'auto',
              }}
              title='New Experiment'
              data-test-id='create-experiment-button'
            />
            <CaretDownSquareIcon
              onClick={() => this.setState({ hidden: true })}
              rotate={90}
              css={{ fontSize: '24px' }}
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
          <div css={classNames.experimentListContainer}>
            <Tree
              treeData={treeData}
              dangerouslySetAntdProps={{
                selectable: true,
                checkable: true,
                multiple: true,
                selectedKeys: activeExperimentIds,
                checkedKeys: activeExperimentIds,
                onCheck: this.handleCheck,
                titleRender: this.renderListItem,
              }}
            />
          </div>
        </div>
      </div>
    );
  }
}

const classNames = {
  experimentListOuterContainer: {
    boxSizing: 'border-box',
    marginLeft: '64px',
  },
  experimentListContainer: {
    overflowY: 'scroll',
    overflowX: 'hidden',
    width: ' 100%',
    height: '90vh',
    marginTop: '8px',
    // Remove an empty space (transparent switcher) in the tree node to align the experiment name
    // to the left.
    '.du-bois-light-tree-switcher': {
      display: 'none',
    },
    '.du-bois-light-tree-checkbox': {
      marginLeft: '4px',
    },
  },
};

export default withRouter(ExperimentListView);

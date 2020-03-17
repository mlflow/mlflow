import React from 'react';
import PropTypes from 'prop-types';
import ModelVersionTable from './ModelVersionTable';
import Utils from '../../utils/Utils';
import { Link } from 'react-router-dom';
import { modelListPageRoute, getCompareModelVersionsPageRoute } from '../routes';
import { Radio, Icon, Descriptions, Menu, Dropdown, Modal } from 'antd';
import { ACTIVE_STAGES } from '../constants';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';
import { EditableNote } from '../../common/components/EditableNote';
import {Button} from "react-bootstrap";

const Stages = {
  ALL: 'ALL',
  ACTIVE: 'ACTIVE',
};

export class ModelView extends React.Component {
  constructor(props) {
    super(props);
    this.onCompare = this.onCompare.bind(this);
  }
  static propTypes = {
    model: PropTypes.shape({
      name: PropTypes.string.isRequired,
      creation_timestamp: PropTypes.number.isRequired,
      last_updated_timestamp: PropTypes.number.isRequired,
    }),
    modelVersions: PropTypes.arrayOf(PropTypes.shape({
      current_stage: PropTypes.string.isRequired,
    })),
    handleEditDescription: PropTypes.func.isRequired,
    handleDelete: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
  };

  state = {
    stageFilter: Stages.ALL,
    showDescriptionEditor: false,
    isDeleteModalVisible: false,
    isDeleteModalConfirmLoading: false,
    runsSelected: {},
  };

  handleStageFilterChange = (e) => {
    this.setState({ stageFilter: e.target.value });
  };

  componentDidMount() {
    document.title = `${this.props.model.name} - MLflow Model`;
  }

  getActiveVersionsCount() {
    const { modelVersions } = this.props;
    return modelVersions
      ? modelVersions.filter((v) => ACTIVE_STAGES.includes(v.current_stage)).length
      : 0;
  }

  handleCancelEditDescription = () => {
    this.setState({ showDescriptionEditor: false });
  };

  handleSubmitEditDescription = (description) => {
    return this.props
      .handleEditDescription(description)
      .then(() => {
        this.setState({ showDescriptionEditor: false });
      });
  };

  startEditingDescription = (e) => {
    e.stopPropagation();
    this.setState({ showDescriptionEditor: true });
  };

  renderBreadCrumbDropdown() {
    const menu = (
      <Menu>
        <Menu.Item onClick={this.showDeleteModal}>Delete</Menu.Item>
      </Menu>
    );
    return (
      <Dropdown overlay={menu} trigger={['click']}>
        <Icon type='caret-down' className='breadcrumb-caret'/>
      </Dropdown>
    );
  }

  showDeleteModal = () => {
    this.setState({ isDeleteModalVisible: true });
  };

  hideDeleteModal = () => {
    this.setState({ isDeleteModalVisible: false });
  };

  showConfirmLoading = () => {
    this.setState({ isDeleteModalConfirmLoading: true });
  };

  hideConfirmLoading = () => {
    this.setState({ isDeleteModalConfirmLoading: false });
  };

  handleDeleteConfirm = () => {
    const { history } = this.props;
    this.showConfirmLoading();
    this.props
      .handleDelete()
      .then(() => {
        history.push(modelListPageRoute);
      })
      .catch((e) => {
        this.hideConfirmLoading();
        Utils.logErrorAndNotifyUser(e);
      });
  };

  onChange = (selectedRowKeys, selectedRows) => {
    const newState = Object.assign({}, this.state);
    newState.runsSelected = {};
    selectedRows.forEach((row) => {
      newState.runsSelected = {
        ...newState.runsSelected,
        [row.run_id]: row.version,
      };
    });
    this.setState(newState);
  };

  onCompare() {
    this.props.history.push(getCompareModelVersionsPageRoute(
        this.props.model.name, this.state.runsSelected));
  }

  render() {
    const { model, modelVersions } = this.props;
    const {
      stageFilter,
      showDescriptionEditor,
      isDeleteModalVisible,
      isDeleteModalConfirmLoading,
    } = this.state;
    const modelName = model.name;
    const chevron = <i className='fas fa-chevron-right breadcrumb-chevron' />;
    const breadcrumbItemClass = 'truncate-text single-line breadcrumb-title';
    const editIcon = <a onClick={this.startEditingDescription}><Icon type='form' /></a>;
    const compareDisabled = Object.keys(this.state.runsSelected).length < 2;
    return (
      <div className='model-view-content'>
        {/* Breadcrumbs */}
        <h1 className='breadcrumb-header'>
          <Link to={modelListPageRoute} className={breadcrumbItemClass}>Registered Models</Link>
          {chevron}
          <span className={breadcrumbItemClass}>{modelName}</span>
          {this.renderBreadCrumbDropdown()}
        </h1>

        {/* Metadata List */}
        <Descriptions className='metadata-list'>
          <Descriptions.Item label='Created Time'>
            {Utils.formatTimestamp(model.creation_timestamp)}
          </Descriptions.Item>
          <Descriptions.Item label='Last Modified'>
            {Utils.formatTimestamp(model.last_updated_timestamp)}
          </Descriptions.Item>
        </Descriptions>

        {/* Page Sections */}
        <CollapsibleSection
          title={<span>Description {showDescriptionEditor ? null : editIcon}</span>}
          forceOpen={showDescriptionEditor}
        >
          <EditableNote
            defaultMarkdown={model.description}
            onSubmit={this.handleSubmitEditDescription}
            onCancel={this.handleCancelEditDescription}
            showEditor={showDescriptionEditor}
          />
        </CollapsibleSection>
        <CollapsibleSection title={(
          <span>
            <div className="ModelView-run-buttons">
            Versions{' '}
              <Radio.Group
                className='active-toggle'
                value={stageFilter}
                onChange={this.handleStageFilterChange}
              >
                <Radio.Button value={Stages.ALL}>All</Radio.Button>
                <Radio.Button value={Stages.ACTIVE}>
                  Active({this.getActiveVersionsCount()})
                </Radio.Button>
              </Radio.Group>
              <Button className="btn-primary" disabled={compareDisabled} onClick={this.onCompare}>
                Compare
              </Button>
            </div>
          </span>
        )}>

          <ModelVersionTable
            activeStageOnly={stageFilter === Stages.ACTIVE}
            modelName={modelName}
            modelVersions={modelVersions}
            onChange={this.onChange}
          />
        </CollapsibleSection>

        {/* Delete Model Dialog */}
        <Modal
          title="Delete Model"
          visible={isDeleteModalVisible}
          confirmLoading={isDeleteModalConfirmLoading}
          onOk={this.handleDeleteConfirm}
          okText='Delete'
          okType='danger'
          onCancel={this.hideDeleteModal}
        >
          <span>Are you sure you want to delete {modelName}? </span>
          <span>This cannot be undone.</span>
        </Modal>
      </div>
    );
  }
}

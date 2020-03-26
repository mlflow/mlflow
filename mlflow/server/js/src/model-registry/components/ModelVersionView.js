import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { modelListPageRoute, getModelPageRoute } from '../routes';
import Utils from '../../common/utils/Utils';
import { ModelStageTransitionDropdown } from './ModelStageTransitionDropdown';
import { Dropdown, Icon, Menu, Modal, Alert, Descriptions, Tooltip } from 'antd';
import {
  ModelVersionStatus,
  StageTagComponents,
  ModelVersionStatusIcons,
  DefaultModelVersionStatusMessages,
  ACTIVE_STAGES,
  MODEL_VERSION_DELETE_MENU_ITEM_DISABLED_TOOLTIP_TEXT,
} from '../constants';
import Routers from '../../experiment-tracking/routes';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';
import { EditableNote } from '../../common/components/EditableNote';

export class ModelVersionView extends React.Component {
  static propTypes = {
    modelName: PropTypes.string,
    modelVersion: PropTypes.object,
    runInfo: PropTypes.object,
    runDisplayName: PropTypes.string,
    handleStageTransitionDropdownSelect: PropTypes.func.isRequired,
    deleteModelVersionApi: PropTypes.func.isRequired,
    handleEditDescription: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
  };

  state = {
    isDeleteModalVisible: false,
    isDeleteModalConfirmLoading: false,
    showDescriptionEditor: false,
  };

  componentDidMount() {
    const pageTitle = `${this.props.modelName} v${this.props.modelVersion.version} - MLflow Model`;
    Utils.updatePageTitle(pageTitle);
  }

  handleDeleteConfirm = () => {
    const { modelName, modelVersion, history } = this.props;
    const version = modelVersion.version;
    this.showConfirmLoading();
    this.props
      .deleteModelVersionApi(modelName, version)
      .then(() => {
        history.push(getModelPageRoute(modelName));
      })
      .catch((e) => {
        this.hideConfirmLoading();
        Utils.logErrorAndNotifyUser(e);
      });
  };

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
        {ACTIVE_STAGES.includes(this.props.modelVersion.current_stage) ?
          (
            <Menu.Item disabled className='delete'>
              <Tooltip title={MODEL_VERSION_DELETE_MENU_ITEM_DISABLED_TOOLTIP_TEXT}>
                Delete
              </Tooltip>
            </Menu.Item>
          ) : (
            <Menu.Item onClick={this.showDeleteModal} className='delete'>
              Delete
            </Menu.Item>
          )}
      </Menu>
    );
    return (
      <Dropdown overlay={menu} trigger={['click']} className='breadcrumb-dropdown'>
        <Icon type='caret-down' className='breadcrumb-caret'/>
      </Dropdown>
    );
  }

  renderStatusAlert() {
    const { status, status_message } = this.props.modelVersion;
    if (status !== ModelVersionStatus.READY) {
      const defaultMessage = DefaultModelVersionStatusMessages[status];
      const type = status === ModelVersionStatus.FAILED_REGISTRATION ? 'error' : 'info';
      return (
        <Alert
          type={type}
          className={`status-alert status-alert-${type}`}
          message={status_message || defaultMessage}
          icon={ModelVersionStatusIcons[status]}
          banner
        />
      );
    }
    return null;
  }

  renderDescriptionEditIcon() {
    return <a onClick={this.startEditingDescription}><Icon type='form' /></a>;
  }

  render() {
    const {
      modelName,
      modelVersion,
      runInfo,
      runDisplayName,
      handleStageTransitionDropdownSelect,
    } = this.props;
    const { status, description } = modelVersion;
    const { isDeleteModalVisible, isDeleteModalConfirmLoading, showDescriptionEditor } = this.state;
    const chevron = <i className='fas fa-chevron-right breadcrumb-chevron' />;
    const breadcrumbItemClass = 'truncate-text single-line breadcrumb-title';
    return (
      <div>
        {/* Breadcrumbs */}
        <h1 className='breadcrumb-header'>
          <Link to={modelListPageRoute} className={breadcrumbItemClass}>Registered Models</Link>
          {chevron}
          <Link to={getModelPageRoute(modelName)} className={breadcrumbItemClass}>
            {modelName}
          </Link>
          {chevron}
          <span className={breadcrumbItemClass}>Version {modelVersion.version}</span>
          {this.renderBreadCrumbDropdown()}
        </h1>
        {this.renderStatusAlert()}

        {/* Metadata List */}
        <Descriptions className='metadata-list'>
          <Descriptions.Item label='Registered At'>
            {Utils.formatTimestamp(modelVersion.creation_timestamp)}
          </Descriptions.Item>
          <Descriptions.Item label='Creator'>{modelVersion.user_id}</Descriptions.Item>
          <Descriptions.Item label='Stage'>
            {status === ModelVersionStatus.READY ? (
              <ModelStageTransitionDropdown
                currentStage={modelVersion.current_stage}
                permissionLevel={modelVersion.permission_level}
                onSelect={handleStageTransitionDropdownSelect}
              />
            ) : StageTagComponents[modelVersion.current_stage]}
          </Descriptions.Item>
          <Descriptions.Item label='Last Modified'>
            {Utils.formatTimestamp(modelVersion.last_updated_timestamp)}
          </Descriptions.Item>
          {runInfo ? (
            <Descriptions.Item label='Source Run'>
              <Link
                to={Routers.getRunPageRoute(runInfo.getExperimentId(), runInfo.getRunUuid())}
              >
                {runDisplayName || runInfo.getRunUuid()}
              </Link>
            </Descriptions.Item>
          ) : null}
        </Descriptions>

        {/* Page Sections */}
        <CollapsibleSection
          title={
            <span>
              Description{' '}
              {!showDescriptionEditor ? this.renderDescriptionEditIcon() : null}
            </span>
          }
          forceOpen={showDescriptionEditor}
        >
          <EditableNote
            defaultMarkdown={description}
            onSubmit={this.handleSubmitEditDescription}
            onCancel={this.handleCancelEditDescription}
            showEditor={showDescriptionEditor}
          />
        </CollapsibleSection>
        <Modal
          title="Delete Model Version"
          visible={isDeleteModalVisible}
          confirmLoading={isDeleteModalConfirmLoading}
          onOk={this.handleDeleteConfirm}
          okText='Delete'
          okType='danger'
          onCancel={this.hideDeleteModal}
        >
          <span>Are you sure you want to delete model version {modelVersion.version}? </span>
          <span>This cannot be undone.</span>
        </Modal>
      </div>
    );
  }
}

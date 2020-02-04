import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { modelListPageRoute, getModelPageRoute } from '../routes';
import Utils from '../../utils/Utils';
import { ModelStageTransitionDropdown} from './ModelStageTransitionDropdown';
import { Dropdown, Icon, Menu, Modal, Alert, Descriptions } from 'antd';
import {
  ModelVersionStatus,
  StageTagComponents,
  ModelVersionStatusIcons,
  DefaultModelVersionStatusMessages,
} from '../constants';
import Routers from '../../Routes';
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

  handleDeleteConfirm = () => {
    const { modelVersion, history } = this.props;
    const { name } = modelVersion.model_version.registered_model;
    this.showConfirmLoading();
    this.props
      .deleteModelVersionApi(modelVersion.model_version)
      .then(() => {
        history.push(getModelPageRoute(name));
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
        <Menu.Item onClick={this.showDeleteModal}>Delete</Menu.Item>
      </Menu>
    );
    return (
      <Dropdown overlay={menu} trigger={['click']}>
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
    const editIcon = <a onClick={this.startEditingDescription}><Icon type='form' /></a>;

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
          {status !== ModelVersionStatus.PENDING_REGISTRATION && this.renderBreadCrumbDropdown()}
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
          title={<span>Description {showDescriptionEditor ? null : editIcon}</span>}
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

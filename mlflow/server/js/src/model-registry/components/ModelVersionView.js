import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { modelListPageRoute, getModelPageRoute } from '../routes';
import { SchemaTable } from './SchemaTable';
import Utils from '../../common/utils/Utils';
import { ModelStageTransitionDropdown } from './ModelStageTransitionDropdown';
import { Dropdown, Icon, Menu, Modal, Alert, Descriptions, Tooltip, message } from 'antd';
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
import { IconButton } from '../../common/components/IconButton';
import EditableTagsTableView from '../../common/components/EditableTagsTableView';
import { getModelVersionTags } from '../reducers';
import { setModelVersionTagApi, deleteModelVersionTagApi } from '../actions';
import { connect } from 'react-redux';

export class ModelVersionViewImpl extends React.Component {
  static propTypes = {
    modelName: PropTypes.string,
    modelVersion: PropTypes.object,
    schema: PropTypes.object,
    runInfo: PropTypes.object,
    runDisplayName: PropTypes.string,
    handleStageTransitionDropdownSelect: PropTypes.func.isRequired,
    deleteModelVersionApi: PropTypes.func.isRequired,
    handleEditDescription: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
    tags: PropTypes.object.isRequired,
    setModelVersionTagApi: PropTypes.func.isRequired,
    deleteModelVersionTagApi: PropTypes.func.isRequired,
  };

  state = {
    isDeleteModalVisible: false,
    isDeleteModalConfirmLoading: false,
    showDescriptionEditor: false,
    isTagsRequestPending: false,
  };

  componentDidMount() {
    const pageTitle = `${this.props.modelName} v${this.props.modelVersion.version} - MLflow Model`;
    Utils.updatePageTitle(pageTitle);
  }

  handleDeleteConfirm = () => {
    const { modelName, modelVersion, history } = this.props;
    const { version } = modelVersion;
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
    return this.props.handleEditDescription(description).then(() => {
      this.setState({ showDescriptionEditor: false });
    });
  };

  startEditingDescription = (e) => {
    e.stopPropagation();
    this.setState({ showDescriptionEditor: true });
  };

  saveFormRef = (formRef) => {
    this.formRef = formRef;
  };

  handleAddTag = (e) => {
    e.preventDefault();
    const { form } = this.formRef.props;
    const { modelName } = this.props;
    const { version } = this.props.modelVersion;
    form.validateFields((err, values) => {
      if (!err) {
        this.setState({ isTagsRequestPending: true });
        this.props
          .setModelVersionTagApi(modelName, version, values.name, values.value)
          .then(() => {
            this.setState({ isTagsRequestPending: false });
            form.resetFields();
          })
          .catch((ex) => {
            this.setState({ isTagsRequestPending: false });
            console.error(ex);
            message.error('Failed to add tag. Error: ' + ex.getUserVisibleError());
          });
      }
    });
  };

  handleSaveEdit = ({ name, value }) => {
    const { modelName } = this.props;
    const { version } = this.props.modelVersion;
    return this.props.setModelVersionTagApi(modelName, version, name, value).catch((ex) => {
      console.error(ex);
      message.error('Failed to set tag. Error: ' + ex.getUserVisibleError());
    });
  };

  handleDeleteTag = ({ name }) => {
    const { modelName } = this.props;
    const { version } = this.props.modelVersion;
    return this.props.deleteModelVersionTagApi(modelName, version, name).catch((ex) => {
      console.error(ex);
      message.error('Failed to delete tag. Error: ' + ex.getUserVisibleError());
    });
  };

  renderBreadCrumbDropdown() {
    const menu = (
      <Menu>
        {ACTIVE_STAGES.includes(this.props.modelVersion.current_stage) ? (
          <Menu.Item disabled className='delete'>
            <Tooltip placement='right' title={MODEL_VERSION_DELETE_MENU_ITEM_DISABLED_TOOLTIP_TEXT}>
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
        <Icon type='caret-down' className='breadcrumb-caret' />
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
    return <IconButton icon={<Icon type='form' />} onClick={this.startEditingDescription} />;
  }

  resolveRunLink() {
    const { modelVersion, runInfo } = this.props;
    if (modelVersion.run_link) {
      return (
        <a target='_blank' href={modelVersion.run_link}>
          {this.resolveRunName()}
        </a>
      );
    } else if (runInfo) {
      return (
        <Link to={Routers.getRunPageRoute(runInfo.getExperimentId(), runInfo.getRunUuid())}>
          {this.resolveRunName()}
        </Link>
      );
    }
    return null;
  }

  resolveRunName() {
    const { modelVersion, runInfo, runDisplayName } = this.props;
    if (modelVersion.run_link) {
      // We use the first 37 chars to stay consistent with runDisplayName, which is typically:
      // Run: [ID]
      return modelVersion.run_link.substr(0, 37) + '...';
    } else if (runInfo) {
      return runDisplayName || runInfo.getRunUuid();
    } else {
      return null;
    }
  }

  render() {
    const {
      modelName,
      modelVersion,
      handleStageTransitionDropdownSelect,
      tags,
      schema,
    } = this.props;
    const { status, description } = modelVersion;
    const {
      isDeleteModalVisible,
      isDeleteModalConfirmLoading,
      showDescriptionEditor,
      isTagsRequestPending,
    } = this.state;
    const chevron = <i className='fas fa-chevron-right breadcrumb-chevron' />;
    const breadcrumbItemClass = 'truncate-text single-line breadcrumb-title';
    return (
      <div>
        {/* Breadcrumbs */}
        <h1 className='breadcrumb-header'>
          <Link to={modelListPageRoute} className={breadcrumbItemClass}>
            Registered Models
          </Link>
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
            ) : (
              StageTagComponents[modelVersion.current_stage]
            )}
          </Descriptions.Item>
          <Descriptions.Item label='Last Modified'>
            {Utils.formatTimestamp(modelVersion.last_updated_timestamp)}
          </Descriptions.Item>
          <Descriptions.Item label='Source Run' className='linked-run'>
            {this.resolveRunLink()}
          </Descriptions.Item>
        </Descriptions>

        {/* Page Sections */}
        <CollapsibleSection
          title={
            <span>
              Description {!showDescriptionEditor ? this.renderDescriptionEditIcon() : null}
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
        <CollapsibleSection title='Tags'>
          <EditableTagsTableView
            wrappedComponentRef={this.saveFormRef}
            handleAddTag={this.handleAddTag}
            handleDeleteTag={this.handleDeleteTag}
            handleSaveEdit={this.handleSaveEdit}
            tags={tags}
            isRequestPending={isTagsRequestPending}
          />
        </CollapsibleSection>
        <CollapsibleSection title='Schema'>
          <SchemaTable schema={schema} />
        </CollapsibleSection>
        <Modal
          title='Delete Model Version'
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

const mapStateToProps = (state, ownProps) => {
  const { modelName } = ownProps;
  const { version } = ownProps.modelVersion;
  const tags = getModelVersionTags(modelName, version, state);
  return { tags };
};
const mapDispatchToProps = { setModelVersionTagApi, deleteModelVersionTagApi };

export const ModelVersionView = connect(mapStateToProps, mapDispatchToProps)(ModelVersionViewImpl);

import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { modelListPageRoute, getModelPageRoute } from '../routes';
import { SchemaTable } from './SchemaTable';
import Utils from '../../common/utils/Utils';
import { ModelStageTransitionDropdown } from './ModelStageTransitionDropdown';
import { Button as AntdButton, Modal, Alert, Descriptions, message } from 'antd';
import {
  ModelVersionStatus,
  ModelVersionStatusIcons,
  DefaultModelVersionStatusMessages
} from '../constants';

import Routers from '../../experiment-tracking/routes';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';
import { EditableNote } from '../../common/components/EditableNote';
import { EditableTagsTableView } from '../../common/components/EditableTagsTableView';
import { getModelVersionTags, getStageComponents } from '../reducers';
import { setModelVersionTagApi, deleteModelVersionTagApi, listModelStagesApi } from '../actions';
import { connect } from 'react-redux';
import { OverflowMenu, PageHeader } from '../../shared/building_blocks/PageHeader';
import { FormattedMessage, injectIntl } from 'react-intl';

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
    listModelStagesApi: PropTypes.func.isRequired,
    stageTagComponents: PropTypes.object,
    modelStageNames: PropTypes.arrayOf(PropTypes.string),
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  state = {
    isDeleteModalVisible: false,
    isDeleteModalConfirmLoading: false,
    showDescriptionEditor: false,
    isTagsRequestPending: false,
  };

  formRef = React.createRef();

  componentDidMount() {
    this.props.listModelStagesApi();
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

  handleAddTag = (values) => {
    const form = this.formRef.current;
    const { modelName } = this.props;
    const { version } = this.props.modelVersion;
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
        message.error(
          this.props.intl.formatMessage(
            {
              defaultMessage: 'Failed to add tag. Error: {userVisibleError}',
              description: 'Text for user visible error when adding tag in model version view',
            },
            {
              userVisibleError: ex.getUserVisibleError(),
            },
          ),
        );
      });
  };

  handleSaveEdit = ({ name, value }) => {
    const { modelName } = this.props;
    const { version } = this.props.modelVersion;
    return this.props.setModelVersionTagApi(modelName, version, name, value).catch((ex) => {
      console.error(ex);
      message.error(
        this.props.intl.formatMessage(
          {
            defaultMessage: 'Failed to set tag. Error: {userVisibleError}',
            description: 'Text for user visible error when setting tag in model version view',
          },
          {
            userVisibleError: ex.getUserVisibleError(),
          },
        ),
      );
    });
  };

  handleDeleteTag = ({ name }) => {
    const { modelName } = this.props;
    const { version } = this.props.modelVersion;
    return this.props.deleteModelVersionTagApi(modelName, version, name).catch((ex) => {
      console.error(ex);
      message.error(
        this.props.intl.formatMessage(
          {
            defaultMessage: 'Failed to delete tag. Error: {userVisibleError}',
            description: 'Text for user visible error when deleting tag in model version view',
          },
          {
            userVisibleError: ex.getUserVisibleError(),
          },
        ),
      );
    });
  };

  shouldHideDeleteOption() {
    return false;
  }

  renderStageDropdown(modelVersion) {
    const { handleStageTransitionDropdownSelect } = this.props;
    return (
      <Descriptions.Item
        key='description-key-stage'
        label={this.props.intl.formatMessage({
          defaultMessage: 'Stage',
          description: 'Label name for stage metadata in model version page',
        })}
      >
        {modelVersion.status === ModelVersionStatus.READY ? (
          <ModelStageTransitionDropdown
            currentStage={modelVersion.current_stage}
            permissionLevel={modelVersion.permission_level}
            onSelect={handleStageTransitionDropdownSelect}
            stageTagComponents={this.props.stageTagComponents}
            modelStageNames={this.props.modelStageNames}
          />
        ) : (
          this.props.stageTagComponents[modelVersion.current_stage]
        )}
      </Descriptions.Item>
    );
  }

  renderRegisteredTimestampDescription(creation_timestamp) {
    return (
      <Descriptions.Item
        key='description-key-register'
        label={this.props.intl.formatMessage({
          defaultMessage: 'Registered At',
          description: 'Label name for registered timestamp metadata in model version page',
        })}
      >
        {Utils.formatTimestamp(creation_timestamp)}
      </Descriptions.Item>
    );
  }

  renderCreatorDescription(user_id) {
    return (
      user_id && (
        <Descriptions.Item
          key='description-key-creator'
          label={this.props.intl.formatMessage({
            defaultMessage: 'Creator',
            description: 'Label name for creator metadata in model version page',
          })}
        >
          {user_id}
        </Descriptions.Item>
      )
    );
  }

  renderLastModifiedDescription(last_updated_timestamp) {
    return (
      <Descriptions.Item
        key='description-key-modified'
        label={this.props.intl.formatMessage({
          defaultMessage: 'Last Modified',
          description: 'Label name for last modified timestamp metadata in model version page',
        })}
      >
        {Utils.formatTimestamp(last_updated_timestamp)}
      </Descriptions.Item>
    );
  }

  renderSourceRunDescription() {
    return (
      <Descriptions.Item
        key='description-key-source-run'
        label={this.props.intl.formatMessage({
          defaultMessage: 'Source Run',
          description: 'Label name for source run metadata in model version page',
        })}
        className='linked-run'
      >
        {this.resolveRunLink()}
      </Descriptions.Item>
    );
  }

  getDescriptions(modelVersion) {
    const defaultOrder = [
      this.renderRegisteredTimestampDescription(modelVersion.creation_timestamp),
      this.renderCreatorDescription(modelVersion.user_id),
      this.renderStageDropdown(modelVersion),
      this.renderLastModifiedDescription(modelVersion.last_updated_timestamp),
      this.renderSourceRunDescription(),
    ];
    return defaultOrder.filter((x) => x !== null);
  }

  renderMetadata(modelVersion) {
    return (
      <Descriptions className='metadata-list'>{this.getDescriptions(modelVersion)}</Descriptions>
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
    return (
      <AntdButton
        data-test-id='descriptionEditButton'
        type='link'
        onClick={this.startEditingDescription}
      >
        <FormattedMessage
          defaultMessage='Edit'
          description='Text for the edit button next to the description section title on
             the model version view page'
        />{' '}
      </AntdButton>
    );
  }

  resolveRunLink() {
    const { modelVersion, runInfo } = this.props;
    if (modelVersion.run_link) {
      return (
        // Reported during ESLint upgrade
        // eslint-disable-next-line react/jsx-no-target-blank
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

  getPageHeader(title, breadcrumbs) {
    const menu = [
      {
        id: 'delete',
        itemName: (
          <FormattedMessage
            defaultMessage='Delete'
            description='Text for delete button on model version view page header'
          />
        ),
        onClick: this.showDeleteModal,
        disabled: this.props.modelStageNames.includes(this.props.modelVersion.current_stage),
      },
    ];
    return (
      <PageHeader title={title} breadcrumbs={breadcrumbs}>
        {!this.shouldHideDeleteOption() && <OverflowMenu menu={menu} />}
      </PageHeader>
    );
  }

  render() {
    const { modelName, modelVersion, tags, schema } = this.props;
    const { description } = modelVersion;
    const {
      isDeleteModalVisible,
      isDeleteModalConfirmLoading,
      showDescriptionEditor,
      isTagsRequestPending,
    } = this.state;
    const title = (
      <FormattedMessage
        defaultMessage='Version {versionNum}'
        description='Title text for model version page'
        values={{ versionNum: modelVersion.version }}
      />
    );
    const breadcrumbs = [
      <Link to={modelListPageRoute}>
        <FormattedMessage
          defaultMessage='Registered Models'
          description='Text for link back to models page under the header on the model version
             view page'
        />
      </Link>,
      <Link data-test-id='breadcrumbRegisteredModel' to={getModelPageRoute(modelName)}>
        {modelName}
      </Link>,
      <span data-test-id='breadcrumbModelVersion'>
        <FormattedMessage
          defaultMessage='Version {versionNum}'
          description='Text for current version under the header on the model version view page'
          values={{ versionNum: modelVersion.version }}
        />
      </span>,
    ];
    return (
      <div>
        {this.getPageHeader(title, breadcrumbs)}
        {this.renderStatusAlert()}

        {/* Metadata List */}
        {this.renderMetadata(modelVersion)}

        {/* Page Sections */}
        <CollapsibleSection
          title={
            <span>
              <FormattedMessage
                defaultMessage='Description'
                description='Title text for the description section on the model version view page'
              />{' '}
              {!showDescriptionEditor ? this.renderDescriptionEditIcon() : null}
            </span>
          }
          forceOpen={showDescriptionEditor}
          defaultCollapsed={!description}
          data-test-id='model-version-description-section'
        >
          <EditableNote
            defaultMarkdown={description}
            onSubmit={this.handleSubmitEditDescription}
            onCancel={this.handleCancelEditDescription}
            showEditor={showDescriptionEditor}
          />
        </CollapsibleSection>
        <div data-test-id='tags-section'>
          <CollapsibleSection
            title={
              <FormattedMessage
                defaultMessage='Tags'
                description='Title text for the tags section on the model versions view page'
              />
            }
            defaultCollapsed={Utils.getVisibleTagValues(tags).length === 0}
            data-test-id='model-version-tags-section'
          >
            <EditableTagsTableView
              innerRef={this.formRef}
              handleAddTag={this.handleAddTag}
              handleDeleteTag={this.handleDeleteTag}
              handleSaveEdit={this.handleSaveEdit}
              tags={tags}
              isRequestPending={isTagsRequestPending}
            />
          </CollapsibleSection>
        </div>
        <CollapsibleSection
          title={
            <FormattedMessage
              defaultMessage='Schema'
              description='Title text for the schema section on the model versions view page'
            />
          }
          data-test-id='model-version-schema-section'
        >
          <SchemaTable schema={schema} />
        </CollapsibleSection>
        <Modal
          title={this.props.intl.formatMessage({
            defaultMessage: 'Delete Model Version',
            description: 'Title text for model version deletion modal in model versions view page',
          })}
          visible={isDeleteModalVisible}
          confirmLoading={isDeleteModalConfirmLoading}
          onOk={this.handleDeleteConfirm}
          okText={this.props.intl.formatMessage({
            defaultMessage: 'Delete',
            description:
              'OK button text for model version deletion modal in model versions view page',
          })}
          okType='danger'
          onCancel={this.hideDeleteModal}
          cancelText={this.props.intl.formatMessage({
            defaultMessage: 'Cancel',
            description:
              'Cancel button text for model version deletion modal in model versions' +
              ' view page',
          })}
        >
          <span>
            <FormattedMessage
              defaultMessage='Are you sure you want to delete model version {versionNum}? This
                 cannot be undone.'
              description='Comment text for model version deletion modal in model versions view
                 page'
              values={{ versionNum: modelVersion.version }}
            />
          </span>
        </Modal>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { modelName } = ownProps;
  const { version } = ownProps.modelVersion;
  const tags = getModelVersionTags(modelName, version, state);
  const stages = getStageComponents(state)
  return { tags, ...stages };
};
const mapDispatchToProps = { setModelVersionTagApi, deleteModelVersionTagApi, listModelStagesApi };

export const ModelVersionView = connect(
  mapStateToProps,
  mapDispatchToProps,
)(injectIntl(ModelVersionViewImpl));

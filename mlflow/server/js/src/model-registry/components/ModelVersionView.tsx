/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Link, NavigateFunction } from '../../common/utils/RoutingUtils';
import { ModelRegistryRoutes } from '../routes';
import { PromoteModelButton } from './PromoteModelButton';
import { SchemaTable } from './SchemaTable';
import Utils from '../../common/utils/Utils';
import { ModelStageTransitionDropdown } from './ModelStageTransitionDropdown';
import { message } from 'antd';
import { Descriptions } from '../../common/components/Descriptions';
import { modelStagesMigrationGuideLink } from '../../common/constants';
import { Alert, Modal, Button, InfoIcon, Tooltip, Typography } from '@databricks/design-system';
import {
  ModelVersionStatus,
  StageLabels,
  StageTagComponents,
  ModelVersionStatusIcons,
  DefaultModelVersionStatusMessages,
  ACTIVE_STAGES,
} from '../constants';
import Routers from '../../experiment-tracking/routes';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';
import { EditableNote } from '../../common/components/EditableNote';
import { EditableTagsTableView } from '../../common/components/EditableTagsTableView';
import { getModelVersionTags } from '../reducers';
import { setModelVersionTagApi, deleteModelVersionTagApi } from '../actions';
import { connect } from 'react-redux';
import { OverflowMenu, PageHeader } from '../../shared/building_blocks/PageHeader';
import { FormattedMessage, type IntlShape, injectIntl } from 'react-intl';
import { extractArtifactPathFromModelSource } from '../utils/VersionUtils';
import { withNextModelsUIContext } from '../hooks/useNextModelsUI';
import { ModelsNextUIToggleSwitch } from './ModelsNextUIToggleSwitch';
import { shouldShowModelsNextUI } from '../../common/utils/FeatureUtils';
import { ModelVersionViewAliasEditor } from './aliases/ModelVersionViewAliasEditor';
import type { ModelEntity } from '../../experiment-tracking/types';

type ModelVersionViewImplProps = {
  modelName?: string;
  modelVersion?: any;
  modelEntity?: ModelEntity;
  schema?: any;
  activities?: Record<string, unknown>[];
  transitionRequests?: Record<string, unknown>[];
  onCreateComment: (...args: any[]) => any;
  onEditComment: (...args: any[]) => any;
  onDeleteComment: (...args: any[]) => any;
  runInfo?: any;
  runDisplayName?: string;
  handleStageTransitionDropdownSelect: (...args: any[]) => any;
  deleteModelVersionApi: (...args: any[]) => any;
  handleEditDescription: (...args: any[]) => any;
  onAliasesModified: () => void;
  navigate: NavigateFunction;
  tags: any;
  setModelVersionTagApi: (...args: any[]) => any;
  deleteModelVersionTagApi: (...args: any[]) => any;
  intl: IntlShape;
  usingNextModelsUI: boolean;
};

type ModelVersionViewImplState = any;

export class ModelVersionViewImpl extends React.Component<ModelVersionViewImplProps, ModelVersionViewImplState> {
  state = {
    isDeleteModalVisible: false,
    isDeleteModalConfirmLoading: false,
    showDescriptionEditor: false,
    isTagsRequestPending: false,
  };

  formRef = React.createRef();

  componentDidMount() {
    const pageTitle = `${this.props.modelName} v${this.props.modelVersion.version} - MLflow Model`;
    Utils.updatePageTitle(pageTitle);
  }

  handleDeleteConfirm = () => {
    const { modelName = '', modelVersion, navigate } = this.props;
    const { version } = modelVersion;
    this.showConfirmLoading();
    this.props
      .deleteModelVersionApi(modelName, version)
      .then(() => {
        navigate(ModelRegistryRoutes.getModelPageRoute(modelName));
      })
      .catch((e: any) => {
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

  handleSubmitEditDescription = (description: any) => {
    return this.props.handleEditDescription(description).then(() => {
      this.setState({ showDescriptionEditor: false });
    });
  };

  startEditingDescription = (e: any) => {
    e.stopPropagation();
    this.setState({ showDescriptionEditor: true });
  };

  handleAddTag = (values: any) => {
    const form = this.formRef.current;
    const { modelName } = this.props;
    const { version } = this.props.modelVersion;
    this.setState({ isTagsRequestPending: true });
    this.props
      .setModelVersionTagApi(modelName, version, values.name, values.value)
      .then(() => {
        this.setState({ isTagsRequestPending: false });
        (form as any).resetFields();
      })
      .catch((ex: any) => {
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

  handleSaveEdit = ({ name, value }: any) => {
    const { modelName } = this.props;
    const { version } = this.props.modelVersion;
    return this.props.setModelVersionTagApi(modelName, version, name, value).catch((ex: any) => {
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

  handleDeleteTag = ({ name }: any) => {
    const { modelName } = this.props;
    const { version } = this.props.modelVersion;
    return this.props.deleteModelVersionTagApi(modelName, version, name).catch((ex: any) => {
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

  renderStageDropdown(modelVersion: any) {
    const { handleStageTransitionDropdownSelect } = this.props;
    return (
      <Descriptions.Item
        key="description-key-stage"
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
          />
        ) : (
          StageTagComponents[modelVersion.current_stage]
        )}
      </Descriptions.Item>
    );
  }

  renderDisabledStage(modelVersion: any) {
    const tooltipContent = (
      <FormattedMessage
        defaultMessage="Stages have been deprecated in the new Model Registry UI. Learn how to 
      migrate models <link>here</link>."
        description="Tooltip content for the disabled stage metadata in model version page"
        values={{
          link: (chunks: any) => (
            <Typography.Link href={modelStagesMigrationGuideLink} openInNewTab>
              {chunks}
            </Typography.Link>
          ),
        }}
      />
    );
    return (
      <Descriptions.Item
        key="description-key-stage-disabled"
        label={this.props.intl.formatMessage({
          defaultMessage: 'Stage (deprecated)',
          description: 'Label name for the deprecated stage metadata in model version page',
        })}
      >
        <div css={{ display: 'flex', alignItems: 'center' }}>
          {StageLabels[modelVersion.current_stage]}
          <Tooltip title={tooltipContent} placement="bottom">
            <InfoIcon css={{ paddingLeft: '4px' }} />
          </Tooltip>
        </div>
      </Descriptions.Item>
    );
  }

  renderRegisteredTimestampDescription(creation_timestamp: any) {
    return (
      <Descriptions.Item
        key="description-key-register"
        label={this.props.intl.formatMessage({
          defaultMessage: 'Registered At',
          description: 'Label name for registered timestamp metadata in model version page',
        })}
      >
        {Utils.formatTimestamp(creation_timestamp)}
      </Descriptions.Item>
    );
  }

  renderCreatorDescription(user_id: any) {
    return (
      user_id && (
        <Descriptions.Item
          key="description-key-creator"
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

  renderLastModifiedDescription(last_updated_timestamp: any) {
    return (
      <Descriptions.Item
        key="description-key-modified"
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
        key="description-key-source-run"
        label={this.props.intl.formatMessage({
          defaultMessage: 'Source Run',
          description: 'Label name for source run metadata in model version page',
        })}
        // @ts-expect-error TS(2322): Type '{ children: Element | null; key: string; lab... Remove this comment to see the full error message
        className="linked-run"
      >
        {this.resolveRunLink()}
      </Descriptions.Item>
    );
  }

  renderCopiedFromLink() {
    const { source } = this.props.modelVersion;
    const modelUriRegex = /^models:\/[^/]+\/[^/]+$/;
    if (!source || !modelUriRegex.test(source)) {
      return null;
    }
    const sourceParts = source.split('/');
    const sourceModelName = sourceParts[1];
    const sourceModelVersion = sourceParts[2];
    const link = (
      <>
        <Link
          data-test-id="copied-from-link"
          to={ModelRegistryRoutes.getModelVersionPageRoute(sourceModelName, sourceModelVersion)}
        >
          {sourceModelName}
        </Link>
        &nbsp;
        <FormattedMessage
          defaultMessage="(Version {sourceModelVersion})"
          description="Version number of the source model version"
          values={{ sourceModelVersion }}
        />
      </>
    );
    return (
      <Descriptions.Item
        key="description-key-copied-from"
        label={this.props.intl.formatMessage({
          defaultMessage: 'Copied from',
          description: 'Label name for source model version metadata in model version page',
        })}
      >
        {link}
      </Descriptions.Item>
    );
  }

  renderAliasEditor = () => {
    // Extract aliases for the currently displayed model version from the model entity object
    const currentVersion = this.props.modelVersion.version;
    const currentVersionAliases =
      this.props.modelEntity?.aliases?.filter(({ version }) => version === currentVersion).map(({ alias }) => alias) ||
      [];
    return (
      <Descriptions.Item
        key="description-key-aliases"
        label={this.props.intl.formatMessage({
          defaultMessage: 'Aliases',
          description: 'Aliases section in the metadata on model version page',
        })}
      >
        <ModelVersionViewAliasEditor
          aliases={currentVersionAliases}
          version={this.props.modelVersion.version}
          modelEntity={this.props.modelEntity}
          onAliasesModified={this.props.onAliasesModified}
        />
      </Descriptions.Item>
    );
  };

  getDescriptions(modelVersion: any) {
    const { usingNextModelsUI } = this.props;

    const defaultOrder = [
      this.renderRegisteredTimestampDescription(modelVersion.creation_timestamp),
      this.renderCreatorDescription(modelVersion.user_id),
      this.renderLastModifiedDescription(modelVersion.last_updated_timestamp),
      this.renderSourceRunDescription(),
      this.renderCopiedFromLink(),
      usingNextModelsUI ? this.renderAliasEditor() : this.renderStageDropdown(modelVersion),
      usingNextModelsUI ? this.renderDisabledStage(modelVersion) : null,
    ];
    return defaultOrder.filter((item) => item !== null);
  }

  renderMetadata(modelVersion: any) {
    return (
      // @ts-expect-error TS(2322): Type '{ children: any[]; className: string; }' is ... Remove this comment to see the full error message
      <Descriptions className="metadata-list">{this.getDescriptions(modelVersion)}</Descriptions>
    );
  }

  renderStatusAlert() {
    const { status, status_message } = this.props.modelVersion;
    if (status !== ModelVersionStatus.READY) {
      const defaultMessage = DefaultModelVersionStatusMessages[status];
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore - OSS specific ignore
      const type = status === ModelVersionStatus.FAILED_REGISTRATION ? 'error' : 'info';
      return (
        <Alert
          type={type}
          className={`status-alert status-alert-${type}`}
          message={status_message || defaultMessage}
          // @ts-expect-error TS(2322): Type '{ type: "error" | "info"; className: string;... Remove this comment to see the full error message
          icon={ModelVersionStatusIcons[status]}
          banner
        />
      );
    }
    return null;
  }

  renderDescriptionEditIcon() {
    return (
      <Button
        componentId="codegen_mlflow_app_src_model-registry_components_modelversionview.tsx_516"
        data-test-id="descriptionEditButton"
        type="link"
        onClick={this.startEditingDescription}
      >
        <FormattedMessage
          defaultMessage="Edit"
          description="Text for the edit button next to the description section title on
             the model version view page"
        />{' '}
      </Button>
    );
  }

  resolveRunLink() {
    const { modelVersion, runInfo } = this.props;
    if (modelVersion.run_link) {
      return (
        // Reported during ESLint upgrade
        // eslint-disable-next-line react/jsx-no-target-blank
        <a target="_blank" href={modelVersion.run_link}>
          {this.resolveRunName()}
        </a>
      );
    } else if (runInfo) {
      let artifactPath = null;
      const modelSource = this.props.modelVersion?.source;
      if (modelSource) {
        artifactPath = extractArtifactPathFromModelSource(modelSource, runInfo.getRunUuid());
      }
      return (
        <Link to={Routers.getRunPageRoute(runInfo.getExperimentId(), runInfo.getRunUuid(), artifactPath)}>
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

  renderPomoteModelButton() {
    const { modelVersion, usingNextModelsUI, navigate } = this.props;
    return usingNextModelsUI ? <PromoteModelButton modelVersion={modelVersion} /> : null;
  }

  getPageHeader(title: any, breadcrumbs: any) {
    const menu = [
      {
        id: 'delete',
        itemName: (
          <FormattedMessage
            defaultMessage="Delete"
            description="Text for delete button on model version view page header"
          />
        ),
        onClick: this.showDeleteModal,
        disabled: ACTIVE_STAGES.includes(this.props.modelVersion.current_stage),
      },
    ];
    return (
      <PageHeader title={title} breadcrumbs={breadcrumbs}>
        {!this.shouldHideDeleteOption() && <OverflowMenu menu={menu} />}
        {this.renderPomoteModelButton()}
      </PageHeader>
    );
  }

  render() {
    const { modelName = '', modelVersion, tags, schema } = this.props;
    const { description } = modelVersion;
    const { isDeleteModalVisible, isDeleteModalConfirmLoading, showDescriptionEditor, isTagsRequestPending } =
      this.state;
    const title = (
      <FormattedMessage
        defaultMessage="Version {versionNum}"
        description="Title text for model version page"
        values={{ versionNum: modelVersion.version }}
      />
    );
    const breadcrumbs = [
      <Link to={ModelRegistryRoutes.modelListPageRoute}>
        <FormattedMessage
          defaultMessage="Registered Models"
          description="Text for link back to models page under the header on the model version
             view page"
        />
      </Link>,
      <Link data-test-id="breadcrumbRegisteredModel" to={ModelRegistryRoutes.getModelPageRoute(modelName)}>
        {modelName}
      </Link>,
    ];
    return (
      <div>
        {this.getPageHeader(title, breadcrumbs)}
        {this.renderStatusAlert()}

        {/* Metadata List */}
        {this.renderMetadata(modelVersion)}

        {/* New models UI switch */}
        {shouldShowModelsNextUI() && (
          <div css={{ marginTop: 8, display: 'flex', justifyContent: 'flex-end' }}>
            <ModelsNextUIToggleSwitch />
          </div>
        )}

        {/* Page Sections */}
        <CollapsibleSection
          title={
            <span>
              <FormattedMessage
                defaultMessage="Description"
                description="Title text for the description section on the model version view page"
              />{' '}
              {!showDescriptionEditor ? this.renderDescriptionEditIcon() : null}
            </span>
          }
          forceOpen={showDescriptionEditor}
          defaultCollapsed={!description}
          data-test-id="model-version-description-section"
        >
          <EditableNote
            defaultMarkdown={description}
            onSubmit={this.handleSubmitEditDescription}
            onCancel={this.handleCancelEditDescription}
            showEditor={showDescriptionEditor}
          />
        </CollapsibleSection>
        <div data-test-id="tags-section">
          <CollapsibleSection
            title={
              <FormattedMessage
                defaultMessage="Tags"
                description="Title text for the tags section on the model versions view page"
              />
            }
            defaultCollapsed={Utils.getVisibleTagValues(tags).length === 0}
            data-test-id="model-version-tags-section"
          >
            <EditableTagsTableView
              // @ts-expect-error TS(2322): Type '{ innerRef: RefObject<unknown>; handleAddTag... Remove this comment to see the full error message
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
              defaultMessage="Schema"
              description="Title text for the schema section on the model versions view page"
            />
          }
          data-test-id="model-version-schema-section"
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
            description: 'OK button text for model version deletion modal in model versions view page',
          })}
          // @ts-expect-error TS(2322): Type '{ children: Element; title: any; visible: bo... Remove this comment to see the full error message
          okType="danger"
          onCancel={this.hideDeleteModal}
          cancelText={this.props.intl.formatMessage({
            defaultMessage: 'Cancel',
            description: 'Cancel button text for model version deletion modal in model versions view page',
          })}
        >
          <span>
            <FormattedMessage
              defaultMessage="Are you sure you want to delete model version {versionNum}? This
                 cannot be undone."
              description="Comment text for model version deletion modal in model versions view
                 page"
              values={{ versionNum: modelVersion.version }}
            />
          </span>
        </Modal>
      </div>
    );
  }
}

const mapStateToProps = (state: any, ownProps: any) => {
  const { modelName } = ownProps;
  const { version } = ownProps.modelVersion;
  const tags = getModelVersionTags(modelName, version, state);
  return { tags };
};
const mapDispatchToProps = { setModelVersionTagApi, deleteModelVersionTagApi };

export const ModelVersionView = connect(
  mapStateToProps,
  mapDispatchToProps,
)(withNextModelsUIContext(injectIntl<'intl', ModelVersionViewImplProps>(ModelVersionViewImpl)));

/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { sortBy } from 'lodash';
import type { NavigateFunction } from '../../common/utils/RoutingUtils';
import { Link } from '../../common/utils/RoutingUtils';
import { ModelRegistryRoutes } from '../routes';
import { TagAssignmentModal } from '../../common/components/TagAssignmentModal';
import { TagList } from '../../common/components/TagList';
import { PromoteModelButton } from './PromoteModelButton';
import { SchemaTable } from './SchemaTable';
import Utils from '../../common/utils/Utils';
import { ModelStageTransitionDropdown } from './ModelStageTransitionDropdown';
import { Descriptions } from '../../common/components/Descriptions';
import { modelStagesMigrationGuideLink } from '../../common/constants';
import { Alert, Modal, Button, InfoSmallIcon, LegacyTooltip, Typography } from '@databricks/design-system';
import {
  ModelVersionStatus,
  StageLabels,
  StageTagComponents,
  ModelVersionStatusIcons,
  DefaultModelVersionStatusMessages,
  ACTIVE_STAGES,
  type ModelVersionActivity,
  type PendingModelVersionActivity,
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
import { shouldShowModelsNextUI, shouldUseSharedTaggingUI } from '../../common/utils/FeatureUtils';
import { ModelVersionViewAliasEditor } from './aliases/ModelVersionViewAliasEditor';
import type { ModelEntity, RunInfoEntity } from '../../experiment-tracking/types';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import type { KeyValueEntity } from '../../common/types';

type ModelVersionViewImplProps = {
  modelName?: string;
  modelVersion?: any;
  modelEntity?: ModelEntity;
  schema?: any;
  activities?: ModelVersionActivity[];
  transitionRequests?: Record<string, unknown>[];
  onCreateComment: (...args: any[]) => any;
  onEditComment: (...args: any[]) => any;
  onDeleteComment: (...args: any[]) => any;
  runInfo?: RunInfoEntity;
  runDisplayName?: string;
  handleStageTransitionDropdownSelect: (
    activity: PendingModelVersionActivity,
    comment?: string,
    archiveExistingVersions?: boolean,
  ) => void;
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
    isTagAssignmentModalVisible: false,
    isSavingTags: false,
    tagSavingError: undefined,
  };

  formRef = React.createRef();

  sharedTaggingUIEnabled = shouldUseSharedTaggingUI();

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

  getTags = () =>
    sortBy(
      Utils.getVisibleTagValues(this.props.tags).map(([key, value]) => ({
        key,
        name: key,
        value,
      })),
      'name',
    );

  handleCloseTagAssignmentModal = () => {
    this.setState({ isTagAssignmentModalVisible: false, tagSavingError: undefined });
  };

  handleEditTags = () => {
    this.setState({ isTagAssignmentModalVisible: true, tagSavingError: undefined });
  };

  handleSaveTags = (newTags: KeyValueEntity[], deletedTags: KeyValueEntity[]): Promise<void> => {
    this.setState({ isSavingTags: true });

    const { modelName } = this.props;
    const { version } = this.props.modelVersion;

    const newTagsToSet = newTags.map(({ key, value }) =>
      this.props.setModelVersionTagApi(modelName, version, key, value),
    );

    const deletedTagsToDelete = deletedTags.map(({ key }) =>
      this.props.deleteModelVersionTagApi(modelName, version, key),
    );

    return Promise.all([...newTagsToSet, ...deletedTagsToDelete])
      .then(() => {
        this.setState({ isSavingTags: false });
      })
      .catch((error: ErrorWrapper | Error) => {
        const message = error instanceof ErrorWrapper ? error.getMessageField() : error.message;

        this.setState({ isSavingTags: false, tagSavingError: message });
      });
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
      .catch((ex: ErrorWrapper | Error) => {
        this.setState({ isTagsRequestPending: false });
        // eslint-disable-next-line no-console -- TODO(FEINF-3587)
        console.error(ex);

        const userVisibleError = ex instanceof ErrorWrapper ? ex.getMessageField() : ex.message;

        Utils.displayGlobalErrorNotification(
          this.props.intl.formatMessage(
            {
              defaultMessage: 'Failed to add tag. Error: {userVisibleError}',
              description: 'Text for user visible error when adding tag in model version view',
            },
            {
              userVisibleError,
            },
          ),
        );
      });
  };

  handleSaveEdit = ({ name, value }: any) => {
    const { modelName } = this.props;
    const { version } = this.props.modelVersion;
    return this.props.setModelVersionTagApi(modelName, version, name, value).catch((ex: ErrorWrapper | Error) => {
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      console.error(ex);

      const userVisibleError = ex instanceof ErrorWrapper ? ex.getMessageField() : ex.message;

      Utils.displayGlobalErrorNotification(
        this.props.intl.formatMessage(
          {
            defaultMessage: 'Failed to set tag. Error: {userVisibleError}',
            description: 'Text for user visible error when setting tag in model version view',
          },
          {
            userVisibleError,
          },
        ),
      );
    });
  };

  handleDeleteTag = ({ name }: any) => {
    const { modelName } = this.props;
    const { version } = this.props.modelVersion;
    return this.props.deleteModelVersionTagApi(modelName, version, name).catch((ex: ErrorWrapper | Error) => {
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      console.error(ex);

      const userVisibleError = ex instanceof ErrorWrapper ? ex.getMessageField() : ex.message;

      Utils.displayGlobalErrorNotification(
        this.props.intl.formatMessage(
          {
            defaultMessage: 'Failed to delete tag. Error: {userVisibleError}',
            description: 'Text for user visible error when deleting tag in model version view',
          },
          {
            userVisibleError,
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
            <Typography.Link
              componentId="codegen_mlflow_app_src_model-registry_components_modelversionview.tsx_301"
              href={modelStagesMigrationGuideLink}
              openInNewTab
            >
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
          <LegacyTooltip title={tooltipContent} placement="bottom">
            <InfoSmallIcon css={{ paddingLeft: '4px' }} />
          </LegacyTooltip>
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
        {Utils.formatTimestamp(creation_timestamp, this.props.intl)}
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
        {Utils.formatTimestamp(last_updated_timestamp, this.props.intl)}
      </Descriptions.Item>
    );
  }

  renderSourceRunDescription() {
    // We don't show the source run link if the model version is not created from a run
    if (!this.props.modelVersion?.run_id) {
      return null;
    }
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
          data-testid="copied-from-link"
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
      <Descriptions columns={5} className="metadata-list">
        {this.getDescriptions(modelVersion)}
      </Descriptions>
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
          className={`mlflow-status-alert mlflow-status-alert-${type}`}
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
        data-testid="descriptionEditButton"
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
        artifactPath = extractArtifactPathFromModelSource(modelSource, runInfo.runUuid);
      }
      return (
        <Link to={Routers.getRunPageRoute(runInfo.experimentId, runInfo.runUuid, artifactPath)}>
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
      return runDisplayName || runInfo.runUuid;
    } else {
      return null;
    }
  }

  renderPomoteModelButton() {
    const { modelVersion, usingNextModelsUI, navigate } = this.props;
    return usingNextModelsUI ? <PromoteModelButton modelVersion={modelVersion} /> : null;
  }

  renderTags() {
    if (!this.sharedTaggingUIEnabled) {
      return null;
    }

    return (
      <Descriptions columns={1} data-testid="model-view-tags">
        <Descriptions.Item label="Tags">
          <TagList tags={this.getTags()} onEdit={this.handleEditTags} />
        </Descriptions.Item>
      </Descriptions>
    );
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
      // eslint-disable-next-line react/jsx-key
      <Link to={ModelRegistryRoutes.modelListPageRoute}>
        <FormattedMessage
          defaultMessage="Registered Models"
          description="Text for link back to models page under the header on the model version
             view page"
        />
      </Link>,
      // eslint-disable-next-line react/jsx-key
      <Link data-testid="breadcrumbRegisteredModel" to={ModelRegistryRoutes.getModelPageRoute(modelName)}>
        {modelName}
      </Link>,
    ];
    return (
      <div>
        <TagAssignmentModal
          isLoading={this.state.isSavingTags}
          error={this.state.tagSavingError}
          visible={this.state.isTagAssignmentModalVisible}
          initialTags={this.getTags()}
          componentIdPrefix="model-version-view"
          onSubmit={this.handleSaveTags}
          onClose={this.handleCloseTagAssignmentModal}
        />
        {this.getPageHeader(title, breadcrumbs)}
        {this.renderStatusAlert()}

        {/* Metadata List */}
        {this.renderMetadata(modelVersion)}
        {this.renderTags()}

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
          data-testid="model-version-description-section"
        >
          <EditableNote
            defaultMarkdown={description}
            onSubmit={this.handleSubmitEditDescription}
            onCancel={this.handleCancelEditDescription}
            showEditor={showDescriptionEditor}
          />
        </CollapsibleSection>
        {!this.sharedTaggingUIEnabled && (
          <div data-testid="tags-section">
            <CollapsibleSection
              title={
                <FormattedMessage
                  defaultMessage="Tags"
                  description="Title text for the tags section on the model versions view page"
                />
              }
              defaultCollapsed={Utils.getVisibleTagValues(tags).length === 0}
              data-testid="model-version-tags-section"
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
        )}
        <CollapsibleSection
          title={
            <FormattedMessage
              defaultMessage="Schema"
              description="Title text for the schema section on the model versions view page"
            />
          }
          data-testid="model-version-schema-section"
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

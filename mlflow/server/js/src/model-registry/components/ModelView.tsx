/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { ModelVersionTable } from './ModelVersionTable';
import Utils from '../../common/utils/Utils';
import type { NavigateFunction } from '../../common/utils/RoutingUtils';
import { Link } from '../../common/utils/RoutingUtils';
import { ModelRegistryRoutes } from '../routes';
import {
  // prettier-ignore
  ACTIVE_STAGES,
  MODEL_VERSIONS_PER_PAGE_COMPACT,
  MODEL_VERSIONS_SEARCH_TIMESTAMP_FIELD,
} from '../constants';
import { CollapsibleSection } from '../../common/components/CollapsibleSection';
import { EditableNote } from '../../common/components/EditableNote';
import { EditableTagsTableView } from '../../common/components/EditableTagsTableView';
import { getRegisteredModelTags } from '../reducers';
import { setRegisteredModelTagApi, deleteRegisteredModelTagApi } from '../actions';
import { connect } from 'react-redux';
import { OverflowMenu, PageHeader } from '../../shared/building_blocks/PageHeader';
import { FormattedMessage, type IntlShape, injectIntl } from 'react-intl';
import {
  Button,
  SegmentedControlGroup,
  SegmentedControlButton,
  DangerModal,
  CursorPagination,
} from '@databricks/design-system';
import type { KeyValueEntity } from '../../common/types';
import { Descriptions } from '../../common/components/Descriptions';
import { TagList } from '../../common/components/TagList';
import type { ModelVersionInfoEntity } from '../../experiment-tracking/types';
import { type ModelEntity } from '../../experiment-tracking/types';
import { shouldShowModelsNextUI, shouldUseSharedTaggingUI } from '../../common/utils/FeatureUtils';
import { ModelsNextUIToggleSwitch } from './ModelsNextUIToggleSwitch';
import { withNextModelsUIContext } from '../hooks/useNextModelsUI';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import type { ColumnSort } from '@tanstack/react-table';
import { sortBy } from 'lodash';

const CREATION_TIMESTAMP_COLUMN_INDEX = 'creation_timestamp';

export const StageFilters = {
  ALL: 'ALL',
  ACTIVE: 'ACTIVE',
};

type ModelViewImplProps = {
  model?: ModelEntity;
  modelVersions?: ModelVersionInfoEntity[];
  handleEditDescription: (...args: any[]) => any;
  handleDelete: (...args: any[]) => any;
  navigate: NavigateFunction;
  showEditPermissionModal: (...args: any[]) => any;
  activePane?: any; // TODO: PropTypes.oneOf(Object.values(PANES))
  emailSubscriptionStatus?: string;
  userLevelEmailSubscriptionStatus?: string;
  handleEmailNotificationPreferenceChange?: (...args: any[]) => any;
  tags: any;
  setRegisteredModelTagApi: (...args: any[]) => any;
  deleteRegisteredModelTagApi: (...args: any[]) => any;
  intl: IntlShape;
  onMetadataUpdated: () => void;
  usingNextModelsUI: boolean;
  orderByKey: string;
  orderByAsc: boolean;
  currentPage: number;
  nextPageToken: string | null;
  onClickNext: () => void;
  onClickPrev: () => void;
  onClickSortableColumn: (fieldName: string | null, isDescending: boolean) => void;
  onSetMaxResult: (key: number) => void;
  maxResultValue: number;
  loading?: boolean;
};

type ModelViewImplState = any;

export class ModelViewImpl extends React.Component<ModelViewImplProps, ModelViewImplState> {
  constructor(props: ModelViewImplProps) {
    super(props);
    this.onCompare = this.onCompare.bind(this);
  }

  state = {
    maxResultsSelection: MODEL_VERSIONS_PER_PAGE_COMPACT,
    stageFilter: StageFilters.ALL,
    showDescriptionEditor: false,
    isDeleteModalVisible: false,
    isDeleteModalConfirmLoading: false,
    runsSelected: {},
    isTagsRequestPending: false,
    updatingEmailPreferences: false,
    isTagAssignmentModalVisible: false,
    isSavingTags: false,
    tagSavingError: undefined,
  };

  formRef = React.createRef();

  sharedTaggingUIEnabled = shouldUseSharedTaggingUI();

  componentDidMount() {
    // @ts-expect-error TS(2532): Object is possibly 'undefined'.
    const pageTitle = `${this.props.model.name} - MLflow Model`;
    Utils.updatePageTitle(pageTitle);
  }

  handleSetMaxResult = ({ key }: { key: number }) => {
    this.props.onSetMaxResult(key);
  };

  getSortFieldName = (column: any) => {
    switch (column) {
      case CREATION_TIMESTAMP_COLUMN_INDEX:
        return MODEL_VERSIONS_SEARCH_TIMESTAMP_FIELD;
      default:
        return null;
    }
  };

  handleSortChange = (params: { sorter: ColumnSort }) => {
    const sorter = params.sorter;
    this.props.onClickSortableColumn(this.getSortFieldName(sorter.id), sorter.desc);
  };

  handleStageFilterChange = (e: any) => {
    this.setState({ stageFilter: e.target.value });
  };

  getActiveVersionsCount() {
    const { modelVersions } = this.props;
    return modelVersions ? modelVersions.filter((v) => ACTIVE_STAGES.includes(v.current_stage)).length : 0;
  }

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

  getOverflowMenuItems() {
    const menuItems = [
      {
        id: 'delete',
        itemName: (
          <FormattedMessage
            defaultMessage="Delete"
            // eslint-disable-next-line max-len
            description="Text for disabled delete button due to active versions on model view page header"
          />
        ),
        onClick: this.showDeleteModal,
        disabled: this.getActiveVersionsCount() > 0,
      },
    ];

    return menuItems;
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
    const { navigate } = this.props;
    this.showConfirmLoading();
    this.props
      .handleDelete()
      .then(() => {
        navigate(ModelRegistryRoutes.modelListPageRoute);
      })
      .catch((e: any) => {
        this.hideConfirmLoading();
        Utils.logErrorAndNotifyUser(e);
      });
  };

  getTags = () =>
    sortBy(
      Utils.getVisibleTagValues(this.props.tags).map((values) => ({
        key: values[0],
        name: values[0],
        value: values[1],
      })),
      'name',
    );

  handleEditTags = () => {
    this.setState({ isTagAssignmentModalVisible: true, tagSavingError: undefined });
  };

  handleCloseTagAssignmentModal = () => {
    this.setState({ isTagAssignmentModalVisible: false, tagSavingError: undefined });
  };

  handleAddTag = (values: any) => {
    const form = this.formRef.current;
    const { model } = this.props;
    // @ts-expect-error TS(2532): Object is possibly 'undefined'.
    const modelName = model.name;
    this.setState({ isTagsRequestPending: true });
    this.props
      .setRegisteredModelTagApi(modelName, values.name, values.value)
      .then(() => {
        this.setState({ isTagsRequestPending: false });
        (form as any).resetFields();
      })
      .catch((ex: ErrorWrapper | Error) => {
        this.setState({ isTagsRequestPending: false });
        // eslint-disable-next-line no-console -- TODO(FEINF-3587)
        console.error(ex);
        const message = ex instanceof ErrorWrapper ? ex.getMessageField() : ex.message;
        Utils.displayGlobalErrorNotification('Failed to add tag. Error: ' + message);
      });
  };

  handleSaveEdit = ({ name, value }: any) => {
    const { model } = this.props;
    // @ts-expect-error TS(2532): Object is possibly 'undefined'.
    const modelName = model.name;
    return this.props.setRegisteredModelTagApi(modelName, name, value).catch((ex: ErrorWrapper | Error) => {
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      console.error(ex);
      const message = ex instanceof ErrorWrapper ? ex.getMessageField() : ex.message;
      Utils.displayGlobalErrorNotification('Failed to set tag. Error: ' + message);
    });
  };

  handleDeleteTag = ({ name }: any) => {
    const { model } = this.props;
    // @ts-expect-error TS(2532): Object is possibly 'undefined'.
    const modelName = model.name;
    return this.props.deleteRegisteredModelTagApi(modelName, name).catch((ex: ErrorWrapper | Error) => {
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      console.error(ex);
      const message = ex instanceof ErrorWrapper ? ex.getMessageField() : ex.message;
      Utils.displayGlobalErrorNotification('Failed to delete tag. Error: ' + message);
    });
  };

  handleSaveTags = (newTags: KeyValueEntity[], deletedTags: KeyValueEntity[]): Promise<void> => {
    this.setState({ isSavingTags: true });
    const { model } = this.props;
    // @ts-expect-error TS(2532): Object is possibly 'undefined'.
    const modelName = model.name;

    const newTagsToSet = newTags.map(({ key, value }) => this.props.setRegisteredModelTagApi(modelName, key, value));

    const deletedTagsToDelete = deletedTags.map(({ key }) => this.props.deleteRegisteredModelTagApi(modelName, key));

    return Promise.all([...newTagsToSet, ...deletedTagsToDelete])
      .then(() => {
        this.setState({ isSavingTags: false });
      })
      .catch((error: ErrorWrapper | Error) => {
        const message = error instanceof ErrorWrapper ? error.getMessageField() : error.message;

        this.setState({ isSavingTags: false, tagSavingError: message });
      });
  };

  onChange = (selectedRowKeys: any, selectedRows: any) => {
    const newState = Object.assign({}, this.state);
    newState.runsSelected = {};
    selectedRows.forEach((row: any) => {
      newState.runsSelected = {
        ...newState.runsSelected,
        [row.version]: row.run_id,
      };
    });
    this.setState(newState);
  };

  onCompare() {
    if (!this.props.model) {
      return;
    }
    this.props.navigate(
      ModelRegistryRoutes.getCompareModelVersionsPageRoute(this.props.model.name, this.state.runsSelected),
    );
  }

  renderDescriptionEditIcon() {
    return (
      <Button
        componentId="codegen_mlflow_app_src_model-registry_components_modelview.tsx_467"
        data-testid="descriptionEditButton"
        type="link"
        css={styles.editButton}
        onClick={this.startEditingDescription}
      >
        <FormattedMessage
          defaultMessage="Edit"
          description="Text for the edit button next to the description section title on
             the model view page"
        />
      </Button>
    );
  }

  renderDetails = () => {
    const { model, modelVersions, tags, currentPage, nextPageToken } = this.props;
    const {
      stageFilter,
      showDescriptionEditor,
      isDeleteModalVisible,
      isDeleteModalConfirmLoading,
      isTagsRequestPending,
    } = this.state;
    // @ts-expect-error TS(2532): Object is possibly 'undefined'.
    const modelName = model.name;
    const compareDisabled = Object.keys(this.state.runsSelected).length < 2;
    return (
      <div css={styles.wrapper}>
        {/* Metadata List */}
        <Descriptions columns={3} data-testid="model-view-metadata">
          <Descriptions.Item
            data-testid="model-view-metadata-item"
            label={this.props.intl.formatMessage({
              defaultMessage: 'Created Time',
              description: 'Label name for the created time under details tab on the model view page',
            })}
          >
            {/* @ts-expect-error TS(2532): Object is possibly 'undefined'. */}
            {Utils.formatTimestamp(model.creation_timestamp, this.props.intl)}
          </Descriptions.Item>
          <Descriptions.Item
            data-testid="model-view-metadata-item"
            label={this.props.intl.formatMessage({
              defaultMessage: 'Last Modified',
              description: 'Label name for the last modified time under details tab on the model view page',
            })}
          >
            {/* @ts-expect-error TS(2532): Object is possibly 'undefined'. */}
            {Utils.formatTimestamp(model.last_updated_timestamp, this.props.intl)}
          </Descriptions.Item>
          {/* Reported during ESLint upgrade */}
          {/* eslint-disable-next-line react/prop-types */}
          {(model as any).user_id && (
            <Descriptions.Item
              data-testid="model-view-metadata-item"
              label={this.props.intl.formatMessage({
                defaultMessage: 'Creator',
                description: 'Lable name for the creator under details tab on the model view page',
              })}
            >
              {/* eslint-disable-next-line react/prop-types */}
              <div>{(model as any).user_id}</div>
            </Descriptions.Item>
          )}
        </Descriptions>
        {this.sharedTaggingUIEnabled && (
          <Descriptions columns={1} data-testid="model-view-tags">
            <Descriptions.Item label="Tags">
              <TagList tags={this.getTags()} onEdit={this.handleEditTags} />
            </Descriptions.Item>
          </Descriptions>
        )}
        {/* Page Sections */}
        <CollapsibleSection
          css={styles.collapsiblePanel}
          title={
            <span>
              <FormattedMessage
                defaultMessage="Description"
                description="Title text for the description section under details tab on the model
                   view page"
              />{' '}
              {!showDescriptionEditor ? this.renderDescriptionEditIcon() : null}
            </span>
          }
          forceOpen={showDescriptionEditor}
          // Reported during ESLint upgrade
          // eslint-disable-next-line react/prop-types
          defaultCollapsed={!(model as any).description}
          data-testid="model-description-section"
        >
          <EditableNote
            defaultMarkdown={(model as any).description}
            onSubmit={this.handleSubmitEditDescription}
            onCancel={this.handleCancelEditDescription}
            showEditor={showDescriptionEditor}
          />
        </CollapsibleSection>
        {!this.sharedTaggingUIEnabled && (
          <div data-testid="tags-section">
            <CollapsibleSection
              css={styles.collapsiblePanel}
              title={
                <FormattedMessage
                  defaultMessage="Tags"
                  description="Title text for the tags section under details tab on the model view
                   page"
                />
              }
              defaultCollapsed={Utils.getVisibleTagValues(tags).length === 0}
              data-testid="model-tags-section"
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
          css={styles.collapsiblePanel}
          title={
            <>
              <div css={styles.versionsTabButtons}>
                <span>
                  <FormattedMessage
                    defaultMessage="Versions"
                    description="Title text for the versions section under details tab on the
                       model view page"
                  />
                </span>
                {!this.props.usingNextModelsUI && (
                  <SegmentedControlGroup
                    componentId="codegen_mlflow_app_src_model-registry_components_modelview.tsx_600"
                    name="stage-filter"
                    value={this.state.stageFilter}
                    onChange={(e) => this.handleStageFilterChange(e)}
                    css={{ fontWeight: 'normal' }}
                  >
                    <SegmentedControlButton value={StageFilters.ALL}>
                      <FormattedMessage
                        defaultMessage="All"
                        description="Tab text to view all versions under details tab on the model view page"
                      />
                    </SegmentedControlButton>
                    <SegmentedControlButton value={StageFilters.ACTIVE}>
                      <FormattedMessage
                        defaultMessage="Active"
                        description="Tab text to view active versions under details tab
                                on the model view page"
                      />{' '}
                      {this.getActiveVersionsCount()}
                    </SegmentedControlButton>
                  </SegmentedControlGroup>
                )}
                <Button
                  componentId="codegen_mlflow_app_src_model-registry_components_modelview.tsx_619"
                  data-testid="compareButton"
                  disabled={compareDisabled}
                  onClick={this.onCompare}
                >
                  <FormattedMessage
                    defaultMessage="Compare"
                    description="Text for compare button to compare versions under details tab
                       on the model view page"
                  />
                </Button>
              </div>
            </>
          }
          data-testid="model-versions-section"
        >
          {shouldShowModelsNextUI() && (
            <div
              css={{
                marginBottom: 8,
                display: 'flex',
                justifyContent: 'flex-end',
              }}
            >
              <ModelsNextUIToggleSwitch />
            </div>
          )}
          <ModelVersionTable
            isLoading={this.props.loading || false}
            activeStageOnly={stageFilter === StageFilters.ACTIVE && !this.props.usingNextModelsUI}
            modelName={modelName}
            modelVersions={modelVersions}
            modelEntity={model}
            onChange={this.onChange}
            onMetadataUpdated={this.props.onMetadataUpdated}
            usingNextModelsUI={this.props.usingNextModelsUI}
            aliases={model?.aliases}
            orderByKey={this.props.orderByKey}
            orderByAsc={this.props.orderByAsc}
            onSortChange={this.handleSortChange}
            getSortFieldName={this.getSortFieldName}
            pagination={
              <div
                data-testid="model-view-pagination-section"
                css={{ width: '100%', alignItems: 'center', display: 'flex' }}
              >
                <div css={{ display: 'flex', justifyContent: 'flex-end', marginTop: 8 }}>
                  <CursorPagination
                    componentId="codegen_mlflow_app_src_model-registry_components_modelview.tsx_646"
                    hasNextPage={Boolean(nextPageToken)}
                    hasPreviousPage={currentPage > 1}
                    onNextPage={this.props.onClickNext}
                    onPreviousPage={this.props.onClickPrev}
                    pageSizeSelect={{
                      onChange: (num) => this.handleSetMaxResult({ key: num }),
                      default: this.props.maxResultValue,
                      options: [10, 25, 50, 100],
                    }}
                  />
                </div>
              </div>
            }
          />
        </CollapsibleSection>

        {/* Delete Model Dialog */}
        <DangerModal
          componentId="codegen_mlflow_app_src_model-registry_components_modelview.tsx_662"
          data-testid="mlflow-input-modal"
          title={this.props.intl.formatMessage({
            defaultMessage: 'Delete Model',
            description: 'Title text for delete model modal on model view page',
          })}
          visible={isDeleteModalVisible}
          confirmLoading={isDeleteModalConfirmLoading}
          onOk={this.handleDeleteConfirm}
          okText={this.props.intl.formatMessage({
            defaultMessage: 'Delete',
            description: 'OK text for delete model modal on model view page',
          })}
          cancelText={this.props.intl.formatMessage({
            defaultMessage: 'Cancel',
            description: 'Cancel text for delete model modal on model view page',
          })}
          onCancel={this.hideDeleteModal}
        >
          <span>
            <FormattedMessage
              defaultMessage="Are you sure you want to delete {modelName}? This cannot be undone."
              description="Confirmation message for delete model modal on model view page"
              values={{ modelName: modelName }}
            />
          </span>
        </DangerModal>
      </div>
    );
  };

  renderMainPanel() {
    return this.renderDetails();
  }

  render() {
    const { model } = this.props;
    // @ts-expect-error TS(2532): Object is possibly 'undefined'.
    const modelName = model.name;

    const breadcrumbs = [
      <Link to={ModelRegistryRoutes.modelListPageRoute} key="registered-models">
        <FormattedMessage
          defaultMessage="Registered Models"
          description="Text for link back to model page under the header on the model view page"
        />
      </Link>,
    ];
    return (
      <div>
        <PageHeader title={modelName} breadcrumbs={breadcrumbs}>
          <OverflowMenu menu={this.getOverflowMenuItems()} />
        </PageHeader>
        {this.renderMainPanel()}
      </div>
    );
  }
}

const mapStateToProps = (state: any, ownProps: any) => {
  const modelName = ownProps.model.name;
  const tags = getRegisteredModelTags(modelName, state);
  return { tags };
};
const mapDispatchToProps = { setRegisteredModelTagApi, deleteRegisteredModelTagApi };

const styles = {
  emailNotificationPreferenceDropdown: (theme: any) => ({
    width: 300,
    marginBottom: theme.spacing.md,
  }),
  emailNotificationPreferenceTip: (theme: any) => ({
    paddingLeft: theme.spacing.sm,
    paddingRight: theme.spacing.sm,
  }),
  collapsiblePanel: (theme: any) => ({
    marginBottom: theme.spacing.md,
  }),
  wrapper: (theme: any) => ({
    /**
     * This seems to be a best and most stable method to catch
     * antd's collapsible section buttons without hacks
     * and using class names.
     */
    'div[role="button"][aria-expanded]': {
      height: theme.general.buttonHeight,
    },
  }),
  editButton: (theme: any) => ({
    marginLeft: theme.spacing.md,
  }),
  versionsTabButtons: (theme: any) => ({
    display: 'flex',
    gap: theme.spacing.md,
    alignItems: 'center',
  }),
};

export const ModelView = connect(
  mapStateToProps,
  mapDispatchToProps,
)(withNextModelsUIContext(injectIntl(ModelViewImpl)));

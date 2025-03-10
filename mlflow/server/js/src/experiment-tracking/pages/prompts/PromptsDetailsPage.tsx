import invariant from 'invariant';
import { usePromptDetailsQuery } from './hooks/usePromptDetailsQuery';
import { Link, useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import { ScrollablePageWrapper } from '../../../common/components/ScrollablePageWrapper';
import {
  Breadcrumb,
  Button,
  ColumnsIcon,
  DropdownMenu,
  GenericSkeleton,
  Header,
  OverflowIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  Space,
  Spacer,
  TableIcon,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
  ZoomMarqueeSelection,
} from '@databricks/design-system';
import { PromptDetailsMetadata } from './components/PromptDetailsMetadata';
import { FormattedMessage } from 'react-intl';
import { PromptVersionsTableMode } from './utils';
import { useMemo, useState } from 'react';
import Routes from '../../routes';
import { CreatePromptModalMode, useCreatePromptModal } from './hooks/useCreatePromptModal';
import { useDeletePromptModal } from './hooks/useDeletePromptModal';
import { PromptVersionsTable } from './components/PromptVersionsTable';
import { useEditRegisteredModelAliasesModal } from '../../../model-registry/hooks/useEditRegisteredModelAliasesModal';
import { usePromptDetailsPageViewState } from './hooks/usePromptDetailsPageViewState';
import { PromptContentPreview } from './components/PromptContentPreview';
import { PromptContentCompare } from './components/PromptContentCompare';
import { NotFoundError, PredefinedError, UnauthorizedError } from '../../../shared/web-shared/errors';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { PromptPageErrorHandler } from './components/PromptPageErrorHandler';
import { first, isEmpty } from 'lodash';

const PromptsDetailsPage = () => {
  const { promptName } = useParams<{ promptName: string }>();
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();

  invariant(promptName, 'Prompt name should be defined');

  const { data: promptDetailsData, refetch, isLoading, error: promptLoadError } = usePromptDetailsQuery({ promptName });

  const { CreatePromptModal, openModal: openCreateVersionModal } = useCreatePromptModal({
    mode: CreatePromptModalMode.CreatePromptVersion,
    registeredPrompt: promptDetailsData?.prompt,
    latestVersion: first(promptDetailsData?.versions),
    onSuccess: async ({ promptVersion }) => {
      await refetch();
      if (promptVersion) {
        setPreviewMode({ version: promptVersion });
      }
    },
  });

  const { DeletePromptModal, openModal: openDeleteModal } = useDeletePromptModal({
    registeredPrompt: promptDetailsData?.prompt,
    onSuccess: () => navigate(Routes.promptsPageRoute),
  });

  const {
    setCompareMode,
    setPreviewMode,
    setTableMode,
    switchSides,
    viewState,
    setSelectedVersion,
    setComparedVersion,
  } = usePromptDetailsPageViewState(promptDetailsData);

  const { mode } = viewState;

  const isEmptyVersions = !isLoading && !promptDetailsData?.versions.length;

  const showPreviewPane =
    !isLoading && !isEmptyVersions && [PromptVersionsTableMode.PREVIEW, PromptVersionsTableMode.COMPARE].includes(mode);

  const selectedVersionEntity = promptDetailsData?.versions.find(
    ({ version }) => version === viewState.selectedVersion,
  );

  const comparedVersionEntity = promptDetailsData?.versions.find(
    ({ version }) => version === viewState.comparedVersion,
  );

  const aliasesByVersion = useMemo(() => {
    const result: Record<string, string[]> = {};
    promptDetailsData?.prompt?.aliases?.forEach(({ alias, version }) => {
      if (!result[version]) {
        result[version] = [];
      }
      result[version].push(alias);
    });
    return result;
  }, [promptDetailsData]);

  const { EditAliasesModal, showEditAliasesModal } = useEditRegisteredModelAliasesModal({
    model: promptDetailsData?.prompt || null,
    onSuccess: refetch,
  });

  // If the load error occurs, throw the error into the error boundary
  if (promptLoadError) {
    throw promptLoadError;
  }

  const breadcrumbs = (
    <Breadcrumb>
      <Breadcrumb.Item>
        <Link to={Routes.promptsPageRoute}>Prompts</Link>
      </Breadcrumb.Item>
    </Breadcrumb>
  );

  if (isLoading) {
    return (
      <ScrollablePageWrapper>
        <PromptsDetailsPage.Skeleton breadcrumbs={breadcrumbs} />
      </ScrollablePageWrapper>
    );
  }

  return (
    <ScrollablePageWrapper css={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        breadcrumbs={breadcrumbs}
        title={promptDetailsData?.prompt?.name}
        buttons={
          <>
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <Button
                  componentId="mlflow.prompts.details.actions"
                  icon={<OverflowIcon />}
                  aria-label="More actions"
                />
              </DropdownMenu.Trigger>
              <DropdownMenu.Content>
                <DropdownMenu.Item componentId="mlflow.prompts.details.actions.delete" onClick={openDeleteModal}>
                  <FormattedMessage
                    defaultMessage="Delete"
                    description="Label for the delete prompt action on the registered prompt details page"
                  />
                </DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Root>
            <Button componentId="mlflow.prompts.details.create" type="primary" onClick={openCreateVersionModal}>
              <FormattedMessage
                defaultMessage="Create prompt version"
                description="Label for the create prompt action on the registered prompt details page"
              />
            </Button>
          </>
        }
      />
      <Spacer shrinks={false} />
      <PromptDetailsMetadata
        promptEntity={promptDetailsData?.prompt}
        promptVersions={promptDetailsData?.versions}
        onTagsUpdated={refetch}
      />
      <div css={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <div css={{ flex: showPreviewPane ? '0 0 320px' : 1, display: 'flex', flexDirection: 'column' }}>
          <Typography.Title level={3}>Prompt versions</Typography.Title>
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <SegmentedControlGroup
              name="mlflow.prompts.details.mode"
              componentId="mlflow.prompts.details.mode"
              value={mode}
              disabled={isLoading}
            >
              <SegmentedControlButton value={PromptVersionsTableMode.PREVIEW} onClick={() => setPreviewMode()}>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <ZoomMarqueeSelection />
                  <FormattedMessage
                    defaultMessage="Preview"
                    description="Label for the preview mode on the registered prompt details page"
                  />
                </div>
              </SegmentedControlButton>
              <SegmentedControlButton value={PromptVersionsTableMode.TABLE} onClick={setTableMode}>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <TableIcon />{' '}
                  <FormattedMessage
                    defaultMessage="List"
                    description="Label for the list mode on the registered prompt details page"
                  />
                </div>
              </SegmentedControlButton>
              <SegmentedControlButton
                disabled={Boolean(!promptDetailsData?.versions.length || promptDetailsData?.versions.length < 2)}
                value={PromptVersionsTableMode.COMPARE}
                onClick={setCompareMode}
              >
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <ColumnsIcon />{' '}
                  <FormattedMessage
                    defaultMessage="Compare"
                    description="Label for the compare mode on the registered prompt details page"
                  />
                </div>
              </SegmentedControlButton>
            </SegmentedControlGroup>
          </div>
          <Spacer shrinks={false} size="sm" />
          <PromptVersionsTable
            isLoading={isLoading}
            registeredPrompt={promptDetailsData?.prompt}
            promptVersions={promptDetailsData?.versions}
            selectedVersion={viewState.selectedVersion}
            comparedVersion={viewState.comparedVersion}
            showEditAliasesModal={showEditAliasesModal}
            aliasesByVersion={aliasesByVersion}
            onUpdateSelectedVersion={setSelectedVersion}
            onUpdateComparedVersion={setComparedVersion}
            mode={mode}
          />
        </div>
        {showPreviewPane && (
          <div css={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <div css={{ borderLeft: `1px solid ${theme.colors.border}`, flex: 1, overflow: 'hidden', display: 'flex' }}>
              {mode === PromptVersionsTableMode.PREVIEW && (
                <PromptContentPreview
                  promptVersion={selectedVersionEntity}
                  onUpdatedContent={refetch}
                  onDeletedVersion={async () => {
                    await refetch().then(({ data }) => {
                      if (!isEmpty(data?.versions) && data?.versions[0].version) {
                        setSelectedVersion(data?.versions[0].version);
                      } else {
                        setTableMode();
                      }
                    });
                  }}
                  aliasesByVersion={aliasesByVersion}
                  showEditAliasesModal={showEditAliasesModal}
                  registeredPrompt={promptDetailsData?.prompt}
                />
              )}
              {mode === PromptVersionsTableMode.COMPARE && (
                <PromptContentCompare
                  baselineVersion={selectedVersionEntity}
                  comparedVersion={comparedVersionEntity}
                  onSwitchSides={switchSides}
                  onEditVersion={setPreviewMode}
                  showEditAliasesModal={showEditAliasesModal}
                  registeredPrompt={promptDetailsData?.prompt}
                  aliasesByVersion={aliasesByVersion}
                />
              )}
            </div>
          </div>
        )}
      </div>
      <Spacer shrinks={false} />
      {EditAliasesModal}
      {CreatePromptModal}
      {DeletePromptModal}
    </ScrollablePageWrapper>
  );
};

PromptsDetailsPage.Skeleton = function PromptsDetailsPageSkeleton({ breadcrumbs }: { breadcrumbs?: React.ReactNode }) {
  const { theme } = useDesignSystemTheme();
  return (
    <>
      <Spacer shrinks={false} />
      <Header
        breadcrumbs={breadcrumbs}
        title={<GenericSkeleton css={{ height: theme.general.heightBase, width: 200 }} />}
        buttons={<GenericSkeleton css={{ height: theme.general.heightBase, width: 120 }} />}
      />
      <Spacer shrinks={false} />
      <TableSkeleton lines={4} />
      <Spacer shrinks={false} />
      <div css={{ display: 'flex', gap: theme.spacing.lg }}>
        <div css={{ flex: '0 0 320px' }}>
          <TableSkeleton lines={6} />
        </div>
        <div css={{ flex: 1 }}>
          <TableSkeleton lines={4} />
        </div>
      </div>
    </>
  );
};

export default withErrorBoundary(
  ErrorUtils.mlflowServices.EXPERIMENTS,
  PromptsDetailsPage,
  undefined,
  PromptPageErrorHandler,
);

import {
  Breadcrumb,
  Button,
  ColumnsIcon,
  DropdownMenu,
  Header,
  OverflowIcon,
  PageWrapper,
  SegmentedControlButton,
  SegmentedControlGroup,
  Spacer,
  TableIcon,
  Typography,
  useDesignSystemTheme,
  ZoomMarqueeSelection,
} from '@databricks/design-system';
import invariant from 'invariant';
import { first } from 'lodash';
import { useCallback, useMemo, useReducer } from 'react';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapperStyles } from '../../../common/components/ScrollablePageWrapper';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { Link, useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import { useEditRegisteredModelAliasesModal } from '../../../model-registry/hooks/useEditRegisteredModelAliasesModal';
import Routes from '../../routes';
import { PromptContentCompare } from './components/PromptContentCompare';
import { PromptContentPreview } from './components/PromptContentPreview';
import { PromptDetailsMetadata } from './components/PromptDetailsMetadata';
import { PromptVersionsTable } from './components/PromptVersionsTable';
import { CreatePromptVersionModalMode, useCreatePromptVersionModal } from './hooks/useCreatePromptVersionModal';
import { useDeletePromptModal } from './hooks/useDeletePromptModal';
import { usePromptDetailsQuery } from './hooks/usePromptDetailsQuery';
import { PromptVersionsTableMode } from './utils';
import { PromptPageErrorHandler } from './components/PromptPageErrorHandler';
import { usePromptDetailsPageViewState } from './hooks/usePromptDetailsPageViewState';

const PromptDetailsPage = () => {
  const { promptName } = useParams<{ promptName: string }>();
  const { theme } = useDesignSystemTheme();

  const navigate = useNavigate();

  invariant(promptName, 'Prompt name should be defined');

  const { data: promptDetailsData, refetch, isLoading, error: promptLoadError } = usePromptDetailsQuery({ promptName });

  const { CreatePromptModal, openModal: openCreateVersionModal } = useCreatePromptVersionModal({
    mode: CreatePromptVersionModalMode.CreatePromptVersion,
    registeredPrompt: promptDetailsData?.prompt,
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

  const { EditAliasesModal, showEditAliasesModal } = useEditRegisteredModelAliasesModal({
    model: promptDetailsData?.prompt || null,
    onSuccess: () => refetch(),
  });

  const isEmptyVersions = !isLoading && !promptDetailsData?.versions.length;

  const showPreviewPane =
    !isLoading && !isEmptyVersions && [PromptVersionsTableMode.PREVIEW, PromptVersionsTableMode.COMPARE].includes(mode);

  const selectedVersionEntity = promptDetailsData?.versions.find(
    ({ version }) => version === viewState.selectedVersion,
  );
  const comparedVersionEntity = promptDetailsData?.versions.find(
    ({ version }) => version === viewState.comparedVersion,
  );

  if (promptLoadError) {
    return <PromptPageErrorHandler error={promptLoadError} />;
  }

  return (
    <PageWrapper css={{ ...ScrollablePageWrapperStyles, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <Spacer shrinks={false} />
      <Header
        breadcrumbs={
          <Breadcrumb>
            <Breadcrumb.Item>
              <Link to={Routes.promptsPageRoute}>Prompts</Link>
            </Breadcrumb.Item>
          </Breadcrumb>
        }
        title={promptDetailsData?.prompt?.name}
        buttons={
          <>
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <Button componentId="TODO" icon={<OverflowIcon />} />
              </DropdownMenu.Trigger>
              <DropdownMenu.Content>
                <DropdownMenu.Item componentId="TODO" onClick={openDeleteModal}>
                  <FormattedMessage defaultMessage="Delete" description="TODO" />
                </DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Root>
            <Button componentId="TODO" type="primary" onClick={openCreateVersionModal}>
              <FormattedMessage defaultMessage="Create prompt version" description="TODO" />
            </Button>
          </>
        }
      />
      <Spacer shrinks={false} />
      <PromptDetailsMetadata promptEntity={promptDetailsData?.prompt} onTagsUpdated={refetch} />
      <div css={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <div css={{ flex: showPreviewPane ? '0 0 320px' : 1, display: 'flex', flexDirection: 'column' }}>
          <Typography.Title level={3}>Prompt versions</Typography.Title>
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <SegmentedControlGroup name="TEMP" componentId="TEMP" value={mode} disabled={isLoading}>
              <SegmentedControlButton value={PromptVersionsTableMode.PREVIEW} onClick={() => setPreviewMode()}>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <ZoomMarqueeSelection />
                  <FormattedMessage defaultMessage="Preview" description="TODO" />
                </div>
              </SegmentedControlButton>
              <SegmentedControlButton value={PromptVersionsTableMode.TABLE} onClick={setTableMode}>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <TableIcon /> <FormattedMessage defaultMessage="List" description="TODO" />
                </div>
              </SegmentedControlButton>
              <SegmentedControlButton
                disabled={Boolean(!promptDetailsData?.versions.length || promptDetailsData?.versions.length < 2)}
                value={PromptVersionsTableMode.COMPARE}
                onClick={setCompareMode}
              >
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <ColumnsIcon /> <FormattedMessage defaultMessage="Compare" description="TODO" />
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
    </PageWrapper>
  );
};

export default withErrorBoundary(
  ErrorUtils.mlflowServices.EXPERIMENTS,
  PromptDetailsPage,
  undefined,
  PromptPageErrorHandler,
);

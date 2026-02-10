import React, { Fragment, useCallback, useMemo, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  BeakerIcon,
  Button,
  ChainIcon,
  CloudModelIcon,
  GearIcon,
  HomeIcon,
  KeyIcon,
  ModelsIcon,
  SegmentedControlGroup,
  SegmentedControlButton,
  Tag,
  TextBoxIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  Typography,
  DesignSystemEventProviderComponentTypes,
  DesignSystemEventProviderAnalyticsEventTypes,
  SidebarCollapseIcon,
  SidebarExpandIcon,
  InfoBookIcon,
  CodeIcon,
} from '@databricks/design-system';
import type { Location } from '../utils/RoutingUtils';
import { Link, matchPath, useLocation, useNavigate, useParams, useSearchParams } from '../utils/RoutingUtils';
import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { ModelRegistryRoutes } from '../../model-registry/routes';
import GatewayRoutes from '../../gateway/routes';
import Routes from '../../experiment-tracking/routes';
import { FormattedMessage } from 'react-intl';
import { useLogTelemetryEvent } from '../../telemetry/hooks/useLogTelemetryEvent';
import { useWorkflowType, WorkflowType } from '../contexts/WorkflowTypeContext';
import {
  getExperimentPageSideNavSectionLabel,
  type ExperimentPageSideNavSectionKey,
  useExperimentPageSideNavConfig,
} from '../../experiment-tracking/pages/experiment-page-tabs/side-nav/constants';
import { ExperimentPageTabName } from '../../experiment-tracking/constants';
import { shouldEnableWorkflowBasedNavigation, shouldEnableWorkspaces } from '../utils/FeatureUtils';
import { AssistantSparkleIcon } from '../../assistant/AssistantIconButton';
import { useAssistant } from '../../assistant/AssistantContext';
import { useExperimentEvaluationRunsData } from '../../experiment-tracking/components/experiment-page/hooks/useExperimentEvaluationRunsData';
import { getExperimentKindForWorkflowType } from '../../experiment-tracking/utils/ExperimentKindUtils';
import { extractWorkspaceFromSearchParams } from '../../workspaces/utils/WorkspaceUtils';
import { MlflowSidebarLink } from './MlflowSidebarLink';
import { MlflowLogo } from './MlflowLogo';
import { HomePageDocsUrl, Version } from '../constants';
import { WorkspaceSelector } from '../../workspaces/components/WorkspaceSelector';
import { MlflowSidebarExperimentItems } from './MlflowSidebarExperimentItems';
import { MlflowSidebarGatewayItems } from './MlflowSidebarGatewayItems';

const isInsideExperiment = (location: Location) =>
  Boolean(matchPath('/experiments/:experimentId/*', location.pathname));
const isHomeActive = (location: Location) => Boolean(matchPath({ path: '/', end: true }, location.pathname));
const isExperimentsActive = (location: Location) =>
  Boolean(
    matchPath({ path: '/experiments', end: true }, location.pathname) ||
    matchPath('/compare-experiments/*', location.pathname),
  );
const isModelsActive = (location: Location) => Boolean(matchPath('/models/*', location.pathname));
const isPromptsActive = (location: Location) => Boolean(matchPath('/prompts/*', location.pathname));
const isGatewayActive = (location: Location) => Boolean(matchPath('/gateway/*', location.pathname));
const isSettingsActive = (location: Location) => Boolean(matchPath('/settings/*', location.pathname));

type MlFlowSidebarMenuDropdownComponentId =
  | 'mlflow_sidebar.create_experiment_button'
  | 'mlflow_sidebar.create_model_button'
  | 'mlflow_sidebar.create_prompt_button';

type NestedMenuItem = {
  key: string;
  icon: React.ReactNode;
  label: React.ReactNode;
  to: string;
  componentId: string;
  isActive: (location: Location) => boolean;
};

type NestedItemsGroup = {
  sectionKey: ExperimentPageSideNavSectionKey;
  items: NestedMenuItem[];
};

type MenuItemWithNested = {
  key: string;
  icon: React.ReactNode;
  linkProps: {
    to: string;
    isActive: (location: Location) => boolean;
    children: React.ReactNode;
  };
  componentId: string;
  dropdownProps?: {
    componentId: MlFlowSidebarMenuDropdownComponentId;
    onClick: () => void;
    children: React.ReactNode;
  };
  nestedItems?: React.ReactNode;
};

const NESTED_ITEMS_UL_CSS = {
  listStyleType: 'none' as const,
  padding: 0,
  margin: 0,
};

const shouldShowGenAIFeatures = (enableWorkflowBasedNavigation: boolean, workflowType: WorkflowType) =>
  !enableWorkflowBasedNavigation || (enableWorkflowBasedNavigation && workflowType === WorkflowType.GENAI);

export function MlflowSidebar({
  showSidebar,
  setShowSidebar,
}: {
  showSidebar: boolean;
  setShowSidebar: (showSidebar: boolean) => void;
}) {
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const { theme } = useDesignSystemTheme();
  const viewId = useMemo(() => uuidv4(), []);
  const enableWorkflowBasedNavigation = shouldEnableWorkflowBasedNavigation();
  // WorkflowType context is always available, but UI is guarded by feature flag
  const { workflowType, setWorkflowType } = useWorkflowType();
  const { experimentId } = useParams();
  const logTelemetryEvent = useLogTelemetryEvent();
  const toggleSidebar = useCallback(() => {
    setShowSidebar(!showSidebar);
  }, [setShowSidebar, showSidebar]);

  // Persist the last selected experiment ID so the nested experiment view
  // stays visible when navigating away from experiment pages
  const lastSelectedExperimentIdRef = useRef<string | null>(null);

  // Update the ref when we're inside an experiment
  if (experimentId && isInsideExperiment(location)) {
    lastSelectedExperimentIdRef.current = experimentId;
  }

  // Callback to clear the persisted experiment ID (used by back button)
  const clearLastSelectedExperiment = useCallback(() => {
    lastSelectedExperimentIdRef.current = null;
  }, []);

  // Use the current experimentId if inside an experiment, otherwise use the persisted one
  const activeExperimentId = isInsideExperiment(location) ? experimentId : lastSelectedExperimentIdRef.current;
  const showNestedExperimentItems = Boolean(activeExperimentId) && shouldEnableWorkflowBasedNavigation();

  const { trainingRuns } = useExperimentEvaluationRunsData({
    experimentId: experimentId || '',
    enabled: Boolean(experimentId) && workflowType === WorkflowType.GENAI,
    filter: '', // not important in this case, we show the runs tab if there are any training runs
  });

  const config = useExperimentPageSideNavConfig({
    experimentKind: getExperimentKindForWorkflowType(workflowType),
    hasTrainingRuns: (trainingRuns?.length ?? 0) > 0,
  });

  const { openPanel, closePanel, isPanelOpen, isLocalServer } = useAssistant();
  const [isAssistantHovered, setIsAssistantHovered] = useState(false);

  const handleAssistantToggle = useCallback(() => {
    if (isPanelOpen) {
      closePanel();
    } else {
      openPanel();
    }
    logTelemetryEvent({
      componentId: 'mlflow.sidebar.assistant_button',
      componentViewId: viewId,
      componentType: DesignSystemEventProviderComponentTypes.Button,
      componentSubType: null,
      eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
    });
  }, [isPanelOpen, closePanel, openPanel, logTelemetryEvent, viewId]);

  const menuItems: MenuItemWithNested[] = useMemo(
    () => [
      {
        key: 'home',
        icon: <HomeIcon />,
        linkProps: {
          to: ExperimentTrackingRoutes.rootRoute,
          isActive: isHomeActive,
          children: <FormattedMessage defaultMessage="Home" description="Sidebar link for home page" />,
        },
        componentId: 'mlflow.sidebar.home_tab_link',
      },
      {
        key: 'experiments',
        icon: <BeakerIcon />,
        linkProps: {
          to: ExperimentTrackingRoutes.experimentsObservatoryRoute,
          isActive: isExperimentsActive,
          children: <FormattedMessage defaultMessage="Experiments" description="Sidebar link for experiments tab" />,
        },
        componentId: 'mlflow.sidebar.experiments_tab_link',
        nestedItems: showNestedExperimentItems ? (
          <MlflowSidebarExperimentItems
            experimentId={activeExperimentId ?? undefined}
            workflowType={workflowType}
            onBackClick={clearLastSelectedExperiment}
          />
        ) : undefined,
      },
      ...(workflowType === WorkflowType.MACHINE_LEARNING || !enableWorkflowBasedNavigation
        ? [
            {
              key: 'models',
              icon: <ModelsIcon />,
              linkProps: {
                to: ModelRegistryRoutes.modelListPageRoute,
                isActive: isModelsActive,
                children: (
                  <FormattedMessage defaultMessage="Model registry" description="Sidebar link for model registry tab" />
                ),
              },
              componentId: 'mlflow.sidebar.models_tab_link',
            },
          ]
        : []),
      ...(shouldShowGenAIFeatures(enableWorkflowBasedNavigation, workflowType) && !showNestedExperimentItems
        ? [
            {
              key: 'prompts',
              icon: <TextBoxIcon />,
              linkProps: {
                to: ExperimentTrackingRoutes.promptsPageRoute,
                isActive: isPromptsActive,
                children: <FormattedMessage defaultMessage="Prompts" description="Sidebar link for prompts tab" />,
              },
              componentId: 'mlflow.sidebar.prompts_tab_link',
            },
          ]
        : []),
      ...(shouldShowGenAIFeatures(enableWorkflowBasedNavigation, workflowType)
        ? [
            {
              key: 'gateway',
              icon: <CloudModelIcon />,
              linkProps: {
                to: GatewayRoutes.gatewayPageRoute,
                isActive: (location: Location) => !enableWorkflowBasedNavigation && isGatewayActive(location),
                children: (
                  <FormattedMessage defaultMessage="AI Gateway" description="Sidebar link for gateway configuration" />
                ),
              },
              componentId: 'mlflow.sidebar.gateway_tab_link',
              nestedItems:
                shouldEnableWorkflowBasedNavigation() && isGatewayActive(location) ? (
                  <MlflowSidebarGatewayItems />
                ) : undefined,
            },
          ]
        : []),
    ],
    [
      showNestedExperimentItems,
      activeExperimentId,
      workflowType,
      clearLastSelectedExperiment,
      enableWorkflowBasedNavigation,
      location,
    ],
  );

  // Workspace support
  const workspacesEnabled = shouldEnableWorkspaces();
  const workspaceFromUrl = extractWorkspaceFromSearchParams(searchParams);
  // Only show workspace-specific menu items when: workspaces are disabled OR a workspace is selected
  const showWorkspaceMenuItems = !workspacesEnabled || workspaceFromUrl !== null;

  return (
    <aside
      css={{
        width: showSidebar ? (enableWorkflowBasedNavigation ? 230 : 200) : 36,
        flexShrink: 0,
        padding: theme.spacing.sm,
        paddingRight: 0,
        display: 'inline-flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {showSidebar && (
          <Link to={ExperimentTrackingRoutes.rootRoute}>
            <MlflowLogo
              css={{
                display: 'block',
                height: theme.spacing.lg,
                color: theme.colors.textPrimary,
                marginLeft: -(theme.spacing.sm + theme.spacing.xs),
                marginRight: -theme.spacing.lg,
              }}
            />
          </Link>
        )}
        {showSidebar && (
          <Typography.Text size="sm" color="secondary">
            {Version}
          </Typography.Text>
        )}
        <Button
          componentId="mlflow_header.toggle_sidebar_button"
          onClick={toggleSidebar}
          aria-label="Toggle sidebar"
          icon={showSidebar ? <SidebarCollapseIcon /> : <SidebarExpandIcon />}
        />
      </div>
      {workspacesEnabled && showSidebar && <WorkspaceSelector />}
      {enableWorkflowBasedNavigation && showWorkspaceMenuItems && showSidebar && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
          }}
        >
          <Typography.Title level={4} withoutMargins color="info" css={{ textTransform: 'uppercase' }}>
            <FormattedMessage
              defaultMessage="Workflow type"
              description="Label for the workflow type selector in the sidebar"
            />
          </Typography.Title>
          <SegmentedControlGroup
            value={workflowType}
            onChange={(e) => {
              if (e.target.value) {
                setWorkflowType(e.target.value as WorkflowType);
              }
            }}
            name="workflow-type-selector"
            componentId="mlflow.sidebar.workflow_type_selector"
            css={{ width: '100%', display: 'flex' }}
          >
            <SegmentedControlButton value={WorkflowType.GENAI}>
              <FormattedMessage defaultMessage="GenAI" description="Label for GenAI workflow type option" />
            </SegmentedControlButton>
            <SegmentedControlButton value={WorkflowType.MACHINE_LEARNING} css={{ whiteSpace: 'nowrap' }}>
              <FormattedMessage
                defaultMessage="Machine Learning"
                description="Label for Machine Learning workflow type option"
              />
            </SegmentedControlButton>
          </SegmentedControlGroup>
        </div>
      )}

      <nav css={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '100%' }}>
        <ul
          css={{
            listStyleType: 'none',
            padding: 0,
            margin: 0,
          }}
        >
          {showWorkspaceMenuItems &&
            menuItems.map(({ key, icon, linkProps, componentId }) => (
              <MlflowSidebarLink
                to={linkProps.to}
                componentId={componentId}
                isActive={linkProps.isActive}
                icon={icon}
                collapsed={!showSidebar}
              >
                {linkProps.children}
              </MlflowSidebarLink>
            ))}
        </ul>
        <div>
          {isLocalServer && (
            <div
              css={{
                padding: 2,
                marginBottom: theme.spacing.xs,
                borderRadius: theme.borders.borderRadiusMd,
                background:
                  'linear-gradient(90deg, rgba(232, 72, 85, 0.7), rgba(155, 93, 229, 0.7), rgba(67, 97, 238, 0.7))',
              }}
            >
              <div
                role="button"
                tabIndex={0}
                aria-pressed={isPanelOpen}
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  paddingInline: showSidebar ? theme.spacing.md : 0,
                  paddingBlock: theme.spacing.xs,
                  borderRadius: theme.borders.borderRadiusMd - 2,
                  justifyContent: showSidebar ? 'flex-start' : 'center',
                  cursor: 'pointer',
                  background: theme.colors.backgroundSecondary,
                  color: isPanelOpen ? theme.colors.actionDefaultIconHover : theme.colors.actionDefaultIconDefault,
                }}
                onClick={handleAssistantToggle}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    handleAssistantToggle();
                  }
                }}
                onMouseEnter={() => setIsAssistantHovered(true)}
                onMouseLeave={() => setIsAssistantHovered(false)}
              >
                <AssistantSparkleIcon isHovered={isAssistantHovered} />
                {showSidebar && (
                  <>
                    <Typography.Text color="primary">
                      <FormattedMessage defaultMessage="Assistant" description="Sidebar button for AI assistant" />
                    </Typography.Text>
                    <Tag componentId="mlflow.sidebar.assistant_beta_tag" color="turquoise" css={{ marginLeft: 'auto' }}>
                      Beta
                    </Tag>
                  </>
                )}
              </div>
            </div>
          )}
          <MlflowSidebarLink
            disableWorkspacePrefix
            css={{ paddingBlock: theme.spacing.sm }}
            to={HomePageDocsUrl}
            componentId="mlflow.sidebar.docs_link"
            isActive={() => false}
            icon={<InfoBookIcon />}
            collapsed={!showSidebar}
            openInNewTab
          >
            <FormattedMessage defaultMessage="Docs" description="Sidebar link for docs page" />
          </MlflowSidebarLink>
          <MlflowSidebarLink
            disableWorkspacePrefix
            css={{ paddingBlock: theme.spacing.sm }}
            to="https://github.com/mlflow/mlflow"
            componentId="mlflow.sidebar.github_link"
            isActive={() => false}
            icon={<CodeIcon />}
            collapsed={!showSidebar}
            openInNewTab
          >
            <FormattedMessage defaultMessage="GitHub" description="Sidebar link for GitHub page" />
          </MlflowSidebarLink>
          <MlflowSidebarLink
            disableWorkspacePrefix
            css={{ paddingBlock: theme.spacing.sm }}
            to={ExperimentTrackingRoutes.settingsPageRoute}
            componentId="mlflow.sidebar.settings_tab_link"
            isActive={isSettingsActive}
            icon={<GearIcon />}
            collapsed={!showSidebar}
          >
            <FormattedMessage defaultMessage="Settings" description="Sidebar link for settings page" />
          </MlflowSidebarLink>
        </div>
      </nav>
    </aside>
  );
}

import React, { useCallback, useMemo, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  BeakerIcon,
  Button,
  CloudModelIcon,
  GearIcon,
  HomeIcon,
  ModelsIcon,
  Tag,
  TextBoxIcon,
  Typography,
  useDesignSystemTheme,
  DesignSystemEventProviderComponentTypes,
  DesignSystemEventProviderAnalyticsEventTypes,
  SidebarCollapseIcon,
  SidebarExpandIcon,
  InfoBookIcon,
  Tooltip,
  NewWindowIcon,
} from '@databricks/design-system';
import type { Location } from '../utils/RoutingUtils';
import { Link, matchPath, useLocation, useParams, useSearchParams } from '../utils/RoutingUtils';
import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { ModelRegistryRoutes } from '../../model-registry/routes';
import GatewayRoutes from '../../gateway/routes';
import { FormattedMessage } from 'react-intl';
import { useLogTelemetryEvent } from '../../telemetry/hooks/useLogTelemetryEvent';
import { useWorkflowType, WorkflowType } from '../contexts/WorkflowTypeContext';
import { shouldEnableWorkflowBasedNavigation, shouldEnableWorkspaces } from '../utils/FeatureUtils';
import { AssistantSparkleIcon } from '../../assistant/AssistantIconButton';
import { useAssistant } from '../../assistant/AssistantContext';
import { extractWorkspaceFromSearchParams } from '../../workspaces/utils/WorkspaceUtils';
import { MlflowSidebarLink } from './MlflowSidebarLink';
import { MlflowLogo } from './MlflowLogo';
import { DOCS_ROOT, GenAIDocsUrl, MLDocsUrl, Version } from '../constants';
import { WorkspaceSelector } from '../../workspaces/components/WorkspaceSelector';
import { MlflowSidebarExperimentItems } from './MlflowSidebarExperimentItems';
import { MlflowSidebarGatewayItems } from './MlflowSidebarGatewayItems';
import { MlflowSidebarWorkflowSwitch } from './MlflowSidebarWorkflowSwitch';

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
      ...(!showNestedExperimentItems
        ? [
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
          ]
        : []),
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
            collapsed={!showSidebar}
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
                  <MlflowSidebarGatewayItems collapsed={!showSidebar} />
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
      showSidebar,
    ],
  );

  // Workspace support
  const workspacesEnabled = shouldEnableWorkspaces();
  const workspaceFromUrl = extractWorkspaceFromSearchParams(searchParams);
  // Only show workspace-specific menu items when: workspaces are disabled OR a workspace is selected
  const showWorkspaceMenuItems = !workspacesEnabled || workspaceFromUrl !== null;

  // Select appropriate docs URL based on workflow type
  const docsUrl = enableWorkflowBasedNavigation
    ? workflowType === WorkflowType.GENAI
      ? GenAIDocsUrl
      : MLDocsUrl
    : DOCS_ROOT;

  return (
    <aside
      css={{
        width: showSidebar ? 190 : theme.spacing.lg + theme.spacing.md,
        flexShrink: 0,
        padding: theme.spacing.sm,
        paddingRight: 0,
        display: 'inline-flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        {showSidebar && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
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
            <Typography.Text size="sm" css={{ paddingLeft: theme.spacing.sm }} color="secondary">
              {Version}
            </Typography.Text>
          </div>
        )}
        <Button
          componentId="mlflow_header.toggle_sidebar_button"
          onClick={toggleSidebar}
          aria-label="Toggle sidebar"
          icon={showSidebar ? <SidebarCollapseIcon /> : <SidebarExpandIcon />}
        />
      </div>
      {workspacesEnabled && showSidebar && <WorkspaceSelector />}
      {workspacesEnabled && !showWorkspaceMenuItems && (
        <MlflowSidebarLink
          key="mlflow.sidebar.workspace_home_link"
          to={ExperimentTrackingRoutes.rootRoute}
          componentId="mlflow.sidebar.workspace_home_link"
          isActive={isHomeActive}
          icon={<HomeIcon />}
          collapsed={!showSidebar}
        >
          <FormattedMessage defaultMessage="Home" description="Sidebar link for home page" />
        </MlflowSidebarLink>
      )}
      {enableWorkflowBasedNavigation && showWorkspaceMenuItems && showSidebar && (
        <MlflowSidebarWorkflowSwitch workflowType={workflowType} setWorkflowType={setWorkflowType} />
      )}

      <nav
        css={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          height: '100%',
          overflow: 'auto',
        }}
      >
        <ul
          css={{
            listStyleType: 'none',
            padding: 0,
            margin: 0,
          }}
        >
          {showWorkspaceMenuItems &&
            menuItems.map(
              ({ key, icon, linkProps, componentId, nestedItems }) =>
                nestedItems ?? (
                  <MlflowSidebarLink
                    key={componentId}
                    to={linkProps.to}
                    componentId={componentId}
                    isActive={linkProps.isActive}
                    icon={icon}
                    collapsed={!showSidebar}
                  >
                    {linkProps.children}
                  </MlflowSidebarLink>
                ),
            )}
        </ul>
        <div>
          {isLocalServer && (
            <Tooltip
              componentId="mlflow.sidebar.assistant_tooltip"
              content={<FormattedMessage defaultMessage="Assistant" description="Tooltip for assistant button" />}
              open={isAssistantHovered && !showSidebar}
              side="right"
              delayDuration={0}
            >
              <div
                role="button"
                tabIndex={0}
                aria-pressed={isPanelOpen}
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  paddingInline: showSidebar ? theme.spacing.sm : theme.spacing.xs,
                  paddingBlock: theme.spacing.sm,
                  borderRadius: theme.borders.borderRadiusMd - 2,
                  justifyContent: showSidebar ? 'flex-start' : 'center',
                  cursor: 'pointer',
                  color: isPanelOpen ? theme.colors.actionDefaultIconHover : theme.colors.actionDefaultIconDefault,
                  '&:hover': {
                    color: theme.colors.actionLinkHover,
                    backgroundColor: theme.colors.actionDefaultBackgroundHover,
                  },
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
            </Tooltip>
          )}
          <MlflowSidebarLink
            disableWorkspacePrefix
            css={{ paddingBlock: theme.spacing.sm }}
            to={docsUrl}
            componentId="mlflow.sidebar.docs_link"
            isActive={() => false}
            icon={<InfoBookIcon />}
            collapsed={!showSidebar}
            openInNewTab
          >
            <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <FormattedMessage defaultMessage="Docs" description="Sidebar link for docs page" />
              <NewWindowIcon css={{ fontSize: theme.typography.fontSizeBase }} />
            </span>
          </MlflowSidebarLink>
          <MlflowSidebarLink
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

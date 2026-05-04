import React, { useCallback, useMemo, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  Avatar,
  BeakerIcon,
  Button,
  CloudModelIcon,
  DropdownMenu,
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
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import type { Location } from '../utils/RoutingUtils';
import { Link, matchPath, useLocation, useNavigate, useParams, useSearchParams } from '../utils/RoutingUtils';
import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { ModelRegistryRoutes } from '../../model-registry/routes';
import GatewayRoutes from '../../gateway/routes';
import AccountRoutes from '../../account/routes';
import { useCurrentUserQuery, useIsBasicAuth } from '../../account/hooks';
import { performLogout } from '../../account/auth-utils';
import { GatewayLabel, GatewayNewTag } from './GatewayNewTag';
import { FormattedMessage } from 'react-intl';
import { useLogTelemetryEvent } from '../../telemetry/hooks/useLogTelemetryEvent';
import { useWorkflowType, WorkflowType } from '../contexts/WorkflowTypeContext';
import { shouldEnableWorkflowBasedNavigation, shouldEnableWorkspaces } from '../utils/FeatureUtils';
import { AssistantSparkleIcon } from '../../assistant/AssistantIconButton';
import { useAssistant } from '../../assistant/AssistantContext';
import { extractWorkspaceFromSearchParams, useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { SETTINGS_RETURN_TO_PARAM, SETTINGS_SECTION_GENERAL } from '../../settings/settingsSectionConstants';
import { getSidebarItemStyles, MlflowSidebarLink } from './MlflowSidebarLink';
import { MlflowLogo } from './MlflowLogo';
import { DOCS_ROOT, GenAIDocsUrl, MLDocsUrl, Version } from '../constants';
import { WorkspaceSelector } from '../../workspaces/components/WorkspaceSelector';
import { MlflowSidebarExperimentItems } from './MlflowSidebarExperimentItems';
import { MlflowSidebarGatewayItems } from './MlflowSidebarGatewayItems';
import { MlflowSidebarSettingsItems } from './MlflowSidebarSettingsItems';
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
const isSettingsActive = (location: Location) =>
  Boolean(
    matchPath({ path: '/settings', end: true }, location.pathname) ||
    matchPath('/settings/:section', location.pathname),
  );

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

  // One useCurrentUserQuery read here covers both the auth-availability flag
  // and the username display - useIsAuthAvailable() would internally call
  // the same query, which React Query dedupes but obscures the data flow.
  const { data: currentUserData, isError: currentUserIsError, isLoading: currentUserIsLoading } = useCurrentUserQuery();
  const username = currentUserData?.user?.username ?? '';
  const isAuthAvailable = !currentUserIsError && (currentUserIsLoading || Boolean(currentUserData?.user));
  // Logout uses an HTTP Basic Auth realm-cache trick that no-ops under
  // custom ``authorization_function`` deployments, so hide it there.
  const isBasicAuth = useIsBasicAuth();
  const queryClient = useQueryClient();
  const navigate = useNavigate({ bypassWorkspacePrefix: true });

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
  const showNestedSettingsItems = isSettingsActive(location);

  const { openPanel, closePanel, isPanelOpen, isLocalServer } = useAssistant();
  const [isAssistantHovered, setIsAssistantHovered] = useState(false);

  // Radix restores focus to the trigger on dismiss, but the browser
  // keeps ``:focus-visible`` - leaving a stale outline. Skip auto-focus
  // return on pointer dismiss; keyboard dismiss still restores focus.
  const accountDropdownClosedByPointerRef = useRef(false);

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
                  <>
                    <GatewayLabel />
                    <GatewayNewTag />
                  </>
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
  // Global routes (e.g. /account) don't carry ``?workspace=`` in the URL but
  // preserve the in-memory active workspace so the sidebar's workspace-scoped
  // links resume in the same workspace. Fall back to the active workspace.
  const activeWorkspace = useActiveWorkspace();
  const effectiveWorkspace = workspaceFromUrl ?? activeWorkspace;
  // Only show workspace-specific menu items when: workspaces are disabled OR a workspace is selected
  const showWorkspaceMenuItems = !workspacesEnabled || effectiveWorkspace !== null;

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
            <Link componentId="mlflow.sidebar.logo_home_link" to={ExperimentTrackingRoutes.rootRoute}>
              <MlflowLogo
                css={{
                  display: 'block',
                  height: theme.spacing.lg,
                  color: theme.colors.textPrimary,
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
            (showNestedSettingsItems ? (
              <MlflowSidebarSettingsItems collapsed={!showSidebar} />
            ) : (
              menuItems.map(
                ({ icon, linkProps, componentId, nestedItems }) =>
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
              )
            ))}
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
          {showWorkspaceMenuItems && !showNestedSettingsItems && (
            <MlflowSidebarLink
              css={{ paddingBlock: theme.spacing.sm }}
              to={`${ExperimentTrackingRoutes.getSettingsSectionRoute(SETTINGS_SECTION_GENERAL)}?${SETTINGS_RETURN_TO_PARAM}=${encodeURIComponent(location.pathname + location.search)}`}
              componentId="mlflow.sidebar.settings_tab_link"
              isActive={isSettingsActive}
              icon={<GearIcon />}
              collapsed={!showSidebar}
            >
              <FormattedMessage defaultMessage="Settings" description="Sidebar link for settings page" />
            </MlflowSidebarLink>
          )}
          {isAuthAvailable && username && (
            <div
              css={{
                marginTop: theme.spacing.sm,
                paddingTop: theme.spacing.sm,
                borderTop: `1px solid ${theme.colors.border}`,
              }}
            >
              <DropdownMenu.Root>
                <DropdownMenu.Trigger asChild>
                  {/*
                    Why a native ``<button>`` instead of DuBois ``Button``:
                    the trigger has to be a full-width row whose content
                    (avatar + optional username) is left-aligned when the
                    sidebar is open, matching ``MlflowSidebarLink``'s row
                    layout. DuBois ``Button`` (AntD-derived) wraps its
                    label content in an internal flex span that centers
                    it; that span doesn't yield to outer flex overrides,
                    so an aligned full-width avatar+label fights the
                    primitive's intent. ``getSidebarItemStyles`` is the
                    shared style helper from ``MlflowSidebarLink`` so the
                    raw button still picks up the same spacing / hover /
                    focus / aria-current tokens, keeping the row visually
                    consistent with the rest of the sidebar.
                  */}
                  <button
                    type="button"
                    aria-label={`Account menu for ${username}`}
                    css={{
                      ...getSidebarItemStyles(theme, !showSidebar),
                      background: 'transparent',
                      border: 'none',
                      cursor: 'pointer',
                      // Suppress the UA dotted underline + help cursor that
                      // ``Avatar`` inherits from its inner ``<abbr title>``.
                      '& abbr[title]': {
                        textDecoration: 'none',
                        cursor: 'inherit',
                      },
                    }}
                  >
                    <Avatar type="user" size="sm" label={username} />
                    {showSidebar && (
                      <Typography.Text
                        css={{
                          flex: 1,
                          textAlign: 'left',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {username}
                      </Typography.Text>
                    )}
                  </button>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content
                  align="start"
                  side="top"
                  minWidth={180}
                  onPointerDownOutside={() => {
                    accountDropdownClosedByPointerRef.current = true;
                  }}
                  onCloseAutoFocus={(event) => {
                    if (accountDropdownClosedByPointerRef.current) {
                      event.preventDefault();
                      accountDropdownClosedByPointerRef.current = false;
                    }
                  }}
                >
                  <DropdownMenu.Item
                    componentId="mlflow.sidebar.account"
                    onPointerDown={() => {
                      accountDropdownClosedByPointerRef.current = true;
                    }}
                    onClick={() => navigate(AccountRoutes.accountPageRoute)}
                  >
                    <FormattedMessage defaultMessage="Account" description="Sidebar account menu item" />
                  </DropdownMenu.Item>
                  {isBasicAuth && (
                    <>
                      <DropdownMenu.Separator />
                      <DropdownMenu.Item
                        componentId="mlflow.sidebar.logout"
                        onPointerDown={() => {
                          accountDropdownClosedByPointerRef.current = true;
                        }}
                        onClick={() => performLogout(queryClient)}
                      >
                        <FormattedMessage defaultMessage="Logout" description="Sidebar logout menu item" />
                      </DropdownMenu.Item>
                    </>
                  )}
                </DropdownMenu.Content>
              </DropdownMenu.Root>
            </div>
          )}
        </div>
      </nav>
    </aside>
  );
}

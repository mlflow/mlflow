import React, { Fragment, useCallback, useMemo, useState } from 'react';
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
import { HomePageDocsUrl } from '../constants';
import { WorkspaceSelector } from '../../workspaces/components/WorkspaceSelector';

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
  nestedItems?: NestedMenuItem[];
  nestedItemsGroups?: NestedItemsGroup[];
};

const buildNestedItemsFromConfig = (
  items: Array<{ tabName: ExperimentPageTabName; icon: React.ReactNode; label: React.ReactNode; componentId: string }>,
  experimentId?: string,
): NestedMenuItem[] => {
  return items.map((item) => ({
    key: `experiments-${item.tabName}`,
    icon: item.icon,
    label: item.label,
    to: experimentId
      ? Routes.getExperimentPageTabRoute(experimentId, item.tabName)
      : ExperimentTrackingRoutes.experimentsObservatoryRoute,
    componentId: item.componentId,
    isActive: (loc) =>
      Boolean(experimentId && matchPath(`/experiments/${experimentId}/${item.tabName}/*`, loc.pathname)),
  }));
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

  const renderNestedItemLink = useCallback(
    (nestedItem: NestedMenuItem, isDisabled: boolean) => {
      const isNestedActive = nestedItem.isActive(location);
      const linkElement = (
        <Link
          to={nestedItem.to}
          aria-current={isNestedActive ? 'page' : undefined}
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            color: isDisabled ? theme.colors.textSecondary : theme.colors.textPrimary,
            paddingInline: theme.spacing.md,
            paddingLeft: 40,
            paddingBlock: theme.spacing.xs,
            borderRadius: theme.borders.borderRadiusSm,
            cursor: isDisabled ? 'not-allowed' : 'pointer',
            opacity: isDisabled ? 0.5 : 1,
            '&:hover': isDisabled
              ? {}
              : {
                  color: theme.colors.actionLinkHover,
                  backgroundColor: theme.colors.actionDefaultBackgroundHover,
                },
            '&[aria-current="page"]': {
              backgroundColor: theme.colors.actionDefaultBackgroundPress,
              color: theme.isDarkMode ? theme.colors.blue300 : theme.colors.blue700,
              fontWeight: theme.typography.typographyBoldFontWeight,
            },
          }}
          onClick={(e) => {
            if (isDisabled) {
              e.preventDefault();
              return;
            }
            logTelemetryEvent({
              componentId: nestedItem.componentId,
              componentViewId: viewId,
              componentType: DesignSystemEventProviderComponentTypes.TypographyLink,
              componentSubType: null,
              eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
            });
          }}
        >
          {nestedItem.icon}
          {nestedItem.label}
        </Link>
      );

      if (isDisabled) {
        return (
          <Tooltip
            componentId={`mlflow.sidebar.nested-item.disabled-tooltip.${nestedItem.key}`}
            content={
              <FormattedMessage
                defaultMessage="Select an experiment to view this tab"
                description="Tooltip shown when nested experiment items are disabled because no experiment is selected"
              />
            }
            side="right"
          >
            {linkElement}
          </Tooltip>
        );
      }
      return linkElement;
    },
    [location, theme, logTelemetryEvent, viewId],
  );

  const experimentNestedItemsGroups = useMemo((): NestedItemsGroup[] => {
    if (!enableWorkflowBasedNavigation) {
      return [];
    }

    const groups: NestedItemsGroup[] = Object.entries(config).map(([sectionKey, items]) => ({
      sectionKey: sectionKey as ExperimentPageSideNavSectionKey,
      items: buildNestedItemsFromConfig(items, experimentId),
    }));

    return groups;
  }, [enableWorkflowBasedNavigation, config, experimentId]);

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
        nestedItemsGroups: experimentNestedItemsGroups.length > 0 ? experimentNestedItemsGroups : undefined,
      },
      ...(workflowType === WorkflowType.MACHINE_LEARNING || !enableWorkflowBasedNavigation
        ? [
            {
              key: 'models',
              icon: <ModelsIcon />,
              linkProps: {
                to: ModelRegistryRoutes.modelListPageRoute,
                isActive: isModelsActive,
                children: <FormattedMessage defaultMessage="Models" description="Sidebar link for models tab" />,
              },
              componentId: 'mlflow.sidebar.models_tab_link',
            },
          ]
        : []),
      ...(shouldShowGenAIFeatures(enableWorkflowBasedNavigation, workflowType)
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
                enableWorkflowBasedNavigation && workflowType === WorkflowType.GENAI
                  ? [
                      {
                        key: 'gateway-endpoints',
                        icon: <ChainIcon />,
                        label: (
                          <FormattedMessage defaultMessage="Endpoints" description="Gateway side nav > Endpoints tab" />
                        ),
                        to: GatewayRoutes.gatewayPageRoute,
                        componentId: 'mlflow.sidebar.gateway.endpoints',
                        isActive: (loc: Location) =>
                          Boolean(matchPath('/gateway', loc.pathname) && !matchPath('/gateway/api-keys', loc.pathname)),
                      },
                      {
                        key: 'gateway-api-keys',
                        icon: <KeyIcon />,
                        label: (
                          <FormattedMessage defaultMessage="API Keys" description="Gateway side nav > API Keys tab" />
                        ),
                        to: GatewayRoutes.apiKeysPageRoute,
                        componentId: 'mlflow.sidebar.gateway.api-keys',
                        isActive: (loc: Location) => Boolean(matchPath('/gateway/api-keys', loc.pathname)),
                      },
                    ]
                  : undefined,
            },
          ]
        : []),
    ],
    [enableWorkflowBasedNavigation, workflowType, experimentNestedItemsGroups],
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
              }}
            />
          </Link>
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
            menuItems.map(({ key, icon, linkProps, componentId, nestedItemsGroups, nestedItems }) => (
              <>
                <MlflowSidebarLink
                  to={linkProps.to}
                  componentId={componentId}
                  isActive={linkProps.isActive}
                  icon={icon}
                  collapsed={!showSidebar}
                >
                  {linkProps.children}
                </MlflowSidebarLink>
                {nestedItemsGroups && nestedItemsGroups.length > 0 && (
                  <ul css={NESTED_ITEMS_UL_CSS}>
                    {nestedItemsGroups.map((group) => (
                      <Fragment key={group.sectionKey}>
                        {group.sectionKey !== 'top-level' && (
                          <li
                            css={{
                              display: 'flex',
                              marginTop: theme.spacing.xs,
                              marginBottom: theme.spacing.xs,
                              position: 'relative',
                              height: theme.typography.lineHeightBase,
                              paddingLeft: 40,
                            }}
                          >
                            <Typography.Text size="sm" color="secondary">
                              {getExperimentPageSideNavSectionLabel(group.sectionKey, [])}
                            </Typography.Text>
                          </li>
                        )}
                        {group.items.map((nestedItem) => {
                          const isDisabled = !experimentId && key === 'experiments';
                          return <li key={nestedItem.key}>{renderNestedItemLink(nestedItem, isDisabled)}</li>;
                        })}
                      </Fragment>
                    ))}
                  </ul>
                )}
                {nestedItems && nestedItems.length > 0 && (
                  <ul css={NESTED_ITEMS_UL_CSS}>
                    {nestedItems.map((nestedItem) => (
                      <li key={nestedItem.key}>{renderNestedItemLink(nestedItem, false)}</li>
                    ))}
                  </ul>
                )}
              </>
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

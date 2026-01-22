import React, { Fragment, useCallback, useMemo, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  BeakerIcon,
  Button,
  CloudModelIcon,
  DropdownMenu,
  GearIcon,
  HomeIcon,
  ModelsIcon,
  PlusIcon,
  SegmentedControlGroup,
  SegmentedControlButton,
  Tag,
  TextBoxIcon,
  Tooltip,
  useDesignSystemTheme,
  DesignSystemEventProviderComponentTypes,
  DesignSystemEventProviderAnalyticsEventTypes,
  Typography,
} from '@databricks/design-system';
import type { Location } from '../utils/RoutingUtils';
import { Link, matchPath, useLocation, useNavigate } from '../utils/RoutingUtils';
import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { ModelRegistryRoutes } from '../../model-registry/routes';
import GatewayRoutes from '../../gateway/routes';
import { CreateExperimentModal } from '../../experiment-tracking/components/modals/CreateExperimentModal';
import { useInvalidateExperimentList } from '../../experiment-tracking/components/experiment-page/hooks/useExperimentListQuery';
import { CreateModelModal } from '../../model-registry/components/CreateModelModal';
import {
  CreatePromptModalMode,
  useCreatePromptModal,
} from '../../experiment-tracking/pages/prompts/hooks/useCreatePromptModal';
import Routes from '../../experiment-tracking/routes';
import { FormattedMessage } from 'react-intl';
import { useLogTelemetryEvent } from '../../telemetry/hooks/useLogTelemetryEvent';
import { useWorkflowType, WorkflowType } from '../contexts/WorkflowTypeContext';
import {
  ExperimentPageSideNavGenAIConfig,
  ExperimentPageSideNavCustomModelConfig,
  getExperimentPageSideNavSectionLabel,
  type ExperimentPageSideNavSectionKey,
} from '../../experiment-tracking/pages/experiment-page-tabs/side-nav/constants';
import { ExperimentPageTabName } from '../../experiment-tracking/constants';
import { ChainIcon, KeyIcon } from '@databricks/design-system';
import { useParams } from '../utils/RoutingUtils';
import { shouldEnableWorkflowBasedNavigation } from '../utils/FeatureUtils';
import { AssistantSparkleIcon } from '../../assistant/AssistantIconButton';
import { useAssistant } from '../../assistant/AssistantContext';

const isHomeActive = (location: Location) => Boolean(matchPath({ path: '/', end: true }, location.pathname));
const isExperimentsActive = (location: Location) =>
  Boolean(
    matchPath({ path: '/experiments', end: true }, location.pathname) ||
    matchPath('/compare-experiments/*', location.pathname),
  );
const isModelsActive = (location: Location) => Boolean(matchPath('/models/*', location.pathname));
const isPromptsActive = (location: Location) => Boolean(matchPath('/prompts/*', location.pathname));
const isGatewayActive = (location: Location) => Boolean(matchPath({ path: '/gateway', end: true }, location.pathname));
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

export function MlflowSidebar() {
  const location = useLocation();
  const { theme } = useDesignSystemTheme();
  const invalidateExperimentList = useInvalidateExperimentList();
  const navigate = useNavigate();
  const viewId = useMemo(() => uuidv4(), []);
  const enableWorkflowBasedNavigation = shouldEnableWorkflowBasedNavigation();
  // WorkflowType context is always available, but UI is guarded by feature flag
  const { workflowType, setWorkflowType } = useWorkflowType();
  const { experimentId } = useParams();
  const logTelemetryEvent = useLogTelemetryEvent();

  const [showCreateExperimentModal, setShowCreateExperimentModal] = useState(false);
  const [showCreateModelModal, setShowCreateModelModal] = useState(false);
  const { CreatePromptModal, openModal: openCreatePromptModal } = useCreatePromptModal({
    mode: CreatePromptModalMode.CreatePrompt,
    onSuccess: ({ promptName }) => navigate(Routes.getPromptDetailsPageRoute(promptName)),
  });
  const { openPanel, closePanel, isPanelOpen } = useAssistant();
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

    if (workflowType === WorkflowType.GENAI) {
      const config = ExperimentPageSideNavGenAIConfig;
      const groups: NestedItemsGroup[] = [];

      if (config.observability) {
        groups.push({
          sectionKey: 'observability',
          items: buildNestedItemsFromConfig(config.observability, experimentId),
        });
      }
      if (config.evaluation) {
        groups.push({
          sectionKey: 'evaluation',
          items: buildNestedItemsFromConfig(config.evaluation, experimentId),
        });
      }
      if (config['prompts-versions']) {
        groups.push({
          sectionKey: 'prompts-versions',
          items: buildNestedItemsFromConfig(config['prompts-versions'], experimentId),
        });
      }

      return groups;
    }

    if (workflowType === WorkflowType.MACHINE_LEARNING) {
      const config = ExperimentPageSideNavCustomModelConfig;
      if (config['top-level']) {
        return [
          {
            sectionKey: 'top-level',
            items: buildNestedItemsFromConfig(config['top-level'], experimentId),
          },
        ];
      }
    }

    return [];
  }, [enableWorkflowBasedNavigation, workflowType, experimentId]);

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
        dropdownProps: {
          componentId: 'mlflow_sidebar.create_experiment_button' as MlFlowSidebarMenuDropdownComponentId,
          onClick: () => setShowCreateExperimentModal(true),
          children: (
            <FormattedMessage
              defaultMessage="Experiment"
              description="Sidebar button inside the 'new' popover to create new experiment"
            />
          ),
        },
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
              dropdownProps: {
                componentId: 'mlflow_sidebar.create_model_button' as MlFlowSidebarMenuDropdownComponentId,
                onClick: () => setShowCreateModelModal(true),
                children: (
                  <FormattedMessage
                    defaultMessage="Model"
                    description="Sidebar button inside the 'new' popover to create new model"
                  />
                ),
              },
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
              dropdownProps: {
                componentId: 'mlflow_sidebar.create_prompt_button' as MlFlowSidebarMenuDropdownComponentId,
                onClick: openCreatePromptModal,
                children: (
                  <FormattedMessage
                    defaultMessage="Prompt"
                    description="Sidebar button inside the 'new' popover to create new prompt"
                  />
                ),
              },
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
    [enableWorkflowBasedNavigation, workflowType, experimentNestedItemsGroups, openCreatePromptModal],
  );

  return (
    <aside
      css={{
        width: enableWorkflowBasedNavigation ? 230 : 200,
        flexShrink: 0,
        padding: theme.spacing.sm,
        display: 'inline-flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
      }}
    >
      {enableWorkflowBasedNavigation && (
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

      <DropdownMenu.Root modal={false}>
        <DropdownMenu.Trigger asChild>
          <Button componentId="mlflow.sidebar.new_button" icon={<PlusIcon />}>
            <FormattedMessage
              defaultMessage="New"
              description="Sidebar create popover button to create new experiment, model or prompt"
            />
          </Button>
        </DropdownMenu.Trigger>

        <DropdownMenu.Content side="right" sideOffset={theme.spacing.sm} align="start">
          {menuItems
            .filter((item) => item.dropdownProps !== undefined)
            .map(({ key, icon, dropdownProps }) => (
              <DropdownMenu.Item
                key={key}
                componentId={(dropdownProps?.componentId ?? `${key}-dropdown-item`) as string}
                onClick={dropdownProps?.onClick}
              >
                <DropdownMenu.IconWrapper>{icon}</DropdownMenu.IconWrapper>
                {dropdownProps?.children}
              </DropdownMenu.Item>
            ))}
        </DropdownMenu.Content>
      </DropdownMenu.Root>

      <nav css={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '100%' }}>
        <ul
          css={{
            listStyleType: 'none',
            padding: 0,
            margin: 0,
          }}
        >
          {menuItems.map(({ key, icon, linkProps, componentId, nestedItemsGroups, nestedItems }) => (
            <li key={key}>
              <Link
                to={linkProps.to}
                aria-current={linkProps.isActive(location) ? 'page' : undefined}
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  color: theme.colors.textPrimary,
                  paddingInline: theme.spacing.md,
                  paddingBlock: theme.spacing.xs,
                  borderRadius: theme.borders.borderRadiusSm,
                  '&:hover': {
                    color: theme.colors.actionLinkHover,
                    backgroundColor: theme.colors.actionDefaultBackgroundHover,
                  },
                  '&[aria-current="page"]': {
                    backgroundColor: theme.colors.actionDefaultBackgroundPress,
                    color: theme.isDarkMode ? theme.colors.blue300 : theme.colors.blue700,
                    fontWeight: theme.typography.typographyBoldFontWeight,
                  },
                }}
                onClick={() =>
                  logTelemetryEvent({
                    componentId,
                    componentViewId: viewId,
                    componentType: DesignSystemEventProviderComponentTypes.TypographyLink,
                    componentSubType: null,
                    eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
                  })
                }
              >
                {icon}
                {linkProps.children}
              </Link>
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
            </li>
          ))}
        </ul>
        <div>
          {enableWorkflowBasedNavigation && (
            <div
              role="button"
              tabIndex={0}
              aria-pressed={isPanelOpen}
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                borderRadius: theme.borders.borderRadiusSm,
                cursor: 'pointer',
                backgroundColor: isPanelOpen ? theme.colors.actionDefaultBackgroundHover : undefined,
                color: isPanelOpen ? theme.colors.actionDefaultIconHover : theme.colors.actionDefaultIconDefault,
                height: theme.typography.lineHeightBase,
                boxSizing: 'content-box',
                ':hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
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
              <Tooltip
                componentId="mlflow.sidebar.assistant_tooltip"
                content={<FormattedMessage defaultMessage="Assistant" description="Tooltip for assistant button" />}
                side="right"
                delayDuration={0}
              >
                <AssistantSparkleIcon isHovered={isAssistantHovered} />
              </Tooltip>
              <Typography.Text bold={isPanelOpen} color="primary">
                <FormattedMessage defaultMessage="Assistant" description="Sidebar button for AI assistant" />
              </Typography.Text>
              <Tag componentId="mlflow.sidebar.assistant_beta_tag" color="turquoise" css={{ marginLeft: 'auto' }}>
                Beta
              </Tag>
            </div>
          )}
          <Link
            to={ExperimentTrackingRoutes.settingsPageRoute}
            aria-current={isSettingsActive(location) ? 'page' : undefined}
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              color: theme.colors.textPrimary,
              paddingInline: theme.spacing.md,
              paddingBlock: theme.spacing.sm,
              borderRadius: theme.borders.borderRadiusSm,
              '&:hover': {
                color: theme.colors.actionLinkHover,
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
              },
              '&[aria-current="page"]': {
                backgroundColor: theme.colors.actionDefaultBackgroundPress,
                color: theme.isDarkMode ? theme.colors.blue300 : theme.colors.blue700,
                fontWeight: theme.typography.typographyBoldFontWeight,
              },
            }}
            onClick={() =>
              logTelemetryEvent({
                componentId: 'mlflow.sidebar.settings_tab_link',
                componentViewId: viewId,
                componentType: DesignSystemEventProviderComponentTypes.TypographyLink,
                componentSubType: null,
                eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
              })
            }
          >
            <GearIcon />
            <FormattedMessage defaultMessage="Settings" description="Sidebar link for settings page" />
          </Link>
        </div>
      </nav>

      <CreateExperimentModal
        isOpen={showCreateExperimentModal}
        onClose={() => setShowCreateExperimentModal(false)}
        onExperimentCreated={invalidateExperimentList}
      />
      <CreateModelModal modalVisible={showCreateModelModal} hideModal={() => setShowCreateModelModal(false)} />
      {CreatePromptModal}
    </aside>
  );
}

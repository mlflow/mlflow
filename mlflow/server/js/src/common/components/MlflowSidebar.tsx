import { v4 as uuidv4 } from 'uuid';
import {
  BeakerIcon,
  Button,
  ChainIcon,
  CloudModelIcon,
  DropdownMenu,
  GearIcon,
  HomeIcon,
  KeyIcon,
  ModelsIcon,
  PlusIcon,
  TextBoxIcon,
  useDesignSystemTheme,
  DesignSystemEventProviderComponentTypes,
  DesignSystemEventProviderAnalyticsEventTypes,
  SegmentedControlGroup,
  SegmentedControlButton,
  ChartLineIcon,
  DatabaseIcon,
  ForkHorizontalIcon,
  GavelIcon,
  ListIcon,
  PlusMinusSquareIcon,
  SpeechBubbleIcon,
  Tooltip,
  UserGroupIcon,
  Popover,
  CloseIcon,
} from '@databricks/design-system';
import type { Location } from '../utils/RoutingUtils';
import { Link, matchPath, useLocation, useNavigate, useParams } from '../utils/RoutingUtils';
import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { ModelRegistryRoutes } from '../../model-registry/routes';
import GatewayRoutes from '../../gateway/routes';
import { CreateExperimentModal } from '../../experiment-tracking/components/modals/CreateExperimentModal';
import { useMemo, useState } from 'react';
import { useInvalidateExperimentList } from '../../experiment-tracking/components/experiment-page/hooks/useExperimentListQuery';
import { CreateModelModal } from '../../model-registry/components/CreateModelModal';
import {
  CreatePromptModalMode,
  useCreatePromptModal,
} from '../../experiment-tracking/pages/prompts/hooks/useCreatePromptModal';
import Routes from '../../experiment-tracking/routes';
import { FormattedMessage } from 'react-intl';
import { useLogTelemetryEvent } from '../../telemetry/hooks/useLogTelemetryEvent';
import { ExperimentKind, ExperimentPageTabName } from '../../experiment-tracking/constants';
import { enableScorersUI, shouldEnableExperimentOverviewTab } from '../utils/FeatureUtils';
import { useLocalStorage } from '../../shared/web-shared/hooks';
import { useEffect } from 'react';

const isHomeActive = (location: Location) => matchPath({ path: '/', end: true }, location.pathname);
// POC: Experiments link should only be active on the experiments list page, not when viewing a specific experiment
const isExperimentsActive = (location: Location) => {
  const isExperimentsList = matchPath('/experiments', location.pathname);
  const isCompareExperiments = matchPath('/compare-experiments/*', location.pathname);
  const isSingleExperiment = matchPath('/experiments/:experimentId/*', location.pathname);
  return (isExperimentsList || isCompareExperiments) && !isSingleExperiment;
};
const isModelsActive = (location: Location) => matchPath('/models/*', location.pathname);
const isPromptsActive = (location: Location) => matchPath('/prompts/*', location.pathname);
// Gateway parent link should not be highlighted when nested tabs are active
const isGatewayActive = () => false;
const isSettingsActive = (location: Location) => matchPath('/settings/*', location.pathname);

// Helper to determine active gateway tab
type GatewayTabName = 'endpoints' | 'api-keys';
const getActiveGatewayTab = (location: Location): GatewayTabName | null => {
  if (matchPath('/gateway/api-keys', location.pathname)) {
    return 'api-keys';
  }
  // Endpoints tab is active on /gateway and any /gateway/endpoints/* routes (details, create, etc.)
  if (
    matchPath('/gateway', location.pathname) ||
    matchPath('/gateway/', location.pathname) ||
    matchPath('/gateway/endpoints/*', location.pathname)
  ) {
    return 'endpoints';
  }
  return null;
};

// Helper to check if we're in an experiment page
const isInExperimentPage = (location: Location) => {
  const match = matchPath('/experiments/:experimentId/*', location.pathname);
  return Boolean(match && match.params.experimentId);
};

// Helper to determine active experiment tab
const getActiveExperimentTab = (location: Location): ExperimentPageTabName | null => {
  if (matchPath('/experiments/:experimentId/overview', location.pathname)) {
    return ExperimentPageTabName.Overview;
  }
  if (matchPath('/experiments/:experimentId/runs', location.pathname)) {
    return ExperimentPageTabName.Runs;
  }
  if (matchPath('/experiments/:experimentId/traces', location.pathname)) {
    return ExperimentPageTabName.Traces;
  }
  if (matchPath('/experiments/:experimentId/chat-sessions*', location.pathname)) {
    return ExperimentPageTabName.ChatSessions;
  }
  if (matchPath('/experiments/:experimentId/datasets', location.pathname)) {
    return ExperimentPageTabName.Datasets;
  }
  if (matchPath('/experiments/:experimentId/evaluation-runs', location.pathname)) {
    return ExperimentPageTabName.EvaluationRuns;
  }
  if (matchPath('/experiments/:experimentId/judges', location.pathname)) {
    return ExperimentPageTabName.Judges;
  }
  if (matchPath('/experiments/:experimentId/prompts*', location.pathname)) {
    return ExperimentPageTabName.Prompts;
  }
  if (matchPath('/experiments/:experimentId/models', location.pathname)) {
    return ExperimentPageTabName.Models;
  }
  return null;
};

const EXPERIMENT_TYPE_POSITION_STORAGE_KEY = 'mlflow.sidebar.experimentTypeSelectorPosition';
const EXPERIMENT_TYPE_POSITION_STORAGE_VERSION = 1;
const WORKFLOW_SELECTOR_GUIDANCE_STORAGE_KEY = 'mlflow.sidebar.workflowSelectorGuidanceShown';
const WORKFLOW_SELECTOR_GUIDANCE_STORAGE_VERSION = 1;

export function MlflowSidebar() {
  const location = useLocation();
  const { theme } = useDesignSystemTheme();
  const invalidateExperimentList = useInvalidateExperimentList();
  const navigate = useNavigate();
  const viewId = useMemo(() => uuidv4(), []);
  const { experimentId } = useParams();

  const [showCreateExperimentModal, setShowCreateExperimentModal] = useState(false);
  const [showCreateModelModal, setShowCreateModelModal] = useState(false);
  const { CreatePromptModal, openModal: openCreatePromptModal } = useCreatePromptModal({
    mode: CreatePromptModalMode.CreatePrompt,
    onSuccess: ({ promptName }) => navigate(Routes.getPromptDetailsPageRoute(promptName)),
  });

  // Track selected experiment type (GenAI or Machine Learning)
  // Use localStorage to persist selection across page reloads
  const [selectedExperimentType, setSelectedExperimentType] = useState<'genai' | 'ml'>(() => {
    const stored = localStorage.getItem('mlflow.sidebar.experimentType');
    return (stored === 'genai' || stored === 'ml') ? stored : 'genai';
  });

  // Read the experiment type selector position setting
  const [isExperimentTypeSelectorAtTop] = useLocalStorage({
    key: EXPERIMENT_TYPE_POSITION_STORAGE_KEY,
    version: EXPERIMENT_TYPE_POSITION_STORAGE_VERSION,
    initialValue: false,
  });

  // Track whether the user has seen the workflow selector guidance
  const [hasSeenGuidance, setHasSeenGuidance] = useLocalStorage({
    key: WORKFLOW_SELECTOR_GUIDANCE_STORAGE_KEY,
    version: WORKFLOW_SELECTOR_GUIDANCE_STORAGE_VERSION,
    initialValue: false,
  });

  // Control popover visibility
  const [showGuidancePopover, setShowGuidancePopover] = useState(false);

  // Show guidance popover on first visit
  useEffect(() => {
    if (!hasSeenGuidance) {
      // Delay slightly to ensure the component is mounted and visible
      const timer = setTimeout(() => {
        setShowGuidancePopover(true);
      }, 500);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [hasSeenGuidance]);

  const handleDismissGuidance = () => {
    setShowGuidancePopover(false);
    setHasSeenGuidance(true);
  };

  const handleExperimentTypeChange = (e: any) => {
    const value = e.target.value;
    if (value === 'genai' || value === 'ml') {
      setSelectedExperimentType(value);
      localStorage.setItem('mlflow.sidebar.experimentType', value);
    }
  };

  const inExperimentPage = isInExperimentPage(location);
  const activeExperimentTab = getActiveExperimentTab(location);
  const activeGatewayTab = getActiveGatewayTab(location);

  const topLevelMenuItems = [
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
        componentId: 'mlflow_sidebar.create_experiment_button',
        onClick: () => setShowCreateExperimentModal(true),
        children: (
          <FormattedMessage
            defaultMessage="Experiment"
            description="Sidebar button inside the 'new' popover to create new experiment"
          />
        ),
      },
    },
  ];

  const bottomMenuItems = [
    // Only show Models (model registry) in ML workflow context
    ...(selectedExperimentType === 'ml'
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
              componentId: 'mlflow_sidebar.create_model_button',
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
        componentId: 'mlflow_sidebar.create_prompt_button',
        onClick: openCreatePromptModal,
        children: (
          <FormattedMessage
            defaultMessage="Prompt"
            description="Sidebar button inside the 'new' popover to create new prompt"
          />
        ),
      },
    },
    {
      key: 'gateway',
      icon: <CloudModelIcon />,
      linkProps: {
        to: GatewayRoutes.gatewayPageRoute,
        isActive: isGatewayActive,
        children: <FormattedMessage defaultMessage="AI Gateway" description="Sidebar link for gateway configuration" />,
      },
      componentId: 'mlflow.sidebar.gateway_tab_link',
    },
  ];

  const mainMenuItems = [...topLevelMenuItems, ...bottomMenuItems];

  // Experiment-level tabs with section groupings - replicating useExperimentPageSideNavConfig structure
  const experimentTabsGenAI = {
    'top-level': [
      ...(shouldEnableExperimentOverviewTab()
        ? [
            {
              key: 'exp-overview',
              icon: <ChartLineIcon />,
              linkProps: {
                to: experimentId
                  ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Overview)
                  : ExperimentTrackingRoutes.experimentsObservatoryRoute,
                isActive: () => activeExperimentTab === ExperimentPageTabName.Overview,
                children: (
                  <FormattedMessage defaultMessage="Overview" description="Sidebar link for experiment overview tab" />
                ),
              },
              componentId: 'mlflow.sidebar.experiment.overview',
              disabled: !experimentId,
            },
          ]
        : []),
    ],
    observability: [
      {
        key: 'exp-traces',
        icon: <ForkHorizontalIcon />,
        linkProps: {
          to: experimentId
            ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Traces)
            : ExperimentTrackingRoutes.experimentsObservatoryRoute,
          isActive: () => activeExperimentTab === ExperimentPageTabName.Traces,
          children: <FormattedMessage defaultMessage="Traces" description="Sidebar link for experiment traces tab" />,
        },
        componentId: 'mlflow.sidebar.experiment.traces',
        disabled: !experimentId,
      },
      {
        key: 'exp-sessions',
        icon: <SpeechBubbleIcon />,
        linkProps: {
          to: experimentId
            ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.ChatSessions)
            : ExperimentTrackingRoutes.experimentsObservatoryRoute,
          isActive: () => activeExperimentTab === ExperimentPageTabName.ChatSessions,
          children: (
            <FormattedMessage defaultMessage="Sessions" description="Sidebar link for experiment sessions tab" />
          ),
        },
        componentId: 'mlflow.sidebar.experiment.sessions',
        disabled: !experimentId,
      },
    ],
    evaluation: [
      {
        key: 'exp-datasets',
        icon: <DatabaseIcon />,
        linkProps: {
          to: experimentId
            ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Datasets)
            : ExperimentTrackingRoutes.experimentsObservatoryRoute,
          isActive: () => activeExperimentTab === ExperimentPageTabName.Datasets,
          children: (
            <FormattedMessage defaultMessage="Datasets" description="Sidebar link for experiment datasets tab" />
          ),
        },
        componentId: 'mlflow.sidebar.experiment.datasets',
        disabled: !experimentId,
      },
      {
        key: 'exp-eval-runs',
        icon: <PlusMinusSquareIcon />,
        linkProps: {
          to: experimentId
            ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.EvaluationRuns)
            : ExperimentTrackingRoutes.experimentsObservatoryRoute,
          isActive: () => activeExperimentTab === ExperimentPageTabName.EvaluationRuns,
          children: (
            <FormattedMessage
              defaultMessage="Evaluation runs"
              description="Sidebar link for experiment evaluation runs tab"
            />
          ),
        },
        componentId: 'mlflow.sidebar.experiment.evaluation-runs',
        disabled: !experimentId,
      },
      ...(enableScorersUI()
        ? [
            {
              key: 'exp-judges',
              icon: <GavelIcon />,
              linkProps: {
                to: experimentId
                  ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Judges)
                  : ExperimentTrackingRoutes.experimentsObservatoryRoute,
                isActive: () => activeExperimentTab === ExperimentPageTabName.Judges,
                children: (
                  <FormattedMessage defaultMessage="Judges" description="Sidebar link for experiment judges tab" />
                ),
              },
              componentId: 'mlflow.sidebar.experiment.judges',
              disabled: !experimentId,
            },
          ]
        : []),
    ],
    'prompts-versions': [
      {
        key: 'exp-prompts',
        icon: <TextBoxIcon />,
        linkProps: {
          to: experimentId
            ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Prompts)
            : ExperimentTrackingRoutes.experimentsObservatoryRoute,
          isActive: () => activeExperimentTab === ExperimentPageTabName.Prompts,
          children: (
            <FormattedMessage defaultMessage="Prompts" description="Sidebar link for experiment prompts tab" />
          ),
        },
        componentId: 'mlflow.sidebar.experiment.prompts',
        disabled: !experimentId,
      },
      {
        key: 'exp-models',
        icon: <ModelsIcon />,
        linkProps: {
          to: experimentId
            ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Models)
            : ExperimentTrackingRoutes.experimentsObservatoryRoute,
          isActive: () => activeExperimentTab === ExperimentPageTabName.Models,
          children: (
            <FormattedMessage
              defaultMessage="Agent versions"
              description="Sidebar link for experiment agent versions tab"
            />
          ),
        },
        componentId: 'mlflow.sidebar.experiment.models',
        disabled: !experimentId,
      },
    ],
  };

  const experimentTabsML = {
    'top-level': [
      {
        key: 'exp-runs',
        icon: <ListIcon />,
        linkProps: {
          to: experimentId
            ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Runs)
            : ExperimentTrackingRoutes.experimentsObservatoryRoute,
          isActive: () => activeExperimentTab === ExperimentPageTabName.Runs,
          children: <FormattedMessage defaultMessage="Runs" description="Sidebar link for experiment runs tab" />,
        },
        componentId: 'mlflow.sidebar.experiment.runs',
        disabled: !experimentId,
      },
      {
        key: 'exp-models-ml',
        icon: <ModelsIcon />,
        linkProps: {
          to: experimentId
            ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Models)
            : ExperimentTrackingRoutes.experimentsObservatoryRoute,
          isActive: () => activeExperimentTab === ExperimentPageTabName.Models,
          children: <FormattedMessage defaultMessage="Models" description="Sidebar link for experiment models tab" />,
        },
        componentId: 'mlflow.sidebar.experiment.models',
        disabled: !experimentId,
      },
      {
        key: 'exp-traces-ml',
        icon: <ForkHorizontalIcon />,
        linkProps: {
          to: experimentId
            ? Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Traces)
            : ExperimentTrackingRoutes.experimentsObservatoryRoute,
          isActive: () => activeExperimentTab === ExperimentPageTabName.Traces,
          children: <FormattedMessage defaultMessage="Traces" description="Sidebar link for experiment traces tab" />,
        },
        componentId: 'mlflow.sidebar.experiment.traces',
        disabled: !experimentId,
      },
    ],
  };

  const experimentTabs = selectedExperimentType === 'genai' ? experimentTabsGenAI : experimentTabsML;

  // Gateway tabs - nested under AI Gateway link
  const gatewayTabs = [
    {
      key: 'gateway-endpoints',
      icon: <ChainIcon />,
      linkProps: {
        to: GatewayRoutes.gatewayPageRoute,
        isActive: () => activeGatewayTab === 'endpoints',
        children: <FormattedMessage defaultMessage="Endpoints" description="Sidebar link for gateway endpoints tab" />,
      },
      componentId: 'mlflow.sidebar.gateway.endpoints',
    },
    {
      key: 'gateway-api-keys',
      icon: <KeyIcon />,
      linkProps: {
        to: GatewayRoutes.apiKeysPageRoute,
        isActive: () => activeGatewayTab === 'api-keys',
        children: <FormattedMessage defaultMessage="API Keys" description="Sidebar link for gateway API keys tab" />,
      },
      componentId: 'mlflow.sidebar.gateway.api-keys',
    },
  ];

  // Helper function to get section label
  const getSectionLabel = (sectionKey: string): React.ReactNode | undefined => {
    switch (sectionKey) {
      case 'observability':
        return (
          <FormattedMessage
            defaultMessage="Observability"
            description="Label for the observability section in the MLflow experiment navbar"
          />
        );
      case 'evaluation':
        return (
          <FormattedMessage
            defaultMessage="Evaluation"
            description="Label for the evaluation section in the MLflow experiment navbar"
          />
        );
      case 'prompts-versions':
        return (
          <FormattedMessage
            defaultMessage="Prompts & versions"
            description="Label for the versions section in the MLflow experiment navbar"
          />
        );
      default:
        return undefined;
    }
  };

  const logTelemetryEvent = useLogTelemetryEvent();

  // Workflow switcher component - can be rendered at top or bottom
  // When at top: no top margin, border at bottom
  // When at bottom: top margin, border at top
  const getWorkflowSwitcher = (position: 'top' | 'bottom') => {
    const switcherContent = (
      <div
        css={{
          marginTop: position === 'bottom' ? theme.spacing.sm : 0,
          paddingTop: position === 'bottom' ? theme.spacing.md : 0,
          paddingBottom: position === 'top' ? theme.spacing.md : 0,
          borderTop: position === 'bottom' ? `1px solid ${theme.colors.border}` : 'none',
          borderBottom: position === 'top' ? `1px solid ${theme.colors.border}` : 'none',
          backgroundColor: theme.colors.backgroundSecondary,
        }}
      >
        <SegmentedControlGroup
          name="experiment-type-switcher"
          componentId="mlflow.sidebar.experiment_type_switcher"
          value={selectedExperimentType}
          onChange={handleExperimentTypeChange}
          css={{ width: '100%', padding: theme.spacing.xs }}
        >
          <SegmentedControlButton value="genai" css={{ flex: 1, fontSize: theme.typography.fontSizeSm }}>
            <FormattedMessage defaultMessage="GenAI" description="GenAI workflow option" />
          </SegmentedControlButton>
          <SegmentedControlButton value="ml" css={{ flex: 1, fontSize: theme.typography.fontSizeSm }}>
            <FormattedMessage defaultMessage="Machine Learning" description="Machine Learning workflow option" />
          </SegmentedControlButton>
        </SegmentedControlGroup>
      </div>
    );

    // Wrap in Popover if guidance should be shown
    if (showGuidancePopover) {
      return (
        <Popover.Root
          componentId="mlflow.sidebar.workflow_guidance"
          open={showGuidancePopover}
          onOpenChange={(open) => {
            // Only allow closing via explicit dismiss actions, not by clicking outside
            if (!open) {
              return;
            }
            setShowGuidancePopover(open);
          }}
          modal
        >
          <Popover.Trigger asChild>
            <div css={{ position: 'relative', zIndex: 1001 }}>{switcherContent}</div>
          </Popover.Trigger>
          <Popover.Content
            side="right"
            align="start"
            onEscapeKeyDown={(e) => e.preventDefault()}
            onPointerDownOutside={(e) => e.preventDefault()}
            onInteractOutside={(e) => e.preventDefault()}
            css={{
              maxWidth: 280,
              padding: theme.spacing.md,
              zIndex: 1000,
              boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.5), 0 8px 24px rgba(0, 0, 0, 0.4)',
            }}
          >
            <Popover.Arrow css={{ fill: theme.colors.backgroundPrimary }} />
            <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: theme.spacing.sm }}>
              <div css={{ fontWeight: theme.typography.typographyBoldFontWeight, fontSize: theme.typography.fontSizeMd }}>
                <FormattedMessage defaultMessage="Choose your workflow" description="Workflow selector guidance title" />
              </div>
              <Button
                componentId="mlflow.sidebar.workflow_guidance.dismiss"
                icon={<CloseIcon />}
                onClick={handleDismissGuidance}
                css={{
                  padding: 0,
                  minWidth: 'auto',
                  border: 'none',
                  background: 'transparent',
                }}
              />
            </div>
            <div css={{ fontSize: theme.typography.fontSizeSm, color: theme.colors.textSecondary }}>
              <FormattedMessage
                defaultMessage="Select your workflow type to customize which features and tabs are shown in the sidebar. You can switch between GenAI and Machine Learning workflows at any time."
                description="Workflow selector guidance message"
              />
            </div>
            <Button
              componentId="mlflow.sidebar.workflow_guidance.got_it"
              onClick={handleDismissGuidance}
              css={{ marginTop: theme.spacing.md, width: '100%' }}
            >
              <FormattedMessage defaultMessage="Got it" description="Workflow selector guidance dismiss button" />
            </Button>
          </Popover.Content>
        </Popover.Root>
      );
    }

    return switcherContent;
  };

  return (
    <aside
      css={{
        width: 230,
        flexShrink: 0,
        padding: theme.spacing.sm,
        display: 'inline-flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
      }}
    >
      {/* Render workflow switcher at top (above New button) if setting enabled */}
      {isExperimentTypeSelectorAtTop && getWorkflowSwitcher('top')}

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
          {mainMenuItems
            .filter((item) => item.dropdownProps !== undefined)
            .map(({ key, icon, dropdownProps }) => (
              <DropdownMenu.Item key={key} componentId={dropdownProps!.componentId} onClick={dropdownProps!.onClick}>
                <DropdownMenu.IconWrapper>{icon}</DropdownMenu.IconWrapper>
                {dropdownProps!.children}
              </DropdownMenu.Item>
            ))}
        </DropdownMenu.Content>
      </DropdownMenu.Root>

      <nav css={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '100%' }}>
        <div>
          <ul
            css={{
              listStyleType: 'none',
              padding: 0,
              margin: 0,
            }}
          >
            {/* Top-level items (Home, Experiments) */}
            {topLevelMenuItems.map(({ key, icon, linkProps, componentId }) => (
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

                {/* Nested experiment tabs under Experiments with section groupings */}
                {key === 'experiments' && (
                  <div
                    css={{
                      paddingLeft: theme.spacing.lg,
                      paddingBottom: theme.spacing.sm,
                    }}
                  >
                    {Object.entries(experimentTabs).map(([sectionKey, sectionItems]) => {
                      const sectionLabel = getSectionLabel(sectionKey);
                      if (!sectionItems || sectionItems.length === 0) return null;

                      return (
                        <div key={sectionKey}>
                          {sectionLabel && (
                            <div
                              css={{
                                fontSize: theme.typography.fontSizeSm,
                                color: theme.colors.textSecondary,
                                paddingInline: theme.spacing.md,
                                paddingTop: theme.spacing.sm,
                                paddingBottom: theme.spacing.xs,
                              }}
                            >
                              {sectionLabel}
                            </div>
                          )}
                          <ul
                            css={{
                              listStyleType: 'none',
                              padding: 0,
                              margin: 0,
                            }}
                          >
                            {sectionItems.map(
                              ({
                                key: tabKey,
                                icon: tabIcon,
                                linkProps: tabLinkProps,
                                componentId: tabComponentId,
                                disabled,
                              }) => {
                                const linkContent = (
                                  <Link
                                    to={tabLinkProps.to}
                                    aria-current={!disabled && tabLinkProps.isActive() ? 'page' : undefined}
                                    css={{
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: theme.spacing.sm,
                                      color: disabled ? theme.colors.textSecondary : theme.colors.textPrimary,
                                      opacity: disabled ? 0.5 : 1,
                                      paddingInline: theme.spacing.md,
                                      paddingBlock: theme.spacing.xs,
                                      borderRadius: theme.borders.borderRadiusSm,
                                      pointerEvents: disabled ? 'none' : 'auto',
                                      cursor: disabled ? 'not-allowed' : 'pointer',
                                      '&:hover': disabled
                                        ? {}
                                        : {
                                            color: theme.colors.actionLinkHover,
                                            backgroundColor: theme.colors.actionDefaultBackgroundHover,
                                          },
                                      '&[aria-current="page"]': disabled
                                        ? {}
                                        : {
                                            backgroundColor: theme.colors.actionDefaultBackgroundPress,
                                            color: theme.isDarkMode ? theme.colors.blue300 : theme.colors.blue700,
                                            fontWeight: theme.typography.typographyBoldFontWeight,
                                          },
                                    }}
                                    onClick={(e) => {
                                      if (disabled) {
                                        e.preventDefault();
                                        return;
                                      }
                                      logTelemetryEvent({
                                        componentId: tabComponentId,
                                        componentViewId: viewId,
                                        componentType: DesignSystemEventProviderComponentTypes.TypographyLink,
                                        componentSubType: null,
                                        eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
                                      });
                                    }}
                                  >
                                    {tabIcon}
                                    {tabLinkProps.children}
                                  </Link>
                                );

                                return (
                                  <li key={tabKey}>
                                    {disabled ? (
                                      <Tooltip
                                        componentId={`${tabComponentId}.tooltip`}
                                        content={
                                          <FormattedMessage
                                            defaultMessage="Select an experiment to view this tab"
                                            description="Tooltip for disabled experiment tab"
                                          />
                                        }
                                      >
                                        <div>{linkContent}</div>
                                      </Tooltip>
                                    ) : (
                                      linkContent
                                    )}
                                  </li>
                                );
                              },
                            )}
                          </ul>
                        </div>
                      );
                    })}
                  </div>
                )}
              </li>
            ))}

            {/* Bottom items (Models, Prompts, Gateway) */}
            {bottomMenuItems.map(({ key, icon, linkProps, componentId }) => (
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

                {/* Nested gateway tabs under AI Gateway */}
                {key === 'gateway' && (
                  <div
                    css={{
                      paddingLeft: theme.spacing.lg,
                      paddingBottom: theme.spacing.sm,
                    }}
                  >
                    <ul
                      css={{
                        listStyleType: 'none',
                        padding: 0,
                        margin: 0,
                      }}
                    >
                      {gatewayTabs.map(
                        ({ key: tabKey, icon: tabIcon, linkProps: tabLinkProps, componentId: tabComponentId }) => (
                          <li key={tabKey}>
                            <Link
                              to={tabLinkProps.to}
                              aria-current={tabLinkProps.isActive() ? 'page' : undefined}
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
                                  componentId: tabComponentId,
                                  componentViewId: viewId,
                                  componentType: DesignSystemEventProviderComponentTypes.TypographyLink,
                                  componentSubType: null,
                                  eventType: DesignSystemEventProviderAnalyticsEventTypes.OnClick,
                                })
                              }
                            >
                              {tabIcon}
                              {tabLinkProps.children}
                            </Link>
                          </li>
                        ),
                      )}
                    </ul>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>

        <div>
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

          {/* Render workflow switcher at bottom if setting disabled */}
          {!isExperimentTypeSelectorAtTop && getWorkflowSwitcher('bottom')}
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

import {
  BeakerIcon,
  Button,
  DatabaseIcon,
  DropdownMenu,
  GearIcon,
  HomeIcon,
  ModelsIcon,
  PlusIcon,
  TextBoxIcon,
  useDesignSystemTheme,
  DesignSystemEventProviderComponentTypes,
  DesignSystemEventProviderAnalyticsEventTypes,
} from '@databricks/design-system';
import type { Location } from '../utils/RoutingUtils';
import { Link, matchPath, useLocation, useNavigate } from '../utils/RoutingUtils';
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

const isHomeActive = (location: Location) => matchPath({ path: '/', end: true }, location.pathname);
const isExperimentsActive = (location: Location) =>
  matchPath('/experiments/*', location.pathname) || matchPath('/compare-experiments/*', location.pathname);
const isModelsActive = (location: Location) => matchPath('/models/*', location.pathname);
const isPromptsActive = (location: Location) => matchPath('/prompts/*', location.pathname);
const isGatewayActive = (location: Location) => matchPath('/gateway/*', location.pathname);
const isSettingsActive = (location: Location) => matchPath('/settings/*', location.pathname);

export function MlflowSidebar() {
  const location = useLocation();
  const { theme } = useDesignSystemTheme();
  const invalidateExperimentList = useInvalidateExperimentList();
  const navigate = useNavigate();
  const viewId = useMemo(() => crypto.randomUUID(), []);

  const [showCreateExperimentModal, setShowCreateExperimentModal] = useState(false);
  const [showCreateModelModal, setShowCreateModelModal] = useState(false);
  const { CreatePromptModal, openModal: openCreatePromptModal } = useCreatePromptModal({
    mode: CreatePromptModalMode.CreatePrompt,
    onSuccess: ({ promptName }) => navigate(Routes.getPromptDetailsPageRoute(promptName)),
  });

  const menuItems = [
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
      icon: <DatabaseIcon />,
      linkProps: {
        to: GatewayRoutes.gatewayPageRoute,
        isActive: isGatewayActive,
        children: <FormattedMessage defaultMessage="Gateway" description="Sidebar link for gateway configuration" />,
      },
      componentId: 'mlflow.sidebar.gateway_tab_link',
    },
  ];

  const logTelemetryEvent = useLogTelemetryEvent();

  return (
    <aside
      css={{
        width: 200,
        flexShrink: 0,
        padding: theme.spacing.sm,
        display: 'inline-flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
      }}
    >
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
              <DropdownMenu.Item key={key} componentId={dropdownProps.componentId} onClick={dropdownProps.onClick}>
                <DropdownMenu.IconWrapper>{icon}</DropdownMenu.IconWrapper>
                {dropdownProps.children}
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
          {menuItems.map(({ key, icon, linkProps, componentId }) => (
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
            </li>
          ))}
        </ul>
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

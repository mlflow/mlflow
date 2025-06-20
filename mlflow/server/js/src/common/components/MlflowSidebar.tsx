import {
  BeakerIcon,
  Button,
  DropdownMenu,
  ModelsIcon,
  PlusIcon,
  TextBoxIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { Link, matchPath, useLocation, Location, useNavigate } from '../../common/utils/RoutingUtils';
import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { ModelRegistryRoutes } from '../../model-registry/routes';
import { Interpolation } from '@emotion/react';
import { Theme } from '@databricks/design-system/dist/theme';
import { CreateExperimentModal } from '../../experiment-tracking/components/modals/CreateExperimentModal';
import { useState } from 'react';
import { useInvalidateExperimentList } from '../../experiment-tracking/components/experiment-page/hooks/useExperimentListQuery';
import { CreateModelModal } from '../../model-registry/components/CreateModelModal';
import {
  CreatePromptModalMode,
  useCreatePromptModal,
} from '../../experiment-tracking/pages/prompts/hooks/useCreatePromptModal';
import Routes from '../../experiment-tracking/routes';
import { FormattedMessage } from 'react-intl';

const isExperimentsActive = (location: Location) =>
  matchPath('/', location.pathname) ||
  matchPath('/experiments/*', location.pathname) ||
  matchPath('/compare-experiments/*', location.pathname);
const isModelsActive = (location: Location) => matchPath('/models/*', location.pathname);
const isPromptsActive = (location: Location) => matchPath('/prompts/*', location.pathname);

export function MlflowSidebar() {
  const location = useLocation();
  const { theme } = useDesignSystemTheme();
  const invalidateExperimentList = useInvalidateExperimentList();
  const navigate = useNavigate();

  const [showCreateExperimentModal, setShowCreateExperimentModal] = useState(false);
  const [showCreateModelModal, setShowCreateModelModal] = useState(false);
  const { CreatePromptModal, openModal: openCreatePromptModal } = useCreatePromptModal({
    mode: CreatePromptModalMode.CreatePrompt,
    onSuccess: ({ promptName }) => navigate(Routes.getPromptDetailsPageRoute(promptName)),
  });

  const ulStyles: Interpolation<Theme> = {
    listStyleType: 'none',
    padding: 0,
    margin: 0,
  };

  const linkStyles: Interpolation<Theme> = {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.sm,
    color: theme.colors.textPrimary,
    paddingInline: theme.spacing.md,
    paddingBlock: theme.spacing.xs,
    borderRadius: theme.borders.borderRadiusSm,
    '&:hover': {
      color: theme.colors.actionLinkHover,
    },
    '&[aria-current="page"]': {
      backgroundColor: theme.colors.actionDefaultBackgroundPress,
      color: theme.isDarkMode ? theme.colors.blue300 : theme.colors.blue700,
      fontWeight: theme.typography.typographyBoldFontWeight,
    },
  };

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
          <Button componentId="mlflow_sidebar.new_button" icon={<PlusIcon />}>
            <FormattedMessage
              defaultMessage="New"
              description="Sidebar create popover button to create new experiment, model or prompt"
            />
          </Button>
        </DropdownMenu.Trigger>

        <DropdownMenu.Content side="right" sideOffset={theme.spacing.sm} align="start">
          <DropdownMenu.Item
            componentId="mlflow_sidebar.create_experiment_button"
            onClick={() => setShowCreateExperimentModal(true)}
          >
            <DropdownMenu.IconWrapper>
              <BeakerIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Experiment"
              description="Sidebar button inside the 'new' popover to create new experiment"
            />
          </DropdownMenu.Item>
          <DropdownMenu.Item
            componentId="mlflow_sidebar.create_model_button"
            onClick={() => setShowCreateModelModal(true)}
          >
            <DropdownMenu.IconWrapper>
              <ModelsIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Model"
              description="Sidebar button inside the 'new' popover to create new model"
            />
          </DropdownMenu.Item>
          <DropdownMenu.Item componentId="mlflow_sidebar.create_prompt_button" onClick={openCreatePromptModal}>
            <DropdownMenu.IconWrapper>
              <TextBoxIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Prompt"
              description="Sidebar button inside the 'new' popover to create new prompt"
            />
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root>

      <nav>
        <ul css={ulStyles}>
          <li>
            <Link
              to={ExperimentTrackingRoutes.experimentsObservatoryRoute}
              aria-current={isExperimentsActive(location) ? 'page' : undefined}
              css={linkStyles}
            >
              <BeakerIcon />
              <FormattedMessage defaultMessage="Experiments" description="Sidebar link for experiments tab" />
            </Link>
          </li>
          <li>
            <Link
              to={ModelRegistryRoutes.modelListPageRoute}
              aria-current={isModelsActive(location) ? 'page' : undefined}
              css={linkStyles}
            >
              <ModelsIcon />
              <FormattedMessage defaultMessage="Models" description="Sidebar link for models tab" />
            </Link>
          </li>
          <li>
            <Link
              to={ExperimentTrackingRoutes.promptsPageRoute}
              aria-current={isPromptsActive(location) ? 'page' : undefined}
              css={linkStyles}
            >
              <TextBoxIcon />
              <FormattedMessage defaultMessage="Prompts" description="Sidebar link for prompts tab" />
            </Link>
          </li>
        </ul>
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

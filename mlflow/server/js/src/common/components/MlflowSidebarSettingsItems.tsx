import { ArrowLeftIcon, GearIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { matchPath, useSearchParams } from '../utils/RoutingUtils';
import type { Location } from '../utils/RoutingUtils';
import { MlflowSidebarLink } from './MlflowSidebarLink';
import {
  SETTINGS_RETURN_TO_PARAM,
  SETTINGS_SECTION_GENERAL,
  SETTINGS_SECTION_LLM_CONNECTIONS,
  SETTINGS_SECTION_WEBHOOKS,
} from '../../settings/settingsSectionConstants';

const matchSettingsSection =
  (section: string) =>
  (location: Location): boolean =>
    Boolean(
      matchPath({ path: ExperimentTrackingRoutes.getSettingsSectionRoute(section), end: true }, location.pathname),
    );

const isSettingsExitLinkActive = () => false;

export const MlflowSidebarSettingsItems = ({ collapsed }: { collapsed: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const [searchParams] = useSearchParams();

  const returnToParam = searchParams.get(SETTINGS_RETURN_TO_PARAM) ?? undefined;
  const exitTo = returnToParam ?? ExperimentTrackingRoutes.rootRoute;

  const sectionTo = (section: string) => {
    const path = ExperimentTrackingRoutes.getSettingsSectionRoute(section);
    return returnToParam ? `${path}?${SETTINGS_RETURN_TO_PARAM}=${encodeURIComponent(returnToParam)}` : path;
  };

  return (
    <>
      <MlflowSidebarLink
        css={{
          border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
          marginBottom: theme.spacing.sm,
        }}
        to={exitTo}
        componentId="mlflow.sidebar.settings_back_link"
        isActive={isSettingsExitLinkActive}
        icon={<ArrowLeftIcon />}
        collapsed={collapsed}
        tooltipContent={
          <FormattedMessage
            defaultMessage="Return to main navigation"
            description="Tooltip for leaving Settings sub-sidebar to Home and other items"
          />
        }
      >
        <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <GearIcon />
          <FormattedMessage
            defaultMessage="Settings"
            description="Settings sub-sidebar: return to main nav (same label as main Settings)"
          />
        </span>
      </MlflowSidebarLink>
      <MlflowSidebarLink
        css={{ paddingLeft: collapsed ? undefined : theme.spacing.lg }}
        to={sectionTo(SETTINGS_SECTION_GENERAL)}
        componentId="mlflow.sidebar.settings_general_link"
        isActive={matchSettingsSection(SETTINGS_SECTION_GENERAL)}
        collapsed={collapsed}
      >
        <FormattedMessage defaultMessage="General" description="Sidebar link: Settings > General" />
      </MlflowSidebarLink>
      <MlflowSidebarLink
        css={{ paddingLeft: collapsed ? undefined : theme.spacing.lg }}
        to={sectionTo(SETTINGS_SECTION_LLM_CONNECTIONS)}
        componentId="mlflow.sidebar.settings_llm_connections_link"
        isActive={matchSettingsSection(SETTINGS_SECTION_LLM_CONNECTIONS)}
        collapsed={collapsed}
      >
        <FormattedMessage defaultMessage="LLM Connections" description="Sidebar link: Settings > LLM Connections" />
      </MlflowSidebarLink>
      <MlflowSidebarLink
        css={{ paddingLeft: collapsed ? undefined : theme.spacing.lg }}
        to={sectionTo(SETTINGS_SECTION_WEBHOOKS)}
        componentId="mlflow.sidebar.settings_webhooks_link"
        isActive={matchSettingsSection(SETTINGS_SECTION_WEBHOOKS)}
        collapsed={collapsed}
      >
        <FormattedMessage defaultMessage="Webhooks" description="Sidebar link: Settings > Webhooks" />
      </MlflowSidebarLink>
    </>
  );
};

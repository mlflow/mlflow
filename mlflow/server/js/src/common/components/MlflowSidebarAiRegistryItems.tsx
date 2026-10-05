import { McpIcon, PuzzleIcon, SparkleDoubleIcon, TextBoxIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import MCPRegistryRoutes from '../../mcp-registry/routes';
import { MCPRegistryBetaTag } from '../../mcp-registry/components/MCPRegistryBetaTag';
import SkillRegistryRoutes from '../../skill-registry/routes';
import { SkillRegistryBetaTag } from '../../skill-registry/components/SkillRegistryBetaTag';
import { matchPath } from '../utils/RoutingUtils';
import type { Location } from '../utils/RoutingUtils';
import { MlflowSidebarLink } from './MlflowSidebarLink';

const isPromptsActive = (location: Location) => Boolean(matchPath('/prompts/*', location.pathname));
const isMCPRegistryActive = (location: Location) => Boolean(matchPath('/mcp-registry/*', location.pathname));
const isSkillRegistryActive = (location: Location) =>
  Boolean(matchPath({ path: '/skills', end: true }, location.pathname) || matchPath('/skills/*', location.pathname));

export const isAiRegistryActive = (location: Location) =>
  isPromptsActive(location) || isMCPRegistryActive(location) || isSkillRegistryActive(location);

export const MlflowSidebarAiRegistryItems = ({ collapsed }: { collapsed: boolean }) => {
  const { theme } = useDesignSystemTheme();

  // Rendered inside the sidebar's <ul>, so every top-level child must be an <li>.
  return (
    <>
      <li
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          justifyContent: collapsed ? 'center' : 'flex-start',
          paddingInline: collapsed ? 0 : theme.spacing.sm,
          paddingBlock: collapsed ? 7 : theme.spacing.sm,
          border: collapsed ? `1px solid ${theme.colors.actionDefaultBorderDefault}` : 'none',
          borderRadius: theme.borders.borderRadiusSm,
          marginBottom: collapsed ? theme.spacing.sm : 0,
          boxSizing: 'border-box',
        }}
      >
        <SparkleDoubleIcon />
        {!collapsed && (
          <FormattedMessage defaultMessage="AI Registry" description="Sidebar label for the AI Registry section" />
        )}
      </li>
      <MlflowSidebarLink
        css={{ paddingLeft: collapsed ? undefined : theme.spacing.lg }}
        to={ExperimentTrackingRoutes.promptsPageRoute}
        componentId="mlflow.sidebar.prompts_tab_link"
        isActive={isPromptsActive}
        icon={<TextBoxIcon />}
        collapsed={collapsed}
      >
        <FormattedMessage defaultMessage="Prompts" description="Sidebar link for prompts tab" />
      </MlflowSidebarLink>
      <MlflowSidebarLink
        css={{ paddingLeft: collapsed ? undefined : theme.spacing.lg }}
        to={MCPRegistryRoutes.mcpRegistryPageRoute}
        componentId="mlflow.sidebar.mcp_registry_tab_link"
        isActive={isMCPRegistryActive}
        icon={<McpIcon />}
        collapsed={collapsed}
      >
        <FormattedMessage defaultMessage="MCP" description="Sidebar link for MCP registry page" />
        <span css={{ marginLeft: 'auto' }}>
          <MCPRegistryBetaTag />
        </span>
      </MlflowSidebarLink>
      <MlflowSidebarLink
        css={{ paddingLeft: collapsed ? undefined : theme.spacing.lg }}
        to={SkillRegistryRoutes.skillRegistryPageRoute}
        componentId="mlflow.sidebar.skill_registry_tab_link"
        isActive={isSkillRegistryActive}
        icon={<PuzzleIcon />}
        collapsed={collapsed}
      >
        <FormattedMessage defaultMessage="Skills" description="Sidebar link for Skill Registry page" />
        <span css={{ marginLeft: 'auto' }}>
          <SkillRegistryBetaTag />
        </span>
      </MlflowSidebarLink>
    </>
  );
};

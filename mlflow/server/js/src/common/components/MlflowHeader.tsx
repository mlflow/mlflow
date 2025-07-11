import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { Link } from '../utils/RoutingUtils';
import { HomePageDocsUrl, Version } from '../constants';
import { DarkThemeSwitch } from '@mlflow/mlflow/src/common/components/DarkThemeSwitch';

const colors = {
  headerBg: '#0b3574',
  headerText: '#e7f1fb',
  headerActiveLink: '#43C9ED',
};

const classNames = {
  activeNavLink: { borderBottom: `4px solid ${colors.headerActiveLink}` },
};

const isExperimentsActive = (location: Location) => matchPath('/experiments/*', location.pathname);
const isModelsActive = (location: Location) => matchPath('/models/*', location.pathname);
const isPromptsActive = (location: Location) => matchPath('/prompts/*', location.pathname);

export const MlflowHeader = ({
  isDarkTheme = false,
  setIsDarkTheme = (val: boolean) => { },
  sidebarOpen,
  toggleSidebar,
}: {
  isDarkTheme?: boolean;
  setIsDarkTheme?: (isDarkTheme: boolean) => void;
  sidebarOpen: boolean;
  toggleSidebar: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <header
      css={{
        backgroundColor: theme.colors.backgroundSecondary,
        color: theme.colors.textSecondary,
        display: 'flex',
        paddingLeft: theme.spacing.sm,
        paddingRight: theme.spacing.md,
        paddingTop: theme.spacing.sm + theme.spacing.xs,
        paddingBottom: theme.spacing.xs,
        a: {
          color: theme.colors.textSecondary,
        },
        alignItems: 'center',
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
        }}
      >
        <Button
          type="tertiary"
          componentId="mlflow_header.toggle_sidebar_button"
          onClick={toggleSidebar}
          aria-label="Toggle sidebar"
          aria-pressed={sidebarOpen}
          icon={<MenuIcon />}
        />
        <Link to={ExperimentTrackingRoutes.rootRoute}>
          <MlflowLogo
            css={{
              display: 'block',
              height: theme.spacing.md * 2,
              color: theme.colors.textPrimary,
            }}
          />
        </Link>
        <span
          css={{
            fontSize: theme.typography.fontSizeSm,
          }}
        >
          {Version}
        </span>
      </div>
      <div
        css={{
          display: 'flex',
          paddingTop: 20,
          fontSize: 16,
          gap: 24,
        }}
      >
        <Link
          to={ExperimentTrackingRoutes.rootRoute}
          style={isExperimentsActive(location) ? classNames.activeNavLink : undefined}
        >
          Experiments
        </Link>
        <Link
          to={ModelRegistryRoutes.modelListPageRoute}
          style={isModelsActive(location) ? classNames.activeNavLink : undefined}
        >
          Models
        </Link>
        <Link
          to={ExperimentTrackingRoutes.promptsPageRoute}
          style={isPromptsActive(location) ? classNames.activeNavLink : undefined}
        >
          Prompts
        </Link>
      </div>
      <div css={{ flex: 1 }} />
      <div css={{ display: 'flex', gap: theme.spacing.lg, alignItems: 'center' }}>
        <DarkThemeSwitch isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} />
        <a href="https://github.com/mlflow/mlflow">GitHub</a>
        <a href={HomePageDocsUrl}>Docs</a>
      </div>
    </header>
  );
};

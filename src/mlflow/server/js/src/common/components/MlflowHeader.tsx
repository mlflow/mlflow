import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { Link, Location, matchPath, useLocation } from '../utils/RoutingUtils';
import logo from '../../common/static/home-logo.png';
import { ModelRegistryRoutes } from '../../model-registry/routes';
import { HomePageDocsUrl, Version } from '../constants';
import { DarkThemeSwitch } from 'common/components/DarkThemeSwitch';

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

export const MlflowHeader = ({
  isDarkTheme = false,
  setIsDarkTheme = (val: boolean) => {},
}: {
  isDarkTheme?: boolean;
  setIsDarkTheme?: (isDarkTheme: boolean) => void;
}) => {
  const location = useLocation();
  return (
    <header
      css={{
        backgroundColor: colors.headerBg,
        height: '60px',
        color: colors.headerText,
        display: 'flex',
        gap: 24,
        a: {
          color: colors.headerText,
        },
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'flex-end',
        }}
      >
        <Link to={ExperimentTrackingRoutes.rootRoute}>
          <img
            css={{
              height: 40,
              marginLeft: 24,
              marginTop: 10,
              marginBottom: 10,
            }}
            alt="MLflow"
            src={logo}
          />
        </Link>
        <span
          css={{
            fontSize: 12,
            marginLeft: 5,
            marginBottom: 13,
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
      </div>
      <div css={{ flex: 1 }} />
      <div css={{ display: 'flex', gap: 24, paddingTop: 20, fontSize: 16, marginRight: 24 }}>
        <DarkThemeSwitch isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} />
        <a href="https://github.com/mlflow/mlflow">GitHub</a>
        <a href={HomePageDocsUrl}>Docs</a>
      </div>
    </header>
  );
};

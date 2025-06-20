import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { Link } from '../utils/RoutingUtils';
import { HomePageDocsUrl, Version } from '../constants';
import { DarkThemeSwitch } from '@mlflow/mlflow/src/common/components/DarkThemeSwitch';
import { useDesignSystemTheme } from '@databricks/design-system';
import { MlflowLogo } from './MlflowLogo';

export const MlflowHeader = ({
  isDarkTheme = false,
  setIsDarkTheme = (val: boolean) => {},
}: {
  isDarkTheme?: boolean;
  setIsDarkTheme?: (isDarkTheme: boolean) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <header
      css={{
        backgroundColor: theme.colors.backgroundSecondary,
        height: '60px',
        color: theme.colors.textSecondary,
        display: 'flex',
        gap: 24,
        a: {
          color: theme.colors.textSecondary,
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
          <MlflowLogo
            css={{
              display: 'block',
              height: 40,
              marginLeft: 24,
              marginTop: 10,
              marginBottom: 10,
              color: theme.colors.textPrimary,
            }}
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
      <div css={{ flex: 1 }} />
      <div css={{ display: 'flex', gap: 24, paddingTop: 20, fontSize: 16, marginRight: 24 }}>
        <DarkThemeSwitch isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} />
        <a href="https://github.com/mlflow/mlflow">GitHub</a>
        <a href={HomePageDocsUrl}>Docs</a>
      </div>
    </header>
  );
};

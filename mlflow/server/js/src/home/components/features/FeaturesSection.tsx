import { ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { featureDefinitions } from './feature-definitions';
import { LaunchDemoCard } from './LaunchDemoCard';
import { FeatureCard } from './FeatureCard';
import { useLocalStorage } from '../../../shared/web-shared/hooks';

const COLLAPSED_KEY = 'mlflow.home.getting-started.collapsed';
const COLLAPSED_KEY_VERSION = 1;

export const FeaturesSection = () => {
  const { theme } = useDesignSystemTheme();

  const [isCollapsed, setIsCollapsed] = useLocalStorage({
    key: COLLAPSED_KEY,
    version: COLLAPSED_KEY_VERSION,
    initialValue: false,
  });

  return (
    <section css={{ display: 'flex', flexDirection: 'column' }}>
      <button
        type="button"
        onClick={() => setIsCollapsed(!isCollapsed)}
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          background: 'none',
          border: 'none',
          padding: 0,
          cursor: 'pointer',
          font: 'inherit',
          color: 'inherit',
        }}
      >
        <Typography.Title level={3} css={{ '&&': { margin: 0 } }}>
          <FormattedMessage defaultMessage="Getting Started" description="Home page features section title" />
        </Typography.Title>
        <span css={{ display: 'inline-flex', alignItems: 'center', color: theme.colors.textSecondary }}>
          {isCollapsed ? <ChevronRightIcon /> : <ChevronDownIcon />}
        </span>
      </button>
      {!isCollapsed && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: theme.spacing.md }}>
          <LaunchDemoCard />
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.sm,
              overflowX: 'auto',
              paddingBottom: theme.spacing.xs,
              scrollbarWidth: 'thin',
              '&::-webkit-scrollbar': {
                height: 4,
              },
              '&::-webkit-scrollbar-thumb': {
                background: theme.colors.borderDecorative,
                borderRadius: 2,
              },
            }}
          >
            {featureDefinitions.map((feature) => (
              <FeatureCard key={feature.id} feature={feature} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
};

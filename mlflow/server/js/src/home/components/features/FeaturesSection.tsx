import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { featureDefinitions } from './feature-definitions';
import { LaunchDemoCard } from './LaunchDemoCard';
import { FeatureCard } from './FeatureCard';

export const FeaturesSection = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Typography.Title level={3} css={{ margin: 0 }}>
        <FormattedMessage defaultMessage="Getting Started" description="Home page features section title" />
      </Typography.Title>
      <LaunchDemoCard />
      <Typography.Text bold color="secondary">
        <FormattedMessage defaultMessage="Explore Features" description="Feature cards section title" />
      </Typography.Text>
      <div
        css={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: theme.spacing.sm,
        }}
      >
        {featureDefinitions.map((feature) => (
          <FeatureCard key={feature.id} feature={feature} />
        ))}
      </div>
    </section>
  );
};

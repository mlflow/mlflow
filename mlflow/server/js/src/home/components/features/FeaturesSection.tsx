import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { FeatureCard } from './FeatureCard';
import { featureDefinitions } from './feature-definitions';
import { DemoBanner } from '../DemoBanner';

export const FeaturesSection = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', alignItems: 'baseline', gap: theme.spacing.sm }}>
        <Typography.Title level={3} css={{ margin: 0 }}>
          <FormattedMessage defaultMessage="Explore MLflow" description="Home page features section title" />
        </Typography.Title>
        <Typography.Text color="secondary" size="sm">
          <FormattedMessage
            defaultMessage="Try out features with demo data or dive into the docs"
            description="Home page features section subtitle"
          />
        </Typography.Text>
      </div>
      <DemoBanner />
      <section
        css={{
          marginBottom: theme.spacing.lg,
          width: '100%',
          minWidth: 0,
          display: 'flex',
          gap: theme.spacing.md,
          flexWrap: 'wrap',
        }}
      >
        {featureDefinitions.map((feature) => (
          <FeatureCard key={feature.id} feature={feature} />
        ))}
      </section>
    </section>
  );
};

import { useState, useCallback, useMemo } from 'react';
import { Typography, useDesignSystemTheme, Button, Spinner, ArrowRightIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import Utils from '../../../common/utils/Utils';
import { getAjaxUrl } from '../../../common/utils/FetchUtils';
import type { FeatureDefinition } from './feature-definitions';

interface FeatureCardProps {
  feature: FeatureDefinition;
}

export const FeatureCard = ({ feature }: FeatureCardProps) => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);

  const handleExploreDemo = useCallback(async () => {
    if (!feature.demoFeatureId) {
      navigate(feature.navigationPath);
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/demo/generate'), {
        method: 'POST',
      });
      const data = await response.json();
      const experimentId = data.experiment_id;

      // Construct navigation URL based on feature type
      let targetUrl = feature.navigationPath;
      if (experimentId) {
        if (feature.demoFeatureId === 'traces') {
          targetUrl = `/experiments/${experimentId}/traces`;
        } else if (feature.demoFeatureId === 'evaluation') {
          targetUrl = `/experiments/${experimentId}/evaluation-runs`;
        } else if (feature.demoFeatureId === 'judges') {
          targetUrl = `/experiments/${experimentId}/judges`;
        }
        // prompts uses the global /prompts page, so keep navigationPath
      }
      navigate(targetUrl);
    } catch (error) {
      Utils.logErrorAndNotifyUser(error);
    } finally {
      setIsLoading(false);
    }
  }, [feature.demoFeatureId, feature.navigationPath, navigate]);

  const handleNavigate = useCallback(() => {
    navigate(feature.navigationPath);
  }, [feature.navigationPath, navigate]);

  const styles = useMemo(
    () => ({
      card: {
        overflow: 'hidden',
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusMd,
        background: theme.colors.backgroundPrimary,
        width: 360,
        minWidth: 360,
        boxSizing: 'border-box' as const,
        boxShadow: theme.shadows.sm,
      },
      header: {
        display: 'flex',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
        paddingBottom: theme.spacing.sm,
        height: 48,
        boxSizing: 'border-box' as const,
      },
      iconWrapper: {
        borderRadius: theme.borders.borderRadiusSm,
        background: theme.colors.actionDefaultBackgroundHover,
        padding: theme.spacing.xs,
        color: theme.colors.blue500,
        height: 'min-content',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      },
      content: {
        display: 'flex',
        flexDirection: 'column' as const,
        gap: theme.spacing.xs,
      },
      summary: {
        padding: `0 ${theme.spacing.md}px`,
        paddingBottom: theme.spacing.md,
        height: 72,
        boxSizing: 'border-box' as const,
        overflow: 'hidden',
      },
      section: {
        borderTop: `1px solid ${theme.colors.borderDecorative}`,
        padding: theme.spacing.md,
        height: 72,
        boxSizing: 'border-box' as const,
      },
      sectionHeader: {
        marginBottom: theme.spacing.xs,
      },
      actionsRow: {
        display: 'flex',
        gap: theme.spacing.sm,
        alignItems: 'center',
        height: 28,
      },
    }),
    [theme],
  );

  return (
    <div css={styles.card}>
      <div css={styles.header}>
        <div css={styles.iconWrapper}>
          <feature.icon />
        </div>
        <div css={styles.content}>
          <span role="heading" aria-level={3}>
            <Typography.Text bold size="lg">
              {feature.title}
            </Typography.Text>
          </span>
        </div>
      </div>

      <div css={styles.summary}>
        <Typography.Text color="secondary" size="sm">
          {feature.summary}
        </Typography.Text>
      </div>

      <div css={styles.section}>
        <div css={styles.sectionHeader}>
          <Typography.Text bold size="sm" color="secondary">
            <FormattedMessage defaultMessage="Try it out" description="Feature card try it out section header" />
          </Typography.Text>
        </div>
        <div css={styles.actionsRow}>
          {feature.demoFeatureId ? (
            isLoading ? (
              <Spinner size="small" />
            ) : (
              <Button
                componentId={`mlflow.home.feature.${feature.id}.explore`}
                size="small"
                type="primary"
                onClick={handleExploreDemo}
                icon={<ArrowRightIcon />}
              >
                <FormattedMessage defaultMessage="Explore demo" description="Feature card explore button" />
              </Button>
            )
          ) : (
            <Button
              componentId={`mlflow.home.feature.${feature.id}.go`}
              size="small"
              type="primary"
              onClick={handleNavigate}
              icon={<ArrowRightIcon />}
            >
              <FormattedMessage
                defaultMessage="Go to {feature}"
                description="Feature card go button"
                values={{ feature: feature.title }}
              />
            </Button>
          )}
        </div>
      </div>

      <div css={styles.section}>
        <div css={styles.sectionHeader}>
          <Typography.Text bold size="sm" color="secondary">
            <FormattedMessage defaultMessage="Learn more" description="Feature card learn more section header" />
          </Typography.Text>
        </div>
        <div css={styles.actionsRow}>
          <Typography.Link componentId={`mlflow.home.feature.${feature.id}.docs`} href={feature.docsLink} openInNewTab>
            <FormattedMessage defaultMessage="Read the docs" description="Feature card docs link" />
          </Typography.Link>
        </div>
      </div>
    </div>
  );
};

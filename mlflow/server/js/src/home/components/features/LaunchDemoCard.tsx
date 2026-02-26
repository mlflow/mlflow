import { useState, useCallback } from 'react';
import { ArrowRightIcon, Button, Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useNavigate } from '../../../common/utils/RoutingUtils';
import { fetchAPI, getAjaxUrl } from '../../../common/utils/FetchUtils';
import { WorkflowType, useWorkflowType } from '../../../common/contexts/WorkflowTypeContext';
import demoScreenshot from '../../../common/static/demo-tracing-screenshot.png';
import demoScreenshotDark from '../../../common/static/demo-tracing-screenshot-dark.png';

export const DEMO_BANNER_DISMISSED_KEY = 'mlflow.demo.banner.dismissed';

export const LaunchDemoCard = () => {
  const navigate = useNavigate();
  const { theme } = useDesignSystemTheme();
  const [isLoading, setIsLoading] = useState(false);
  const { setWorkflowType } = useWorkflowType();

  const handleLaunchDemo = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/demo/generate'), {
        method: 'POST',
      });
      const url = (data.navigation_url || '/experiments').replace(/^#\//, '/');
      setWorkflowType(WorkflowType.GENAI);
      navigate(url);
    } catch (error) {
      console.error('Failed to generate demo:', error);
      navigate('/experiments');
    } finally {
      setIsLoading(false);
    }
  }, [navigate, setWorkflowType]);

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.md,
        padding: theme.spacing.md,
        background: theme.colors.backgroundPrimary,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.borderDecorative}`,
        boxShadow: theme.shadows.sm,
      }}
    >
      <img
        src={theme.isDarkMode ? demoScreenshotDark : demoScreenshot}
        alt="MLflow Tracing UI"
        css={{
          width: 200,
          height: 'auto',
          borderRadius: theme.borders.borderRadiusMd,
          flexShrink: 0,
        }}
      />
      <div css={{ flex: 1 }}>
        <div
          css={{
            fontSize: 14,
            fontWeight: theme.typography.typographyBoldFontWeight,
            color: theme.colors.textPrimary,
            marginBottom: theme.spacing.xs,
          }}
        >
          <FormattedMessage defaultMessage="New to MLflow?" description="Demo banner title" />
        </div>
        <div css={{ color: theme.colors.textSecondary }}>
          <FormattedMessage
            defaultMessage="Explore MLflow's core features with pre-populated sample data including traces, evaluations, and prompts."
            description="Demo banner description"
          />
        </div>
      </div>
      <Button
        componentId="mlflow.home.demo-banner.launch"
        type="primary"
        onClick={handleLaunchDemo}
        disabled={isLoading}
        endIcon={isLoading ? undefined : <ArrowRightIcon />}
        css={{ flexShrink: 0, marginRight: theme.spacing.md }}
      >
        {isLoading ? (
          <Spinner size="small" />
        ) : (
          <FormattedMessage defaultMessage="Start Demo" description="Demo banner launch button" />
        )}
      </Button>
    </div>
  );
};

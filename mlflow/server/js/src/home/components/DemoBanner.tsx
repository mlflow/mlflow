import { useState, useCallback } from 'react';
import { Button, Empty, Spinner, BeakerIcon, CloseIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useNavigate } from '../../common/utils/RoutingUtils';
import Utils from '../../common/utils/Utils';
import { fetchAPI, getAjaxUrl } from '../../common/utils/FetchUtils';

const DEMO_BANNER_DISMISSED_KEY = 'mlflow.demo.banner.dismissed';

export const DemoBanner = () => {
  const navigate = useNavigate();
  const { theme } = useDesignSystemTheme();
  const [isDismissed, setIsDismissed] = useState(() => localStorage.getItem(DEMO_BANNER_DISMISSED_KEY) === 'true');
  const [isLoading, setIsLoading] = useState(false);

  const handleLaunchDemo = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/demo/generate'), {
        method: 'POST',
      });
      navigate(data.navigation_url || '/experiments');
    } catch (error) {
      Utils.logErrorAndNotifyUser(error);
    } finally {
      setIsLoading(false);
    }
  }, [navigate]);

  const handleDismiss = useCallback(() => {
    localStorage.setItem(DEMO_BANNER_DISMISSED_KEY, 'true');
    setIsDismissed(true);
  }, []);

  if (isDismissed) {
    return null;
  }

  return (
    <div
      css={{
        position: 'relative',
        padding: theme.spacing.lg,
        background: theme.colors.backgroundSecondary,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.borderDecorative}`,
      }}
    >
      <button
        onClick={handleDismiss}
        aria-label="Dismiss"
        css={{
          position: 'absolute',
          top: theme.spacing.sm,
          right: theme.spacing.sm,
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: theme.spacing.xs,
          color: theme.colors.textSecondary,
          '&:hover': {
            color: theme.colors.textPrimary,
          },
        }}
      >
        <CloseIcon />
      </button>
      <Empty
        image={<BeakerIcon css={{ fontSize: 48, color: theme.colors.actionPrimaryBackgroundDefault }} />}
        title={<FormattedMessage defaultMessage="New to MLflow?" description="Demo banner title" />}
        description={
          <FormattedMessage
            defaultMessage="Explore GenAI features with pre-populated sample data including traces, evaluations, and prompts."
            description="Demo banner description"
          />
        }
        button={
          <Button
            componentId="mlflow.home.demo-banner.launch"
            type="primary"
            onClick={handleLaunchDemo}
            disabled={isLoading}
          >
            {isLoading ? (
              <Spinner size="small" />
            ) : (
              <FormattedMessage defaultMessage="Launch Demo" description="Demo banner launch button" />
            )}
          </Button>
        }
      />
    </div>
  );
};

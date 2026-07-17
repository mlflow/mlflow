import { Alert, Button, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

/**
 * Banner shown while the user is viewing a shared experiment view. It signals that the view is
 * read-only with respect to local storage — nothing the user does is saved — and hosts the
 * explicit actions to either override the user's own saved view with the current one, or discard
 * the shared view and restore the user's own.
 */
export const ExperimentViewSharedViewBanner = ({
  onOverride,
  onDiscard,
}: {
  onOverride: () => void;
  onDiscard: () => void;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Alert
      componentId="mlflow.experiment_page.shared_view.banner"
      type="info"
      closable={false}
      css={{ marginBottom: theme.spacing.sm }}
      message={
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: theme.spacing.md,
            flexWrap: 'wrap',
          }}
        >
          <span>
            <FormattedMessage
              defaultMessage="You're viewing a shared view. Your changes won't be saved unless you override your own saved view."
              description="Experiment page > shared view banner > explanation that the shared view is read-only"
            />
          </span>
          <div css={{ display: 'flex', gap: theme.spacing.sm, flexShrink: 0 }}>
            <Button
              componentId="mlflow.experiment_page.shared_view.override"
              type="primary"
              size="small"
              onClick={onOverride}
            >
              <FormattedMessage
                defaultMessage="Override saved view"
                description="Experiment page > shared view banner > button that overwrites the user's own saved view with the shared view"
              />
            </Button>
            <Button componentId="mlflow.experiment_page.shared_view.discard" size="small" onClick={onDiscard}>
              <FormattedMessage
                defaultMessage="Discard"
                description="Experiment page > shared view banner > button that discards the shared view and restores the user's own view"
              />
            </Button>
          </div>
        </div>
      }
    />
  );
};

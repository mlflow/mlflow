import { FormattedMessage } from 'react-intl';
import { SharedViewBanner } from '../saved-views/SharedViewBanner';

/**
 * Banner shown while the user is viewing a shared experiment (runs) view. Thin runs-specific wrapper
 * over the shared {@link SharedViewBanner}: it supplies the runs componentId namespace, message, and
 * the Override action (adopt the shared view into the user's own saved view). Discard restores the
 * user's own view.
 */
export const ExperimentViewSharedViewBanner = ({
  onOverride,
  onDiscard,
}: {
  onOverride: () => void;
  onDiscard: () => void;
}) => (
  <SharedViewBanner
    componentId="mlflow.experiment_page.shared_view"
    message={
      <FormattedMessage
        defaultMessage="You're viewing a shared view. Your changes won't be saved unless you override your own saved view."
        description="Experiment page > shared view banner > explanation that the shared view is read-only"
      />
    }
    overrideLabel={
      <FormattedMessage
        defaultMessage="Override saved view"
        description="Experiment page > shared view banner > button that overwrites the user's own saved view with the shared view"
      />
    }
    onOverride={onOverride}
    onDiscard={onDiscard}
  />
);

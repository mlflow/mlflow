import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { Button, LinkIcon } from '@databricks/design-system';

import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { ExperimentGetShareLinkModal } from './ExperimentGetShareLinkModal';

/**
 * "Share" button in the runs controls toolbar. Opens the "Save & share view" modal, which names the
 * current view (columns / filters / sort / charts), persists it as a named saved view, then hands
 * back a link. Sharing is always something you do to a *named* view — there is no anonymous
 * current-state link — so this is just a more discoverable, top-level entry point to the same flow
 * as the Views dropdown's "Save current view…" item.
 */
export const ExperimentViewShareButton = ({
  experimentId,
  searchFacetsState,
  uiState,
}: {
  experimentId: string;
  searchFacetsState?: ExperimentPageSearchFacetsState;
  uiState?: ExperimentPageUIState;
}) => {
  const [showModal, setShowModal] = useState(false);

  return (
    <>
      <Button
        componentId="mlflow.experiment_page.share_current_view"
        icon={<LinkIcon />}
        data-testid="experiment-share-button"
        onClick={() => setShowModal(true)}
      >
        <FormattedMessage
          defaultMessage="Share"
          description="Label for the button that opens the save-and-share-view modal in the experiment runs toolbar"
        />
      </Button>
      <ExperimentGetShareLinkModal
        experimentId={experimentId}
        searchFacetsState={searchFacetsState}
        uiState={uiState}
        visible={showModal}
        onCancel={() => setShowModal(false)}
      />
    </>
  );
};

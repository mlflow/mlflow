import { Button } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { GetLinkModal } from '../../../modals/GetLinkModal';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { ExperimentGetShareLinkModal } from './ExperimentGetShareLinkModal';

/**
 * Experiment page header part responsible for displaying button
 * that displays modal for sharing the link
 */
export const ExperimentViewHeaderShareButton = ({
  searchFacetsState,
  uiState,
  experimentIds,
}: {
  searchFacetsState?: ExperimentPageSearchFacetsState;
  uiState?: ExperimentPageUIState;
  experimentIds?: string[];
}) => {
  const [showGetLinkModal, setShowGetLinkModal] = useState(false);

  return (
    <>
      {searchFacetsState && uiState && experimentIds ? (
        <ExperimentGetShareLinkModal
          searchFacetsState={searchFacetsState}
          uiState={uiState}
          visible={showGetLinkModal}
          onCancel={() => setShowGetLinkModal(false)}
          experimentIds={experimentIds}
        />
      ) : (
        <GetLinkModal
          link={window.location.href}
          visible={showGetLinkModal}
          onCancel={() => setShowGetLinkModal(false)}
        />
      )}
      {/* TODO: ensure that E2E tests are working after refactor is complete */}
      <Button
        componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_header_experimentviewheadersharebutton.tsx_44"
        type="primary"
        onClick={() => setShowGetLinkModal(true)}
        data-testid="share-button"
      >
        <FormattedMessage defaultMessage="Share" description="Text for share button on experiment view page header" />
      </Button>
    </>
  );
};

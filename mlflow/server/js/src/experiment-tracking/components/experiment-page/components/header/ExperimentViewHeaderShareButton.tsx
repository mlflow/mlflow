import { Button } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { GetLinkModal } from '../../../modals/GetLinkModal';
import { shouldEnableShareExperimentViewByTags } from '../../../../../common/utils/FeatureUtils';
import { ExperimentPageSearchFacetsStateV2 } from '../../models/ExperimentPageSearchFacetsStateV2';
import { ExperimentPageUIStateV2 } from '../../models/ExperimentPageUIStateV2';
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
  searchFacetsState?: ExperimentPageSearchFacetsStateV2;
  uiState?: ExperimentPageUIStateV2;
  experimentIds?: string[];
}) => {
  const shareExperimentViewStateByTagsEnabled = shouldEnableShareExperimentViewByTags();
  const [showGetLinkModal, setShowGetLinkModal] = useState(false);

  return (
    <>
      {shareExperimentViewStateByTagsEnabled && searchFacetsState && uiState && experimentIds ? (
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
      <Button type="primary" onClick={() => setShowGetLinkModal(true)} data-test-id="share-button">
        <FormattedMessage defaultMessage="Share" description="Text for share button on experiment view page header" />
      </Button>
    </>
  );
};

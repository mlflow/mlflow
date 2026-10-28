import { Button } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { GetLinkModal } from '../../../modals/GetLinkModal';

/**
 * Share button for the compare-experiments header: copies the current URL (whose search facets
 * already round-trip through query params) via the generic Get Link modal. Single-experiment
 * sharing lives in the runs toolbar's Views dropdown / Share button instead (it saves a named
 * view), so this button only ever needs the plain URL-copy path.
 */
export const ExperimentViewHeaderShareButton = ({ type }: { type?: 'primary' | 'link' | 'tertiary' }) => {
  const [showGetLinkModal, setShowGetLinkModal] = useState(false);

  return (
    <>
      <GetLinkModal
        link={window.location.href}
        visible={showGetLinkModal}
        onCancel={() => setShowGetLinkModal(false)}
      />
      {/* TODO: ensure that E2E tests are working after refactor is complete */}
      <Button
        componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_header_experimentviewheadersharebutton.tsx_44"
        type={type}
        onClick={() => setShowGetLinkModal(true)}
        data-testid="share-button"
      >
        <FormattedMessage defaultMessage="Share" description="Text for share button on experiment view page header" />
      </Button>
    </>
  );
};

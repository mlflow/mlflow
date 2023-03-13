import { Button } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { GetLinkModal } from '../../../modals/GetLinkModal';

/**
 * Experiment page header part responsible for displaying button
 * that displays modal for sharing the link
 */
export const ExperimentViewHeaderShareButton = () => {
  const [showGetLinkModal, setShowGetLinkModal] = useState(false);

  return (
    <>
      <GetLinkModal
        link={window.location.href}
        visible={showGetLinkModal}
        onCancel={() => setShowGetLinkModal(false)}
      />
      {/* TODO: ensure that E2E tests are working after refactor is complete */}
      <Button type='primary' onClick={() => setShowGetLinkModal(true)} data-test-id='share-button'>
        <FormattedMessage
          defaultMessage='Share'
          description='Text for share button on experiment view page header'
        />
      </Button>
    </>
  );
};

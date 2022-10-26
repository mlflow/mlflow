import { Alert, Spacer } from '@databricks/design-system';
import React, { useCallback } from 'react';
import { FormattedMessage } from 'react-intl';
import { useExperimentViewLocalStore } from '../hooks/useExperimentViewLocalStore';
import { ExperimentTrackingDocUrl, onboarding as ONBOARDING } from '../../../../common/constants';

const showTrackingHelperStoreKey = 'showTrackingHelper';

const getLearnMoreLinkUrl = () => ExperimentTrackingDocUrl;

export const ExperimentViewOnboarding = React.memo(() => {
  const onboardingLocalStore = useExperimentViewLocalStore(ONBOARDING);

  const disableOnboardingHelper = useCallback(() => {
    onboardingLocalStore.setItem(showTrackingHelperStoreKey, 'false');
  }, [onboardingLocalStore]);

  const displayOnboarding = onboardingLocalStore.getItem(showTrackingHelperStoreKey) !== 'false';

  return displayOnboarding ? (
    <>
      <Alert
        message={
          <FormattedMessage
            defaultMessage='Track machine learning training runs in experiments. <link>Learn more</link>'
            description='Information banner text to provide more information about experiments runs page'
            values={{
              link: (chunks: any) => (
                <a href={getLearnMoreLinkUrl()} target='_blank' rel='noopener noreferrer'>
                  {chunks}
                </a>
              ),
            }}
          />
        }
        type='info'
        closable
        onClose={disableOnboardingHelper}
      />
      <Spacer size='medium' />
    </>
  ) : null;
});

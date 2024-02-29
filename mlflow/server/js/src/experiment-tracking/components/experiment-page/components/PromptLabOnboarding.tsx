import { Button, Modal } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { useCallback, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useExperimentViewLocalStore } from '../hooks/useExperimentViewLocalStore';

import OnboardingGifPromptLab from '../../../../common/static/onboarding-promptlab.gif';
const PROMPTLAB_ONBOARDING_STORAGE_KEY = 'promptlabOnboarding';

export const PromptLabOnboarding = ({ onDismissed }: { onDismissed?: () => void }) => {
  const { formatMessage } = useIntl();
  const visitedFlagStore = useExperimentViewLocalStore(PROMPTLAB_ONBOARDING_STORAGE_KEY);
  // TODO: replace this with the new link once it's ready
  // const learnMoreUrl = 'https://www.google.com/search?q=MLflow+Promptlab';

  const [modalVisible, setModalVisible] = useState(() => {
    // Hide the modal if it was visited already
    if (visitedFlagStore.getItem('visited')) {
      return false;
    }

    return true;
  });

  const closeModal = useCallback(() => {
    visitedFlagStore.setItem('visited', true);
    setModalVisible(false);
    onDismissed?.();
  }, [visitedFlagStore, onDismissed]);

  const promptlabDescription = formatMessage({
    defaultMessage: 'Introducing prompt engineering tools to help you build better LLMs faster.',
    description: 'Text in the modal for the prompt engineering onboarding, above the first GIF',
  });

  return (
    <Modal
      visible={modalVisible}
      onCancel={closeModal}
      onOk={closeModal}
      footer={
        <>
          {/* <Button href={learnMoreUrl} target='_blank'>
            <FormattedMessage
              defaultMessage='Learn more'
              description='"Learn more" button in the modal for the prompt engineering onboarding'
            />
          </Button> */}
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_promptlabonboarding.tsx_58"
            onClick={closeModal}
            type="primary"
          >
            <FormattedMessage
              defaultMessage="Try it now"
              description='"Try it now" button in the modal for the prompt engineering onboarding'
            />
          </Button>
        </>
      }
      size="wide"
      title={
        <FormattedMessage
          defaultMessage="🎉 Introducing Prompt Engineering in Experiment Tracking 🎉"
          description="Title of the modal for the prompt engineering onboarding"
        />
      }
    >
      <section css={styles.wrapper}>
        <div css={styles.label}>{promptlabDescription}</div>
        <div css={styles.gifContainer}>
          <img src={OnboardingGifPromptLab} alt={promptlabDescription} css={{ maxWidth: '100%', maxHeight: '100%' }} />
        </div>
      </section>
    </Modal>
  );
};

const styles = {
  label: (theme: Theme) => ({
    marginTop: theme.spacing.sm,
  }),
  wrapper: (theme: Theme) => ({
    display: 'flex',
    flexDirection: 'column' as const,
    gap: theme.spacing.sm,
  }),
  gifContainer: () => ({
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: 200,
  }),
};

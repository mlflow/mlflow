import { useCallback, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { ModelsNextUIPromoModal } from './ModelsNextUIPromoModal';
import { Modal, Switch } from '@databricks/design-system';
import { useNextModelsUIContext } from '../hooks/useNextModelsUI';

const promoModalSeenStorageKey = '_mlflow_promo_modal_dismissed';

export const ModelsNextUIToggleSwitch = () => {
  const { usingNextModelsUI, setUsingNextModelsUI } = useNextModelsUIContext();

  const promoModalVisited = window.localStorage.getItem(promoModalSeenStorageKey) === 'true';

  const [promoModalVisible, setPromoModalVisible] = useState(!promoModalVisited);
  const [confirmDisableModalVisible, setConfirmDisableModalVisible] = useState(false);

  const setPromoModalVisited = useCallback(() => {
    setPromoModalVisible(false);
    window.localStorage.setItem(promoModalSeenStorageKey, 'true');
  }, []);

  const intl = useIntl();
  const label = intl.formatMessage({
    defaultMessage: 'New model registry UI',
    description:
      'Model registry > Switcher for the new model registry UI containing aliases > label',
  });
  const switchNextUI = (newUsingNewUIValue: boolean) => {
    if (!newUsingNewUIValue) {
      setConfirmDisableModalVisible(true);
    } else {
      setUsingNextModelsUI(true);
    }
  };
  return (
    <>
      <Switch
        checked={usingNextModelsUI}
        aria-label={label}
        label={label}
        onChange={switchNextUI}
      />
      <ModelsNextUIPromoModal
        visible={promoModalVisible}
        onClose={() => {
          setPromoModalVisited();
        }}
        onTryItNow={() => {
          setUsingNextModelsUI(true);
          setPromoModalVisited();
        }}
      />
      <Modal
        visible={confirmDisableModalVisible}
        title={
          <FormattedMessage
            defaultMessage='Disable the new model stages'
            description='Model registry > Switcher for the new model registry UI containing aliases > disable confirmation modal title'
          />
        }
        okText='Disable'
        onCancel={() => {
          setConfirmDisableModalVisible(false);
        }}
        onOk={() => {
          setUsingNextModelsUI(false);
          setConfirmDisableModalVisible(false);
        }}
      >
        <FormattedMessage
          defaultMessage='
          In order to improve the product experience, we would love to get your feedback on the new model layout.'
          description='Model registry > Switcher for the new model registry UI containing aliases > disable confirmation modal content'
        />
      </Modal>
    </>
  );
};

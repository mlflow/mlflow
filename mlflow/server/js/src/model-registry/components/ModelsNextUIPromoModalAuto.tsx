import { useState } from 'react';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { ModelsNextUIPromoModal } from './ModelsNextUIPromoModal';
import { useNextModelsUIContext } from '../hooks/useNextModelsUI';

const promoModalSeenStorageKey = '_mlflow_model_registry_promo_modal_dismissed';

export const ModelsNextUIPromoModalAuto = () => {
  const { setUsingNextModelsUI } = useNextModelsUIContext();
  const [promoModalVisited, setPromoModalVisited] = useLocalStorage({
    key: promoModalSeenStorageKey,
    version: 1,
    initialValue: false,
  });
  const [promoModalVisible, setPromoModalVisible] = useState(!promoModalVisited);

  const dismiss = () => {
    setPromoModalVisible(false);
    setPromoModalVisited(true);
  };

  return (
    <ModelsNextUIPromoModal
      visible={promoModalVisible}
      onClose={dismiss}
      onTryItNow={() => {
        setUsingNextModelsUI(true);
        dismiss();
      }}
    />
  );
};

import { useCallback, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { ModelsNextUIPromoModal } from './ModelsNextUIPromoModal';
import { Modal, Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useNextModelsUIContext } from '../hooks/useNextModelsUI';

const promoModalSeenStorageKey = '_mlflow_model_registry_promo_modal_dismissed';

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
    description: 'Model registry > Switcher for the new model registry UI containing aliases > label',
  });
  const switchNextUI = (newUsingNewUIValue: boolean) => {
    if (!newUsingNewUIValue) {
      setConfirmDisableModalVisible(true);
    } else {
      setUsingNextModelsUI(true);
    }
  };
  const { theme } = useDesignSystemTheme();
  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <label>{label}</label>
        <Switch
          componentId="codegen_mlflow_app_src_model-registry_components_modelsnextuitoggleswitch.tsx_39"
          checked={usingNextModelsUI}
          aria-label={label}
          onChange={switchNextUI}
        />
      </div>
      <ModelsNextUIPromoModal
        visible={promoModalVisible}
        onClose={() => {
          setPromoModalVisited();
        }}
        onTryItNow={() => {
          setPromoModalVisited();
        }}
      />
      <Modal
        componentId="codegen_mlflow_app_src_model-registry_components_modelsnextuitoggleswitch.tsx_50"
        visible={confirmDisableModalVisible}
        title={
          <FormattedMessage
            defaultMessage="Disable the new model stages"
            description="Model registry > Switcher for the new model registry UI containing aliases > disable confirmation modal title"
          />
        }
        okText="Disable"
        onCancel={() => {
          setConfirmDisableModalVisible(false);
        }}
        onOk={() => {
          setUsingNextModelsUI(false);
          setConfirmDisableModalVisible(false);
        }}
      >
        <FormattedMessage
          defaultMessage="
          Thank you for exploring the new Model Registry UI. We are dedicated to providing the best experience, and your feedback is invaluable.
          Please share your thoughts with us <link>here</link>."
          description="Model registry > Switcher for the new model registry UI containing aliases > disable confirmation modal content"
          values={{
            link: (chunks) => (
              <Typography.Link
                componentId="codegen_mlflow_app_src_model-registry_components_modelsnextuitoggleswitch.tsx_74"
                href="https://forms.gle/aMB4qDrhMeEm2r359"
                openInNewTab
              >
                {chunks}
              </Typography.Link>
            ),
          }}
        />
      </Modal>
    </>
  );
};

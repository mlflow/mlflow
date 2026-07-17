import { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { Modal, Switch, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useNextModelsUIContext } from '../hooks/useNextModelsUI';

export const ModelsNextUIToggleSwitch = () => {
  const { usingNextModelsUI, setUsingNextModelsUI } = useNextModelsUIContext();

  const [confirmDisableModalVisible, setConfirmDisableModalVisible] = useState(false);

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

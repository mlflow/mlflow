import React, { useState } from 'react';
import type { ButtonProps } from '@databricks/design-system';
import { Button } from '@databricks/design-system';
import { CreateModelModal } from './CreateModelModal';
import { FormattedMessage } from 'react-intl';

type Props = {
  buttonType?: ButtonProps['type'];
  buttonText?: React.ReactNode;
};

export function CreateModelButton({
  buttonType = 'primary',
  buttonText = <FormattedMessage defaultMessage="Create Model" description="Create button to register a new model" />,
}: Props) {
  const [modalVisible, setModalVisible] = useState<boolean>(false);

  const hideModal = () => {
    setModalVisible(false);
  };

  const showModal = () => {
    setModalVisible(true);
  };

  return (
    <div css={styles.wrapper}>
      <Button
        componentId="codegen_mlflow_app_src_model-registry_components_CreateModelButton.tsx_28"
        className="create-model-btn"
        css={styles.getButtonSize(buttonType)}
        type={buttonType}
        onClick={showModal}
        data-testid="create-model-button"
      >
        {buttonText}
      </Button>
      <CreateModelModal modalVisible={modalVisible} hideModal={hideModal} />
    </div>
  );
}

const styles = {
  getButtonSize: (buttonType: string) =>
    buttonType === 'primary'
      ? {
          height: '40px',
          width: 'fit-content',
        }
      : { padding: '0px' },
  wrapper: { display: 'inline' },
};

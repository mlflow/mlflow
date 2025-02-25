import { Button, Modal, Typography } from '@databricks/design-system';
import { ReactComponent as PromoContentSvg } from '../../common/static/promo-modal-content.svg';
import { FormattedMessage } from 'react-intl';
import { modelStagesMigrationGuideLink } from '../../common/constants';

export const ModelsNextUIPromoModal = ({
  visible,
  onClose,
  onTryItNow,
}: {
  visible: boolean;
  onClose: () => void;
  onTryItNow: () => void;
}) => (
  <Modal
    componentId="codegen_mlflow_app_src_model-registry_components_modelsnextuipromomodal.tsx_15"
    visible={visible}
    title={
      <FormattedMessage
        defaultMessage="Flexible, governed deployments with the new Model Registry UI"
        description="Model registry > OSS Promo modal for model version aliases > modal title"
      />
    }
    onCancel={onClose}
    footer={
      <>
        <Button
          componentId="codegen_mlflow_app_src_model-registry_components_modelsnextuipromomodal.tsx_26"
          href={modelStagesMigrationGuideLink}
          rel="noopener"
          target="_blank"
        >
          <FormattedMessage
            defaultMessage="Learn more"
            description="Model registry > OSS Promo modal for model version aliases > learn more link"
          />
        </Button>
        <Button
          componentId="codegen_mlflow_app_src_model-registry_components_modelsnextuipromomodal.tsx_32"
          type="primary"
          onClick={onTryItNow}
        >
          <FormattedMessage
            defaultMessage="Try it now"
            description="Model registry > OSS Promo modal for model version aliases > try it now button label"
          />
        </Button>
      </>
    }
  >
    <PromoContentSvg width="100%" />
    <Typography.Text>
      <FormattedMessage
        defaultMessage={`With the latest Model Registry UI, you can use <b>Model Aliases</b> for flexible 
        references to specific model versions, streamlining deployment in a given environment. Use 
        <b>Model Tags</b> to annotate model versions with metadata, like the status of pre-deployment checks.`}
        description="Model registry > OSS Promo modal for model version aliases > description paragraph body"
        values={{
          b: (chunks: any) => <b>{chunks}</b>,
        }}
      />
    </Typography.Text>
  </Modal>
);

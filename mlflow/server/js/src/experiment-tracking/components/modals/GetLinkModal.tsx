import { FormattedMessage } from 'react-intl';
import { Modal } from '@databricks/design-system';
import { CopyBox } from '../../../shared/building_blocks/CopyBox';

type Props = {
  visible: boolean;
  onCancel: () => void;
  link: string;
};

export const GetLinkModal = ({ visible, onCancel, link }: Props) => {
  return (
    <Modal
      componentId="codegen_mlflow_app_src_experiment-tracking_components_modals_getlinkmodal.tsx_21"
      title={<FormattedMessage defaultMessage="Get Link" description="Title text for get-link modal" />}
      visible={visible}
      onCancel={onCancel}
    >
      <CopyBox copyText={link} />
    </Modal>
  );
};

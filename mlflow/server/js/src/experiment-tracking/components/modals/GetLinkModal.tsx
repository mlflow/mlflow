import React from 'react';
import { FormattedMessage } from 'react-intl';
import { Modal } from '@databricks/design-system';
import { CopyBox } from '../../../shared/building_blocks/CopyBox';

type Props = {
  visible: boolean;
  onCancel: (...args: any[]) => any;
  link: string;
};

export const GetLinkModal = ({ visible, onCancel, link }: Props) => {
  return (
    <Modal
      title={
        <FormattedMessage defaultMessage='Get Link' description={'Title text for get-link modal'} />
      }
      visible={visible}
      onCancel={onCancel}
    >
      <CopyBox copyText={link} />
    </Modal>
  );
};

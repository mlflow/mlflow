import React from 'react';
import PropTypes from 'prop-types';
import { FormattedMessage } from 'react-intl';
import { Modal } from '@databricks/design-system';
import { CopyBox } from '../../../shared/building_blocks/CopyBox';

export const GetLinkModal = ({ visible, onCancel, link }) => {
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

GetLinkModal.propTypes = {
  visible: PropTypes.bool.isRequired,
  onCancel: PropTypes.func.isRequired,
  link: PropTypes.string.isRequired,
};

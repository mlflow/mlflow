import React from 'react';
import PropTypes from 'prop-types';
import { Input } from '@databricks/design-system';
import { CopyButton } from './CopyButton';

export const CopyBox = ({ copyText }) => (
  <>
    <Input
      readOnly
      value={copyText}
      style={{ width: 'calc(100% - 75px)' }}
      data-test-id='copy-box'
    />{' '}
    <CopyButton copyText={copyText} />
  </>
);

CopyBox.propTypes = {
  copyText: PropTypes.string.isRequired,
};

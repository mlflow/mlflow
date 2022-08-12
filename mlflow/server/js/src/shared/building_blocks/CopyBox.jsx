import React from 'react';
import PropTypes from 'prop-types';
import { Input } from '@databricks/design-system';
import { CopyButton } from './CopyButton';

export const CopyBox = ({ copyText }) => (
  <div css={{ display: 'flex', gap: 4 }}>
    <Input readOnly value={copyText} data-test-id='copy-box' />
    <CopyButton copyText={copyText} />
  </div>
);

CopyBox.propTypes = {
  copyText: PropTypes.string.isRequired,
};

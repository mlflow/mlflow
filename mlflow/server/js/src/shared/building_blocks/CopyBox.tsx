import React from 'react';
import { Input } from '@databricks/design-system';
import { CopyButton } from './CopyButton';

type Props = {
  copyText: string;
};

export const CopyBox = ({ copyText }: Props) => (
  <div css={{ display: 'flex', gap: 4 }}>
    <Input readOnly value={copyText} data-test-id='copy-box' />
    <CopyButton copyText={copyText} />
  </div>
);

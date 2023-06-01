/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

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

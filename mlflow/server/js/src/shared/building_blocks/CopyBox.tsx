import { Input } from '@databricks/design-system';
import { CopyButton } from './CopyButton';

type Props = {
  copyText: string;
};

export const CopyBox = ({ copyText }: Props) => (
  <div css={{ display: 'flex', gap: 4 }}>
    <Input
      componentId="codegen_mlflow_app_src_shared_building_blocks_copybox.tsx_18"
      readOnly
      value={copyText}
      data-testid="copy-box"
    />
    <CopyButton copyText={copyText} />
  </div>
);

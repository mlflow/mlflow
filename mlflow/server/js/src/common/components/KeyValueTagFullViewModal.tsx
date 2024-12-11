import React from 'react';
import { Modal, Typography, CopyIcon, useDesignSystemTheme } from '@databricks/design-system';
const { Paragraph } = Typography;
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';

export interface KeyValueTagFullViewModalProps {
  tagKey: string;
  tagValue: string;
  setIsKeyValueTagFullViewModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
  isKeyValueTagFullViewModalVisible: boolean;
}

export const KeyValueTagFullViewModal = React.memo((props: KeyValueTagFullViewModalProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Modal
      componentId="codegen_mlflow_app_src_common_components_keyvaluetagfullviewmodal.tsx_17"
      title={'Tag: ' + props.tagKey}
      visible={props.isKeyValueTagFullViewModalVisible}
      onCancel={() => props.setIsKeyValueTagFullViewModalVisible(false)}
    >
      <div css={{ display: 'flex' }}>
        <Paragraph css={{ flexGrow: 1 }}>
          <pre
            css={{
              backgroundColor: theme.colors.backgroundPrimary,
              marginTop: theme.spacing.sm,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
            }}
          >
            {props.tagValue}
          </pre>
        </Paragraph>
        <div
          css={{
            marginTop: theme.spacing.sm,
          }}
        >
          <CopyButton copyText={props.tagValue} showLabel={false} icon={<CopyIcon />} aria-label="Copy" />
        </div>
      </div>
    </Modal>
  );
});

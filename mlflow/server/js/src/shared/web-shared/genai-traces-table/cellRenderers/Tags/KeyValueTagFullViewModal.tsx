import React from 'react';

import { Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { CopyActionButton } from '@databricks/web-shared/copy';
const { Paragraph } = Typography;

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
      <div css={{ display: 'flex', alignItems: 'center' }}>
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
            marginBottom: theme.spacing.md,
          }}
        >
          <CopyActionButton
            componentId="mlflow.genai-traces-table.tag_view_modal.tag_value_copy_button"
            copyText={props.tagValue}
          />
        </div>
      </div>
    </Modal>
  );
});

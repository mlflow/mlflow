/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

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
      componentId="codegen_mlflow_app_src_experiment-tracking_components_modals_getlinkmodal.tsx_21"
      title={<FormattedMessage defaultMessage="Get Link" description="Title text for get-link modal" />}
      visible={visible}
      onCancel={onCancel}
    >
      <CopyBox copyText={link} />
    </Modal>
  );
};

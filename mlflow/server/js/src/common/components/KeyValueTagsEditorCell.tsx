import { Button, PencilIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { KeyValueEntity } from '../types';
import { KeyValueTag } from './KeyValueTag';

interface KeyValueTagsEditorCellProps {
  tags?: KeyValueEntity[];
  onAddEdit: () => void;
}

/**
 * A cell renderer used in tables, displaying a list of key-value tags with button for editing those
 */
export const KeyValueTagsEditorCell = ({ tags = [], onAddEdit }: KeyValueTagsEditorCellProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexWrap: 'wrap',
        '> *': {
          marginRight: '0 !important',
        },
        gap: theme.spacing.xs,
      }}
    >
      {tags.length < 1 ? (
        <Button
          componentId="codegen_mlflow_app_src_common_components_keyvaluetagseditorcell.tsx_29"
          size="small"
          type="link"
          onClick={onAddEdit}
        >
          <FormattedMessage defaultMessage="Add" description="Key-value tag table cell > 'add' button label" />
        </Button>
      ) : (
        <>
          {tags.map((tag) => (
            <KeyValueTag tag={tag} key={`${tag.key}-${tag.value}`} />
          ))}
          <Button
            componentId="codegen_mlflow_app_src_common_components_keyvaluetagseditorcell.tsx_37"
            size="small"
            icon={<PencilIcon />}
            onClick={onAddEdit}
          />
        </>
      )}
    </div>
  );
};

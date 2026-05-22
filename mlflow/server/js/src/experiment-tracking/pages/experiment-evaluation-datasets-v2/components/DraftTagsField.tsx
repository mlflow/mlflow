import { useCallback, useMemo, useState } from 'react';
import { Button, PlusIcon, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { KeyValueTagFullViewModal } from '../../experiment-evaluation-datasets/components/KeyValueTagFullViewModal';

interface DraftTagsFieldProps {
  tags: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
}

interface TagModalState {
  open: boolean;
  /** Empty key when adding a new tag. */
  key: string;
  value: string;
}

/**
 * Draft-tag editor for the create flow. Parallels `TagsCell`'s pill + add-button + modal
 * layout but operates purely on local state: every edit calls `onChange` with a new map,
 * and the create-side `useRecordCreateState` submits the whole map alongside inputs and
 * expectations when the user clicks "Add record". No upsert mutation, no concurrency queue —
 * there's nothing to race against until a record actually exists.
 */
export const DraftTagsField = ({ tags, onChange }: DraftTagsFieldProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [modalState, setModalState] = useState<TagModalState>({ open: false, key: '', value: '' });
  const entries = useMemo(() => Object.entries(tags), [tags]);

  const handleSaveTag = useCallback(
    async (key: string, value: string): Promise<void> => {
      const next = { ...tags };
      // A rename drops the old key. Matches TagsCell.
      if (modalState.key && modalState.key !== key) {
        delete next[modalState.key];
      }
      next[key] = value;
      onChange(next);
    },
    [tags, modalState.key, onChange],
  );

  const handleDeleteTag = useCallback(
    async (key: string): Promise<void> => {
      const next = { ...tags };
      delete next[key];
      onChange(next);
    },
    [tags, onChange],
  );

  const handlePillClose = useCallback(
    (key: string) => {
      const next = { ...tags };
      delete next[key];
      onChange(next);
    },
    [tags, onChange],
  );

  return (
    <div
      css={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        gap: theme.spacing.xs,
      }}
    >
      {entries.map(([key, value]) => (
        <Tooltip
          key={key}
          componentId="mlflow.eval-datasets-v2.records.tag.draft.pill-tooltip"
          content={`${key}: ${value}`}
        >
          <Tag
            componentId="mlflow.eval-datasets-v2.records.tag.draft.pill"
            color="default"
            closable
            onClose={() => handlePillClose(key)}
            onClick={() => setModalState({ open: true, key, value })}
            css={{ cursor: 'pointer', maxWidth: 220 }}
          >
            <Typography.Text ellipsis css={{ maxWidth: 180 }}>
              {key}: {value}
            </Typography.Text>
          </Tag>
        </Tooltip>
      ))}
      <Button
        componentId="mlflow.eval-datasets-v2.records.tag.draft.add"
        size="small"
        type="tertiary"
        icon={<PlusIcon />}
        onClick={() => setModalState({ open: true, key: '', value: '' })}
        aria-label={intl.formatMessage({
          defaultMessage: 'Add tag',
          description: 'Aria label for the add-tag button in the V2 dataset record side panel (create mode)',
        })}
      >
        {entries.length === 0 ? (
          <FormattedMessage
            defaultMessage="Add tag"
            description="Button text for adding a tag in the V2 dataset record side panel (create mode) when no tags exist"
          />
        ) : null}
      </Button>
      <KeyValueTagFullViewModal
        tagKey={modalState.key}
        tagValue={modalState.value}
        isKeyValueTagFullViewModalVisible={modalState.open}
        setIsKeyValueTagFullViewModalVisible={(visible) => setModalState((prev) => ({ ...prev, open: visible }))}
        onSave={handleSaveTag}
        onDelete={modalState.key ? handleDeleteTag : undefined}
      />
    </div>
  );
};

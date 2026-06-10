import { useMemo, useState } from 'react';

import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxHintRow,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSearch,
  DialogComboboxTrigger,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { LabelSchema } from '../../components/label-schemas';

/**
 * Multi-select, searchable checklist of the experiment's questions (label
 * schemas), shared by the create-queue and manage-queue modals. Owns the search
 * box state; the caller owns which questions are checked. The trigger shows the
 * caller-provided summary (a count), not the joined names — checked state comes
 * from each item's `checked` prop, not from `value`.
 */
export const QuestionChecklistCombobox = ({
  componentId,
  schemas,
  checkedIds,
  onToggle,
  triggerValue,
  disabled,
  dropdownZIndex,
}: {
  componentId: string;
  schemas: LabelSchema[];
  checkedIds: Set<string>;
  onToggle: (schemaId: string) => void;
  /** Summary string(s) shown in the closed trigger (e.g. "3 questions selected"). */
  triggerValue: string[];
  disabled?: boolean;
  dropdownZIndex?: number;
}) => {
  const intl = useIntl();
  const [search, setSearch] = useState('');
  const visibleSchemas = useMemo(() => {
    const query = search.trim().toLowerCase();
    return query ? schemas.filter((s) => s.name.toLowerCase().includes(query)) : schemas;
  }, [schemas, search]);

  return (
    <DialogCombobox
      componentId={componentId}
      label={intl.formatMessage({
        defaultMessage: 'Questions',
        description: 'Review queue: questions dropdown label',
      })}
      multiSelect
      value={triggerValue}
    >
      <DialogComboboxTrigger
        allowClear={false}
        disabled={disabled}
        placeholder={intl.formatMessage({
          defaultMessage: 'Select questions',
          description: 'Review queue: questions dropdown placeholder',
        })}
      />
      <DialogComboboxContent maxHeight={240} matchTriggerWidth style={{ zIndex: dropdownZIndex }}>
        <DialogComboboxOptionList>
          <DialogComboboxOptionListSearch controlledValue={search} setControlledValue={setSearch}>
            {visibleSchemas.length === 0 ? (
              <DialogComboboxHintRow>
                <FormattedMessage
                  defaultMessage="No matching questions"
                  description="Review queue: no questions match the search"
                />
              </DialogComboboxHintRow>
            ) : (
              visibleSchemas.map((schema) => (
                <DialogComboboxOptionListCheckboxItem
                  key={schema.schema_id}
                  value={schema.name}
                  checked={checkedIds.has(schema.schema_id)}
                  onChange={() => onToggle(schema.schema_id)}
                >
                  {schema.name}
                </DialogComboboxOptionListCheckboxItem>
              ))
            )}
          </DialogComboboxOptionListSearch>
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};

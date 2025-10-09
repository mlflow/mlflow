import type { Dispatch, SetStateAction } from 'react';
import { useCallback, useMemo, useState } from 'react';

import {
  FormUI,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxRoot,
  useComboboxState,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import type { AssessmentSchema } from '../contexts/AssessmentSchemaContext';
import { useAssessmentSchemas } from '../contexts/AssessmentSchemaContext';

const getDefaultSchema = (name: string): AssessmentSchema => ({
  name,
  assessmentType: 'feedback',
  dataType: 'boolean',
});

export const AssessmentCreateNameTypeahead = ({
  name,
  setName,
  nameError,
  setNameError,
  handleChangeSchema,
}: {
  name: string;
  setName: Dispatch<SetStateAction<string>>;
  nameError: React.ReactNode | null;
  setNameError: Dispatch<SetStateAction<React.ReactNode | null>>;
  handleChangeSchema: (schema: AssessmentSchema | null) => void;
}) => {
  const { schemas } = useAssessmentSchemas();
  const intl = useIntl();
  const schemaNames = schemas.map((schema) => schema.name ?? '');

  const [selectedItem, setSelectedItem] = useState<AssessmentSchema | null>(null);
  const [itemsTest, setItemsTest] = useState<(AssessmentSchema | null)[]>(schemas);

  const items = useMemo(() => {
    const filteredItems = [...itemsTest];

    // hack to allow creating a new assessment name even if it's not in
    // the schemas. basically creates a fake schema with the name of the
    // input value so it always shows up in the typeahead
    if (name && !schemaNames.includes(name)) {
      const newSchema = getDefaultSchema(name);
      filteredItems.unshift(newSchema);
    }

    return filteredItems;
  }, [name, itemsTest, schemaNames]);

  const formOnChange = useCallback(
    (newSelectedItem: AssessmentSchema | null) => {
      setSelectedItem(newSelectedItem);
      handleChangeSchema(newSelectedItem);
      setNameError(null);
    },
    [handleChangeSchema, setNameError],
  );

  const comboboxState = useComboboxState<AssessmentSchema | null>({
    componentId: 'shared.model-trace-explorer.assessment-name-typeahead',
    allItems: schemas,
    items,
    setItems: setItemsTest,
    multiSelect: false,
    setInputValue: (value) => {
      setName(value);
      setNameError(null);
    },
    itemToString: (item) => item?.name ?? '',
    matcher: (item, query) => item?.name?.toLowerCase().includes(query.toLowerCase()) ?? false,
    formValue: selectedItem,
    formOnChange,
    preventUnsetOnBlur: true,
  });

  return (
    <TypeaheadComboboxRoot
      onKeyDown={(e) => {
        // disable left and right to prevent the previous/next
        // trace interaction while typing an assessment name,
        // but still allow up and down for tabbing through
        // typeahead options
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
          e.stopPropagation();
        }
      }}
      id="shared.model-trace-explorer.assessment-name-typeahead"
      comboboxState={comboboxState}
    >
      <TypeaheadComboboxInput
        data-testid="assessment-name-typeahead-input"
        placeholder={intl.formatMessage({
          defaultMessage: 'Enter an assessment name',
          description: 'Placeholder for the assessment name typeahead',
        })}
        validationState={nameError ? 'error' : undefined}
        comboboxState={comboboxState}
        formOnChange={formOnChange}
        onPressEnter={() => {
          if (items.length > 0) {
            formOnChange(items[0]);
          }
        }}
        allowClear
        showComboboxToggleButton
      />
      {nameError && <FormUI.Message type="error" message={nameError} />}
      <TypeaheadComboboxMenu comboboxState={comboboxState}>
        {items.map((item, index) => (
          <TypeaheadComboboxMenuItem
            data-testid={`assessment-name-typeahead-item-${item?.name ?? ''}`}
            key={`assessment-name-typeahead-${item?.name ?? ''}`}
            item={item}
            index={index}
            comboboxState={comboboxState}
          >
            {item?.name ?? ''}
          </TypeaheadComboboxMenuItem>
        ))}
      </TypeaheadComboboxMenu>
    </TypeaheadComboboxRoot>
  );
};

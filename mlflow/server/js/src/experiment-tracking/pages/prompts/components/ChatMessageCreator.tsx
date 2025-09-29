import {
  Button,
  PlusIcon,
  TrashIcon,
  RHFControlledComponents,
  useDesignSystemTheme,
  FormUI,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxRoot,
  useComboboxState,
} from '@databricks/design-system';
import { Fragment } from 'react';
import { Controller, useFieldArray, useFormContext } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ChatPromptMessage } from '../types';

const SUGGESTED_ROLES = ['system', 'user', 'assistant'];

const ChatRoleTypeaheadField = ({
  id,
  value,
  onChange,
  onBlur,
  placeholder,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
  onBlur: () => void;
  placeholder: string;
}) => {
  const comboboxState = useComboboxState<string>({
    componentId: id,
    items: SUGGESTED_ROLES,
    allItems: SUGGESTED_ROLES,
    setItems: () => {},
    setInputValue: (val) => {
      if (typeof val === 'string') {
        onChange(val);
      }
    },
    multiSelect: false,
    allowNewValue: true,
    preventUnsetOnBlur: true,
    itemToString: (item) => item ?? '',
    matcher: (item, query) => item?.toLowerCase().includes(query.toLowerCase()) ?? false,
    formValue: value ?? '',
    initialInputValue: value ?? '',
    formOnChange: (item) => {
      if (typeof item === 'string') {
        onChange(item);
      } else if (!item) {
        onChange('');
      }
    },
  });

  return (
    <TypeaheadComboboxRoot id={id} comboboxState={comboboxState}>
      <TypeaheadComboboxInput
        placeholder={placeholder}
        comboboxState={comboboxState}
        formOnChange={(item) => {
          if (typeof item === 'string') {
            onChange(item);
          } else if (!item) {
            onChange('');
          }
        }}
        onBlur={onBlur}
        allowClear
        showComboboxToggleButton
      />
      <TypeaheadComboboxMenu comboboxState={comboboxState} matchTriggerWidth>
        {SUGGESTED_ROLES.map((role, roleIndex) => (
          <TypeaheadComboboxMenuItem key={role} item={role} index={roleIndex} comboboxState={comboboxState}>
            {role}
          </TypeaheadComboboxMenuItem>
        ))}
      </TypeaheadComboboxMenu>
    </TypeaheadComboboxRoot>
  );
};

/**
 * Provides a small UI for composing chat-style prompts as a list of role/content pairs.
 */
export const ChatMessageCreator = ({ name }: { name: string }) => {
  const { control, formState } = useFormContext<{ [key: string]: ChatPromptMessage[] }>();
  const { fields, insert, remove, replace } = useFieldArray({ control, name });
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

  const addMessageAfter = (index: number) => insert(index + 1, { role: 'user', content: '' });

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {fields.map((field, index) => (
        <div
          key={field.id}
          css={{ display: 'grid', gridTemplateColumns: '120px 1fr auto auto', gap: theme.spacing.sm }}
        >
          <Fragment>
            <Controller
              control={control}
              name={`${name}.${index}.role`}
              render={({ field: { value, onChange, onBlur } }) => (
                <ChatRoleTypeaheadField
                  id={`mlflow.prompts.chat_creator.role_${index}`}
                  placeholder={formatMessage({
                    defaultMessage: 'role',
                    description: 'Placeholder for chat message role input',
                  })}
                  value={value ?? ''}
                  onChange={onChange}
                  onBlur={onBlur}
                />
              )}
            />
          </Fragment>
          <RHFControlledComponents.TextArea
            componentId={`mlflow.prompts.chat_creator.content_${index}`}
            name={`${name}.${index}.content`}
            control={control}
            autoSize={{ minRows: 1, maxRows: 6 }}
            css={{ width: '100%' }}
          />
          <Button
            componentId={`mlflow.prompts.chat_creator.add_after_${index}`}
            type="tertiary"
            icon={<PlusIcon />}
            aria-label={formatMessage({
              defaultMessage: 'Add message',
              description: 'Button to insert a new chat message row',
            })}
            onClick={() => addMessageAfter(index)}
          />
          <Button
            componentId={`mlflow.prompts.chat_creator.remove_${index}`}
            type="tertiary"
            icon={<TrashIcon />}
            aria-label={formatMessage({
              defaultMessage: 'Remove message',
              description: 'Button to remove a chat message row',
            })}
            onClick={() => remove(index)}
            disabled={fields.length === 1}
          />
        </div>
      ))}
      {formState.errors?.[name] && <FormUI.Message type="error" message={(formState.errors as any)[name]?.message} />}
    </div>
  );
};

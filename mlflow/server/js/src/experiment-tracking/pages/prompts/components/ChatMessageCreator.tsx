import {
  Button,
  PlusIcon,
  TrashIcon,
  RHFControlledComponents,
  useDesignSystemTheme,
  FormUI,
} from '@databricks/design-system';
import { Fragment } from 'react';
import { useFieldArray, useFormContext } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ChatPromptMessage } from '../types';

const SUGGESTED_ROLES = ['system', 'user', 'assistant'];

/**
 * Provides a small UI for composing chat-style prompts as a list of role/content pairs.
 */
export const ChatMessageCreator = ({ name }: { name: string }) => {
  const { control, formState } = useFormContext<{ [key: string]: ChatPromptMessage[] }>();
  const { fields, insert, remove, replace } = useFieldArray({ control, name });
  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

  const addMessageAfter = (index: number) => insert(index + 1, { role: 'user', content: '' });

  const clearAll = () => replace([{ role: 'user', content: '' }]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {fields.map((field, index) => (
        <div
          key={field.id}
          css={{ display: 'grid', gridTemplateColumns: '120px 1fr auto auto', gap: theme.spacing.sm }}
        >
          <Fragment>
            <RHFControlledComponents.Input
              componentId={`mlflow.prompts.chat_creator.role_${index}`}
              name={`${name}.${index}.role`}
              control={control}
              list={`chat-role-suggestions-${index}`}
              placeholder={formatMessage({
                defaultMessage: 'role',
                description: 'Placeholder for chat message role input',
              })}
              css={{ width: '100%' }}
            />
            <datalist id={`chat-role-suggestions-${index}`}>
              {SUGGESTED_ROLES.map((role) => (
                <option key={role} value={role} />
              ))}
            </datalist>
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
      <div css={{ display: 'flex', gap: theme.spacing.sm }}>
        <Button componentId="mlflow.prompts.chat_creator.clear_all" onClick={clearAll}>
          <FormattedMessage
            defaultMessage="Clear all"
            description="Button to clear all chat messages in the chat prompt creator"
          />
        </Button>
      </div>
      {formState.errors?.[name] && <FormUI.Message type="error" message={(formState.errors as any)[name]?.message} />}
    </div>
  );
};

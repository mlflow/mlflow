import { useState } from 'react';
import {
  Button,
  FormUI,
  Input,
  PlusIcon,
  SimpleSelect,
  SimpleSelectOption,
  TrashIcon,
  Switch,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';
import { jsonSchemaToProperties, nextSchemaPropertyId, propertiesToJsonSchema } from '../utils';
import { SCHEMA_PROPERTY_TYPES } from '../types';
import type { EditableSchemaProperty, SchemaProperty, SchemaPropertyType } from '../types';

interface Props {
  schemaText: string;
  onSchemaChange: (next: string) => void;
}

/**
 * Visual (field-by-field) editor for the structured output JSON schema, shown as an
 * alternative to the raw JSON textarea in the prompt creation modal.
 * Edits are merged onto the schema it was opened with, so anything it does not model survives.
 */
export const ResponseFormatSchemaBuilder = ({ schemaText, onSchemaChange }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  // Parsed once on mount; `schemaText` is only re-read when switching back into this mode
  const [properties, setProperties] = useState<SchemaProperty[]>(() => jsonSchemaToProperties(schemaText));
  const [baseSchema] = useState<unknown>(() => parseJSONSafe(schemaText));

  const emit = (next: SchemaProperty[]) => {
    setProperties(next);
    // No rows left means no structured output, rather than an empty object schema
    onSchemaChange(next.length === 0 ? '' : propertiesToJsonSchema(next, baseSchema));
  };

  const updateProperty = (id: string, patch: Partial<Omit<EditableSchemaProperty, 'id' | 'kind'>>) =>
    emit(properties.map((p) => (p.id === id ? { ...p, ...patch } : p)));

  const removeProperty = (id: string) => emit(properties.filter((p) => p.id !== id));

  const addProperty = () =>
    emit([
      ...properties,
      {
        id: nextSchemaPropertyId(),
        kind: 'editable',
        name: '',
        type: 'string',
        isArray: false,
        required: false,
        enumValues: [],
      },
    ]);

  const updateEnumValue = (prop: EditableSchemaProperty, index: number, value: string) =>
    updateProperty(prop.id, {
      enumValues: prop.enumValues.map((v, i) => (i === index ? value : v)),
    });

  const removeEnumValue = (prop: EditableSchemaProperty, index: number) =>
    updateProperty(prop.id, { enumValues: prop.enumValues.filter((_, i) => i !== index) });

  const addEnumValue = (prop: EditableSchemaProperty) =>
    updateProperty(prop.id, { enumValues: [...prop.enumValues, ''] });

  const duplicateNames = new Set(
    properties.map((p) => p.name.trim()).filter((name, index, names) => name && names.indexOf(name) !== index),
  );

  const nameValidation = (prop: SchemaProperty) => {
    const name = prop.name.trim();
    if (duplicateNames.has(name)) {
      return {
        validationState: 'error' as const,
        message: intl.formatMessage({
          defaultMessage: 'Duplicate property name. Only the last field with this name is saved.',
          description: 'Validation error for a repeated property name in the visual JSON schema editor',
        }),
      };
    }
    if (!name) {
      return {
        validationState: 'warning' as const,
        message: intl.formatMessage({
          defaultMessage: 'Properties without a name are not saved.',
          description: 'Validation warning for an unnamed property in the visual JSON schema editor',
        }),
      };
    }
    return undefined;
  };

  const hasReadOnlyRows = properties.some((prop) => prop.kind === 'readOnly');

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {hasReadOnlyRows && (
        <FormUI.Hint>
          <FormattedMessage
            defaultMessage="Fields this editor cannot represent (nested objects, $ref, oneOf/anyOf/allOf) are shown read-only and left unchanged. Switch to the code editor to edit them."
            description="Hint explaining read-only rows in the visual JSON schema editor"
          />
        </FormUI.Hint>
      )}
      {properties.map((prop) => {
        const isReadOnly = prop.kind === 'readOnly';
        const validation = isReadOnly ? undefined : nameValidation(prop);
        return (
          <div key={prop.id} css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
              <Input
                componentId="mlflow.prompts.create.response_format.property_name"
                css={{ flex: 2 }}
                value={prop.name}
                disabled={isReadOnly}
                validationState={validation?.validationState}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Property',
                  description: 'Schema property name input',
                })}
                onChange={(e) => updateProperty(prop.id, { name: e.target.value })}
              />
              <div css={{ flex: 1 }}>
                {isReadOnly ? (
                  <Input
                    componentId="mlflow.prompts.create.response_format.property_type_readonly"
                    value={prop.label}
                    disabled
                    aria-label={intl.formatMessage({
                      defaultMessage: 'Property type (read-only)',
                      description:
                        'Label for the read-only type of a property the visual JSON schema editor cannot edit',
                    })}
                  />
                ) : (
                  <SimpleSelect
                    id={`mlflow.prompts.create.response_format.property_type.${prop.id}`}
                    componentId="mlflow.prompts.create.response_format.property_type"
                    value={prop.type}
                    onChange={(e) => updateProperty(prop.id, { type: e.target.value as SchemaPropertyType })}
                  >
                    {SCHEMA_PROPERTY_TYPES.map((t) => (
                      <SimpleSelectOption key={t} value={t}>
                        {t}
                      </SimpleSelectOption>
                    ))}
                  </SimpleSelect>
                )}
              </div>
              {/* Tooltip wraps a <span> since Switch isn't guaranteed to forward hover/focus handlers */}
              <Tooltip
                componentId="mlflow.prompts.create.response_format.property_is_array_tooltip"
                content={intl.formatMessage({
                  defaultMessage: 'Allow multiple values',
                  description: 'Tooltip for the array toggle in the visual JSON schema editor',
                })}
              >
                <span>
                  <Switch
                    componentId="mlflow.prompts.create.response_format.property_is_array"
                    checked={!isReadOnly && prop.isArray}
                    disabled={isReadOnly}
                    onChange={(checked) => updateProperty(prop.id, { isArray: checked })}
                    label="[ ]"
                  />
                </span>
              </Tooltip>
              {/* `required` lives on the schema root, so it stays editable on read-only rows */}
              <Tooltip
                componentId="mlflow.prompts.create.response_format.property_required_tooltip"
                content={intl.formatMessage({
                  defaultMessage: 'Required field',
                  description: 'Tooltip for the required toggle in the visual JSON schema editor',
                })}
              >
                <span>
                  <Switch
                    componentId="mlflow.prompts.create.response_format.property_required"
                    checked={prop.required}
                    onChange={(checked) => updateProperty(prop.id, { required: checked })}
                    label="*"
                  />
                </span>
              </Tooltip>
              <Button
                componentId="mlflow.prompts.create.response_format.remove_property"
                icon={<TrashIcon />}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Remove property',
                  description: 'Button to remove a property row in the visual JSON schema editor',
                })}
                onClick={() => removeProperty(prop.id)}
              />
            </div>
            {validation && <FormUI.Message type={validation.validationState} message={validation.message} />}
            {/* Enum values are edited as individual rows (not a comma-separated field) so that
                values containing commas can be entered */}
            {!isReadOnly && prop.type === 'enum' && (
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                  gap: theme.spacing.xs,
                  marginLeft: theme.spacing.lg,
                  maxWidth: theme.spacing.lg * 20,
                  width: '100%',
                }}
              >
                {prop.enumValues.map((value, index) => (
                  // Keyed by index: enum values have no identity of their own, and lists are short enough
                  // that the rare focus jump on mid-list delete beats threading ids through the conversion.
                  <div key={index} css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
                    <Input
                      componentId="mlflow.prompts.create.response_format.property_enum_value"
                      value={value}
                      placeholder={intl.formatMessage({
                        defaultMessage: 'Allowed value',
                        description: 'Placeholder for a single enum value input in the visual JSON schema editor',
                      })}
                      onChange={(e) => updateEnumValue(prop, index, e.target.value)}
                    />
                    <Button
                      componentId="mlflow.prompts.create.response_format.remove_enum_value"
                      icon={<TrashIcon />}
                      aria-label={intl.formatMessage({
                        defaultMessage: 'Remove enum value',
                        description: 'Button to remove a single enum value in the visual JSON schema editor',
                      })}
                      onClick={() => removeEnumValue(prop, index)}
                    />
                  </div>
                ))}
                <Button
                  componentId="mlflow.prompts.create.response_format.add_enum_value"
                  icon={<PlusIcon />}
                  onClick={() => addEnumValue(prop)}
                >
                  <FormattedMessage
                    defaultMessage="Add enum value"
                    description="Button to add a new enum value in the visual JSON schema editor"
                  />
                </Button>
              </div>
            )}
          </div>
        );
      })}
      <Button
        componentId="mlflow.prompts.create.response_format.add_property"
        icon={<PlusIcon />}
        onClick={addProperty}
      >
        <FormattedMessage
          defaultMessage="Add property"
          description="Button to add a schema property in the visual JSON schema editor"
        />
      </Button>
    </div>
  );
};

import { FormUI, Input, useDesignSystemTheme, Typography } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import type { ChangeEvent } from 'react';
import { MaskedApiKeyInput } from './MaskedApiKeyInput';
import type { AuthConfigField } from './routeConstants';

export interface AuthConfigFieldsProps {
  fields: AuthConfigField[];
  values: Record<string, string>;
  errors?: Record<string, string>;
  onChange: (name: string, value: string) => void;
  componentIdPrefix: string;
}

export const AuthConfigFields = ({ fields, values, errors, onChange, componentIdPrefix }: AuthConfigFieldsProps) => {
  const { theme } = useDesignSystemTheme();

  if (!fields || fields.length === 0) {
    return null;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Typography.Text css={{ fontWeight: 600, color: theme.colors.textPrimary }}>
        <FormattedMessage defaultMessage="Additional Configuration" description="Auth config section title" />
      </Typography.Text>
      {fields.map((field) => (
        <div key={field.name}>
          <FormUI.Label htmlFor={`${componentIdPrefix}-${field.name}`}>
            {field.label}
            {field.required && <span css={{ color: theme.colors.textValidationDanger }}> *</span>}
          </FormUI.Label>
          {field.multiline ? (
            <Input.TextArea
              componentId={`${componentIdPrefix}.${field.name}`}
              id={`${componentIdPrefix}-${field.name}`}
              placeholder={field.placeholder}
              value={values[field.name] || ''}
              onChange={(e: ChangeEvent<HTMLTextAreaElement>) => {
                onChange(field.name, e.target.value);
              }}
              validationState={errors?.[field.name] ? 'error' : undefined}
              autoSize={{ minRows: 4, maxRows: 8 }}
            />
          ) : field.sensitive ? (
            <MaskedApiKeyInput
              componentId={`${componentIdPrefix}.${field.name}`}
              id={`${componentIdPrefix}-${field.name}`}
              placeholder={field.placeholder}
              value={values[field.name] || ''}
              onChange={(value) => {
                onChange(field.name, value);
              }}
            />
          ) : (
            <Input
              componentId={`${componentIdPrefix}.${field.name}`}
              id={`${componentIdPrefix}-${field.name}`}
              placeholder={field.placeholder}
              value={values[field.name] || ''}
              onChange={(e) => {
                onChange(field.name, e.target.value);
              }}
              validationState={errors?.[field.name] ? 'error' : undefined}
            />
          )}
          {errors?.[field.name] && <FormUI.Message type="error" message={errors[field.name]} />}
          {field.helpText && <FormUI.Hint css={{ marginTop: theme.spacing.sm }}>{field.helpText}</FormUI.Hint>}
        </div>
      ))}
    </div>
  );
};

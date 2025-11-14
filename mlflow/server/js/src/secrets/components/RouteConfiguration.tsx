import { FormUI, Input, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import {
  TagAssignmentRoot,
  TagAssignmentRow,
  TagAssignmentLabel,
  TagAssignmentKey,
  TagAssignmentValue,
  TagAssignmentRemoveButton,
} from '@databricks/web-shared/unified-tagging';

interface RouteConfigurationProps {
  routeName: string;
  onChangeRouteName: (name: string) => void;
  routeNameError?: string;
  envVarKey?: string;
  onChangeEnvVarKey?: (key: string) => void;
  envVarKeyError?: string;
  description: string;
  onChangeDescription: (desc: string) => void;
  tagsFieldArray: any; // UseTagAssignmentFormReturn type from unified-tagging
  componentIdPrefix?: string;
}

export const RouteConfiguration = ({
  routeName,
  onChangeRouteName,
  routeNameError,
  envVarKey,
  onChangeEnvVarKey,
  envVarKeyError,
  description,
  onChangeDescription,
  tagsFieldArray,
  componentIdPrefix = 'mlflow.routes.configuration',
}: RouteConfigurationProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {/* Route Name */}
      <div>
        <FormUI.Label htmlFor="route-name-input">
          <FormattedMessage defaultMessage="Route Name" description="Route name label" />
          <span css={{ color: theme.colors.textValidationDanger }}> *</span>
        </FormUI.Label>
        <Input
          componentId={`${componentIdPrefix}.route_name`}
          id="route-name-input"
          placeholder={intl.formatMessage({
            defaultMessage: 'e.g., my-gpt4-route',
            description: 'Route name placeholder',
          })}
          value={routeName}
          onChange={(e) => onChangeRouteName(e.target.value)}
          validationState={routeNameError ? 'error' : undefined}
        />
        {routeNameError && <FormUI.Message type="error" message={routeNameError} />}
      </div>

      {/* Environment Variable (only show if provided) */}
      {envVarKey !== undefined && onChangeEnvVarKey && (
        <div>
          <FormUI.Label htmlFor="env-var-key-input">
            <FormattedMessage defaultMessage="Environment Variable" description="Environment variable label" />
            <span css={{ color: theme.colors.textValidationDanger }}> *</span>
          </FormUI.Label>
          <Input
            componentId={`${componentIdPrefix}.env_var_key`}
            id="env-var-key-input"
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g., OPENAI_API_KEY',
              description: 'Environment variable placeholder',
            })}
            value={envVarKey}
            onChange={(e) => onChangeEnvVarKey(e.target.value)}
            validationState={envVarKeyError ? 'error' : undefined}
          />
          {envVarKeyError && <FormUI.Message type="error" message={envVarKeyError} />}
        </div>
      )}

      {/* Description */}
      <div>
        <FormUI.Label htmlFor="route-description-input">
          <FormattedMessage defaultMessage="Description (optional)" description="Route description label" />
        </FormUI.Label>
        <Input
          componentId={`${componentIdPrefix}.description`}
          id="route-description-input"
          placeholder={intl.formatMessage({
            defaultMessage: 'Describe this route...',
            description: 'Route description placeholder',
          })}
          value={description}
          onChange={(e) => onChangeDescription(e.target.value)}
        />
      </div>

      {/* Tags */}
      <div>
        <FormUI.Label>
          <FormattedMessage defaultMessage="Tags (optional)" description="Tags label" />
        </FormUI.Label>
        <TagAssignmentRoot {...tagsFieldArray}>
          <TagAssignmentRow>
            <TagAssignmentLabel>
              <FormattedMessage defaultMessage="Key" description="Tag key header label" />
            </TagAssignmentLabel>
            <TagAssignmentLabel>
              <FormattedMessage defaultMessage="Value" description="Tag value header label" />
            </TagAssignmentLabel>
          </TagAssignmentRow>
          {tagsFieldArray.fields.map((field: any, index: number) => (
            <TagAssignmentRow key={field.id}>
              <TagAssignmentKey index={index} />
              <TagAssignmentValue index={index} />
              <TagAssignmentRemoveButton componentId={`${componentIdPrefix}.tag_remove_${index}`} index={index} />
            </TagAssignmentRow>
          ))}
        </TagAssignmentRoot>
      </div>
    </div>
  );
};

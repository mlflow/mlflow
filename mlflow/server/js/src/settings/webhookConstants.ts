import { defineMessages } from '@databricks/i18n';

export const eventLabels = defineMessages({
  'REGISTERED_MODEL.CREATED': { defaultMessage: 'Registered model created', description: 'Webhook event label' },
  'MODEL_VERSION.CREATED': { defaultMessage: 'Model version created', description: 'Webhook event label' },
  'MODEL_VERSION_TAG.SET': { defaultMessage: 'Model version tag set', description: 'Webhook event label' },
  'MODEL_VERSION_TAG.DELETED': { defaultMessage: 'Model version tag deleted', description: 'Webhook event label' },
  'MODEL_VERSION_ALIAS.CREATED': {
    defaultMessage: 'Model version alias created',
    description: 'Webhook event label',
  },
  'MODEL_VERSION_ALIAS.DELETED': {
    defaultMessage: 'Model version alias deleted',
    description: 'Webhook event label',
  },
  'PROMPT.CREATED': { defaultMessage: 'Prompt created', description: 'Webhook event label' },
  'PROMPT_VERSION.CREATED': { defaultMessage: 'Prompt version created', description: 'Webhook event label' },
  'PROMPT_TAG.SET': { defaultMessage: 'Prompt tag set', description: 'Webhook event label' },
  'PROMPT_TAG.DELETED': { defaultMessage: 'Prompt tag deleted', description: 'Webhook event label' },
  'PROMPT_VERSION_TAG.SET': { defaultMessage: 'Prompt version tag set', description: 'Webhook event label' },
  'PROMPT_VERSION_TAG.DELETED': {
    defaultMessage: 'Prompt version tag deleted',
    description: 'Webhook event label',
  },
  'PROMPT_ALIAS.CREATED': { defaultMessage: 'Prompt alias created', description: 'Webhook event label' },
  'PROMPT_ALIAS.DELETED': { defaultMessage: 'Prompt alias deleted', description: 'Webhook event label' },
  'BUDGET_POLICY.EXCEEDED': { defaultMessage: 'Budget policy exceeded', description: 'Webhook event label' },
});

export const VALID_EVENTS: { entity: string; action: string }[] = [
  { entity: 'REGISTERED_MODEL', action: 'CREATED' },
  { entity: 'MODEL_VERSION', action: 'CREATED' },
  { entity: 'MODEL_VERSION_TAG', action: 'SET' },
  { entity: 'MODEL_VERSION_TAG', action: 'DELETED' },
  { entity: 'MODEL_VERSION_ALIAS', action: 'CREATED' },
  { entity: 'MODEL_VERSION_ALIAS', action: 'DELETED' },
  { entity: 'PROMPT', action: 'CREATED' },
  { entity: 'PROMPT_VERSION', action: 'CREATED' },
  { entity: 'PROMPT_TAG', action: 'SET' },
  { entity: 'PROMPT_TAG', action: 'DELETED' },
  { entity: 'PROMPT_VERSION_TAG', action: 'SET' },
  { entity: 'PROMPT_VERSION_TAG', action: 'DELETED' },
  { entity: 'PROMPT_ALIAS', action: 'CREATED' },
  { entity: 'PROMPT_ALIAS', action: 'DELETED' },
  { entity: 'BUDGET_POLICY', action: 'EXCEEDED' },
];

export const WEBHOOK_NAME_REGEX = /^[a-z0-9]([a-z0-9._-]*[a-z0-9])?$/i;

export const eventKey = (entity: string, action: string) => `${entity}.${action}`;

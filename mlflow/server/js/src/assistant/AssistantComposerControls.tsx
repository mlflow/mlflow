/**
 * In-composer controls for the active model and execution mode, rendered as two quiet,
 * borderless pills in the composer toolbar.
 *
 * The model pill is a status + entry point: it shows the selected provider's model and opens the
 * scoped "Model settings" sub-view (where any provider/model can be changed) — the same outcome
 * for every provider, so the pill's affordance is consistent. The execution-mode pill flips
 * `permissions.full_access` inline; that's composer-local fast-toggle state with no setup
 * equivalent. Both persist via `updateConfig` and take effect on the next message.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Button,
  ChevronDownIcon,
  DropdownMenu,
  LockIcon,
  ShieldIcon,
  SparkleIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { updateConfig } from './AssistantService';
import { useAssistantConfigQuery } from './hooks/useAssistantConfigQuery';

// Friendly names for providers whose `model` is a fixed/free-form string rather than a
// selectable value (so the pill reads "Claude Code" instead of the literal "default").
const PROVIDER_LABELS = {
  claude_code: 'Claude Code',
  codex: 'OpenAI Codex',
  mlflow_gateway: 'MLflow Gateway',
  ollama: 'Ollama',
} satisfies Record<string, string>;

export const AssistantComposerControls = ({ onOpenSettings }: { onOpenSettings: () => void }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { config, refetch } = useAssistantConfigQuery();

  const selected = useMemo(() => {
    const entry = Object.entries(config?.providers ?? {}).find(([, p]) => p.selected);
    return entry ? { name: entry[0], ...entry[1] } : null;
  }, [config]);

  const [fullAccess, setFullAccess] = useState(false);
  useEffect(() => {
    setFullAccess(selected?.permissions?.full_access ?? false);
  }, [selected]);

  const handleExecutionModeChange = useCallback(
    async (value: string) => {
      if (!selected) return;
      const next = value === 'full';
      setFullAccess(next);
      // Send the full permissions object: the backend rebuilds PermissionsConfig from the posted
      // fields, so omitting allow_edit_files/allow_read_docs would reset them to defaults.
      await updateConfig({
        providers: {
          [selected.name]: { permissions: { ...selected.permissions, full_access: next } },
        },
      });
      await refetch();
    },
    [selected, refetch],
  );

  if (!selected) return null;

  const providerLabel = PROVIDER_LABELS[selected.name as keyof typeof PROVIDER_LABELS] ?? selected.name;
  const modelDisplay = selected.model && selected.model !== 'default' ? selected.model : providerLabel;
  const settingsLabel = intl.formatMessage({
    defaultMessage: 'Change model in Settings',
    description: 'Tooltip/label on the assistant model pill; opens the Model settings view to change the model',
  });
  const fullAccessLabel = intl.formatMessage({
    defaultMessage: 'Full access',
    description: 'Assistant execution mode granting full tool, file, and command access',
  });
  const readOnlyLabel = intl.formatMessage({
    defaultMessage: 'Read-only',
    description: 'Assistant execution mode restricting the assistant to read-only/limited commands',
  });

  const pillLabelCss = {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
    maxWidth: 120,
  };
  const caretCss = { fontSize: 12, opacity: 0.7, flex: '0 0 auto' };
  const pillContentCss = { display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs, minWidth: 0 };

  return (
    <div css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs, minWidth: 0 }}>
      <Tooltip componentId="mlflow.assistant.chat_panel.model_settings.tooltip" content={settingsLabel}>
        <Button
          componentId="mlflow.assistant.chat_panel.model_settings"
          type="tertiary"
          size="small"
          onClick={onOpenSettings}
          aria-label={settingsLabel}
        >
          <span css={pillContentCss}>
            <SparkleIcon css={{ fontSize: 14 }} />
            <span css={pillLabelCss}>{modelDisplay}</span>
            <ChevronDownIcon css={caretCss} />
          </span>
        </Button>
      </Tooltip>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button componentId="mlflow.assistant.chat_panel.execution_mode" type="tertiary" size="small">
            <span css={pillContentCss}>
              {fullAccess ? <ShieldIcon css={{ fontSize: 14 }} /> : <LockIcon css={{ fontSize: 14 }} />}
              <span>{fullAccess ? fullAccessLabel : readOnlyLabel}</span>
              <ChevronDownIcon css={caretCss} />
            </span>
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content>
          <DropdownMenu.RadioGroup
            componentId="mlflow.assistant.chat_panel.execution_mode_radio"
            value={fullAccess ? 'full' : 'restricted'}
            onValueChange={handleExecutionModeChange}
          >
            <DropdownMenu.RadioItem value="full">
              <DropdownMenu.ItemIndicator />
              {fullAccessLabel}
            </DropdownMenu.RadioItem>
            <DropdownMenu.RadioItem value="restricted">
              <DropdownMenu.ItemIndicator />
              {readOnlyLabel}
            </DropdownMenu.RadioItem>
          </DropdownMenu.RadioGroup>
          <DropdownMenu.Arrow />
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    </div>
  );
};

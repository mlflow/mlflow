/**
 * Final step: Project configuration for MLflow Assistant setup.
 */

import { useState, useCallback, useEffect } from 'react';
import { Typography, useDesignSystemTheme, Input, Checkbox, Spinner } from '@databricks/design-system';

import { updateConfig } from '../AssistantService';
import { useAssistantConfigQuery } from '../hooks/useAssistantConfigQuery';
import { WizardFooter } from './WizardFooter';

interface SetupStepProjectProps {
  experimentId?: string;
  onBack: () => void;
  onComplete: () => void;
  /** Custom label for the save/finish button */
  nextLabel?: string;
  /** Custom label for the back button */
  backLabel?: string;
}

export const SetupStepProject = ({
  experimentId,
  onBack,
  onComplete,
  nextLabel = 'Finish',
  backLabel,
}: SetupStepProjectProps) => {
  const { theme } = useDesignSystemTheme();
  const { config, isLoading: isLoadingConfig } = useAssistantConfigQuery();

  const [projectPath, setProjectPath] = useState<string>('');
  // Permissions state
  const [editFiles, setEditFiles] = useState(true);
  const [readDocs, setReadDocs] = useState(true);
  const [fullPermission, setFullPermission] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize form state from loaded config
  useEffect(() => {
    if (!config) return;

    const provider = config.providers?.['claude_code'];
    if (provider?.permissions) {
      setEditFiles(provider.permissions.allow_edit_files ?? true);
      setReadDocs(provider.permissions.allow_read_docs ?? true);
      setFullPermission(provider.permissions.full_access ?? false);
    }
    if (experimentId && config.projects?.[experimentId]) {
      setProjectPath(config.projects[experimentId].location || '');
    }
  }, [config, experimentId]);

  const handleSave = useCallback(async () => {
    setIsSaving(true);
    setError(null);

    try {
      // Build config update - always mark provider as selected (setup complete)
      const configUpdate: Parameters<typeof updateConfig>[0] = {
        providers: {
          claude_code: {
            model: 'default',
            selected: true,
            permissions: {
              allow_edit_files: editFiles,
              allow_read_docs: readDocs,
              full_access: fullPermission,
            },
          },
        },
      };

      // Add project mapping if experiment and path provided
      if (experimentId && projectPath.trim()) {
        configUpdate.projects = {
          [experimentId]: { type: 'local' as const, location: projectPath.trim() },
        };
      }

      await updateConfig(configUpdate);
      onComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save configuration');
      setIsSaving(false);
    }
  }, [experimentId, projectPath, editFiles, readDocs, fullPermission, onComplete]);

  if (isLoadingConfig) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <Spinner size="large" />
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div css={{ flex: 1 }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          {/* Permissions Section */}
          <div>
            <Typography.Text bold css={{ fontSize: 18, marginBottom: theme.spacing.sm, display: 'block' }}>
              Permissions
            </Typography.Text>
            <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
              Configure what actions the assistant can perform on your behalf.
            </Typography.Text>

            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              <div>
                <Checkbox
                  componentId={`mlflow.assistant.setup.project.perm_mlflow_cli`}
                  isChecked
                  disabled
                  onChange={() => {}}
                >
                  <Typography.Text>Execute MLflow CLI (required)</Typography.Text>
                </Checkbox>
                <Typography.Text
                  color="secondary"
                  css={{ fontSize: theme.typography.fontSizeSm, marginLeft: 24, display: 'block' }}
                >
                  Allow running MLflow commands to fetch traces, runs, and experiment data.
                </Typography.Text>
              </div>

              <div>
                <Checkbox
                  componentId="mlflow.assistant.setup.project.perm_edit_files"
                  isChecked={editFiles}
                  onChange={(checked) => setEditFiles(checked)}
                >
                  <Typography.Text>Edit project code</Typography.Text>
                </Checkbox>
                <Typography.Text
                  color="secondary"
                  css={{ fontSize: theme.typography.fontSizeSm, marginLeft: 24, display: 'block' }}
                >
                  Allow modifying files in your project directory.
                </Typography.Text>
              </div>

              <div>
                <Checkbox
                  componentId="mlflow.assistant.setup.project.perm_read_docs"
                  isChecked={readDocs}
                  onChange={(checked) => setReadDocs(checked)}
                >
                  <Typography.Text>Read MLflow documentation</Typography.Text>
                </Checkbox>
                <Typography.Text
                  color="secondary"
                  css={{ fontSize: theme.typography.fontSizeSm, marginLeft: 24, display: 'block' }}
                >
                  Allow fetching content from mlflow.org documentation.
                </Typography.Text>
              </div>

              <div>
                <Checkbox
                  componentId="mlflow.assistant.setup.project.perm_full"
                  isChecked={fullPermission}
                  onChange={(checked) => setFullPermission(checked)}
                >
                  <Typography.Text>Full access</Typography.Text>
                </Checkbox>
                <Typography.Text
                  color="secondary"
                  css={{ fontSize: theme.typography.fontSizeSm, marginLeft: 24, display: 'block' }}
                >
                  Bypass all permission checks. Use with caution.
                </Typography.Text>
              </div>
            </div>
          </div>

          {/* Project Configuration Section */}
          <div>
            <Typography.Text bold css={{ fontSize: 18, marginBottom: theme.spacing.sm, display: 'block' }}>
              Project Path (Optional)
            </Typography.Text>
            <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
              Link this experiment to your local codebase. This enables the assistant to understand your project context
              and provide more accurate suggestions and fixes.
            </Typography.Text>

            {experimentId ? (
              <Input
                componentId="mlflow.assistant.setup.project.path_input"
                value={projectPath}
                onChange={(e) => setProjectPath(e.target.value)}
                placeholder="/Users/me/projects/my-llm-project"
                css={{ width: '100%' }}
              />
            ) : (
              <div
                css={{
                  backgroundColor: theme.colors.backgroundSecondary,
                  borderRadius: theme.borders.borderRadiusMd,
                  padding: theme.spacing.md,
                }}
              >
                <Typography.Text color="secondary">
                  No experiment selected. You can configure project mappings later in Settings.
                </Typography.Text>
              </div>
            )}
          </div>

          {error && <Typography.Text css={{ color: theme.colors.textValidationDanger }}>{error}</Typography.Text>}
        </div>
      </div>

      <WizardFooter
        onBack={onBack}
        onNext={handleSave}
        nextLabel={nextLabel}
        backLabel={backLabel}
        isLoading={isSaving}
      />
    </div>
  );
};

/**
 * Final step: Project configuration for MLflow Assistant setup.
 */

import { useState, useCallback, useEffect } from 'react';
import {
  Typography,
  useDesignSystemTheme,
  Input,
  Checkbox,
  Spinner,
  Radio,
  Tooltip,
  QuestionMarkIcon,
  Alert,
} from '@databricks/design-system';

import { updateConfig, installSkills } from '../AssistantService';
import { useAssistantConfigQuery } from '../hooks/useAssistantConfigQuery';
import { WizardFooter } from './WizardFooter';

type SkillsLocation = 'global' | 'project' | 'custom';

interface SetupStepProjectProps {
  experimentId?: string;
  onBack: () => void;
  onComplete: () => void;
  /** Custom label for the save/finish button */
  nextLabel?: string;
  /** Custom label for the back button */
  backLabel?: string;
}

const GLOBAL_SKILLS_PATH = '~/.claude/skills';

const deriveSkillsLocation = (
  skillsLocation: string | undefined,
  projectPath: string,
): { location: SkillsLocation; customPath: string } => {
  if (!skillsLocation) {
    return { location: 'global', customPath: '' };
  }
  const globalPath = GLOBAL_SKILLS_PATH.replace('~', '');
  if (skillsLocation.endsWith(globalPath)) {
    return { location: 'global', customPath: '' };
  }
  if (projectPath && skillsLocation.endsWith(`${projectPath}/.claude/skills`)) {
    return { location: 'project', customPath: '' };
  }
  return { location: 'custom', customPath: skillsLocation };
};

export const SetupStepProject = ({
  experimentId,
  onBack,
  onComplete,
  nextLabel = 'Finish',
  backLabel,
}: SetupStepProjectProps) => {
  const { theme } = useDesignSystemTheme();
  const { config, isLoading: isLoadingConfig, refetch: refetchConfig } = useAssistantConfigQuery();

  const [projectPath, setProjectPath] = useState<string>('');
  // Permissions state
  const [editFiles, setEditFiles] = useState(true);
  const [readDocs, setReadDocs] = useState(true);
  const [fullPermission, setFullPermission] = useState(false);

  // Skills state
  const [skillsLocation, setSkillsLocation] = useState<SkillsLocation>('global');
  const [customSkillsPath, setCustomSkillsPath] = useState<string>('');
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

    let currentProjectPath = '';
    if (experimentId && config.projects?.[experimentId]) {
      currentProjectPath = config.projects[experimentId].location || '';
      setProjectPath(currentProjectPath);
    }

    // Initialize skills location from config
    const { location, customPath } = deriveSkillsLocation(config.skills_location, currentProjectPath);
    setSkillsLocation(location);
    setCustomSkillsPath(customPath);
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

      // Handle project mapping - add if path provided, remove if cleared
      if (experimentId) {
        if (projectPath.trim()) {
          configUpdate.projects = {
            [experimentId]: { type: 'local' as const, location: projectPath.trim() },
          };
        } else {
          // Send null to remove the project mapping
          configUpdate.projects = {
            [experimentId]: null,
          };
        }
      }

      await updateConfig(configUpdate);

      // Install skills based on selected location
      try {
        await installSkills(
          skillsLocation,
          skillsLocation === 'custom' ? customSkillsPath.trim() : undefined,
          skillsLocation === 'project' ? experimentId : undefined,
        );
        await refetchConfig();
      } catch {
        // Silently ignore skills installation errors - user can install later
      }

      onComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save configuration');
      setIsSaving(false);
    }
  }, [
    experimentId,
    projectPath,
    skillsLocation,
    customSkillsPath,
    onComplete,
    refetchConfig,
    editFiles,
    readDocs,
    fullPermission,
  ]);

  if (isLoadingConfig) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <Spinner size="large" />
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div css={{ flex: 1, overflow: 'auto' }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          {/* Permissions Section */}
          <div>
            <Typography.Text bold css={{ fontSize: 18, marginBottom: theme.spacing.sm, display: 'block' }}>
              Permissions
            </Typography.Text>
            <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
              Configure what actions the assistant can perform.
            </Typography.Text>

            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <Checkbox
                  componentId={`mlflow.assistant.setup.project.perm_mlflow_cli`}
                  isChecked
                  disabled
                  onChange={() => {}}
                >
                  <Typography.Text>Execute MLflow CLI (required)</Typography.Text>
                </Checkbox>
                <Tooltip
                  componentId="mlflow.assistant.setup.project.perm_mlflow_cli_tooltip"
                  content="Allow running MLflow commands to fetch traces, runs, and experiment data. This is required for the assistant to work properly."
                >
                  <QuestionMarkIcon
                    css={{ color: theme.colors.actionPrimaryBackgroundDefault, fontSize: 14, cursor: 'help' }}
                  />
                </Tooltip>
              </div>

              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <Checkbox
                  componentId="mlflow.assistant.setup.project.perm_read_docs"
                  isChecked={readDocs}
                  onChange={(checked) => setReadDocs(checked)}
                >
                  <Typography.Text>Read MLflow doc</Typography.Text>
                </Checkbox>
                <Tooltip
                  componentId="mlflow.assistant.setup.project.perm_read_docs_tooltip"
                  content="Allow fetching content from mlflow.org documentation to get the latest information about MLflow and make accurate suggestions."
                >
                  <QuestionMarkIcon
                    css={{ color: theme.colors.actionPrimaryBackgroundDefault, fontSize: 14, cursor: 'help' }}
                  />
                </Tooltip>
              </div>

              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <Checkbox
                  componentId="mlflow.assistant.setup.project.perm_edit_files"
                  isChecked={editFiles}
                  onChange={(checked) => setEditFiles(checked)}
                >
                  <Typography.Text>Edit project code</Typography.Text>
                </Checkbox>
                <Tooltip
                  componentId="mlflow.assistant.setup.project.perm_edit_files_tooltip"
                  content="Allow modifying files in your project directory. Required if you want to use the assistant to work on your project code, e.g., writing evaluation scripts, fixing bugs, etc."
                >
                  <QuestionMarkIcon
                    css={{ color: theme.colors.actionPrimaryBackgroundDefault, fontSize: 14, cursor: 'help' }}
                  />
                </Tooltip>
              </div>

              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <Checkbox
                  componentId="mlflow.assistant.setup.project.perm_full"
                  isChecked={fullPermission}
                  onChange={(checked) => setFullPermission(checked)}
                >
                  <Typography.Text>Full access</Typography.Text>
                </Checkbox>
                <Tooltip
                  componentId="mlflow.assistant.setup.project.perm_full_tooltip"
                  content="Bypass all permission checks. Use with caution."
                >
                  <QuestionMarkIcon
                    css={{ color: theme.colors.actionPrimaryBackgroundDefault, fontSize: 14, cursor: 'help' }}
                  />
                </Tooltip>
              </div>
            </div>
          </div>

          {/* Project Configuration Section */}
          <div>
            <Typography.Text bold css={{ fontSize: 18, marginBottom: theme.spacing.sm, display: 'block' }}>
              Project Path
            </Typography.Text>
            <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
              Link this experiment to your local codebase (optional) to enable the assistant to understand your project
              context.
            </Typography.Text>

            {experimentId ? (
              <Input
                componentId="mlflow.assistant.setup.project.path_input"
                value={projectPath}
                onChange={(e) => {
                  setProjectPath(e.target.value);
                  if (error) setError(null);
                }}
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

          {/* Skills Installation Section */}
          <div>
            <Typography.Text bold css={{ fontSize: 18, marginBottom: theme.spacing.sm, display: 'block' }}>
              Skills Location
            </Typography.Text>
            <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.md }}>
              Extend the assistant with specialized MLflow workflows. See{' '}
              <Typography.Link
                componentId="mlflow.assistant.setup.project.skills_link"
                href="https://github.com/mlflow/skills"
                target="_blank"
              >
                MLflow Skills
              </Typography.Link>{' '}
              to find list of skills to be installed.
            </Typography.Text>

            <Radio.Group
              componentId="mlflow.assistant.setup.project.skills_location"
              name="skills-location"
              value={skillsLocation}
              onChange={(e) => setSkillsLocation(e.target.value as SkillsLocation)}
            >
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                <Radio componentId="mlflow.assistant.setup.project.skills_global" value="global">
                  <Typography.Text>Global</Typography.Text>
                  <Typography.Text color="secondary" css={{ marginLeft: theme.spacing.xs }}>
                    (~/.claude/skills/)
                  </Typography.Text>
                </Radio>

                <Radio
                  componentId="mlflow.assistant.setup.project.skills_project"
                  value="project"
                  disabled={!projectPath.trim()}
                >
                  <Typography.Text color={!projectPath.trim() ? 'secondary' : undefined}>Project</Typography.Text>
                  <Typography.Text color="secondary" css={{ marginLeft: theme.spacing.xs }}>
                    {projectPath.trim() ? `(${projectPath.trim()}/.claude/skills/)` : '(requires project path)'}
                  </Typography.Text>
                </Radio>

                <div>
                  <Radio componentId="mlflow.assistant.setup.project.skills_custom" value="custom">
                    <Typography.Text>Custom location</Typography.Text>
                  </Radio>
                  {skillsLocation === 'custom' && (
                    <div css={{ marginTop: theme.spacing.sm, paddingLeft: 24 }}>
                      <Input
                        componentId="mlflow.assistant.setup.project.custom_skills_path"
                        value={customSkillsPath}
                        onChange={(e) => {
                          setCustomSkillsPath(e.target.value);
                          if (error) setError(null);
                        }}
                        placeholder="/path/to/skills"
                        css={{ width: '100%' }}
                      />
                    </div>
                  )}
                </div>
              </div>
            </Radio.Group>
          </div>
        </div>
      </div>

      {error && (
        <Alert
          componentId="mlflow.assistant.setup.project.error"
          type="error"
          message={error}
          closable={false}
          css={{ marginTop: theme.spacing.md }}
        />
      )}

      <WizardFooter
        onBack={onBack}
        onNext={handleSave}
        nextLabel={nextLabel}
        backLabel={backLabel}
        isLoading={isSaving}
        nextDisabled={!!error}
      />
    </div>
  );
};

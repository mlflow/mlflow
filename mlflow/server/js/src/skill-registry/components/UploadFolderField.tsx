import {
  Checkbox,
  FormUI,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import {
  totalFileSize,
  type ExceededContentLimit,
  type LocalSkillChoice,
  type SkillManifestProblem,
} from '../localSkillFolder';
import { formatFileSize, formatSizeLimit } from '../skillFiles';

/** The folder picker of the upload source, with its limits and what is wrong with the selected folder. */
export const UploadFolderField = ({
  files,
  hasSkillManifest,
  exceededLimit,
  manifestName,
  manifestProblem,
  maxBytes,
  maxFiles,
  onSelect,
  choices,
  selectedRoots,
  completedRoots,
  onSelectionChange,
  isVersion,
  discovering,
  disabled,
}: {
  files: File[];
  hasSkillManifest: boolean;
  exceededLimit: ExceededContentLimit;
  manifestName?: string;
  manifestProblem?: SkillManifestProblem;
  maxBytes?: number;
  maxFiles?: number;
  onSelect: (files: File[]) => void;
  choices: LocalSkillChoice[];
  selectedRoots: string[];
  completedRoots: string[];
  onSelectionChange: (roots: string[]) => void;
  isVersion: boolean;
  discovering: boolean;
  disabled: boolean;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const folderError =
    exceededLimit === 'files' ? (
      <FormattedMessage
        defaultMessage="This folder has {count} files. The server accepts up to {max}."
        description="Error when a skill folder has more files than the server accepts"
        values={{ count: files.length, max: maxFiles }}
      />
    ) : exceededLimit === 'bytes' && maxBytes !== undefined ? (
      <FormattedMessage
        defaultMessage="This folder is {size}. The server accepts up to {max} of files."
        description="Error when a skill folder is larger than the server accepts"
        values={{ size: formatFileSize(totalFileSize(files)), max: formatSizeLimit(maxBytes) }}
      />
    ) : !discovering && files.length > 0 && !hasSkillManifest ? (
      <FormattedMessage
        defaultMessage="No skills found. Select a folder containing SKILL.md or skill subfolders."
        description="Error when a selected skill folder has no SKILL.md"
      />
    ) : manifestProblem === 'unreadable' ? (
      <FormattedMessage
        defaultMessage="Couldn't read SKILL.md. Select the folder again."
        description="Error when a skill folder's SKILL.md cannot be read from disk"
      />
    ) : manifestProblem === 'unparsable' ? (
      <FormattedMessage
        defaultMessage="The frontmatter in SKILL.md couldn't be parsed. It must be a YAML mapping between {fence} lines, without aliases or merge keys."
        description="Error when a skill folder's SKILL.md frontmatter is not a valid YAML mapping"
        values={{ fence: <code>---</code> }}
      />
    ) : manifestProblem === 'missing' ? (
      <FormattedMessage
        defaultMessage="SKILL.md needs a name in its frontmatter, as in {example}."
        description="Error when a skill folder's SKILL.md has no name in its frontmatter"
        values={{ example: <code>name: my-skill</code> }}
      />
    ) : manifestProblem === 'invalid' ? (
      <FormattedMessage
        defaultMessage='The name in SKILL.md, "{name}", is not a valid skill name. Use lowercase letters, digits, and single hyphens.'
        description="Error when a skill folder's SKILL.md declares an invalid name"
        values={{ name: manifestName }}
      />
    ) : undefined;

  return (
    <>
      <input
        id="mlflow.skill_registry.register_modal.folder"
        aria-label={intl.formatMessage({
          defaultMessage: 'Skill folder',
          description: 'Aria label for the skill directory picker',
        })}
        type="file"
        disabled={disabled}
        multiple
        {...{ webkitdirectory: '', directory: '' }}
        onChange={(event) => onSelect(event.target.files ? Array.from(event.target.files) : [])}
      />
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="Select a skill folder or a parent directory. Skills are discovered recursively."
          description="Hint for the skill directory picker"
        />
      </Typography.Text>
      {discovering && (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Finding skills…"
            description="Progress while reading discovered skill manifests"
          />
        </Typography.Text>
      )}
      {choices.length > 1 && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <FormUI.Label htmlFor="mlflow.skill_registry.register_modal.discovered_skills">
            <FormattedMessage
              defaultMessage="Discovered skills"
              description="Label for skills found in a selected directory"
            />
          </FormUI.Label>
          <Typography.Text color="secondary">
            {isVersion ? (
              <FormattedMessage
                defaultMessage="Choose one skill for this version."
                description="Instruction for selecting one discovered skill when creating a version"
              />
            ) : (
              <FormattedMessage
                defaultMessage="Select the skills to register. Each skill gets its own registry entry."
                description="Instruction for selecting discovered skills to register"
              />
            )}
          </Typography.Text>
          {isVersion ? (
            <SimpleSelect
              id="mlflow.skill_registry.register_modal.discovered_skills"
              componentId="mlflow.skill_registry.register_modal.discovered_skill"
              aria-label={intl.formatMessage({
                defaultMessage: 'Discovered skills',
                description: 'Label for skills found in a selected directory',
              })}
              placeholder={intl.formatMessage({
                defaultMessage: 'Select a skill',
                description: 'Placeholder when choosing a discovered local skill',
              })}
              value={selectedRoots[0]}
              disabled={disabled}
              onChange={({ target }) => onSelectionChange([target.value])}
            >
              {choices.map((choice) => (
                <SimpleSelectOption
                  key={choice.rootPath}
                  value={choice.rootPath}
                  disabled={Boolean(choice.problem || choice.exceededLimit)}
                >
                  {choice.manifest?.name ?? choice.path} · {choice.path}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          ) : (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.sm,
                maxHeight: 240,
                overflowY: 'auto',
              }}
            >
              {choices.map((choice) => (
                <div key={choice.rootPath} css={{ display: 'flex', flexDirection: 'column' }}>
                  <Checkbox
                    componentId="mlflow.skill_registry.register_modal.select_discovered_skill"
                    isChecked={selectedRoots.includes(choice.rootPath)}
                    isDisabled={disabled || Boolean(choice.problem || choice.exceededLimit)}
                    onChange={(checked) =>
                      onSelectionChange(
                        checked
                          ? [...selectedRoots, choice.rootPath]
                          : selectedRoots.filter((root) => root !== choice.rootPath),
                      )
                    }
                  >
                    {choice.manifest?.name ?? choice.path}
                    <Typography.Text color="secondary" css={{ marginLeft: theme.spacing.sm }}>
                      {choice.path}
                    </Typography.Text>
                  </Checkbox>
                  {(choice.problem || choice.exceededLimit) && (
                    <Typography.Text color="error">
                      <FormattedMessage
                        defaultMessage="Cannot register: invalid SKILL.md or folder exceeds the upload limit."
                        description="Inline reason a discovered local skill cannot be selected"
                      />
                    </Typography.Text>
                  )}
                  {completedRoots.includes(choice.rootPath) && (
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="Registered"
                        description="Status of a successfully registered local skill in a partial batch"
                      />
                    </Typography.Text>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      <Typography.Text color="secondary">
        {maxBytes !== undefined ? (
          <FormattedMessage
            defaultMessage="Up to {max} of files per skill."
            description="Hint for the server's size limit of an uploaded skill folder"
            values={{ max: formatSizeLimit(maxBytes) }}
          />
        ) : (
          <FormattedMessage
            defaultMessage="Up to 25 MB of files per skill, unless your server sets a different limit."
            description="Hint for the default size limit of an uploaded skill folder"
          />
        )}
      </Typography.Text>
      {folderError && <FormUI.Message type="error" message={folderError} />}
    </>
  );
};

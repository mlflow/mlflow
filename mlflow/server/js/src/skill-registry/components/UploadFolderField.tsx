import { FormUI, Typography } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { totalFileSize, type ExceededContentLimit } from '../localSkillFolder';
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
}: {
  files: File[];
  hasSkillManifest: boolean;
  exceededLimit: ExceededContentLimit;
  manifestName?: string;
  manifestProblem?: 'unreadable' | 'unparsable' | 'missing' | 'invalid';
  maxBytes?: number;
  maxFiles?: number;
  onSelect: (files: File[]) => void;
}) => {
  const intl = useIntl();
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
    ) : files.length > 0 && !hasSkillManifest ? (
      <FormattedMessage
        defaultMessage="This folder has no SKILL.md at its top level."
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
        multiple
        {...{ webkitdirectory: '', directory: '' }}
        onChange={(event) => onSelect(event.target.files ? Array.from(event.target.files) : [])}
      />
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="Select the directory containing SKILL.md."
          description="Hint for the skill directory picker"
        />
      </Typography.Text>
      <Typography.Text color="secondary">
        {maxBytes !== undefined ? (
          <FormattedMessage
            defaultMessage="Up to {max} of files."
            description="Hint for the server's size limit of an uploaded skill folder"
            values={{ max: formatSizeLimit(maxBytes) }}
          />
        ) : (
          <FormattedMessage
            defaultMessage="Up to 25 MB of files, unless your server sets a different limit."
            description="Hint for the default size limit of an uploaded skill folder"
          />
        )}
      </Typography.Text>
      {folderError && <FormUI.Message type="error" message={folderError} />}
    </>
  );
};

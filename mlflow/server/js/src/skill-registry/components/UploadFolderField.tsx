import { FormUI, Typography } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { totalFileSize, type ExceededContentLimit } from '../localSkillFolder';
import { formatFileSize, formatSizeLimit } from '../skillFiles';

/** The folder picker of the upload source, with its limits and what is wrong with the selected folder. */
export const UploadFolderField = ({
  files,
  hasSkillManifest,
  exceededLimit,
  maxBytes,
  maxFiles,
  onSelect,
}: {
  files: File[];
  hasSkillManifest: boolean;
  exceededLimit: ExceededContentLimit;
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

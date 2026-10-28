import {
  FormUI,
  Input,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { SkillRegistrationSourceType } from '../sourceLocation';
import { isCommitSha } from '../utils';

/** The advanced fields of a remote source: its type, and for Git the ref, plus the path within it. */
export const PointerSourceFields = ({
  sourceType,
  refValue,
  subpath,
  onSourceTypeChange,
  onRefChange,
  onSubpathChange,
}: {
  sourceType?: SkillRegistrationSourceType;
  refValue: string;
  subpath: string;
  onSourceTypeChange: (sourceType: SkillRegistrationSourceType) => void;
  onRefChange: (ref: string) => void;
  onSubpathChange: (subpath: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const ref = refValue.trim();
  const hintCss = { display: 'block', marginTop: theme.spacing.xs };

  return (
    <>
      <div>
        <FormUI.Label htmlFor="mlflow.skill_registry.register_modal.source_type">
          <FormattedMessage defaultMessage="Source type" description="Label for the skill source type override" />
        </FormUI.Label>
        <SimpleSelect
          id="mlflow.skill_registry.register_modal.source_type"
          componentId="mlflow.skill_registry.register_modal.source_type"
          aria-label={intl.formatMessage({
            defaultMessage: 'Source type',
            description: 'Aria label for the skill source type override',
          })}
          value={sourceType}
          placeholder={intl.formatMessage({
            defaultMessage: 'Select a source type',
            description: 'Placeholder for the skill source type override',
          })}
          onChange={({ target }) => onSourceTypeChange(target.value as SkillRegistrationSourceType)}
        >
          <SimpleSelectOption value="git">Git</SimpleSelectOption>
          <SimpleSelectOption value="oci">OCI</SimpleSelectOption>
          <SimpleSelectOption value="zip">ZIP</SimpleSelectOption>
        </SimpleSelect>
      </div>
      <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: theme.spacing.md }}>
        {sourceType !== 'oci' && sourceType !== 'zip' && (
          <div>
            <FormUI.Label htmlFor="mlflow.skill_registry.register_modal.ref">
              <FormattedMessage
                defaultMessage="Branch, tag or commit"
                description="Label for an optional skill git ref"
              />
            </FormUI.Label>
            <Input
              id="mlflow.skill_registry.register_modal.ref"
              componentId="mlflow.skill_registry.register_modal.ref"
              aria-label={intl.formatMessage({
                defaultMessage: 'Branch, tag or commit',
                description: 'Aria label for an optional skill git ref',
              })}
              value={refValue}
              onChange={(event) => onRefChange(event.target.value)}
            />
            <Typography.Hint css={hintCss}>
              {!ref ? (
                <FormattedMessage
                  defaultMessage="Defaults to the repository's default branch. A branch keeps moving, so pulls of this version get whatever it points to then. Use a tag or commit SHA to pin the content."
                  description="Hint for an empty skill git ref field"
                />
              ) : isCommitSha(ref) ? (
                <FormattedMessage
                  defaultMessage="Pinned to this commit, so every pull of this version gets the same content."
                  description="Hint when the skill git ref is a commit SHA"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="If this is a branch, pulls of this version get whatever it points to then. Use a tag or commit SHA to pin the content."
                  description="Hint when the skill git ref may be a moving branch"
                />
              )}
            </Typography.Hint>
          </div>
        )}
        <div>
          <FormUI.Label htmlFor="mlflow.skill_registry.register_modal.subpath">
            <FormattedMessage
              defaultMessage="Path within the source"
              description="Label for an optional skill source subpath"
            />
          </FormUI.Label>
          <Input
            id="mlflow.skill_registry.register_modal.subpath"
            componentId="mlflow.skill_registry.register_modal.subpath"
            aria-label={intl.formatMessage({
              defaultMessage: 'Path within the source',
              description: 'Aria label for an optional skill source subpath',
            })}
            value={subpath}
            onChange={(event) => onSubpathChange(event.target.value)}
          />
          <Typography.Hint css={hintCss}>
            <FormattedMessage
              defaultMessage="The directory holding SKILL.md. Leave blank if it is at the root."
              description="Hint for the skill source subpath field"
            />
          </Typography.Hint>
        </div>
      </div>
    </>
  );
};

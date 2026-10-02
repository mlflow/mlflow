import { useEffect, useRef, useState } from 'react';
import {
  Button,
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  FormUI,
  Input,
  Modal,
  PlusIcon,
  PuzzleIcon,
  Radio,
  Tooltip,
  SegmentedControlButton,
  SegmentedControlGroup,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { CopyButton } from '../../shared/building_blocks/CopyButton';
import { KeyValueTag } from '../../common/components/KeyValueTag';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { resolveIcon, sanitizeHref } from '../../common/utils/registryIcons';
import { SkillRegistryApi } from '../api';
import { useRegisterSkillMutation } from '../hooks/useRegisterSkillMutation';
import { findSkillManifest, packageSkillFolder, readSkillManifest } from '../localSkillFolder';
import {
  buildExternalSkillVersionRequest,
  buildUploadedSkillVersionRequest,
  formatSkillRegisterCli,
  formatSkillRegisterPython,
  parseSkillLocation,
  toRegisterSkillRequest,
  type SkillRegistrationErrorCode,
  type SkillRegistrationFields,
  type SkillRegistrationSourceType,
} from '../sourceLocation';
import { overlayButtonStyles } from '../styles';
import { SkillStatus, type RegistryIcon, type SkillVersion } from '../types';
import { formatSkillIdentity } from '../utils';

type RegistrationMode = 'pointer' | 'upload';
type DialogView = 'form' | 'api';
type SnippetFormat = 'cli' | 'python';
type IconTheme = 'Any' | 'Light' | 'Dark';

const EMPTY_FORM: SkillRegistrationFields = {
  location: '',
  identity: '',
  sourceTypeOverride: '',
  ref: '',
  subpath: '',
  digest: '',
  status: SkillStatus.ACTIVE,
};

const themeToIcon = (theme: IconTheme): string | undefined => (theme === 'Any' ? undefined : theme.toLowerCase());

const errorMessage = (error: SkillRegistrationErrorCode) => {
  switch (error) {
    case 'location_required':
      return (
        <FormattedMessage
          defaultMessage="Enter a source location."
          description="Validation error when skill registration has no source URL"
        />
      );
    case 'name_required':
      return (
        <FormattedMessage
          defaultMessage="Enter a skill name."
          description="Validation error when skill registration has no name"
        />
      );
    case 'name_invalid':
      return (
        <FormattedMessage
          defaultMessage="Names must be lowercase letters, digits, and single hyphens, as in @my-org/my-skill."
          description="Validation error for a skill registration name"
        />
      );
    case 'organization_invalid':
      return (
        <FormattedMessage
          defaultMessage="Organization names must be lowercase letters, digits, hyphens, and periods."
          description="Validation error for a skill registration organization"
        />
      );
    case 'source_type_required':
      return (
        <FormattedMessage
          defaultMessage="Choose a source type in Advanced settings. The location is not a Git, OCI, or ZIP URL."
          description="Validation error when a skill source type cannot be inferred"
        />
      );
    case 'source_type_conflict':
      return (
        <FormattedMessage
          defaultMessage="That source type does not match this location."
          description="Validation error when a skill source type override contradicts the URL"
        />
      );
    case 'credentials':
      return (
        <FormattedMessage
          defaultMessage="Remove credentials from the URL. They would be stored with the skill."
          description="Validation error when a skill source URL contains credentials"
        />
      );
    case 'zip_scheme':
      return (
        <FormattedMessage
          defaultMessage="A ZIP source must be an http(s) URL."
          description="Validation error for a non-HTTP skill ZIP source"
        />
      );
    case 'ref_not_git':
      return (
        <FormattedMessage
          defaultMessage="Ref is only used for Git sources."
          description="Validation error when a skill ref is set for a non-Git source"
        />
      );
    case 'digest_invalid':
      return (
        <FormattedMessage
          defaultMessage="Digest must be 64 lowercase hex characters. You can omit it."
          description="Validation error for a skill content digest"
        />
      );
    case 'status':
      return (
        <FormattedMessage
          defaultMessage="New versions can be Active or Draft."
          description="Validation error when a new skill version status is not active or draft"
        />
      );
    case 'skill_md_required':
      return (
        <FormattedMessage
          defaultMessage="Select the directory containing SKILL.md."
          description="Validation error when a skill folder upload has no SKILL.md"
        />
      );
    default:
      return (
        <FormattedMessage
          defaultMessage="Enter a remote Git, OCI, or ZIP location."
          description="Validation error for an invalid skill source location"
        />
      );
  }
};

const clientSourceType = (sourceType: string | null | undefined): '' | SkillRegistrationSourceType =>
  sourceType === 'git' || sourceType === 'oci' || sourceType === 'zip' ? sourceType : '';

export const RegisterSkillModal = ({
  visible,
  onClose,
  skill,
  sourceVersion,
  nextVersion,
  onRegistered,
}: {
  visible: boolean;
  onClose: () => void;
  /** Set when adding a version to an existing skill. Identity stays fixed. */
  skill?: { name: string; organization: string };
  /** Version whose source is copied into the form. Status stays Active. */
  sourceVersion?: SkillVersion;
  nextVersion?: number;
  onRegistered: (version: SkillVersion) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isVersion = Boolean(skill);
  const fixedIdentity = skill ? formatSkillIdentity(skill.name, skill.organization) : '';
  const [view, setView] = useState<DialogView>('form');
  const [snippetFormat, setSnippetFormat] = useState<SnippetFormat>('cli');
  const [mode, setMode] = useState<RegistrationMode>('pointer');
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [form, setForm] = useState(EMPTY_FORM);
  const [description, setDescription] = useState('');
  const [icons, setIcons] = useState<RegistryIcon[]>([]);
  const [iconUrl, setIconUrl] = useState('');
  const [iconTheme, setIconTheme] = useState<IconTheme>('Any');
  const [tags, setTags] = useState<Record<string, string>>({});
  const [tagKey, setTagKey] = useState('');
  const [tagValue, setTagValue] = useState('');
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const [identityTouched, setIdentityTouched] = useState(false);
  const [descriptionTouched, setDescriptionTouched] = useState(false);
  const [refTouched, setRefTouched] = useState(false);
  const [subpathTouched, setSubpathTouched] = useState(false);
  const [validationError, setValidationError] = useState<SkillRegistrationErrorCode | undefined>();
  const [presentationError, setPresentationError] = useState<string | undefined>();
  const { mutate, isLoading, error, reset } = useRegisterSkillMutation();
  const seeded = useRef(false);

  const parsed = parseSkillLocation(form.location);
  const effectiveSourceType = form.sourceTypeOverride || parsed?.sourceType;
  const hasSkillManifest = Boolean(findSkillManifest(folderFiles));

  const close = () => {
    setView('form');
    setSnippetFormat('cli');
    setMode('pointer');
    setAdvancedOpen(false);
    setForm(EMPTY_FORM);
    setDescription('');
    setIcons([]);
    setIconUrl('');
    setIconTheme('Any');
    setTags({});
    setTagKey('');
    setTagValue('');
    setFolderFiles([]);
    setIdentityTouched(false);
    setDescriptionTouched(false);
    setRefTouched(false);
    setSubpathTouched(false);
    setValidationError(undefined);
    setPresentationError(undefined);
    seeded.current = false;
    reset();
    onClose();
  };

  useEffect(() => {
    if (!visible) {
      seeded.current = false;
      return;
    }
    if (!sourceVersion || seeded.current) return;
    seeded.current = true;
    setForm({
      ...EMPTY_FORM,
      location: sourceVersion.source ?? '',
      sourceTypeOverride: clientSourceType(sourceVersion.source_type),
      ref: sourceVersion.ref ?? '',
      subpath: sourceVersion.subpath ?? '',
    });
    setAdvancedOpen(true);
  }, [visible, sourceVersion]);

  const applyLocation = (location: string) => {
    const nextParsed = parseSkillLocation(location);
    setForm((current) => {
      const suggested =
        !isVersion && !identityTouched && nextParsed?.suggestedName
          ? formatSkillIdentity(nextParsed.suggestedName, nextParsed.suggestedOrganization)
          : current.identity;
      return {
        ...current,
        location,
        identity: suggested,
        ref: refTouched ? current.ref : (nextParsed?.ref ?? ''),
        subpath: subpathTouched ? current.subpath : (nextParsed?.subpath ?? ''),
      };
    });
    setValidationError(undefined);
  };

  const onFolderSelected = async (files: File[]) => {
    setFolderFiles(files);
    setValidationError(undefined);
    const manifestFile = findSkillManifest(files);
    if (!manifestFile) return;
    const manifest = readSkillManifest(await manifestFile.text());
    if (!isVersion && manifest.name && !identityTouched) {
      setForm((current) => ({ ...current, identity: manifest.name ?? current.identity }));
    }
    if (!isVersion && manifest.description && !descriptionTouched) {
      setDescription(manifest.description);
    }
  };

  const addIcon = () => {
    const src = iconUrl.trim();
    if (!src) return;
    const next: RegistryIcon = { src };
    const iconThemeValue = themeToIcon(iconTheme);
    if (iconThemeValue) next.theme = iconThemeValue;
    setIcons((current) => [...current, next]);
    setIconUrl('');
    setIconTheme('Any');
  };

  const addTag = () => {
    const key = tagKey.trim();
    if (!key) return;
    setTags((current) => ({ ...current, [key]: tagValue }));
    setTagKey('');
    setTagValue('');
  };

  const savePresentation = async (version: SkillVersion) => {
    if (isVersion) return;
    const savedIcons = icons.filter((icon) => icon.src.trim());
    if (description.trim() || savedIcons.length) {
      await SkillRegistryApi.updateSkill(
        version.name,
        {
          ...(description.trim() ? { description: description.trim() } : {}),
          ...(savedIcons.length ? { icons: savedIcons } : {}),
        },
        version.organization,
      );
    }
    for (const [key, value] of Object.entries(tags)) {
      await SkillRegistryApi.setSkillTag(version.name, { key, value }, version.organization);
    }
  };

  const submit = () => {
    if (view !== 'form' || isLoading) return;
    const fields: SkillRegistrationFields = { ...form, identity: isVersion ? fixedIdentity : form.identity };
    if (mode === 'upload') {
      if (!hasSkillManifest) {
        setValidationError('skill_md_required');
        return;
      }
      const built = buildUploadedSkillVersionRequest(fields);
      if (!built.ok) {
        setValidationError(built.error);
        return;
      }
      setValidationError(undefined);
      void packageSkillFolder(folderFiles)
        .then((content) => {
          mutate(
            isVersion && skill
              ? {
                  kind: 'version-upload',
                  name: skill.name,
                  organization: skill.organization,
                  request: built.request,
                  content,
                }
              : { kind: 'register-upload', request: toRegisterSkillRequest(built.request, built.identity), content },
            {
              onSuccess: (version) => {
                void savePresentation(version)
                  .then(() => {
                    close();
                    onRegistered(version);
                  })
                  .catch((presentationFailure: unknown) => {
                    setPresentationError(
                      presentationFailure instanceof Error
                        ? presentationFailure.message
                        : intl.formatMessage({
                            defaultMessage:
                              'The version was created, but its description, icon, or tags could not be saved.',
                            description: 'Error when skill presentation metadata fails after registration',
                          }),
                    );
                  });
              },
            },
          );
        })
        .catch((packageError: unknown) => {
          setPresentationError(packageError instanceof Error ? packageError.message : 'Could not package the folder.');
        });
      return;
    }

    const built = buildExternalSkillVersionRequest(fields);
    if (!built.ok) {
      setValidationError(built.error);
      return;
    }
    setValidationError(undefined);
    mutate(
      isVersion && skill
        ? { kind: 'version', name: skill.name, organization: skill.organization, request: built.request }
        : { kind: 'register', request: toRegisterSkillRequest(built.request, built.identity) },
      {
        onSuccess: (version) => {
          void savePresentation(version)
            .then(() => {
              close();
              onRegistered(version);
            })
            .catch((presentationFailure: unknown) => {
              setPresentationError(
                presentationFailure instanceof Error
                  ? presentationFailure.message
                  : intl.formatMessage({
                      defaultMessage: 'The version was created, but its description, icon, or tags could not be saved.',
                      description: 'Error when skill presentation metadata fails after registration',
                    }),
              );
            });
        },
      },
    );
  };

  if (!visible) return null;

  const locationLabel = intl.formatMessage({
    defaultMessage: 'Location',
    description: 'Label for the skill registration source URL',
  });
  const nameLabel = intl.formatMessage({
    defaultMessage: 'Name',
    description: 'Label for the skill registration name',
  });
  const snippetIdentity = isVersion && skill ? { name: skill.name, organization: skill.organization } : {};
  const snippet = (snippetFormat === 'cli' ? formatSkillRegisterCli : formatSkillRegisterPython)({
    sourceType: effectiveSourceType,
    location: form.location,
    local: mode === 'upload',
    ...snippetIdentity,
  });
  const apiLink = (
    <Button componentId="mlflow.skill_registry.register_modal.api_link" type="link" onClick={() => setView('api')}>
      <FormattedMessage
        defaultMessage="create through API →"
        description="Link from skill registration to the API example"
      />
    </Button>
  );

  return (
    <Modal
      componentId="mlflow.skill_registry.register_modal"
      title={
        isVersion ? (
          <FormattedMessage
            defaultMessage="Create skill version {version}"
            description="Title for adding a skill version"
            values={{ version: nextVersion ?? '' }}
          />
        ) : (
          <FormattedMessage defaultMessage="Create skill" description="Title for registering a skill" />
        )
      }
      visible={visible}
      onCancel={close}
      size="wide"
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button componentId="mlflow.skill_registry.register_modal.cancel" onClick={close}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel skill registration" />
          </Button>
          <Button
            componentId="mlflow.skill_registry.register_modal.submit"
            type="primary"
            loading={isLoading}
            disabled={view === 'api' || (mode === 'upload' && !hasSkillManifest)}
            onClick={submit}
          >
            <FormattedMessage defaultMessage="Create" description="Submit button for skill registration" />
          </Button>
        </div>
      }
    >
      {view === 'api' ? (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <div>
            <Button componentId="mlflow.skill_registry.register_modal.back" type="link" onClick={() => setView('form')}>
              <FormattedMessage
                defaultMessage="← Back to form"
                description="Return from the skill API example to the form"
              />
            </Button>
          </div>
          <SegmentedControlGroup
            name="mlflow.skill_registry.register_modal.api_format"
            componentId="mlflow.skill_registry.register_modal.api_format"
            value={snippetFormat}
            onChange={(event) => setSnippetFormat(event.target.value as SnippetFormat)}
          >
            <SegmentedControlButton value="cli">
              <FormattedMessage defaultMessage="CLI" description="CLI example for skill registration" />
            </SegmentedControlButton>
            <SegmentedControlButton value="python">
              <FormattedMessage defaultMessage="Python" description="Python example for skill registration" />
            </SegmentedControlButton>
          </SegmentedControlGroup>
          <Typography.Text color="secondary">
            {snippetFormat === 'cli' ? (
              <FormattedMessage
                defaultMessage="The CLI reads the skill locally, so it can infer the name and record a digest."
                description="Explanation of the skill registration CLI example"
              />
            ) : (
              <FormattedMessage
                defaultMessage="The Python SDK reads the skill locally, so it can infer the name and record a digest."
                description="Explanation of the skill registration Python example"
              />
            )}
          </Typography.Text>
          <div css={{ position: 'relative' }}>
            <CopyButton
              componentId="mlflow.skill_registry.register_modal.api_snippet.copy"
              showLabel={false}
              copyText={snippet}
              icon={<CopyIcon />}
              css={overlayButtonStyles(theme)}
            />
            <CodeSnippet
              language={snippetFormat === 'python' ? 'python' : 'text'}
              theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
              style={{ padding: theme.spacing.sm, paddingRight: theme.spacing.xl + theme.spacing.sm }}
            >
              {snippet}
            </CodeSnippet>
          </div>
        </div>
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <Typography.Text color="secondary">
            {isVersion ? (
              <FormattedMessage
                defaultMessage="Adding a version to {identity}. Its content can come from anywhere, not just where the previous version lives. Or {apiLink}"
                description="Intro for adding an external skill version"
                values={{ identity: fixedIdentity, apiLink }}
              />
            ) : (
              <FormattedMessage
                defaultMessage="Specify a source and name below to register a skill to MLflow, or {apiLink}"
                description="Intro for the skill registration dialog"
                values={{ apiLink }}
              />
            )}
          </Typography.Text>

          <div>
            <FormUI.Label htmlFor="mlflow.skill_registry.register_modal.mode">
              <FormattedMessage defaultMessage="Source" description="Label for the skill registration source choice" />
            </FormUI.Label>
            <Radio.Group
              id="mlflow.skill_registry.register_modal.mode"
              componentId="mlflow.skill_registry.register_modal.mode"
              name="mlflow.skill_registry.register_modal.mode"
              value={mode}
              layout="vertical"
              css={{
                width: '100%',
                '& label': { width: '100%', alignItems: 'flex-start' },
                '& label > span:last-child': { flex: '1 1 auto', minWidth: 0 },
              }}
              onChange={(event) => setMode(event.target.value as RegistrationMode)}
            >
              <Radio value="pointer" css={{ alignItems: 'flex-start', width: '100%' }}>
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, width: '100%' }}>
                  <FormattedMessage
                    defaultMessage="Import from existing source, e.g. Git, OCI"
                    description="Skill registration option for a remote source pointer"
                  />
                  <Typography.Text color="secondary">
                    <FormattedMessage
                      defaultMessage="Content stays where it is. Clients fetch it with their own credentials."
                      description="Explanation of pointer-only skill registration"
                    />
                  </Typography.Text>
                  {mode === 'pointer' && (
                    <div css={{ width: '100%' }}>
                      <FormUI.Label htmlFor="mlflow.skill_registry.register_modal.location">
                        {locationLabel}
                      </FormUI.Label>
                      <Input
                        id="mlflow.skill_registry.register_modal.location"
                        componentId="mlflow.skill_registry.register_modal.location"
                        aria-label={locationLabel}
                        placeholder="https://github.com/redhat-ai/skills-developer.git"
                        value={form.location}
                        onChange={(event) => applyLocation(event.target.value)}
                        css={{ width: '100%' }}
                      />
                    </div>
                  )}
                </div>
              </Radio>
              <Radio value="upload" css={{ alignItems: 'flex-start', width: '100%' }}>
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, width: '100%' }}>
                  <FormattedMessage
                    defaultMessage="Upload a folder"
                    description="Skill registration option that uploads a local skill folder"
                  />
                  <Typography.Text color="secondary">
                    <FormattedMessage
                      defaultMessage="Your browser reads the folder and uploads it. MLflow stores the content."
                      description="Explanation of the local skill folder upload flow"
                    />
                  </Typography.Text>
                  {mode === 'upload' && (
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
                        onChange={(event) => {
                          void onFolderSelected(event.target.files ? Array.from(event.target.files) : []);
                        }}
                      />
                      <Typography.Text color="secondary">
                        <FormattedMessage
                          defaultMessage="Select the directory containing SKILL.md."
                          description="Hint for the skill directory picker"
                        />
                      </Typography.Text>
                    </>
                  )}
                </div>
              </Radio>
            </Radio.Group>
          </div>

          {!isVersion && (
            <div>
              <FormUI.Label htmlFor="mlflow.skill_registry.register_modal.name">{nameLabel}</FormUI.Label>
              <Input
                id="mlflow.skill_registry.register_modal.name"
                componentId="mlflow.skill_registry.register_modal.name"
                aria-label={nameLabel}
                placeholder="@my-org/my-skill"
                value={form.identity}
                onChange={(event) => {
                  setIdentityTouched(true);
                  setForm((current) => ({ ...current, identity: event.target.value }));
                }}
                css={{ width: '100%' }}
              />
              <Typography.Hint css={{ display: 'block', marginTop: theme.spacing.xs }}>
                <FormattedMessage
                  defaultMessage="Group skills with an organization by adding it to the name, e.g. @my-org/my-skill-name."
                  description="Hint for the skill registration name field"
                />
              </Typography.Hint>
            </div>
          )}

          <div>
            <Button
              componentId="mlflow.skill_registry.register_modal.advanced"
              type="tertiary"
              icon={advancedOpen ? <ChevronDownIcon /> : <ChevronRightIcon />}
              onClick={() => setAdvancedOpen((open) => !open)}
            >
              <FormattedMessage
                defaultMessage="Advanced settings (optional)"
                description="Toggle for optional skill registration fields"
              />
            </Button>
            {advancedOpen && (
              <div
                css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, marginTop: theme.spacing.md }}
              >
                {mode === 'pointer' && (
                  <>
                    <div>
                      <FormUI.Label htmlFor="mlflow.skill_registry.register_modal.source_type">
                        <FormattedMessage
                          defaultMessage="Source type"
                          description="Label for the skill source type override"
                        />
                      </FormUI.Label>
                      <SimpleSelect
                        id="mlflow.skill_registry.register_modal.source_type"
                        componentId="mlflow.skill_registry.register_modal.source_type"
                        aria-label={intl.formatMessage({
                          defaultMessage: 'Source type',
                          description: 'Aria label for the skill source type override',
                        })}
                        value={form.sourceTypeOverride || parsed?.sourceType}
                        placeholder={intl.formatMessage({
                          defaultMessage: 'Select a source type',
                          description: 'Placeholder for the skill source type override',
                        })}
                        onChange={({ target }) =>
                          setForm((current) => ({
                            ...current,
                            sourceTypeOverride: target.value as SkillRegistrationSourceType,
                          }))
                        }
                      >
                        <SimpleSelectOption value="git">Git</SimpleSelectOption>
                        <SimpleSelectOption value="oci">OCI</SimpleSelectOption>
                        <SimpleSelectOption value="zip">ZIP</SimpleSelectOption>
                      </SimpleSelect>
                    </div>
                    <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: theme.spacing.md }}>
                      {effectiveSourceType !== 'oci' && effectiveSourceType !== 'zip' && (
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
                            value={form.ref}
                            onChange={(event) => {
                              setRefTouched(true);
                              setForm((current) => ({ ...current, ref: event.target.value }));
                            }}
                          />
                          <Typography.Hint css={{ display: 'block', marginTop: theme.spacing.xs }}>
                            <FormattedMessage
                              defaultMessage="Defaults to the repository's default branch."
                              description="Hint for the skill git ref field"
                            />
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
                          value={form.subpath}
                          onChange={(event) => {
                            setSubpathTouched(true);
                            setForm((current) => ({ ...current, subpath: event.target.value }));
                          }}
                        />
                        <Typography.Hint css={{ display: 'block', marginTop: theme.spacing.xs }}>
                          <FormattedMessage
                            defaultMessage="The directory holding SKILL.md. Leave blank if it is at the root."
                            description="Hint for the skill source subpath field"
                          />
                        </Typography.Hint>
                      </div>
                    </div>
                  </>
                )}
                {!isVersion && (
                  <>
                    <div>
                      <FormUI.Label htmlFor="mlflow.skill_registry.register_modal.description">
                        <FormattedMessage
                          defaultMessage="Description"
                          description="Label for the skill description saved after registration"
                        />
                      </FormUI.Label>
                      <Input.TextArea
                        id="mlflow.skill_registry.register_modal.description"
                        componentId="mlflow.skill_registry.register_modal.description"
                        aria-label={intl.formatMessage({
                          defaultMessage: 'Description',
                          description: 'Aria label for the skill description',
                        })}
                        placeholder="What this skill does and when to use it."
                        value={description}
                        onChange={(event) => {
                          setDescriptionTouched(true);
                          setDescription(event.target.value);
                        }}
                        rows={3}
                      />
                    </div>
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                      <Typography.Text bold>
                        <FormattedMessage defaultMessage="Icon" description="Label for the skill icon editor" />
                      </Typography.Text>
                      <div
                        css={{
                          display: 'flex',
                          flexDirection: 'column',
                          gap: theme.spacing.sm,
                          padding: theme.spacing.md,
                          border: `1px solid ${theme.colors.border}`,
                          borderRadius: theme.general.borderRadiusBase,
                        }}
                      >
                        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                          <Typography.Text color="secondary" size="sm" css={{ flex: 1 }}>
                            <FormattedMessage
                              defaultMessage="Icon URL"
                              description="Column header for a skill icon URL"
                            />
                          </Typography.Text>
                          <Typography.Text color="secondary" size="sm" css={{ width: theme.spacing.xl * 4 }}>
                            <FormattedMessage
                              defaultMessage="Theme"
                              description="Column header for a skill icon theme"
                            />
                          </Typography.Text>
                          <div css={{ width: theme.spacing.xl + theme.spacing.sm }} />
                        </div>
                        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                          <div css={{ flex: 1 }}>
                            <Input
                              id="mlflow.skill_registry.register_modal.icon_url"
                              componentId="mlflow.skill_registry.register_modal.icon_url"
                              aria-label={intl.formatMessage({
                                defaultMessage: 'Icon URL',
                                description: 'Aria label for a skill icon URL',
                              })}
                              placeholder="https://example.com/icon.svg"
                              value={iconUrl}
                              onChange={(event) => setIconUrl(event.target.value)}
                            />
                          </div>
                          <SimpleSelect
                            id="mlflow.skill_registry.register_modal.icon_theme"
                            componentId="mlflow.skill_registry.register_modal.icon_theme"
                            aria-label={intl.formatMessage({
                              defaultMessage: 'Theme',
                              description: 'Aria label for a skill icon theme',
                            })}
                            value={iconTheme}
                            onChange={({ target }) => setIconTheme(target.value as IconTheme)}
                            css={{ width: theme.spacing.xl * 4 }}
                          >
                            <SimpleSelectOption value="Any">
                              <FormattedMessage defaultMessage="Any" description="Theme-agnostic skill icon option" />
                            </SimpleSelectOption>
                            <SimpleSelectOption value="Light">
                              <FormattedMessage defaultMessage="Light" description="Light mode skill icon option" />
                            </SimpleSelectOption>
                            <SimpleSelectOption value="Dark">
                              <FormattedMessage defaultMessage="Dark" description="Dark mode skill icon option" />
                            </SimpleSelectOption>
                          </SimpleSelect>
                          <Tooltip
                            componentId="mlflow.skill_registry.register_modal.icon_add.tooltip"
                            content={intl.formatMessage({
                              defaultMessage: 'Add icon',
                              description: 'Tooltip for adding a skill icon',
                            })}
                          >
                            <Button
                              componentId="mlflow.skill_registry.register_modal.icon_add"
                              aria-label={intl.formatMessage({
                                defaultMessage: 'Add icon',
                                description: 'Aria label for adding a skill icon',
                              })}
                              disabled={!iconUrl.trim()}
                              onClick={addIcon}
                            >
                              <PlusIcon />
                            </Button>
                          </Tooltip>
                        </div>
                        <Typography.Text color="secondary">
                          <FormattedMessage defaultMessage="Preview" description="Label for the skill icon preview" />
                        </Typography.Text>
                        <div css={{ display: 'flex', gap: theme.spacing.md }}>
                          {[false, true].map((isDark) => {
                            const src = sanitizeHref(resolveIcon(icons, isDark)?.src);
                            return (
                              <div
                                key={isDark ? 'dark' : 'light'}
                                css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
                              >
                                <div
                                  css={{
                                    width: 44,
                                    height: 44,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    borderRadius: theme.borders.borderRadiusSm,
                                    border: `1px solid ${theme.colors.border}`,
                                    backgroundColor: isDark ? '#1e1e1e' : '#ffffff',
                                  }}
                                >
                                  {src ? (
                                    <img src={src} alt="" css={{ width: 28, height: 28, objectFit: 'contain' }} />
                                  ) : (
                                    <PuzzleIcon css={{ fontSize: 28, color: theme.colors.textSecondary }} />
                                  )}
                                </div>
                                <Typography.Text color="secondary">
                                  {isDark ? (
                                    <FormattedMessage
                                      defaultMessage="dark theme"
                                      description="Dark skill icon preview label"
                                    />
                                  ) : (
                                    <FormattedMessage
                                      defaultMessage="light theme"
                                      description="Light skill icon preview label"
                                    />
                                  )}
                                </Typography.Text>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </>
                )}
                <div>
                  <FormUI.Label htmlFor="mlflow.skill_registry.register_modal.status">
                    <FormattedMessage
                      defaultMessage="Status"
                      description="Label for the initial skill version status"
                    />
                  </FormUI.Label>
                  <SimpleSelect
                    id="mlflow.skill_registry.register_modal.status"
                    componentId="mlflow.skill_registry.register_modal.status"
                    aria-label={intl.formatMessage({
                      defaultMessage: 'Status',
                      description: 'Aria label for the initial skill version status',
                    })}
                    value={form.status}
                    onChange={({ target }) =>
                      setForm((current) => ({
                        ...current,
                        status: target.value === SkillStatus.DRAFT ? SkillStatus.DRAFT : SkillStatus.ACTIVE,
                      }))
                    }
                  >
                    <SimpleSelectOption value={SkillStatus.ACTIVE}>
                      <FormattedMessage defaultMessage="Active" description="Initial skill version status active" />
                    </SimpleSelectOption>
                    <SimpleSelectOption value={SkillStatus.DRAFT}>
                      <FormattedMessage defaultMessage="Draft" description="Initial skill version status draft" />
                    </SimpleSelectOption>
                  </SimpleSelect>
                  <Typography.Hint css={{ display: 'block', marginTop: theme.spacing.xs }}>
                    <FormattedMessage
                      defaultMessage="A draft is passed over whenever any active version exists. A draft can always be reached by an explicit version pin or an alias."
                      description="Hint for the initial skill version status"
                    />
                  </Typography.Hint>
                </div>
                {!isVersion && (
                  <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                    <Typography.Text bold>
                      <FormattedMessage
                        defaultMessage="Tags"
                        description="Label for skill tags saved after registration"
                      />
                    </Typography.Text>
                    <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'flex-end' }}>
                      <Input
                        componentId="mlflow.skill_registry.register_modal.tag_key"
                        aria-label={intl.formatMessage({
                          defaultMessage: 'Key',
                          description: 'Aria label for a skill tag key',
                        })}
                        placeholder="Key"
                        value={tagKey}
                        onChange={(event) => setTagKey(event.target.value)}
                        css={{ flex: 1 }}
                      />
                      <Input
                        componentId="mlflow.skill_registry.register_modal.tag_value"
                        aria-label={intl.formatMessage({
                          defaultMessage: 'Value',
                          description: 'Aria label for a skill tag value',
                        })}
                        placeholder="Value"
                        value={tagValue}
                        onChange={(event) => setTagValue(event.target.value)}
                        css={{ flex: 1 }}
                      />
                      <Button
                        componentId="mlflow.skill_registry.register_modal.tag_add"
                        icon={<PlusIcon />}
                        aria-label={intl.formatMessage({
                          defaultMessage: 'Add tag',
                          description: 'Aria label for adding a skill tag',
                        })}
                        disabled={!tagKey.trim()}
                        onClick={addTag}
                      />
                    </div>
                    <Typography.Hint>
                      <FormattedMessage
                        defaultMessage="Key/value metadata you can filter the registry by. Editable later from the skill page."
                        description="Hint for skill registration tags"
                      />
                    </Typography.Hint>
                    {Object.keys(tags).length > 0 && (
                      <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
                        {Object.entries(tags).map(([key, value]) => (
                          <KeyValueTag
                            key={key}
                            isClosable
                            tag={{ key, value }}
                            onClose={() =>
                              setTags((current) => {
                                const next = { ...current };
                                delete next[key];
                                return next;
                              })
                            }
                          />
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {validationError && <FormUI.Message type="error" message={errorMessage(validationError)} />}
          {presentationError && <FormUI.Message type="error" message={presentationError} />}
          {error && (
            <FormUI.Message
              type="error"
              message={
                error.message || (
                  <FormattedMessage
                    defaultMessage="Could not register the skill."
                    description="Fallback error when skill registration fails"
                  />
                )
              }
            />
          )}
        </div>
      )}
    </Modal>
  );
};

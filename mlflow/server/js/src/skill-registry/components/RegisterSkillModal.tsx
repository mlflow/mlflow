import { useEffect, useRef, useState } from 'react';
import {
  Alert,
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  FormUI,
  Input,
  Modal,
  Radio,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { defineMessages, FormattedMessage, useIntl } from 'react-intl';

import Utils from '../../common/utils/Utils';
import { SkillIconEditor } from './SkillIconEditor';
import { SkillTagsInput } from './SkillTagsInput';
import { RegisterSkillApiView, RepositoryImportHint } from './RegisterSkillApiView';
import { SkillRegistryApi } from '../api';
import { useRegisterSkillMutation, type RegisterSkillMutationInput } from '../hooks/useRegisterSkillMutation';
import { findSkillManifest, packageSkillFolder, readSkillManifest } from '../localSkillFolder';
import {
  buildExternalSkillVersionRequest,
  buildUploadedSkillVersionRequest,
  parseSkillIdentityInput,
  parseSkillLocation,
  toRegisterSkillRequest,
  type SkillRegistrationErrorCode,
  type SkillRegistrationFields,
  type SkillRegistrationSourceType,
} from '../sourceLocation';
import { formatSkillImportCli, type SkillImportSnippetOptions, type SkillRegisterSnippetOptions } from '../snippets';
import { SkillStatus, type RegistryIcon, type SkillVersion } from '../types';
import { formatSkillIdentity, formatSkillSourceLabel } from '../utils';

type RegistrationMode = 'pointer' | 'upload';

interface RegisterSkillModalProps {
  visible: boolean;
  onClose: () => void;
  /** Set when adding a version to an existing skill. Identity stays fixed. */
  skill?: { name: string; organization: string };
  /** Version whose source is copied into the form. Status stays Active. */
  sourceVersion?: SkillVersion;
  onRegistered: (version: SkillVersion) => void;
}

const EMPTY_FORM: SkillRegistrationFields = {
  location: '',
  identity: '',
  sourceTypeOverride: '',
  ref: '',
  subpath: '',
  status: SkillStatus.ACTIVE,
};

// A Record keyed by every error code, so adding a code without a message fails type checking.
const REGISTRATION_ERROR_MESSAGES: Record<SkillRegistrationErrorCode, { defaultMessage: string; description: string }> =
  defineMessages({
    location_required: {
      defaultMessage: 'Enter a source location.',
      description: 'Validation error when skill registration has no source URL',
    },
    name_required: {
      defaultMessage: 'Enter a skill name.',
      description: 'Validation error when skill registration has no name',
    },
    name_invalid: {
      defaultMessage: 'Names must be lowercase letters, digits, and single hyphens, as in @my-org/my-skill.',
      description: 'Validation error for a skill registration name',
    },
    organization_invalid: {
      defaultMessage: 'Organization names must be lowercase letters, digits, hyphens, and periods.',
      description: 'Validation error for a skill registration organization',
    },
    source_type_required: {
      defaultMessage: 'Choose a source type in Advanced settings. The location is not a Git, OCI, or ZIP URL.',
      description: 'Validation error when a skill source type cannot be inferred',
    },
    source_type_conflict: {
      defaultMessage: 'That source type does not match this location.',
      description: 'Validation error when a skill source type override contradicts the URL',
    },
    credentials: {
      defaultMessage: 'Remove credentials from the URL. They would be stored with the skill.',
      description: 'Validation error when a skill source URL contains credentials',
    },
    zip_scheme: {
      defaultMessage: 'A ZIP source must be an http(s) URL.',
      description: 'Validation error for a non-HTTP skill ZIP source',
    },
    ref_not_git: {
      defaultMessage: 'Ref is only used for Git sources.',
      description: 'Validation error when a skill ref is set for a non-Git source',
    },
    source_invalid: {
      defaultMessage: 'Enter a remote Git, OCI, or ZIP location.',
      description: 'Validation error for an invalid skill source location',
    },
    source_too_long: {
      defaultMessage: 'The location, ref, or path is too long.',
      description: 'Validation error when a skill source field exceeds the length limit',
    },
    status: {
      defaultMessage: 'New versions can be Active or Draft.',
      description: 'Validation error when a new skill version status is not active or draft',
    },
    skill_md_required: {
      defaultMessage: 'Select the directory containing SKILL.md.',
      description: 'Validation error when a skill folder upload has no SKILL.md',
    },
  });

const clientSourceType = (sourceType: string | null | undefined): '' | SkillRegistrationSourceType =>
  sourceType === 'git' || sourceType === 'oci' || sourceType === 'zip' ? sourceType : '';

const initialForm = (sourceVersion?: SkillVersion): SkillRegistrationFields =>
  sourceVersion
    ? {
        ...EMPTY_FORM,
        location: sourceVersion.source ?? '',
        sourceTypeOverride: clientSourceType(sourceVersion.source_type),
        ref: sourceVersion.ref ?? '',
        subpath: sourceVersion.subpath ?? '',
      }
    : EMPTY_FORM;

// The dialog is only mounted while visible, so every open starts from fresh state.
export const RegisterSkillModal = (props: RegisterSkillModalProps) =>
  props.visible ? <RegisterSkillDialog {...props} /> : null;

const RegisterSkillDialog = ({ onClose, skill, sourceVersion, onRegistered }: RegisterSkillModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isVersion = Boolean(skill);
  const fixedIdentity = skill ? formatSkillIdentity(skill.name, skill.organization) : '';
  const [view, setView] = useState<'form' | 'api'>('form');
  const [mode, setMode] = useState<RegistrationMode>('pointer');
  const [advancedOpen, setAdvancedOpen] = useState(Boolean(sourceVersion));
  const [form, setForm] = useState(() => initialForm(sourceVersion));
  const [description, setDescription] = useState('');
  const [icons, setIcons] = useState<RegistryIcon[]>([]);
  const [tags, setTags] = useState<Record<string, string>>({});
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const [identityTouched, setIdentityTouched] = useState(false);
  const [descriptionTouched, setDescriptionTouched] = useState(false);
  const [refTouched, setRefTouched] = useState(false);
  const [subpathTouched, setSubpathTouched] = useState(false);
  const [validationError, setValidationError] = useState<SkillRegistrationErrorCode>();
  const [packageError, setPackageError] = useState<string>();
  const [submitting, setSubmitting] = useState(false);
  const submitErrorRef = useRef<HTMLDivElement>(null);
  // Async work outliving a cancel must not close or navigate a dialog the user already left.
  const closedRef = useRef(false);
  const { mutateAsync, error } = useRegisterSkillMutation();

  useEffect(() => {
    if (!validationError && !packageError && !error) return;
    submitErrorRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [validationError, packageError, error]);

  const parsed = parseSkillLocation(form.location);
  const effectiveSourceType = form.sourceTypeOverride || parsed?.sourceType;
  const hasSkillManifest = Boolean(findSkillManifest(folderFiles));
  const ref = form.ref.trim();
  const subpath = form.subpath.trim();

  const close = () => {
    closedRef.current = true;
    onClose();
  };

  const applyLocation = (location: string) => {
    const nextParsed = parseSkillLocation(location);
    setForm((current) => ({
      ...current,
      location,
      identity:
        !isVersion && !identityTouched && nextParsed?.suggestedName
          ? formatSkillIdentity(nextParsed.suggestedName, nextParsed.suggestedOrganization)
          : current.identity,
      ref: refTouched ? current.ref : (nextParsed?.ref ?? ''),
      subpath: subpathTouched ? current.subpath : (nextParsed?.subpath ?? ''),
    }));
    setValidationError(undefined);
  };

  const onFolderSelected = async (files: File[]) => {
    setFolderFiles(files);
    setValidationError(undefined);
    const manifestFile = findSkillManifest(files);
    if (!manifestFile || isVersion) return;
    const manifest = readSkillManifest(await manifestFile.text());
    if (manifest.name && !identityTouched) {
      setForm((current) => ({ ...current, identity: manifest.name ?? current.identity }));
    }
    if (manifest.description && !descriptionTouched) {
      setDescription(manifest.description);
    }
  };

  // Description, icons and tags are not part of the register payload; they use their own endpoints.
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
    await Promise.all(
      Object.entries(tags).map(([key, value]) =>
        SkillRegistryApi.setSkillTag(version.name, { key, value }, version.organization),
      ),
    );
  };

  const buildMutationInput = async (): Promise<RegisterSkillMutationInput | undefined> => {
    const fields: SkillRegistrationFields = { ...form, identity: isVersion ? fixedIdentity : form.identity };
    if (mode === 'pointer') {
      const built = buildExternalSkillVersionRequest(fields);
      if (!built.ok) {
        setValidationError(built.error);
        return undefined;
      }
      return skill
        ? { kind: 'version', name: skill.name, organization: skill.organization, request: built.request }
        : { kind: 'register', request: toRegisterSkillRequest(built.request, built.identity) };
    }
    if (!hasSkillManifest) {
      setValidationError('skill_md_required');
      return undefined;
    }
    const built = buildUploadedSkillVersionRequest(fields);
    if (!built.ok) {
      setValidationError(built.error);
      return undefined;
    }
    const content = await packageSkillFolder(folderFiles);
    return skill
      ? { kind: 'version-upload', name: skill.name, organization: skill.organization, request: built.request, content }
      : { kind: 'register-upload', request: toRegisterSkillRequest(built.request, built.identity), content };
  };

  const submit = async () => {
    if (view !== 'form' || submitting) return;
    setValidationError(undefined);
    setPackageError(undefined);
    setSubmitting(true);
    let input: RegisterSkillMutationInput | undefined;
    try {
      input = await buildMutationInput();
    } catch (packagingFailure) {
      setPackageError(
        packagingFailure instanceof Error
          ? packagingFailure.message
          : intl.formatMessage({
              defaultMessage: 'Could not package the folder.',
              description: 'Error when a skill folder cannot be packaged for upload',
            }),
      );
    }
    if (!input || closedRef.current) {
      setSubmitting(false);
      return;
    }
    let version: SkillVersion;
    try {
      version = await mutateAsync(input);
    } catch {
      // The mutation error is rendered from `error`.
      setSubmitting(false);
      return;
    }
    try {
      await savePresentation(version);
    } catch (presentationFailure) {
      Utils.displayGlobalErrorNotification(
        presentationFailure instanceof Error
          ? presentationFailure.message
          : intl.formatMessage({
              defaultMessage: 'The version was created, but its description, icon, or tags could not be saved.',
              description: 'Error when skill presentation metadata fails after registration',
            }),
      );
    }
    if (closedRef.current) return;
    close();
    onRegistered(version);
  };

  const locationLabel = intl.formatMessage({
    defaultMessage: 'Location',
    description: 'Label for the skill registration source URL',
  });
  const nameLabel = intl.formatMessage({
    defaultMessage: 'Name',
    description: 'Label for the skill registration name',
  });
  const formIdentity = form.identity.trim() ? parseSkillIdentityInput(form.identity) : undefined;
  const snippetIdentity: { name?: string; organization?: string } = skill
    ? { name: skill.name, organization: skill.organization }
    : formIdentity && !('error' in formIdentity)
      ? formIdentity
      : {};
  const registerSnippet: SkillRegisterSnippetOptions = {
    sourceType: effectiveSourceType,
    location: parsed?.source ?? form.location,
    local: mode === 'upload',
    ref: ref || undefined,
    subpath: subpath || undefined,
    status: form.status,
    ...snippetIdentity,
  };
  // A Git source with no subpath points at a whole repository, which usually holds many skills.
  const repositoryImport: SkillImportSnippetOptions | undefined =
    !isVersion && mode === 'pointer' && effectiveSourceType === 'git' && parsed?.repositoryUrl && !subpath
      ? { source: parsed.repositoryUrl, ref: ref || undefined }
      : undefined;
  const locationSummary = parsed
    ? [
        `${formatSkillSourceLabel(effectiveSourceType)} ${parsed.source}`,
        ref &&
          intl.formatMessage(
            { defaultMessage: 'branch {ref}', description: 'Git ref in the skill location summary' },
            { ref },
          ),
        subpath &&
          intl.formatMessage(
            { defaultMessage: 'path {subpath}', description: 'Subpath in the skill location summary' },
            { subpath },
          ),
      ]
        .filter(Boolean)
        .join(' · ')
    : undefined;
  const apiLink = (
    <Button componentId="mlflow.skill_registry.register_modal.api_link" type="link" onClick={() => setView('api')}>
      <FormattedMessage
        defaultMessage="create through API →"
        description="Link from skill registration to the API example"
      />
    </Button>
  );
  const submitError = validationError
    ? intl.formatMessage(REGISTRATION_ERROR_MESSAGES[validationError])
    : packageError || error?.message;

  return (
    <Modal
      componentId="mlflow.skill_registry.register_modal"
      title={
        isVersion ? (
          <FormattedMessage defaultMessage="Create skill version" description="Title for adding a skill version" />
        ) : (
          <FormattedMessage defaultMessage="Create skill" description="Title for registering a skill" />
        )
      }
      visible
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
            loading={submitting}
            disabled={view === 'api' || submitting || (mode === 'upload' && !hasSkillManifest)}
            onClick={() => void submit()}
          >
            <FormattedMessage defaultMessage="Create" description="Submit button for skill registration" />
          </Button>
        </div>
      }
    >
      {view === 'api' ? (
        <RegisterSkillApiView
          register={registerSnippet}
          repositoryImport={repositoryImport && { ...repositoryImport, organization: snippetIdentity.organization }}
          onBack={() => setView('form')}
        />
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {(submitError || error) && (
            <div ref={submitErrorRef}>
              <Alert
                componentId="mlflow.skill_registry.register_modal.error"
                closable={false}
                type="error"
                message={
                  submitError || (
                    <FormattedMessage
                      defaultMessage="Could not register the skill."
                      description="Fallback error when skill registration fails"
                    />
                  )
                }
              />
            </div>
          )}
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
                      {locationSummary && (
                        <Typography.Hint css={{ display: 'block', marginTop: theme.spacing.xs }}>
                          <FormattedMessage
                            defaultMessage="Registers {summary}"
                            description="Summary of what a skill location registers"
                            values={{ summary: locationSummary }}
                          />
                        </Typography.Hint>
                      )}
                      {repositoryImport && (
                        <div css={{ marginTop: theme.spacing.sm }}>
                          <RepositoryImportHint format="cli" code={formatSkillImportCli(repositoryImport)} />
                        </div>
                      )}
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
                {!identityTouched && form.identity.trim() ? (
                  <FormattedMessage
                    defaultMessage="Filled in from the source. Edit it to rename the skill or change its organization."
                    description="Hint when the skill registration name was suggested from the source"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Group skills with an organization by adding it to the name, e.g. @my-org/my-skill-name."
                    description="Hint for the skill registration name field"
                  />
                )}
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
                        value={effectiveSourceType}
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
                        placeholder={intl.formatMessage({
                          defaultMessage: 'What this skill does and when to use it.',
                          description: 'Placeholder for the skill description editor',
                        })}
                        value={description}
                        onChange={(event) => {
                          setDescriptionTouched(true);
                          setDescription(event.target.value);
                        }}
                        rows={3}
                      />
                    </div>
                    <SkillIconEditor icons={icons} onChange={setIcons} />
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
                {!isVersion && <SkillTagsInput tags={tags} onChange={setTags} />}
              </div>
            )}
          </div>
        </div>
      )}
    </Modal>
  );
};

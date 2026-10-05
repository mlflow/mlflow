import { useEffect, useMemo, useRef, useState } from 'react';
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
import { useArtifactServingEnabled } from '../../experiment-tracking/hooks/useServerInfo';
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
import {
  formatSkillIdentity,
  formatSkillSourceLabel,
  isCommitSha,
  isNotFoundError,
  isPermissionDeniedError,
} from '../utils';

type RegistrationMode = 'pointer' | 'upload';

interface RegisterSkillModalProps {
  visible: boolean;
  onClose: () => void;
  /** Set when adding a version to an existing skill. Identity stays fixed. */
  skill?: { name: string; organization: string };
  /**
   * Version whose source is copied into the form when it can be submitted again. An uploaded version opens the
   * folder picker instead, since its MLflow artifact location is not a source a client can register.
   */
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

// Returns the version's source as form fields only when that source would pass validation as is.
const formFromVersion = (sourceVersion: SkillVersion | undefined, identity: string) => {
  if (!sourceVersion) return undefined;
  const sourceTypeOverride = clientSourceType(sourceVersion.source_type);
  if (!sourceTypeOverride) return undefined;
  const fields: SkillRegistrationFields = {
    ...EMPTY_FORM,
    location: sourceVersion.source ?? '',
    sourceTypeOverride,
    ref: sourceVersion.ref ?? '',
    subpath: sourceVersion.subpath ?? '',
  };
  return buildExternalSkillVersionRequest({ ...fields, identity }).ok ? fields : undefined;
};

// The dialog is only mounted while visible, so every open starts from fresh state.
export const RegisterSkillModal = (props: RegisterSkillModalProps) =>
  props.visible ? <RegisterSkillDialog {...props} /> : null;

const RegisterSkillDialog = ({ onClose, skill, sourceVersion, onRegistered }: RegisterSkillModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isVersion = Boolean(skill);
  const fixedIdentity = skill ? formatSkillIdentity(skill.name, skill.organization) : '';
  const [view, setView] = useState<'form' | 'api'>('form');
  const [seededForm] = useState(() => formFromVersion(sourceVersion, fixedIdentity));
  // RFC-0008 limits MLflow-stored content to servers that serve artifacts; others can only import.
  const uploadEnabled = useArtifactServingEnabled();
  const [mode, setMode] = useState<RegistrationMode>(
    uploadEnabled && sourceVersion?.source_type === 'mlflow' ? 'upload' : 'pointer',
  );
  const [advancedOpen, setAdvancedOpen] = useState(Boolean(seededForm));
  const [form, setForm] = useState(seededForm ?? EMPTY_FORM);
  const [description, setDescription] = useState('');
  const [icons, setIcons] = useState<RegistryIcon[]>([]);
  const [tags, setTags] = useState<Record<string, string>>({});
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  // Server info can answer after the dialog opened in Upload mode; fall back once uploads turn out unsupported.
  if (!uploadEnabled && mode === 'upload') {
    setMode('pointer');
    setFolderFiles([]);
  }
  const [identityTouched, setIdentityTouched] = useState(false);
  const [descriptionTouched, setDescriptionTouched] = useState(false);
  const [refTouched, setRefTouched] = useState(false);
  const [subpathTouched, setSubpathTouched] = useState(false);
  const [validationError, setValidationError] = useState<SkillRegistrationErrorCode>();
  const [buildError, setBuildError] = useState<string>();
  const [takenIdentity, setTakenIdentity] = useState<string>();
  const [submitting, setSubmitting] = useState(false);
  const submitErrorRef = useRef<HTMLDivElement>(null);
  // Async work outliving a cancel must not close or navigate a dialog the user already left.
  const closedRef = useRef(false);
  const { mutateAsync, error } = useRegisterSkillMutation();

  useEffect(() => {
    if (!validationError && !buildError && !error) return;
    submitErrorRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [validationError, buildError, error]);

  const parsed = parseSkillLocation(form.location);
  const effectiveSourceType = form.sourceTypeOverride || parsed?.sourceType;
  const hasSkillManifest = useMemo(() => Boolean(findSkillManifest(folderFiles)), [folderFiles]);
  const ref = form.ref.trim();
  const subpath = form.subpath.trim();

  const showRefSplitWarning =
    effectiveSourceType === 'git' && Boolean(parsed?.refMayIncludePath) && !refTouched && !subpathTouched;

  const close = () => {
    closedRef.current = true;
    onClose();
  };

  // A ref only applies to Git; the field is hidden for other types, so a leftover value must not linger.
  const leavesGit = (sourceType: SkillRegistrationSourceType | '' | undefined) =>
    Boolean(sourceType) && sourceType !== 'git';

  const applyLocation = (location: string) => {
    const nextParsed = parseSkillLocation(location);
    const dropRef = leavesGit(form.sourceTypeOverride || nextParsed?.sourceType);
    if (dropRef) setRefTouched(false);
    setForm((current) => ({
      ...current,
      location,
      identity:
        !isVersion && !identityTouched && nextParsed?.suggestedName
          ? formatSkillIdentity(nextParsed.suggestedName, nextParsed.suggestedOrganization)
          : current.identity,
      ref: dropRef ? '' : refTouched ? current.ref : (nextParsed?.ref ?? ''),
      subpath: subpathTouched ? current.subpath : (nextParsed?.subpath ?? ''),
    }));
    setValidationError(undefined);
    if (!identityTouched) setTakenIdentity(undefined);
  };

  // POST /register reuses an existing skill and silently adds a version to it, so a new skill's name must be
  // confirmed free first. Only a 404 confirms that; a 403 still means the skill exists, and any other failure
  // throws so the create waits until the check succeeds.
  const findTakenIdentity = async (identityInput: string) => {
    const identity = parseSkillIdentityInput(identityInput);
    if ('error' in identity) return undefined;
    try {
      await SkillRegistryApi.getSkill(identity.name, identity.organization);
    } catch (lookupError) {
      if (isNotFoundError(lookupError as Error)) return undefined;
      if (!isPermissionDeniedError(lookupError as Error)) {
        throw new Error(
          intl.formatMessage(
            {
              defaultMessage: "Couldn't check whether this name is already registered: {reason} Try again.",
              description: 'Error when the skill name availability check fails before registration',
            },
            { reason: (lookupError as Error).message },
          ),
        );
      }
    }
    return formatSkillIdentity(identity.name, identity.organization);
  };

  const onFolderSelected = async (files: File[]) => {
    setFolderFiles(files);
    setValidationError(undefined);
    const manifestFile = findSkillManifest(files);
    if (!manifestFile || isVersion) return;
    const manifest = readSkillManifest(await manifestFile.text());
    if (manifest.name && !identityTouched) {
      setForm((current) => ({ ...current, identity: manifest.name ?? current.identity }));
      setTakenIdentity(undefined);
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

  const rejectTakenIdentity = async () => {
    const taken = await findTakenIdentity(form.identity);
    setTakenIdentity(taken);
    return Boolean(taken);
  };

  const buildMutationInput = async (): Promise<RegisterSkillMutationInput | undefined> => {
    const fields: SkillRegistrationFields = { ...form, identity: isVersion ? fixedIdentity : form.identity };
    if (mode === 'pointer') {
      const built = buildExternalSkillVersionRequest(fields);
      if (!built.ok) {
        setValidationError(built.error);
        return undefined;
      }
      if (skill) {
        return { kind: 'version', name: skill.name, organization: skill.organization, request: built.request };
      }
      if (await rejectTakenIdentity()) return undefined;
      return { kind: 'register', request: toRegisterSkillRequest(built.request, built.identity) };
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
    if (!skill && (await rejectTakenIdentity())) return undefined;
    const content = await packageSkillFolder(folderFiles);
    return skill
      ? { kind: 'version-upload', name: skill.name, organization: skill.organization, request: built.request, content }
      : { kind: 'register-upload', request: toRegisterSkillRequest(built.request, built.identity), content };
  };

  const submit = async () => {
    if (view !== 'form' || submitting) return;
    setValidationError(undefined);
    setBuildError(undefined);
    setSubmitting(true);
    let input: RegisterSkillMutationInput | undefined;
    try {
      input = await buildMutationInput();
    } catch (buildFailure) {
      setBuildError(
        buildFailure instanceof Error
          ? buildFailure.message
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
  // The UI cannot see whether a Git location is one skill or a folder of skills, so it always offers the
  // repository import, which discovers every SKILL.md under the path (RFC-0008 `mlflow skills import --subpath`).
  const repositoryImport: SkillImportSnippetOptions | undefined =
    !isVersion && mode === 'pointer' && effectiveSourceType === 'git' && parsed?.repositoryUrl
      ? {
          source: parsed.repositoryUrl,
          ref: ref || undefined,
          subpath: subpath || undefined,
          organization: snippetIdentity.organization || undefined,
        }
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
    : buildError || error?.message;

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
            disabled={
              view === 'api' || submitting || Boolean(takenIdentity) || (mode === 'upload' && !hasSkillManifest)
            }
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
          repositoryImport={repositoryImport}
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
              onChange={(event) => {
                // The folder picker remounts empty, so a folder chosen before switching away must not upload.
                setMode(event.target.value as RegistrationMode);
                setFolderFiles([]);
                setValidationError(undefined);
              }}
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
                      {showRefSplitWarning && (
                        <Typography.Hint css={{ display: 'block', marginTop: theme.spacing.xs }}>
                          <FormattedMessage
                            defaultMessage="GitHub links don't show where a branch name ends, so this assumes the branch is {ref}. If the branch name contains a slash, correct the branch and path under Advanced settings."
                            description="Warning that a GitHub link's branch and path split is a guess"
                            values={{ ref: <code>{ref}</code> }}
                          />
                        </Typography.Hint>
                      )}
                      {repositoryImport && (
                        <div css={{ marginTop: theme.spacing.sm }}>
                          <RepositoryImportHint
                            format="cli"
                            code={formatSkillImportCli(repositoryImport)}
                            underFolder={Boolean(repositoryImport.subpath)}
                          />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </Radio>
              {uploadEnabled && (
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
                        <Typography.Text color="secondary">
                          <FormattedMessage
                            defaultMessage="Up to 25 MB of files, unless your server sets a different limit."
                            description="Hint for the default size limit of an uploaded skill folder"
                          />
                        </Typography.Text>
                      </>
                    )}
                  </div>
                </Radio>
              )}
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
                  setTakenIdentity(undefined);
                  setForm((current) => ({ ...current, identity: event.target.value }));
                }}
                onBlur={() => {
                  // A failed check is retried on submit, which reports it.
                  findTakenIdentity(form.identity).then(
                    (taken) => {
                      if (!closedRef.current) setTakenIdentity(taken);
                    },
                    () => undefined,
                  );
                }}
                validationState={takenIdentity ? 'error' : undefined}
                css={{ width: '100%' }}
              />
              {takenIdentity ? (
                <FormUI.Message
                  type="error"
                  message={
                    <FormattedMessage
                      defaultMessage='A skill named "{name}" is already registered.'
                      description="Error when a new skill's name belongs to an existing skill"
                      values={{ name: takenIdentity }}
                    />
                  }
                />
              ) : (
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
              )}
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
                        onChange={({ target }) => {
                          const sourceTypeOverride = target.value as SkillRegistrationSourceType;
                          if (leavesGit(sourceTypeOverride)) setRefTouched(false);
                          setForm((current) => ({
                            ...current,
                            sourceTypeOverride,
                            ref: leavesGit(sourceTypeOverride) ? '' : current.ref,
                          }));
                        }}
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

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

import { useArtifactServingEnabled, useSkillContentLimits } from '../../experiment-tracking/hooks/useServerInfo';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import { SkillIconEditor } from './SkillIconEditor';
import { SkillTagsInput } from './SkillTagsInput';
import { RegisterSkillApiView, RepositoryImportHint } from './RegisterSkillApiView';
import type { RegisterSkillMutationInput } from '../hooks/useRegisterSkillMutation';
import { findTakenSkillIdentity, useRegisterSkillSubmission } from '../hooks/useRegisterSkillSubmission';
import { exceededContentLimit, findSkillManifest, packageSkillFolder, readSkillManifest } from '../localSkillFolder';
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
import { buildRegistrationSnippets, formatSkillImportCli } from '../snippets';
import { SkillStatus, type RegistryIcon, type SkillVersion } from '../types';
import { formatSkillIdentity, formatSkillSourceLabel, formatSkillStatusLabel } from '../utils';
import { PointerSourceFields } from './PointerSourceFields';
import { UploadFolderField } from './UploadFolderField';

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
  // Content MLflow stores itself needs a server that serves artifacts; other servers can only import.
  const uploadEnabled = useArtifactServingEnabled();
  const workspace = useActiveWorkspace();
  const { maxBytes, maxFiles } = useSkillContentLimits();
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
  const [takenIdentity, setTakenIdentity] = useState<string>();
  // Bumped by every folder selection and source switch, so a slow SKILL.md read cannot apply to a later choice.
  const folderSelectionRef = useRef(0);
  const submitErrorRef = useRef<HTMLDivElement>(null);
  const submission = useRegisterSkillSubmission({ onClose, onRegistered, onNameTaken: setTakenIdentity });

  useEffect(() => {
    if (!validationError && !submission.failed) return;
    submitErrorRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [validationError, submission.failed]);

  const parsed = parseSkillLocation(form.location);
  const effectiveSourceType = form.sourceTypeOverride || parsed?.sourceType;
  const hasSkillManifest = useMemo(() => Boolean(findSkillManifest(folderFiles)), [folderFiles]);
  // Packaging reads every file into memory, so a folder over the server's limits is refused before that.
  const exceededLimit = useMemo(
    () => exceededContentLimit(folderFiles, { maxBytes, maxFiles }),
    [folderFiles, maxBytes, maxFiles],
  );
  const ref = form.ref.trim();
  const subpath = form.subpath.trim();

  const showRefSplitWarning =
    effectiveSourceType === 'git' && Boolean(parsed?.refMayIncludePath) && !refTouched && !subpathTouched;

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

  const changeSourceType = (sourceTypeOverride: SkillRegistrationSourceType) => {
    if (leavesGit(sourceTypeOverride)) setRefTouched(false);
    setForm((current) => ({
      ...current,
      sourceTypeOverride,
      ref: leavesGit(sourceTypeOverride) ? '' : current.ref,
    }));
  };

  // Values filled in from one source, never typed, must not carry over to another source.
  const resetSuggestedFields = (location = '') => {
    if (!identityTouched) {
      const suggestion = parseSkillLocation(location);
      setForm((current) => ({
        ...current,
        identity: suggestion?.suggestedName
          ? formatSkillIdentity(suggestion.suggestedName, suggestion.suggestedOrganization)
          : '',
      }));
      setTakenIdentity(undefined);
    }
    if (!descriptionTouched) setDescription('');
  };

  const changeMode = (nextMode: RegistrationMode) => {
    folderSelectionRef.current += 1;
    // The folder picker remounts empty, so a folder chosen before switching away must not upload.
    setMode(nextMode);
    setFolderFiles([]);
    setValidationError(undefined);
    if (!isVersion) resetSuggestedFields(nextMode === 'pointer' ? form.location : undefined);
  };

  const onFolderSelected = async (files: File[]) => {
    folderSelectionRef.current += 1;
    const selection = folderSelectionRef.current;
    setFolderFiles(files);
    setValidationError(undefined);
    if (isVersion) return;
    resetSuggestedFields();
    const manifestFile = findSkillManifest(files);
    // A folder over the server's limits is refused, so its SKILL.md is not worth reading into memory.
    if (!manifestFile || exceededContentLimit(files, { maxBytes, maxFiles })) return;
    // The cleared fields are filled from SKILL.md only if nothing is typed into them while it is read.
    const identityBefore = identityTouched ? undefined : '';
    const descriptionBefore = descriptionTouched ? undefined : '';
    const manifest = readSkillManifest(await manifestFile.text());
    if (selection !== folderSelectionRef.current || !submission.isActive()) return;
    const { name: manifestName, description: manifestDescription } = manifest;
    if (manifestName) {
      setForm((current) => (current.identity === identityBefore ? { ...current, identity: manifestName } : current));
      setTakenIdentity(undefined);
    }
    if (manifestDescription) {
      setDescription((current) => (current === descriptionBefore ? manifestDescription : current));
    }
  };

  const rejectTakenIdentity = async () => {
    const taken = await findTakenSkillIdentity(form.identity, intl);
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
    // The folder field explains both; Create is disabled meanwhile.
    if (!hasSkillManifest || exceededLimit) return undefined;
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

  const submit = () => {
    if (view !== 'form') return;
    setValidationError(undefined);
    void submission.submit(buildMutationInput, { description, icons, tags });
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
  // A name check can answer after the name changed again, so only a result for the current name counts.
  const nameTaken =
    takenIdentity !== undefined &&
    formIdentity !== undefined &&
    !('error' in formIdentity) &&
    takenIdentity === formatSkillIdentity(formIdentity.name, formIdentity.organization);
  const snippetIdentity: { name?: string; organization?: string } = skill
    ? { name: skill.name, organization: skill.organization }
    : formIdentity && !('error' in formIdentity)
      ? formIdentity
      : {};
  const snippets = buildRegistrationSnippets({
    register: {
      sourceType: effectiveSourceType,
      location: parsed?.source ?? form.location,
      local: mode === 'upload',
      ref: ref || undefined,
      subpath: subpath || undefined,
      status: form.status,
      workspace,
      ...snippetIdentity,
    },
    repositoryUrl: isVersion ? undefined : parsed?.repositoryUrl,
  });
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
    : submission.error;

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
      onCancel={submission.close}
      size="wide"
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button componentId="mlflow.skill_registry.register_modal.cancel" onClick={submission.close}>
            <FormattedMessage defaultMessage="Cancel" description="Cancel skill registration" />
          </Button>
          <Button
            componentId="mlflow.skill_registry.register_modal.submit"
            type="primary"
            loading={submission.submitting}
            disabled={
              view === 'api' ||
              submission.submitting ||
              nameTaken ||
              (mode === 'upload' && (!hasSkillManifest || Boolean(exceededLimit)))
            }
            onClick={submit}
          >
            <FormattedMessage defaultMessage="Create" description="Submit button for skill registration" />
          </Button>
        </div>
      }
    >
      {view === 'api' ? (
        <RegisterSkillApiView
          register={snippets.register}
          repositoryImport={snippets.repositoryImport}
          onBack={() => setView('form')}
        />
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {(submitError || submission.failed) && (
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
              onChange={(event) => changeMode(event.target.value as RegistrationMode)}
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
                      {snippets.repositoryImport && (
                        <div css={{ marginTop: theme.spacing.sm }}>
                          <RepositoryImportHint
                            format="cli"
                            code={formatSkillImportCli(snippets.repositoryImport)}
                            underFolder={Boolean(snippets.repositoryImport.subpath)}
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
                      <UploadFolderField
                        files={folderFiles}
                        hasSkillManifest={hasSkillManifest}
                        exceededLimit={exceededLimit}
                        maxBytes={maxBytes}
                        maxFiles={maxFiles}
                        onSelect={(files) => void onFolderSelected(files)}
                      />
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
                  findTakenSkillIdentity(form.identity, intl).then(
                    (taken) => {
                      if (submission.isActive()) setTakenIdentity(taken);
                    },
                    () => undefined,
                  );
                }}
                validationState={nameTaken ? 'error' : undefined}
                css={{ width: '100%' }}
              />
              {nameTaken ? (
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
                  <PointerSourceFields
                    sourceType={effectiveSourceType}
                    refValue={form.ref}
                    subpath={form.subpath}
                    onSourceTypeChange={changeSourceType}
                    onRefChange={(value) => {
                      setRefTouched(true);
                      setForm((current) => ({ ...current, ref: value }));
                    }}
                    onSubpathChange={(value) => {
                      setSubpathTouched(true);
                      setForm((current) => ({ ...current, subpath: value }));
                    }}
                  />
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
                      {formatSkillStatusLabel(intl, SkillStatus.ACTIVE)}
                    </SimpleSelectOption>
                    <SimpleSelectOption value={SkillStatus.DRAFT}>
                      {formatSkillStatusLabel(intl, SkillStatus.DRAFT)}
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

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

import { useArtifactServingEnabled, useSkillContentLimits } from '../../experiment-tracking/hooks/useServerInfo';
import { useTrackingSnippetConnection } from '../../common/utils/trackingSnippetConnection';
import { SkillIconEditor } from './SkillIconEditor';
import { SkillTagsInput } from './SkillTagsInput';
import { RegisterSkillApiView, RepositoryImportHint } from './RegisterSkillApiView';
import type { RegisterSkillMutationInput } from '../hooks/useRegisterSkillMutation';
import { findTakenSkillIdentity, useRegisterSkillSubmission } from '../hooks/useRegisterSkillSubmission';
import {
  exceededContentLimit,
  discoverSkillFolders,
  findSkillManifest,
  packageSkillFolder,
  readSkillManifest,
  type LocalSkillChoice,
} from '../localSkillFolder';
import {
  buildExternalSkillVersionRequest,
  buildUploadedSkillVersionRequest,
  isValidSkillName,
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
  const snippetConnection = useTrackingSnippetConnection();
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
  const [folderChoices, setFolderChoices] = useState<LocalSkillChoice[]>([]);
  const [selectedRoots, setSelectedRoots] = useState<string[]>([]);
  const [completedRoots, setCompletedRoots] = useState<string[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [batchSubmitting, setBatchSubmitting] = useState(false);
  const batchRunningRef = useRef(false);
  const [batchError, setBatchError] = useState<string>();
  const [organization, setOrganization] = useState('');
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
  const selectedChoices = folderChoices.filter((choice) => selectedRoots.includes(choice.rootPath));
  const selectedChoice = selectedChoices.length === 1 ? selectedChoices[0] : undefined;
  const manifestName = selectedChoice?.manifest?.name;
  const multipleSkills = mode === 'upload' && !isVersion && selectedChoices.length > 1;
  const submitting = submission.submitting || batchSubmitting;
  const selectionLocked = submitting || completedRoots.length > 0;
  const duplicateNames = new Set(selectedChoices.map((choice) => choice.manifest?.name)).size < selectedChoices.length;
  const folderReady =
    !discovering &&
    selectedChoices.length > 0 &&
    !duplicateNames &&
    selectedChoices.every((choice) => choice.manifest && !choice.problem && !choice.exceededLimit);
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
    setFolderChoices([]);
    setSelectedRoots([]);
    setDiscovering(false);
    setBatchError(undefined);
    setValidationError(undefined);
    if (!isVersion) resetSuggestedFields(nextMode === 'pointer' ? form.location : undefined);
  };

  const onFolderSelected = async (files: File[]) => {
    folderSelectionRef.current += 1;
    const selection = folderSelectionRef.current;
    setFolderFiles(files);
    setFolderChoices([]);
    setSelectedRoots([]);
    setDiscovering(true);
    setBatchError(undefined);
    setValidationError(undefined);
    if (!isVersion) resetSuggestedFields();
    const choices = await Promise.all(
      discoverSkillFolders(files).map(async (folder): Promise<LocalSkillChoice> => {
        const choice: LocalSkillChoice = {
          ...folder,
          exceededLimit: exceededContentLimit(folder.files, { maxBytes, maxFiles }),
        };
        // Limits apply to each skill, before reading any of its files into memory.
        if (choice.exceededLimit) return choice;
        const manifestFile = findSkillManifest(folder.files, folder.rootPath);
        if (!manifestFile) return { ...choice, problem: 'unreadable' };
        let text: string;
        try {
          text = await manifestFile.text();
        } catch {
          return { ...choice, problem: 'unreadable' };
        }
        try {
          const manifest = readSkillManifest(text);
          return {
            ...choice,
            manifest,
            problem: !manifest.name ? 'missing' : isValidSkillName(manifest.name) ? undefined : 'invalid',
          };
        } catch {
          return { ...choice, problem: 'unparsable' };
        }
      }),
    );
    if (selection !== folderSelectionRef.current || !submission.isActive()) return;
    setDiscovering(false);
    setFolderChoices(choices);
    if (choices.length === 1) {
      setSelectedRoots([choices[0].rootPath]);
      suggestFolderFields(choices[0]);
    }
  };

  const suggestFolderFields = (choice: LocalSkillChoice | undefined) => {
    if (isVersion || !choice?.manifest) return;
    const { name: suggestedName, description: manifestDescription } = choice.manifest;
    if (suggestedName) {
      setForm((current) =>
        !identityTouched && current.identity === '' ? { ...current, identity: suggestedName } : current,
      );
      setTakenIdentity(undefined);
    }
    if (manifestDescription) {
      setDescription((current) => (!descriptionTouched && current === '' ? manifestDescription : current));
    }
  };

  const selectFolders = (roots: string[]) => {
    setSelectedRoots(roots);
    setTakenIdentity(undefined);
    setBatchError(undefined);
    setValidationError(undefined);
    if (!isVersion) resetSuggestedFields();
    if (roots.length === 1) suggestFolderFields(folderChoices.find((choice) => choice.rootPath === roots[0]));
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
    // The folder field explains what is wrong; Create is disabled meanwhile.
    if (!folderReady || !selectedChoice) return undefined;
    const built = buildUploadedSkillVersionRequest(fields);
    if (!built.ok) {
      setValidationError(built.error);
      return undefined;
    }
    if (!skill && (await rejectTakenIdentity())) return undefined;
    const content = await packageSkillFolder(selectedChoice.files, selectedChoice.rootPath);
    return skill
      ? { kind: 'version-upload', name: skill.name, organization: skill.organization, request: built.request, content }
      : { kind: 'register-upload', request: toRegisterSkillRequest(built.request, built.identity), content };
  };

  const submitBatch = async () => {
    if (batchRunningRef.current || !folderReady) return;
    batchRunningRef.current = true;
    setBatchSubmitting(true);
    setBatchError(undefined);
    setTakenIdentity(undefined);
    const pending = selectedChoices.filter((choice) => !completedRoots.includes(choice.rootPath));
    try {
      // Check every name before creating anything. Retrying a partial batch only checks the remaining names.
      const inputs: {
        choice: LocalSkillChoice;
        input: Omit<Extract<RegisterSkillMutationInput, { kind: 'register-upload' }>, 'content'>;
      }[] = [];
      for (const choice of pending) {
        if (!submission.isActive()) return;
        const name = choice.manifest?.name;
        if (!name) return;
        const identity = formatSkillIdentity(name, organization.trim());
        const built = buildUploadedSkillVersionRequest({ ...form, identity });
        if (!built.ok) {
          setValidationError(built.error);
          return;
        }
        const taken = await findTakenSkillIdentity(identity, intl);
        if (!submission.isActive()) return;
        if (taken) {
          setTakenIdentity(taken);
          return;
        }
        inputs.push({
          choice,
          input: { kind: 'register-upload', request: toRegisterSkillRequest(built.request, built.identity) },
        });
      }
      for (const { choice, input } of inputs) {
        if (!submission.isActive()) return;
        const version = await submission.submit(
          async () => ({ ...input, content: await packageSkillFolder(choice.files, choice.rootPath) }),
          { description: choice.manifest?.description ?? '', icons: [], tags },
          { closeOnSuccess: false },
        );
        if (!submission.isActive() || !version) return;
        setCompletedRoots((current) => [...current, choice.rootPath]);
      }
      // Each successful registration refreshed the catalog. Keep the user there after creating several skills.
      if (submission.isActive()) submission.close();
    } catch (error) {
      if (submission.isActive()) setBatchError((error as Error).message);
    } finally {
      batchRunningRef.current = false;
      if (submission.isActive()) setBatchSubmitting(false);
    }
  };

  const submit = () => {
    if (view !== 'form') return;
    setValidationError(undefined);
    if (multipleSkills) {
      void submitBatch();
      return;
    }
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
      ...snippetConnection,
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
    <Button
      componentId="mlflow.skill_registry.register_modal.api_link"
      type="link"
      disabled={selectionLocked}
      onClick={() => setView('api')}
      css={{ fontSize: 'inherit', lineHeight: 'inherit !important', verticalAlign: 'baseline !important' }}
    >
      <FormattedMessage
        defaultMessage="create through API →"
        description="Link from skill registration to the API example"
      />
    </Button>
  );
  const submitError =
    batchError ??
    (validationError
      ? intl.formatMessage(REGISTRATION_ERROR_MESSAGES[validationError])
      : multipleSkills && takenIdentity
        ? intl.formatMessage(
            {
              defaultMessage: 'A skill named "{name}" is already registered.',
              description: "Error when a new skill's name belongs to an existing skill",
            },
            { name: takenIdentity },
          )
        : submission.error);

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
            loading={submitting}
            disabled={
              view === 'api' || submitting || (!multipleSkills && nameTaken) || (mode === 'upload' && !folderReady)
            }
            onClick={submit}
          >
            {completedRoots.length > 0 ? (
              <FormattedMessage
                defaultMessage="Retry remaining"
                description="Retry a partially completed local skill registration"
              />
            ) : multipleSkills ? (
              <FormattedMessage
                defaultMessage="Create {count} skills"
                description="Submit button for registering several local skills"
                values={{ count: selectedChoices.length }}
              />
            ) : (
              <FormattedMessage defaultMessage="Create" description="Submit button for skill registration" />
            )}
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
          {completedRoots.length > 0 && (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="{completed} of {total} skills registered. Retry skips the completed skills."
                description="Progress after a local skill registration partially succeeds"
                values={{ completed: completedRoots.length, total: selectedChoices.length }}
              />
            </Typography.Text>
          )}
          <fieldset
            disabled={selectionLocked}
            css={{
              border: 0,
              margin: 0,
              padding: 0,
              minWidth: 0,
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.md,
            }}
          >
            <Typography.Text color="secondary">
              {isVersion ? (
                <FormattedMessage
                  defaultMessage="Adding a version to {identity}, or {apiLink}"
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
                <FormattedMessage
                  defaultMessage="Source"
                  description="Label for the skill registration source choice"
                />
              </FormUI.Label>
              <Radio.Group
                id="mlflow.skill_registry.register_modal.mode"
                componentId="mlflow.skill_registry.register_modal.mode"
                name="mlflow.skill_registry.register_modal.mode"
                value={mode}
                disabled={selectionLocked}
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
                    </div>
                  </Radio>
                )}
              </Radio.Group>
            </div>
            {mode === 'upload' && (
              <UploadFolderField
                files={selectedChoice?.files ?? (folderChoices.length === 0 ? folderFiles : [])}
                hasSkillManifest={folderChoices.length > 0}
                exceededLimit={selectedChoice?.exceededLimit}
                manifestName={manifestName}
                manifestProblem={selectedChoice?.problem}
                maxBytes={maxBytes}
                maxFiles={maxFiles}
                onSelect={(files) => void onFolderSelected(files)}
                choices={folderChoices}
                selectedRoots={selectedRoots}
                completedRoots={completedRoots}
                onSelectionChange={selectFolders}
                isVersion={isVersion}
                discovering={discovering}
                disabled={selectionLocked}
              />
            )}
            {duplicateNames && (
              <FormUI.Message
                type="error"
                message={
                  <FormattedMessage
                    defaultMessage="Selected skills have the same name in SKILL.md. Select only one skill for each name."
                    description="Error when discovered skills would create duplicate registry entries"
                  />
                }
              />
            )}
            {multipleSkills && (
              <div>
                <FormUI.Label htmlFor="mlflow.skill_registry.register_modal.organization">
                  <FormattedMessage
                    defaultMessage="Organization (optional)"
                    description="Label for the organization shared by locally imported skills"
                  />
                </FormUI.Label>
                <Input
                  id="mlflow.skill_registry.register_modal.organization"
                  componentId="mlflow.skill_registry.register_modal.organization"
                  value={organization}
                  onChange={(event) => {
                    setOrganization(event.target.value);
                    setTakenIdentity(undefined);
                    setValidationError(undefined);
                  }}
                />
                <Typography.Hint css={{ display: 'block', marginTop: theme.spacing.xs }}>
                  <FormattedMessage
                    defaultMessage="Names and descriptions come from each SKILL.md. Organization, status, and tags apply to all selected skills."
                    description="Hint for metadata applied to multiple local skill registrations"
                  />
                </Typography.Hint>
              </div>
            )}

            {!isVersion && !multipleSkills && (
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
                  {!isVersion && !multipleSkills && (
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
                      disabled={selectionLocked}
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
          </fieldset>
        </div>
      )}
    </Modal>
  );
};

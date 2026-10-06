import { useEffect, useRef, useState } from 'react';
import { useIntl, type IntlShape } from 'react-intl';

import Utils from '../../common/utils/Utils';
import { SkillRegistryApi } from '../api';
import { parseSkillIdentityInput } from '../sourceLocation';
import type { RegistryIcon, SkillVersion } from '../types';
import { formatSkillIdentity, isNotFoundError, isPermissionDeniedError, settleAll } from '../utils';
import { useInvalidateSkillQueries } from './useInvalidateSkillQueries';
import { useRegisterSkillMutation, type RegisterSkillMutationInput } from './useRegisterSkillMutation';

/** What a new skill shows besides its first version. None of it is part of the register payload. */
export interface SkillPresentation {
  description: string;
  icons: RegistryIcon[];
  tags: Record<string, string>;
}

/**
 * The formatted identity when a skill with this name already exists, or undefined when the name is free.
 * POST /register reuses an existing skill and silently adds a version to it, so a new skill's name must be
 * confirmed free first. Only a 404 confirms that; a 403 still means the skill exists, and any other failure
 * throws so the create waits until the check succeeds.
 */
export const findTakenSkillIdentity = async (identityInput: string, intl: IntlShape) => {
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

const createRequest = (name: string, organization: string | undefined, { description, icons }: SkillPresentation) => {
  const trimmedDescription = description.trim();
  const savedIcons = icons.filter((icon) => icon.src.trim());
  return {
    name,
    ...(organization ? { organization } : {}),
    ...(trimmedDescription ? { description: trimmedDescription } : {}),
    ...(savedIcons.length ? { icons: savedIcons } : {}),
  };
};

/**
 * Runs a registration from the dialog. A new skill is created first, with its description and icons, so a name
 * taken in the meantime fails here instead of adding a version to someone else's skill; then the version is
 * registered and the tags are set. The skill queries are refreshed once at the end before the version is handed
 * over. Work that outlives the dialog, whether it was cancelled or unmounted by navigation, never closes it or
 * navigates.
 */
export const useRegisterSkillSubmission = ({
  onClose,
  onRegistered,
  onNameTaken,
}: {
  onClose: () => void;
  onRegistered: (version: SkillVersion) => void;
  /** Called with the formatted identity when creating a new skill finds the name already registered. */
  onNameTaken: (identity: string) => void;
}) => {
  const intl = useIntl();
  const [submitting, setSubmitting] = useState(false);
  const [failure, setFailure] = useState<string>();
  const { mutateAsync, error: registerError, reset } = useRegisterSkillMutation();
  const invalidate = useInvalidateSkillQueries();
  const activeRef = useRef(true);

  useEffect(() => {
    activeRef.current = true;
    return () => {
      activeRef.current = false;
    };
  }, []);

  const isActive = () => activeRef.current;

  const close = () => {
    activeRef.current = false;
    onClose();
  };

  const messageOf = (error: unknown, fallback: string) => (error instanceof Error ? error.message : fallback);

  // True when the skill now exists empty; false (after reporting why) when it could not be created.
  const createSkill = async (name: string, organization: string | undefined, presentation: SkillPresentation) => {
    try {
      await SkillRegistryApi.createSkill(createRequest(name, organization, presentation));
      return true;
    } catch (createFailure) {
      // The create's error code does not reach the client, so a conflict is recognized by looking the name up.
      const taken = await findTakenSkillIdentity(formatSkillIdentity(name, organization), intl).catch(() => undefined);
      if (taken) {
        onNameTaken(taken);
      } else {
        setFailure(messageOf(createFailure, String(createFailure)));
      }
      return false;
    }
  };

  const submit = async (
    buildInput: () => Promise<RegisterSkillMutationInput | undefined>,
    presentation: SkillPresentation,
  ) => {
    if (submitting) return;
    setFailure(undefined);
    reset();
    setSubmitting(true);
    let input: RegisterSkillMutationInput | undefined;
    try {
      input = await buildInput();
    } catch (buildFailure) {
      setFailure(
        messageOf(
          buildFailure,
          intl.formatMessage({
            defaultMessage: 'Could not package the folder.',
            description: 'Error when a skill folder cannot be packaged for upload',
          }),
        ),
      );
    }
    if (!input || !isActive()) {
      setSubmitting(false);
      return;
    }
    const newSkill = input.kind === 'register' || input.kind === 'register-upload' ? input.request : undefined;
    if (newSkill && !(await createSkill(newSkill.name, newSkill.organization, presentation))) {
      setSubmitting(false);
      return;
    }
    let version: SkillVersion;
    try {
      version = await mutateAsync(input);
    } catch {
      // The registration error is reported through `error`. A skill created above stays empty, which would make
      // a retry find its name taken, so it is removed again.
      if (newSkill) await SkillRegistryApi.deleteSkill(newSkill.name, newSkill.organization).catch(() => undefined);
      setSubmitting(false);
      return;
    }
    if (newSkill) {
      // The skill exists now, so its tags are saved even if the dialog has gone away meanwhile.
      try {
        await settleAll(
          Object.entries(presentation.tags).map(([key, value]) =>
            SkillRegistryApi.setSkillTag(version.name, { key, value }, version.organization),
          ),
        );
      } catch (tagFailure) {
        Utils.displayGlobalErrorNotification(
          intl.formatMessage(
            {
              defaultMessage: 'The skill was created, but its tags could not be saved: {reason}',
              description: 'Error when saving a new skill tags fails after registration',
            },
            { reason: messageOf(tagFailure, String(tagFailure)) },
          ),
        );
      }
    }
    await invalidate(version.name, version.organization);
    if (!isActive()) return;
    close();
    onRegistered(version);
  };

  return {
    submit,
    submitting,
    /** The failure to show, if any: building the request, creating the skill, or registering the version. */
    error: failure ?? registerError?.message,
    failed: Boolean(failure || registerError),
    close,
    isActive,
  };
};

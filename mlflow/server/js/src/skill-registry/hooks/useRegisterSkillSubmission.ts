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

const saveSkillPresentation = async (version: SkillVersion, { description, icons, tags }: SkillPresentation) => {
  const trimmedDescription = description.trim();
  const savedIcons = icons.filter((icon) => icon.src.trim());
  await settleAll([
    ...(trimmedDescription || savedIcons.length
      ? [
          SkillRegistryApi.updateSkill(
            version.name,
            {
              ...(trimmedDescription ? { description: trimmedDescription } : {}),
              ...(savedIcons.length ? { icons: savedIcons } : {}),
            },
            version.organization,
          ),
        ]
      : []),
    ...Object.entries(tags).map(([key, value]) =>
      SkillRegistryApi.setSkillTag(version.name, { key, value }, version.organization),
    ),
  ]);
};

/**
 * Runs a registration from the dialog: builds the request, registers it, saves a new skill's description,
 * icons and tags, refreshes the skill queries, then hands the version over. Work that outlives the dialog,
 * whether it was cancelled or unmounted by navigation, never closes it or navigates.
 */
export const useRegisterSkillSubmission = ({
  onClose,
  onRegistered,
}: {
  onClose: () => void;
  onRegistered: (version: SkillVersion) => void;
}) => {
  const intl = useIntl();
  const [submitting, setSubmitting] = useState(false);
  const [buildError, setBuildError] = useState<string>();
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

  const submit = async (
    buildInput: () => Promise<RegisterSkillMutationInput | undefined>,
    presentation?: SkillPresentation,
  ) => {
    if (submitting) return;
    setBuildError(undefined);
    reset();
    setSubmitting(true);
    let input: RegisterSkillMutationInput | undefined;
    try {
      input = await buildInput();
    } catch (failure) {
      setBuildError(
        failure instanceof Error
          ? failure.message
          : intl.formatMessage({
              defaultMessage: 'Could not package the folder.',
              description: 'Error when a skill folder cannot be packaged for upload',
            }),
      );
    }
    if (!input || !isActive()) {
      setSubmitting(false);
      return;
    }
    let version: SkillVersion;
    try {
      version = await mutateAsync(input);
    } catch {
      // The registration error is reported through `error`.
      setSubmitting(false);
      return;
    }
    // The skill exists now, so its presentation is saved even if the dialog has gone away meanwhile.
    if (presentation) {
      try {
        await saveSkillPresentation(version, presentation);
      } catch (failure) {
        Utils.displayGlobalErrorNotification(
          intl.formatMessage(
            {
              defaultMessage: 'The skill was created, but its description, icons, or tags could not be saved: {reason}',
              description: 'Error when saving a new skill description, icons, or tags fails after registration',
            },
            { reason: failure instanceof Error ? failure.message : String(failure) },
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
    /** The build or registration failure to show, if any. */
    error: buildError ?? registerError?.message,
    failed: Boolean(buildError || registerError),
    close,
    isActive,
  };
};

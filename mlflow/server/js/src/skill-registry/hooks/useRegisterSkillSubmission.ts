import { useEffect, useRef, useState } from 'react';
import { useIntl, type IntlShape } from 'react-intl';

import Utils from '../../common/utils/Utils';
import { SkillRegistryApi } from '../api';
import { parseSkillIdentityInput } from '../sourceLocation';
import type { RegistryIcon, SkillVersion } from '../types';
import { settleAll } from '../../common/utils/registryWrites';
import { formatSkillIdentity, isAlreadyExistsError, isNotFoundError, isPermissionDeniedError } from '../utils';
import { useInvalidateSkillQueries } from './useInvalidateSkillQueries';
import { useRegisterSkillMutation, type RegisterSkillMutationInput } from './useRegisterSkillMutation';

/** What a new skill shows besides its first version. None of it is part of the register payload. */
export interface SkillPresentation {
  description: string;
  icons: RegistryIcon[];
  tags: Record<string, string>;
}

/**
 * The formatted identity when a skill with this name already exists, or undefined when the name is free or
 * cannot be checked. A 404 means free; a 403 is treated as unknown, since with workspaces enabled the server
 * also answers 403 for a name that does not exist. Any other failure throws so registration waits until the
 * check succeeds.
 */
export const findTakenSkillIdentity = async (identityInput: string, intl: IntlShape) => {
  const identity = parseSkillIdentityInput(identityInput);
  if ('error' in identity) return undefined;
  try {
    await SkillRegistryApi.getSkill(identity.name, identity.organization);
  } catch (lookupError) {
    if (isNotFoundError(lookupError as Error) || isPermissionDeniedError(lookupError as Error)) return undefined;
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
  return formatSkillIdentity(identity.name, identity.organization);
};

/**
 * Runs a registration from the dialog. Registering the first version creates its parent in the same store
 * transaction. Presentation writes follow the committed version. The skill queries are refreshed once before
 * the version is handed over. Work that outlives the dialog never closes it or navigates.
 */
export const useRegisterSkillSubmission = ({
  onClose,
  onRegistered,
  onNameTaken,
}: {
  onClose: () => void;
  onRegistered: (version: SkillVersion) => void;
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
    let version: SkillVersion;
    try {
      version = await mutateAsync(input);
    } catch (registrationError) {
      if (newSkill && isAlreadyExistsError(registrationError) && isActive()) {
        onNameTaken(formatSkillIdentity(newSkill.name, newSkill.organization));
      }
      setSubmitting(false);
      return;
    }
    if (newSkill) {
      const description = presentation.description.trim();
      const icons = presentation.icons.filter((icon) => icon.src.trim());
      if (description || icons.length) {
        try {
          await SkillRegistryApi.updateSkill(
            version.name,
            { ...(description ? { description } : {}), ...(icons.length ? { icons } : {}) },
            version.organization,
          );
        } catch (metadataFailure) {
          Utils.displayGlobalErrorNotification(
            intl.formatMessage(
              {
                defaultMessage: 'The skill was created, but its description or icons could not be saved: {reason}',
                description: 'Error when saving new skill presentation fails after registration',
              },
              { reason: messageOf(metadataFailure, String(metadataFailure)) },
            ),
          );
        }
      }
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
    /** The failure to show, if any: building the request or registering the version. */
    error: failure ?? (isAlreadyExistsError(registerError) ? undefined : registerError?.message),
    failed: Boolean(failure || (registerError && !isAlreadyExistsError(registerError))),
    close,
    isActive,
  };
};

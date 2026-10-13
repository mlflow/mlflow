import { useCallback, useState } from 'react';

interface UseGuardedTransitionParams {
  isDirty: boolean;
}

interface UseGuardedTransitionResult {
  /**
   * Run `transition` immediately when clean; when dirty, stash it and open the confirmation
   * prompt. The transition only fires once the consumer calls `confirm`.
   */
  requestTransition: (transition: () => void) => void;
  /** True while the confirmation modal should be visible. */
  isPromptOpen: boolean;
  /** Called by the modal's confirm button — fires the stashed transition. */
  confirm: () => void;
  /** Called by the modal's cancel button or overlay dismissal — drops the stashed transition. */
  cancel: () => void;
}

/**
 * Gates arbitrary transitions on a dirty flag. Unlike a fixed-callback close guard, the
 * transition itself is supplied per request, so a single hook instance can serialize close,
 * record-switch, mode-switch, and router-block retries through one prompt.
 */
export const useGuardedTransition = ({ isDirty }: UseGuardedTransitionParams): UseGuardedTransitionResult => {
  const [pendingTransition, setPendingTransition] = useState<(() => void) | null>(null);

  const requestTransition = useCallback(
    (transition: () => void) => {
      if (!isDirty) {
        transition();
        return;
      }
      // Functional setState avoids React invoking `transition` itself when stashing.
      setPendingTransition(() => transition);
    },
    [isDirty],
  );

  const confirm = useCallback(() => {
    pendingTransition?.();
    setPendingTransition(null);
  }, [pendingTransition]);

  const cancel = useCallback(() => setPendingTransition(null), []);

  return { requestTransition, isPromptOpen: pendingTransition !== null, confirm, cancel };
};

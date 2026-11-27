/**
 * Bridge between the Custom View host (which owns the live A2UI surface) and the
 * `render_custom_view` assistant tool. The host is rendered deep inside the
 * trace drawer, while the tool executes inside the assistant runtime; rather
 * than thread a callback through every layer, the host registers an applier on
 * mount and the tool looks it up at execute time.
 */

export type RenderCustomViewSpec = {
  title?: string;
  messages: unknown;
};

export class CustomViewValidationError extends Error {}

export type CustomViewApplyResult = { ok: true } | { ok: false; error: string; retryable: boolean };

export type CustomViewSpecApplier = (spec: RenderCustomViewSpec) => Promise<CustomViewApplyResult>;

// The slot is keyed by a per-host `sessionId` so a pending tool call can be
// bound to the host that was active when it started. Registration is
// last-writer-wins (a single Custom View tab is active at a time).
type Registration = { sessionId: string; applier: CustomViewSpecApplier };

let current: Registration | null = null;

// Waiters ride out the brief window where the active host has unmounted
// (clearing the slot) and is about to re-register — e.g. while the trace modal
// is closed and reopened — so a tool call that lands mid-remount doesn't
// spuriously report "tab not open". Each waiter records the session it is
// waiting for so a DIFFERENT host registering can't satisfy it (which would
// render the spec into the wrong trace explorer).
type Waiter = { expectedSessionId?: string; resolve: (applier: CustomViewSpecApplier | null) => void };
const applierWaiters = new Set<Waiter>();

const matchesSession = (registration: Registration | null, expectedSessionId?: string): boolean =>
  registration !== null && (expectedSessionId === undefined || registration.sessionId === expectedSessionId);

/** Returns an unregister function; registration is last-writer-wins. */
export const registerCustomViewSpecApplier = (sessionId: string, applier: CustomViewSpecApplier): (() => void) => {
  const registration: Registration = { sessionId, applier };
  current = registration;
  for (const waiter of Array.from(applierWaiters)) {
    if (waiter.expectedSessionId === undefined || waiter.expectedSessionId === sessionId) {
      applierWaiters.delete(waiter);
      waiter.resolve(applier);
    }
  }
  return () => {
    if (current === registration) {
      current = null;
    }
  };
};

export const getCustomViewSpecApplier = (): CustomViewSpecApplier | null => current?.applier ?? null;

// The session id of the currently-registered host, captured by the tool at
// execute start so its wait can be scoped to that same host.
export const getCurrentApplierSessionId = (): string | undefined => current?.sessionId;

/**
 * Resolves with the registered applier, waiting up to `timeoutMs` for one to
 * appear if the slot is momentarily empty (host remounting).
 *
 * When `expectedSessionId` is provided, only that host satisfies the wait — a
 * different host registering during the window is ignored, so an in-flight
 * tool call never resumes against the wrong trace explorer. When it is
 * omitted (no host was registered at execute start), any next registration
 * resolves it, preserving the reopen-from-closed behavior.
 */
export const waitForCustomViewSpecApplier = (
  expectedSessionId?: string,
  timeoutMs = 3000,
): Promise<CustomViewSpecApplier | null> => {
  const registration = current;
  if (matchesSession(registration, expectedSessionId) && registration !== null) {
    return Promise.resolve(registration.applier);
  }
  return new Promise((resolve) => {
    let settled = false;
    const waiter: Waiter = {
      expectedSessionId,
      resolve: (applier: CustomViewSpecApplier | null) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timer);
        applierWaiters.delete(waiter);
        resolve(applier);
      },
    };
    const timer = setTimeout(() => {
      const activeRegistration = current;
      waiter.resolve(
        matchesSession(activeRegistration, expectedSessionId) && activeRegistration !== null
          ? activeRegistration.applier
          : null,
      );
    }, timeoutMs);
    applierWaiters.add(waiter);
  });
};

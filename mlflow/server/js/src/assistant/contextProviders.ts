/**
 * Module-level registry of PULL-based context providers, decoupling the generic
 * Assistant page-context store from feature-specific code (e.g. the Custom View
 * host, which pulls in the ESM-only `@a2ui` renderer and should not be part of
 * the Assistant's static module graph).
 *
 * Unlike `useRegisterAssistantContext` (which pushes a value into React state
 * whenever it changes), a provider here is a plain function invoked lazily, at
 * the moment a turn's context is assembled (see `getPageContext` in
 * `AssistantContext.tsx`). This suits context that is expensive/unnecessary to
 * keep reactively in sync (e.g. a large per-trace data snapshot) and context
 * whose value depends on state resolved only at send time (e.g. which output
 * convention to use, based on the provider about to serve the turn).
 */

export type AssistantContextProvider = () => unknown;

const providers = new Map<string, AssistantContextProvider>();

/** Register a pull-based context provider for a key. Returns an unregister function. */
export const registerAssistantContextProvider = (key: string, provider: AssistantContextProvider): (() => void) => {
  providers.set(key, provider);
  return () => {
    if (providers.get(key) === provider) {
      providers.delete(key);
    }
  };
};

/** Invokes every registered provider and returns the non-null/undefined results, keyed. */
export const collectDynamicAssistantContext = (): Record<string, unknown> => {
  const result: Record<string, unknown> = {};
  for (const [key, provider] of providers) {
    const value = provider();
    if (value !== null && value !== undefined) {
      result[key] = value;
    }
  }
  return result;
};

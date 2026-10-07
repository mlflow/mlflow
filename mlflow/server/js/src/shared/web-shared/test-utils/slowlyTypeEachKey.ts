import userEvent from '@testing-library/user-event';

type UserEventInstance = ReturnType<typeof userEvent.setup>;
type TypeOptions = NonNullable<Parameters<UserEventInstance['type']>[2]>;

/**
 * Types `text` into `element` one keystroke at a time, dispatching a real keydown/keyup for every
 * character. This is exactly what `userEvent.type()` does and is intentionally slow — each
 * character is a separate event plus re-render, and under CI CPU stress the accumulated
 * inter-keystroke timing windows are what push long-string `type()` calls over Jest's timeout.
 *
 * Reach for this ONLY when a test genuinely depends on per-keystroke behavior — typeahead /
 * type-to-filter, validation-on-keystroke, debounce, or user-event special keys like `{enter}`
 * that `paste()` cannot replicate. For everything else, prefer `userEvent.paste()`, which inserts
 * the whole string in a single event and does not flake under load. This helper is the sanctioned
 * escape hatch the `@databricks/no-userevent-type` lint rule points to, so the per-keystroke intent
 * is explicit at the call site rather than hidden inside a bare `type()`.
 *
 * Defaults to the bare `userEvent` API. Pass a `user` instance (from `userEvent.setup(...)`) via
 * `options.user` when the test configures user-event — e.g. fake timers through `advanceTimers`.
 */
export async function slowlyTypeEachKey(
  element: Element,
  text: string,
  options?: TypeOptions & { user?: UserEventInstance },
): Promise<void> {
  const { user, ...typeOptions } = options ?? {};
  await (user ?? userEvent).type(element, text, typeOptions);
}

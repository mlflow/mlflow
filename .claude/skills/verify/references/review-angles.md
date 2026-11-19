# Code review angles

- **Line by line.** Read each hunk with its enclosing function. Which inputs, states, or
  timing conditions could produce a wrong result or expose an unhandled error?
- **Removed behavior.** Identify what each deleted guard, validation, or cleanup step did.
  Where does that responsibility belong after the change?
- **Callers and callees.** Trace changes in arguments, return values, exceptions, and
  ordering through the callers that rely on them.
- **Language and framework pitfalls.** Check assumptions the change makes about defaults,
  closures, empty values, async execution, or time zones.
- **Wrappers and proxies.** Follow delegation to the implementation that actually runs.
  Check whether a registry lookup can re-enter the wrapper or a forwarded method skips it.
- **Reuse.** Look for existing implementations of the same rule. If duplication matters,
  identify where the copies disagree or what future change would have to update both.
- **Simplification.** When redundant state, dead paths, or branching obscure an invariant,
  describe the simpler equivalent and how it preserves behavior.
- **Efficiency.** For repeated I/O, avoidable computation, or retained memory, identify the
  affected workload and where the extra cost comes from.
- **Root cause.** Trace a bug fix back to the mechanism that caused the failure. Check other
  entry points that use the same mechanism.
- **Repository conventions.** Cite the applicable rule and the line that violates it. Check
  relevant invariants in neighboring code, such as workspace scoping in the tracking store.

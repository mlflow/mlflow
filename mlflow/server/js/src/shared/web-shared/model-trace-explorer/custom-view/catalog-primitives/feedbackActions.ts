/**
 * Shared action name for the staged-feedback inputs (RadioGroup,
 * FeedbackInputText). Unlike FeedbackThumbsUpDownButtons (which logs an
 * assessment immediately) and FeedbackSubmit (which flushes via the host
 * feedback bridge), these inputs only STAGE their values into a host-side
 * buffer via this action. Kept in its own module so the inputs can import the
 * name without pulling in sibling components (avoids import cycles).
 */

/**
 * Dispatched whenever a staged-feedback input changes. The host merges the
 * action context into its pending buffer keyed by surface id plus `formId`,
 * `name`, and `spanId`.
 * Context: `{ name, value?, rationale?, spanId?, formId? }` — `value` and
 * `rationale` are both optional so an input can stage either field (or both over
 * time). The host reads each field defensively off the untrusted action context,
 * so there is no shared context type to keep in sync.
 */
export const FEEDBACK_STAGED = 'FEEDBACK_STAGED';

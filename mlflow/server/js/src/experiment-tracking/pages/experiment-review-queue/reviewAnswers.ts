import type { LabelSchema } from '../../components/label-schemas';
import type { LabelSchemaValue } from '../../components/label-schemas';
import { sameUser } from './queuePermissions';

/**
 * Whether a label schema's answers are written as feedback or expectation
 * assessments. Mirrors the server's `schema.type` -> assessment-kind mapping.
 */
export type AssessmentKind = 'feedback' | 'expectation';

export const schemaAssessmentKind = (schema: LabelSchema): AssessmentKind =>
  schema.type === 'EXPECTATION' ? 'expectation' : 'feedback';

/**
 * Whether a label-schema widget holds a real answer (not empty / unset). The
 * single source of truth for "answered": the focused-review submit path uses it
 * to gate completion, and prefill uses it to drop empties that could never be
 * re-submitted.
 */
export const isAnswered = (value: LabelSchemaValue): boolean =>
  value !== undefined && value !== null && value !== '' && !(Array.isArray(value) && value.length === 0);

/**
 * A prior answer already recorded on a trace, normalized from a trace
 * assessment into the minimal shape the prefill logic needs.
 */
export interface PriorAnswer {
  name: string;
  kind: AssessmentKind;
  value: LabelSchemaValue;
  /** Free-form rationale recorded alongside the value, if any. */
  rationale?: string;
  /** A trace assessment is valid unless explicitly marked `valid: false`. */
  valid: boolean;
  /** The source assessment's id, so a re-submit can supersede it. */
  assessmentId?: string;
  /**
   * Epoch ms from the assessment's `last_update_time` (falling back to
   * `create_time`); `undefined` when neither is present or parseable. Used to
   * pick the most recent answer per schema instead of trusting array order.
   */
  updatedAt?: number;
}

/**
 * Minimal structural view of a trace assessment as returned by the
 * trace-get endpoint. Kept local (the full `Assessment` type is not part of
 * the web-shared public surface) and intentionally permissive.
 */
export interface RawTraceAssessment {
  assessment_id?: string;
  assessment_name?: string;
  valid?: boolean;
  rationale?: string;
  source?: { source_id?: string; source_type?: string };
  feedback?: { value?: LabelSchemaValue };
  expectation?: { value?: LabelSchemaValue; serialized_value?: { value?: string } };
  /**
   * ISO-8601 timestamps. Used to pick the most recent answer per schema rather
   * than trusting the response array order, which comes from an unordered
   * SQLAlchemy backref and is backend heap-dependent.
   */
  create_time?: string;
  last_update_time?: string;
}

/**
 * Comparable recency for an assessment: `last_update_time` (reflects the latest
 * write) preferred, else `create_time`. `undefined` when neither is present or
 * parseable, which sorts oldest.
 */
const assessmentTime = (assessment: RawTraceAssessment): number | undefined => {
  const raw = assessment.last_update_time ?? assessment.create_time;
  if (!raw) {
    return undefined;
  }
  const ms = Date.parse(raw);
  return Number.isNaN(ms) ? undefined : ms;
};

/**
 * Normalize raw trace assessments into {@link PriorAnswer}s. Feedback and
 * expectation assessments carry their value in different places (and an
 * expectation may be serialized); this collapses them to a single `value`.
 * Assessments without a name or a resolvable value are dropped. When
 * `reviewerSourceId` is given, only that reviewer's own answers are kept, so
 * one reviewer never sees or supersedes another reviewer's answers.
 */
export const extractPriorAnswers = (assessments: RawTraceAssessment[], reviewerSourceId?: string): PriorAnswer[] => {
  // A provided-but-empty source id can't identify a reviewer, so prefill nothing
  // rather than matching source-less assessments (`sameUser('', undefined)` is
  // true). `undefined` means "no source filter"; a non-empty id filters to that
  // reviewer. In practice the reviewer source is always `undefined` or a real
  // username (never ''), so this is a defensive guard.
  if (reviewerSourceId === '') {
    return [];
  }
  const priors: PriorAnswer[] = [];
  for (const assessment of assessments) {
    if (!assessment.assessment_name) {
      continue;
    }
    // Only the reviewer's own *human* answers prefill and supersede. Review
    // answers are always written `source_type: 'HUMAN'`, so an LLM-judge or
    // SDK-written assessment that happens to share the reviewer's source_id
    // (e.g. both `default` on a no-auth server) must not be adopted as theirs.
    if (
      reviewerSourceId !== undefined &&
      (assessment.source?.source_type !== 'HUMAN' || !sameUser(assessment.source?.source_id, reviewerSourceId))
    ) {
      continue;
    }
    let kind: AssessmentKind;
    let value: LabelSchemaValue;
    if (assessment.feedback) {
      kind = 'feedback';
      value = assessment.feedback.value;
    } else if (assessment.expectation) {
      kind = 'expectation';
      value =
        assessment.expectation.value !== undefined
          ? assessment.expectation.value
          : assessment.expectation.serialized_value?.value;
    } else {
      // Neither feedback nor expectation (e.g. an issue reference) — not an answer.
      continue;
    }
    // Drop empties that could never be re-submitted, so a prior doesn't render
    // as a phantom prefill.
    if (!isAnswered(value)) {
      continue;
    }
    priors.push({
      name: assessment.assessment_name,
      kind,
      value,
      rationale: assessment.rationale,
      valid: assessment.valid !== false,
      assessmentId: assessment.assessment_id,
      updatedAt: assessmentTime(assessment),
    });
  }
  return priors;
};

/**
 * The most recent valid prior answer matching a schema's name and kind, by
 * `updatedAt`. Same-name duplicates accumulate across reopen/resubmit, so the
 * latest timestamp wins rather than the last array position (the response order
 * is backend heap-dependent). A missing timestamp sorts oldest, so a timestamped
 * answer always beats an un-timestamped one; among equal timestamps (including
 * when every match lacks one) the last in array order wins, preserving the
 * previous `.at(-1)` behavior.
 */
const pickMostRecent = (priors: PriorAnswer[], name: string, kind: AssessmentKind): PriorAnswer | undefined =>
  priors
    .filter((p) => p.valid && p.name === name && p.kind === kind)
    .reduce<
      PriorAnswer | undefined
    >((best, p) => (best === undefined || (p.updatedAt ?? 0) >= (best.updatedAt ?? 0) ? p : best), undefined);

/**
 * Seed the focused-review widgets from a trace's existing answers.
 *
 * For each schema, picks the most recent valid prior answer whose name and
 * kind match (a FEEDBACK schema only adopts feedback answers, EXPECTATION
 * only expectation), so a reopened trace shows what was previously recorded.
 * Schemas with no matching prior are left unanswered.
 */
export const buildPrefilledAnswers = (
  priors: PriorAnswer[],
  schemas: LabelSchema[],
): Record<string, LabelSchemaValue> => {
  const prefilled: Record<string, LabelSchemaValue> = {};
  for (const schema of schemas) {
    const kind = schemaAssessmentKind(schema);
    const match = pickMostRecent(priors, schema.name, kind);
    if (match) {
      prefilled[schema.name] = match.value;
    }
  }
  return prefilled;
};

/**
 * Seed the focused-review rationale boxes from a trace's existing answers, so a
 * reopened trace shows the rationale recorded alongside each prior answer.
 * Same name/kind/most-recent matching as {@link buildPrefilledAnswers}.
 */
export const buildPrefilledRationales = (priors: PriorAnswer[], schemas: LabelSchema[]): Record<string, string> => {
  const prefilled: Record<string, string> = {};
  for (const schema of schemas) {
    const kind = schemaAssessmentKind(schema);
    const match = pickMostRecent(priors, schema.name, kind);
    if (match?.rationale) {
      prefilled[schema.name] = match.rationale;
    }
  }
  return prefilled;
};

/**
 * Map each schema to the assessment id of the reviewer's most recent valid prior
 * answer for it, so a re-submit can supersede that assessment (via `overrides`)
 * instead of accumulating a duplicate. Same name/kind/most-recent matching as
 * {@link buildPrefilledAnswers}; `priors` should already be scoped to the
 * current reviewer (see {@link extractPriorAnswers}).
 */
export const buildPriorAssessmentIds = (priors: PriorAnswer[], schemas: LabelSchema[]): Record<string, string> => {
  const ids: Record<string, string> = {};
  for (const schema of schemas) {
    const kind = schemaAssessmentKind(schema);
    const match = pickMostRecent(priors, schema.name, kind);
    if (match?.assessmentId) {
      ids[schema.name] = match.assessmentId;
    }
  }
  return ids;
};

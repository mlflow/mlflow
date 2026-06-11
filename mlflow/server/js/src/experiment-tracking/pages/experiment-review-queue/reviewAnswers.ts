import type { LabelSchema } from '../../components/label-schemas';
import type { LabelSchemaValue } from '../../components/label-schemas';

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
}

/**
 * Minimal structural view of a trace assessment as returned by the
 * trace-get endpoint. Kept local (the full `Assessment` type is not part of
 * the web-shared public surface) and intentionally permissive.
 */
export interface RawTraceAssessment {
  assessment_name?: string;
  valid?: boolean;
  rationale?: string;
  feedback?: { value?: LabelSchemaValue };
  expectation?: { value?: LabelSchemaValue; serialized_value?: { value?: string } };
}

/**
 * Normalize raw trace assessments into {@link PriorAnswer}s. Feedback and
 * expectation assessments carry their value in different places (and an
 * expectation may be serialized); this collapses them to a single `value`.
 * Assessments without a name or a resolvable value are dropped.
 */
export const extractPriorAnswers = (assessments: RawTraceAssessment[]): PriorAnswer[] => {
  const priors: PriorAnswer[] = [];
  for (const assessment of assessments) {
    if (!assessment.assessment_name) {
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
    });
  }
  return priors;
};

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
    // Last match wins: later assessments override earlier ones for the same name.
    const match = priors.filter((p) => p.valid && p.name === schema.name && p.kind === kind).at(-1);
    if (match) {
      prefilled[schema.name] = match.value;
    }
  }
  return prefilled;
};

/**
 * Seed the focused-review rationale boxes from a trace's existing answers, so a
 * reopened trace shows the rationale recorded alongside each prior answer.
 * Same name/kind/last-wins matching as {@link buildPrefilledAnswers}.
 */
export const buildPrefilledRationales = (priors: PriorAnswer[], schemas: LabelSchema[]): Record<string, string> => {
  const prefilled: Record<string, string> = {};
  for (const schema of schemas) {
    const kind = schemaAssessmentKind(schema);
    const match = priors.filter((p) => p.valid && p.name === schema.name && p.kind === kind).at(-1);
    if (match?.rationale) {
      prefilled[schema.name] = match.rationale;
    }
  }
  return prefilled;
};

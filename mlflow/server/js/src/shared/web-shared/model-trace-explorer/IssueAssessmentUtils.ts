import type { Assessment, IssueReferenceAssessment } from './ModelTrace.types';

/** Versioned evidence written by issue detection onto a trace's issue reference. */
export const ISSUE_EVIDENCE_METADATA_KEY = 'mlflow.issue.evidence';
const ISSUE_EVIDENCE_SCHEMA_VERSION = 1;

export type IssueEvidenceHighlightTone = 'error' | 'warning' | 'success';

export interface IssueEvidenceStep {
  spanId: string;
  text: string;
  highlights: string[];
  highlightTone: IssueEvidenceHighlightTone;
  mono?: boolean;
}

export interface IssueEvidence {
  schemaVersion: typeof ISSUE_EVIDENCE_SCHEMA_VERSION;
  steps: IssueEvidenceStep[];
}

export interface TraceIssueReference {
  id: string;
  name: string;
  spanId?: string;
  rationale?: string;
  /** Absent for a reference written before structured evidence, whose rationale still reads. */
  evidence?: IssueEvidence;
}

const isIssueReferenceAssessment = (assessment: Assessment): assessment is IssueReferenceAssessment =>
  'issue' in assessment && assessment.issue !== undefined;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const parseRequiredString = (value: unknown): string | undefined => {
  const result = typeof value === 'string' ? value.trim() : '';
  return result || undefined;
};

const parseHighlightTone = (value: unknown): IssueEvidenceHighlightTone | undefined =>
  value === 'error' || value === 'warning' || value === 'success' ? value : undefined;

const parseEvidenceStep = (value: unknown): IssueEvidenceStep | undefined => {
  if (!isRecord(value)) {
    return undefined;
  }
  const spanId = parseRequiredString(value['span_id']);
  const text = parseRequiredString(value['text']);
  const highlightTone = parseHighlightTone(value['highlight_tone']);
  const highlights = Array.isArray(value['highlights'])
    ? value['highlights'].flatMap((highlight) => parseRequiredString(highlight) ?? [])
    : [];
  // Highlights are rendered by searching the step's text, so a highlight that is not an exact
  // substring would emphasize nothing. The writer rejects those; a reader treats the whole step
  // as unreadable rather than showing emphasis that silently went missing.
  if (
    !spanId ||
    !text ||
    !highlightTone ||
    highlights.length === 0 ||
    highlights.some((highlight) => !text.includes(highlight))
  ) {
    return undefined;
  }
  return {
    spanId,
    text,
    highlights,
    highlightTone,
    ...(typeof value['mono'] === 'boolean' ? { mono: value['mono'] } : {}),
  };
};

/**
 * Reads the versioned evidence payload, or `undefined` when it is absent, of another schema
 * version, or malformed. Tolerant on purpose: the reference's rationale still explains the issue,
 * so unreadable evidence degrades to the summary rather than blanking the card.
 */
const parseEvidence = (value?: string): IssueEvidence | undefined => {
  if (!value) {
    return undefined;
  }
  try {
    const parsed: unknown = JSON.parse(value);
    if (
      !isRecord(parsed) ||
      parsed['schema_version'] !== ISSUE_EVIDENCE_SCHEMA_VERSION ||
      !Array.isArray(parsed['steps'])
    ) {
      return undefined;
    }
    const steps: IssueEvidenceStep[] = [];
    for (const rawStep of parsed['steps']) {
      const step = parseEvidenceStep(rawStep);
      if (!step) {
        return undefined;
      }
      steps.push(step);
    }
    if (steps.length === 0) {
      return undefined;
    }
    return { schemaVersion: ISSUE_EVIDENCE_SCHEMA_VERSION, steps };
  } catch {
    return undefined;
  }
};

/**
 * The issue a trace assessment references, for native issue references only. Feedback remains a
 * separate assessment type and is never read as an issue association.
 */
export const getIssueReferenceFromAssessment = (assessment: Assessment): TraceIssueReference | undefined => {
  if (!isIssueReferenceAssessment(assessment)) {
    return undefined;
  }
  return {
    id: assessment.assessment_name,
    name: assessment.issue.issue_name || assessment.assessment_name,
    spanId: assessment.span_id,
    rationale: assessment.rationale,
    evidence: parseEvidence(assessment.metadata?.[ISSUE_EVIDENCE_METADATA_KEY]),
  };
};

/**
 * The newest valid reference per issue, as the assessments themselves.
 *
 * Repeated detection runs leave one reference each on a trace they both flag, and an invalidated
 * one is not an association at all -- so any reader that takes the assessments as they come shows
 * a card per run and a count that grows with every rerun. Every such reader goes through here, so
 * "which reference represents this issue" is decided in one place.
 */
export const getUniqueIssueReferenceAssessments = (assessments: Assessment[]): IssueReferenceAssessment[] => {
  const newest = new Map<string, { assessment: IssueReferenceAssessment; timestamp: number }>();
  for (const assessment of assessments) {
    if (assessment.valid === false || !isIssueReferenceAssessment(assessment)) {
      continue;
    }
    const timestamp = Date.parse(assessment.last_update_time || assessment.create_time) || 0;
    const existing = newest.get(assessment.assessment_name);
    if (!existing || timestamp >= existing.timestamp) {
      newest.set(assessment.assessment_name, { assessment, timestamp });
    }
  }
  return [...newest.values()].map(({ assessment }) => assessment);
};

/** The same selection, projected to what a card needs to render. */
export const getUniqueIssueReferences = (assessments: Assessment[]): TraceIssueReference[] =>
  getUniqueIssueReferenceAssessments(assessments).flatMap(
    (assessment) => getIssueReferenceFromAssessment(assessment) ?? [],
  );

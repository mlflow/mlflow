import { parseJSONSafe } from '../../../common/utils/TagUtils';
import Utils from '../../../common/utils/Utils';

// Written by `mlflow.genai.evaluate` (`log_lineage_tags` in mlflow/genai/evaluation/lineage.py); keep the
// keys and value formats in sync with it.
export const EVAL_RUN_SCORERS_DIGEST_TAG = 'mlflow.genaiEvaluate.scorersDigest';
export const EVAL_RUN_AGENT_FUNCTION_TAG = 'mlflow.genaiEvaluate.agent.function';
export const EVAL_RUN_AGENT_URI_TAG = 'mlflow.genaiEvaluate.agent.uri';
export const EVAL_RUN_AGENT_DIGEST_TAG = 'mlflow.genaiEvaluate.agent.digest';

export type EvalRunScorersDigestCoverage = 'complete' | 'partial' | 'unavailable';

export interface EvalRunScorersDigest {
  /** Present only when at least one scorer definition could be hashed. */
  digest?: string;
  hashed: number;
  total: number;
  coverage: EvalRunScorersDigestCoverage;
}

export interface EvalRunAgentLineage {
  function?: string;
  uri?: string;
  digest?: string;
  loggedModelId?: string;
}

const SHORT_DIGEST_LENGTH = 7;

/** Shortens a `sha256:<hex>` digest to its first hex characters, like a short git commit. */
export const formatShortDigest = (digest: string) => digest.replace(/^sha256:/, '').slice(0, SHORT_DIGEST_LENGTH);

const isCount = (value: unknown): value is number => typeof value === 'number' && Number.isInteger(value) && value >= 0;

const asNonEmptyString = (value: unknown): string | undefined =>
  typeof value === 'string' && value !== '' ? value : undefined;

/** Parses the scorers digest tag, `{"digest": "sha256:…", "hashed": n, "total": m}`; `digest` is absent when n is 0. */
export const parseEvalRunScorersDigest = (value: string | undefined): EvalRunScorersDigest | undefined => {
  const parsed = value ? parseJSONSafe(value) : undefined;
  if (!parsed || typeof parsed !== 'object') {
    return undefined;
  }
  const { hashed, total } = parsed;
  if (!isCount(hashed) || !isCount(total) || total === 0 || hashed > total) {
    return undefined;
  }
  const digest = hashed > 0 ? asNonEmptyString(parsed.digest) : undefined;
  if (hashed > 0 && !digest) {
    return undefined;
  }
  return {
    digest,
    hashed,
    total,
    coverage: hashed === 0 ? 'unavailable' : hashed === total ? 'complete' : 'partial',
  };
};

export const getEvalRunAgentLineage = (
  tags: Record<string, string | undefined>,
  loggedModelId?: string,
): EvalRunAgentLineage | undefined => {
  const lineage: EvalRunAgentLineage = {
    function: asNonEmptyString(tags[EVAL_RUN_AGENT_FUNCTION_TAG]),
    uri: asNonEmptyString(tags[EVAL_RUN_AGENT_URI_TAG]),
    digest: asNonEmptyString(tags[EVAL_RUN_AGENT_DIGEST_TAG]),
    loggedModelId: asNonEmptyString(loggedModelId),
  };
  const hasAnyValue = lineage.function || lineage.uri || lineage.digest || lineage.loggedModelId;
  return hasAnyValue ? lineage : undefined;
};

export const getEvalRunGitCommit = (tags: Record<string, string | undefined>): string | undefined =>
  asNonEmptyString(tags[Utils.gitCommitTag]);

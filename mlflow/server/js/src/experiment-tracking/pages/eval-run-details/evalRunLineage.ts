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

// Group-by keys for lineage values. They use the tag grouping mode. The agent key is not a real tag,
// because its value comes from the URI tag or, for a local function, the function tag.
export const EVAL_RUN_SCORERS_GROUP_BY_KEY = EVAL_RUN_SCORERS_DIGEST_TAG;
export const EVAL_RUN_AGENT_GROUP_BY_KEY = 'mlflow.genaiEvaluate.agent';

const PARTIAL_SCORERS_GROUP_VALUE_PATTERN = /^(\S+) partial:(\d+)\/(\d+)$/;

/**
 * A partial digest covers only the scorers that could be hashed, so `[A]` and `[A, unhashable B]` have the
 * same digest. Adding the counts keeps partial runs out of the groups of complete runs.
 */
const getScorersGroupValue = (scorersDigest: EvalRunScorersDigest | undefined): string | null => {
  if (!scorersDigest?.digest) {
    return null;
  }
  return scorersDigest.coverage === 'partial'
    ? `${scorersDigest.digest} partial:${scorersDigest.hashed}/${scorersDigest.total}`
    : scorersDigest.digest;
};

/** Splits a "Scorers" group value into its digest and, for a partial digest, its counts. */
export const parseScorersGroupValue = (value: string): { digest: string; hashed?: number; total?: number } => {
  const match = value.match(PARTIAL_SCORERS_GROUP_VALUE_PATTERN);
  return match ? { digest: match[1], hashed: Number(match[2]), total: Number(match[3]) } : { digest: value };
};

/** The group value of a lineage group-by key, or `undefined` when the key is not a lineage key. */
export const getEvalRunLineageGroupValue = (
  groupByKey: string,
  getTag: (key: string) => string | undefined,
): string | null | undefined => {
  switch (groupByKey) {
    case EVAL_RUN_SCORERS_GROUP_BY_KEY:
      return getScorersGroupValue(parseEvalRunScorersDigest(getTag(EVAL_RUN_SCORERS_DIGEST_TAG)));
    case EVAL_RUN_AGENT_GROUP_BY_KEY:
      return (
        asNonEmptyString(getTag(EVAL_RUN_AGENT_URI_TAG)) ??
        asNonEmptyString(getTag(EVAL_RUN_AGENT_FUNCTION_TAG)) ??
        null
      );
    default:
      return undefined;
  }
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

import { MlflowService } from '../../sdk/MlflowService';
import { MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG } from '../../constants';
import { ALL_ISSUE_CATEGORIES } from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';

export interface IssueDetectionSettings {
  instructions: string;
  categories: Record<string, boolean>;
}

interface ExperimentTag {
  key: string;
  value?: string | null;
}

const DEFAULT_CATEGORIES: Record<string, boolean> = Object.fromEntries(
  ALL_ISSUE_CATEGORIES.map((category) => [category, true]),
);

export const defaultIssueDetectionSettings = (): IssueDetectionSettings => ({
  instructions: '',
  categories: { ...DEFAULT_CATEGORIES },
});

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const normalizeCategories = (value: unknown): Record<string, boolean> => {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value).flatMap(([category, selected]) => {
      const normalizedCategory = category.trim();
      return normalizedCategory && typeof selected === 'boolean' ? [[normalizedCategory, selected]] : [];
    }),
  );
};

export const readIssueDetectionSettingsFromTags = (
  tags: readonly ExperimentTag[] | undefined | null,
): IssueDetectionSettings => {
  const raw = tags?.find(({ key }) => key === MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG)?.value;
  if (!raw) {
    return defaultIssueDetectionSettings();
  }

  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) {
      return defaultIssueDetectionSettings();
    }
    return {
      instructions: typeof parsed['instructions'] === 'string' ? parsed['instructions'] : '',
      // Taken as-is, matching upstream: a focus area is selected only where the tag says so
      // explicitly. A built-in the tag does not mention is therefore not selected, and a tag
      // that selects nothing resolves to every category in `selectedDetectionCategories`.
      categories: normalizeCategories(parsed['categories']),
    };
  } catch {
    return defaultIssueDetectionSettings();
  }
};

export const loadIssueDetectionSettings = async (experimentId: string): Promise<IssueDetectionSettings> => {
  const { experiment } = await MlflowService.getExperiment({ experiment_id: experimentId });
  return readIssueDetectionSettingsFromTags(experiment?.tags);
};

export const saveIssueDetectionSettings = async (
  experimentId: string,
  settings: IssueDetectionSettings,
): Promise<IssueDetectionSettings> => {
  const normalized = {
    instructions: settings.instructions.trim(),
    categories: normalizeCategories(settings.categories),
  };
  await MlflowService.setExperimentTag({
    experiment_id: experimentId,
    key: MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG,
    value: JSON.stringify(normalized),
  });
  return normalized;
};

/** Whether the settings restrict detection to nothing at all, which no run can honour literally. */
export const noFocusAreaSelected = (settings: IssueDetectionSettings | undefined): boolean =>
  Boolean(settings) && !Object.values(settings?.categories ?? {}).some(Boolean);

/**
 * The categories a detection run should be given.
 *
 * Replaces the upstream prompt builder: detection runs server-side here, so settings travel as
 * a category list and free-text priorities on the invoke request rather than being spliced into
 * a prompt in the browser.
 *
 * **The stored value and the sent value can differ in exactly one case**: settings with every
 * focus area deselected are sent as every built-in category. An all-deselected state is kept
 * storable -- a tag written by hand or by another client can hold it, and the reader has to
 * mean something by it -- but it cannot be sent as-is, because a run with no categories can
 * report nothing.
 *
 * This is a resolution rather than an override. "Detect none of these" and "detect all of
 * these" encode the same intent, "do not restrict by category", since restricting to the empty
 * set is not a thing detection can usefully do. Both surfaces that can produce or launch that
 * state say so before the run starts: the settings modal while nothing is selected, and the
 * detection modal when the saved settings select nothing (see ``noFocusAreaSelected``).
 */
export const selectedDetectionCategories = (settings: IssueDetectionSettings | undefined): string[] => {
  const selected = Object.entries(settings?.categories ?? {})
    .filter(([, isSelected]) => isSelected)
    .map(([category]) => category);
  return selected.length > 0 ? selected : [...ALL_ISSUE_CATEGORIES];
};

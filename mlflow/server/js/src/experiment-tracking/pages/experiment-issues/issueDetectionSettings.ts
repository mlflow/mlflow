import { MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG } from '../../constants';
import { ALL_ISSUE_CATEGORIES } from '../../components/experiment-page/components/traces-v3/IssueDetectionCategories';
import { MlflowService } from '../../sdk/MlflowService';

/**
 * What an experiment's owner has decided detection should look for.
 *
 * Persisted as one JSON experiment tag rather than its own resource: the whole value is read
 * and written together, it is small, and it belongs to the experiment's lifetime, so it needs
 * no schema of its own.
 */
export interface IssueDetectionSettings {
  /** Free text, applied by the stages that decide what to flag and how severe it is. */
  instructions: string;
  /** Category id to whether detection should look for it, including the owner's own ids. */
  categories: Record<string, boolean>;
}

interface ExperimentTag {
  key: string;
  value?: string | null;
}

const allCategoriesSelected = (): Record<string, boolean> =>
  Object.fromEntries(ALL_ISSUE_CATEGORIES.map((category) => [category, true]));

export const defaultIssueDetectionSettings = (): IssueDetectionSettings => ({
  instructions: '',
  categories: allCategoriesSelected(),
});

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** Keeps only entries that could have come from the form, so a hand-edited tag cannot widen the shape. */
const readCategories = (value: unknown): Record<string, boolean> => {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value).flatMap(([category, selected]) => {
      const trimmed = category.trim();
      return trimmed && typeof selected === 'boolean' ? [[trimmed, selected]] : [];
    }),
  );
};

/**
 * The saved settings, or the defaults when the experiment has none.
 *
 * A built-in category the tag does not mention is treated as selected, not as deselected: the
 * tag records the owner's decisions, and a category added to the product after they last saved
 * is not a decision they made. Turning everything off stays expressible, because doing so
 * writes every built-in explicitly false.
 */
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
      categories: { ...allCategoriesSelected(), ...readCategories(parsed['categories']) },
    };
  } catch {
    // A tag that is not the JSON we wrote tells us nothing about what the owner wanted.
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
  const normalized: IssueDetectionSettings = {
    instructions: settings.instructions.trim(),
    categories: readCategories(settings.categories),
  };
  await MlflowService.setExperimentTag({
    experiment_id: experimentId,
    key: MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG,
    value: JSON.stringify(normalized),
  });
  return normalized;
};

/**
 * The categories a detection run should be given.
 *
 * Deselecting everything would otherwise ask for a run that can report nothing, so it falls
 * back to every built-in category. The settings modal says as much while nothing is selected,
 * rather than letting the fallback be a surprise.
 */
export const selectedDetectionCategories = (settings: IssueDetectionSettings | undefined): string[] => {
  const selected = Object.entries(settings?.categories ?? {})
    .filter(([, isSelected]) => isSelected)
    .map(([category]) => category);
  return selected.length > 0 ? selected : [...ALL_ISSUE_CATEGORIES];
};

import { Alert } from '@databricks/design-system';
import { useLocalStorage } from '@databricks/web-shared/hooks';
import { useIntl } from 'react-intl';

/**
 * Baseline comparison is the payoff of this page, but nothing in the product asks
 * anyone to choose a baseline — so without a nudge the Baseline column stays
 * empty and the delta arrows never appear. This is a discoverability fix.
 *
 * It self-resolves: setting a baseline makes `hasBaseline` true, so dismissal
 * only ever means "not now".
 */
export const EvalRunsBaselineNudge = ({
  experimentId,
  hasBaseline,
  runCount,
  isLoading,
}: {
  experimentId: string;
  hasBaseline: boolean;
  runCount: number;
  isLoading: boolean;
}) => {
  const intl = useIntl();
  // Keyed per experiment: a global key would mean one dismissal hides the nudge
  // on every other experiment, so the user could never discover baselines again.
  const [isDismissed, setIsDismissed] = useLocalStorage({
    key: `mlflow.evalRuns.baselineNudge.${experimentId}`,
    version: 1,
    initialValue: false,
  });

  // Never render while loading, or it flashes on every visit. A baseline is also
  // meaningless with fewer than two runs — there would be nothing to compare.
  if (isLoading || hasBaseline || isDismissed || runCount < 2) {
    return null;
  }

  return (
    <Alert
      componentId="mlflow.eval-runs.baseline-nudge"
      type="info"
      closable
      onClose={() => setIsDismissed(true)}
      message={intl.formatMessage({
        defaultMessage: 'Set a baseline to see how each run compares.',
        description: 'Prompt inviting the user to choose a baseline run for the experiment',
      })}
      description={intl.formatMessage({
        defaultMessage: 'Select one run, then choose "Set as baseline" from the Actions menu.',
        description: 'Instruction pointing at the existing Actions menu affordance for setting a baseline',
      })}
    />
  );
};

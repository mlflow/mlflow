/**
 * Test cases tab for the regression-test run page.
 *
 * Reuses the polished GenAI traces table (the same component the run
 * Evaluations tab renders), scoped to this regression-test run. Each test
 * case is one trace produced by an ``assert_behavior`` call; its assertion
 * results show up as the trace's assessment columns. We get input/output
 * previews, latency, tokens, hover/expand, and per-assertion pass/fail for
 * free instead of hand-rolling a table.
 */
import { Empty } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { RegressionTestCasesTable } from '../../../components/evaluations/RegressionTestCasesTable';

const TestCasesTab = ({ experimentId, runUuid }: { experimentId?: string; runUuid?: string }) => {
  if (!experimentId || !runUuid) {
    return (
      <Empty
        title={
          <FormattedMessage
            defaultMessage="No run selected"
            description="Empty state when the regression-test run page has no run id"
          />
        }
        description={null}
      />
    );
  }

  return <RegressionTestCasesTable experimentId={experimentId} runUuid={runUuid} runDisplayName={runUuid} />;
};

export default TestCasesTab;

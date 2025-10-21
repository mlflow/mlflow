import { MlflowClient } from '../../src/clients';
import * as mlflow from '../../src';
import { SpanType } from '../../src/core/constants';
import { LiveSpan } from '../../src/core/entities/span';
import { SpanStatus, SpanStatusCode } from '../../src/core/entities/span_status';
import { TraceState } from '../../src/core/entities/trace_state';
import { convertHrTimeToMs } from '../../src/core/utils';
import { Trace } from '../../src/core/entities/trace';
import { TEST_TRACKING_URI } from '../helper';

describe('API', () => {
  beforeAll(async () => {
    mlflow.init({
      trackingUri: 'databricks',
      location: {
        catalog_name: 'yuki_watanabe_test',
        schema_name: 'shinkansen_prpr_typescript_sdk',
      }
    });
  });

  describe('startSpan', () => {
    it('should create a span with span type', async () => {
      const span = mlflow.startSpan({ name: 'test-span' });
      expect(span).toBeInstanceOf(LiveSpan);

      span.setInputs({ prompt: 'Hello, world!' });
      span.setOutputs({ response: 'Hello, world!' });
      span.setAttributes({ model: 'gpt-4' });
      span.setStatus('OK');
      span.end();

      // Validate traces pushed to the backend
      await mlflow.flushTraces();
    });
  })
});

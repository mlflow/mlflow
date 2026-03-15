import { useState, useCallback, useEffect, useRef } from 'react';
import {
  Alert,
  Button,
  Input,
  Modal,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { fetchOrFail, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { GatewayApi } from '../../api';
import type { Guardrail, TestGuardrailResponse } from '../../types';

interface TraceEntry {
  trace_id: string;
  request_preview?: string;
  response_preview?: string;
  request_time?: string;
  state?: string;
}

interface RegisteredScorerEntry {
  scorer_name: string;
  scorer_version: number;
  experiment_id: string;
  serialized_scorer: string;
}

interface TestGuardrailModalProps {
  open: boolean;
  onClose: () => void;
  guardrail: Guardrail;
  experimentId?: string;
}

type InputMode = 'trace' | 'manual';

// Job status polling interval
const JOB_POLL_INTERVAL = 1500;

interface ScorerInvokeResult {
  score: string | null;
  rationale: string | null;
}

export const TestGuardrailModal = ({ open, onClose, guardrail, experimentId }: TestGuardrailModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [inputMode, setInputMode] = useState<InputMode>(experimentId ? 'trace' : 'manual');
  const [manualText, setManualText] = useState('');
  const [traces, setTraces] = useState<TraceEntry[]>([]);
  const [tracesLoading, setTracesLoading] = useState(false);
  const [tracesError, setTracesError] = useState<string | null>(null);
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<TestGuardrailResponse | null>(null);
  const [invokeResult, setInvokeResult] = useState<ScorerInvokeResult | null>(null);
  const [testing, setTesting] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch traces from the endpoint's experiment
  useEffect(() => {
    if (!open || !experimentId) return;

    const fetchTraces = async () => {
      setTracesLoading(true);
      setTracesError(null);
      try {
        const res = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/traces/search'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: experimentId } }],
            max_results: 20,
            order_by: ['timestamp_ms DESC'],
          }),
        });
        const data = (await res.json()) as { traces?: TraceEntry[] };
        setTraces(data.traces ?? []);
      } catch (e: any) {
        setTracesError(e.message || 'Failed to load traces');
      } finally {
        setTracesLoading(false);
      }
    };

    fetchTraces();
  }, [open, experimentId]);

  // Reset state when modal opens/closes
  useEffect(() => {
    if (open) {
      setTestResult(null);
      setInvokeResult(null);
      setTestError(null);
      setSelectedTraceId(null);
      setManualText('');
      setInputMode(experimentId ? 'trace' : 'manual');
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [open, experimentId]);

  /**
   * Look up the serialized_scorer for this guardrail from the registered scorers list.
   * Returns null for builtin guardrails (which don't have a serialized_scorer).
   */
  const getSerializedScorer = useCallback(async (): Promise<string | null> => {
    const config = guardrail.config;
    if (!config?.registered_scorer) return null;

    try {
      const expId = config.experiment_id ?? '0';
      const res = await fetchOrFail(getAjaxUrl(`ajax-api/3.0/mlflow/scorers/list?experiment_id=${expId}`));
      const data = (await res.json()) as { scorers?: RegisteredScorerEntry[] };
      const scorer = (data.scorers ?? []).find(
        (s) =>
          s.scorer_name === config.registered_scorer &&
          (config.scorer_version == null || s.scorer_version === config.scorer_version),
      );
      return scorer?.serialized_scorer ?? null;
    } catch {
      return null;
    }
  }, [guardrail]);

  /**
   * Poll for job completion and fetch the assessment result from the trace.
   */
  const pollForJobResult = useCallback(
    async (jobIds: string[], traceId: string) => {
      return new Promise<ScorerInvokeResult>((resolve, reject) => {
        const poll = async () => {
          try {
            const res = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/jobs/get'), {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ job_id: jobIds[0] }),
            });
            const data = (await res.json()) as { job?: { status: string; error?: string } };
            const status = data.job?.status;

            if (status === 'SUCCEEDED') {
              if (pollRef.current) clearInterval(pollRef.current);
              // Fetch the updated trace to get the assessment
              try {
                const traceRes = await fetchOrFail(getAjaxUrl(`ajax-api/3.0/mlflow/traces/${traceId}`));
                const traceData = (await traceRes.json()) as {
                  trace_info?: { assessments?: { value?: any; rationale?: string; name?: string }[] };
                };
                const assessments = traceData.trace_info?.assessments ?? [];
                // Get the latest assessment (last one added)
                const latest = assessments[assessments.length - 1];
                resolve({
                  score: latest?.value != null ? String(latest.value) : null,
                  rationale: latest?.rationale ?? null,
                });
              } catch {
                resolve({ score: 'done', rationale: 'Scorer completed. Check trace for results.' });
              }
            } else if (status === 'FAILED') {
              if (pollRef.current) clearInterval(pollRef.current);
              reject(new Error(data.job?.error || 'Scorer job failed'));
            }
            // Otherwise keep polling (PENDING, RUNNING)
          } catch (e: any) {
            if (pollRef.current) clearInterval(pollRef.current);
            reject(e);
          }
        };

        pollRef.current = setInterval(poll, JOB_POLL_INTERVAL);
        poll(); // Run immediately
      });
    },
    [],
  );

  /**
   * Run test via /mlflow/scorer/invoke for registered scorers on traces,
   * or fall back to /gateway/guardrails/test for builtins or manual input.
   */
  const handleTest = useCallback(async () => {
    setTesting(true);
    setTestResult(null);
    setInvokeResult(null);
    setTestError(null);
    if (pollRef.current) clearInterval(pollRef.current);

    try {
      // For trace mode with a registered scorer, use /mlflow/scorer/invoke
      if (inputMode === 'trace' && selectedTraceId && experimentId) {
        const serializedScorer = await getSerializedScorer();

        if (serializedScorer) {
          // Use the scorer invoke endpoint
          const invokeRes = await fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/scorer/invoke'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              experiment_id: experimentId,
              serialized_scorer: serializedScorer,
              trace_ids: [selectedTraceId],
              log_assessments: true,
            }),
          });
          const invokeData = (await invokeRes.json()) as { jobs?: { job_id: string }[] };
          const jobIds = (invokeData.jobs ?? []).map((j) => j.job_id);

          if (jobIds.length === 0) {
            setTestError('No scorer job was created.');
            setTesting(false);
            return;
          }

          // Poll for result
          const result = await pollForJobResult(jobIds, selectedTraceId);
          setInvokeResult(result);
          setTesting(false);
          return;
        }

        // Fall back to /gateway/guardrails/test for builtin scorers
        const result = await GatewayApi.testGuardrail({
          guardrail_id: guardrail.guardrail_id,
          trace_id: selectedTraceId,
          experiment_id: experimentId,
        });
        setTestResult(result);
      } else if (inputMode === 'manual' && manualText) {
        // Manual text always goes through our /test endpoint
        const result = await GatewayApi.testGuardrail({
          guardrail_id: guardrail.guardrail_id,
          text: manualText,
        });
        setTestResult(result);
      } else {
        setTestError('Please provide text or select a trace to test against.');
      }
    } catch (e: any) {
      setTestError(e.message || 'Test failed');
    } finally {
      setTesting(false);
    }
  }, [guardrail, inputMode, selectedTraceId, experimentId, manualText, getSerializedScorer, pollForJobResult]);

  // Determine result state from either path
  const resultScore = testResult?.result?.score ?? invokeResult?.score;
  const resultRationale = testResult?.result?.rationale ?? invokeResult?.rationale;
  const hasResult = testResult != null || invokeResult != null;
  const isPassed = resultScore === 'yes';

  return (
    <Modal
      componentId="mlflow.gateway.guardrails.test-modal"
      visible={open}
      onCancel={onClose}
      title={intl.formatMessage(
        {
          defaultMessage: 'Test guardrail: {name}',
          description: 'Title for test guardrail modal',
        },
        { name: guardrail.scorer_name },
      )}
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button componentId="mlflow.gateway.guardrails.test-cancel" onClick={onClose}>
            <FormattedMessage defaultMessage="Close" description="Close button for test guardrail modal" />
          </Button>
          <Button
            componentId="mlflow.gateway.guardrails.test-run"
            type="primary"
            onClick={handleTest}
            loading={testing}
            disabled={inputMode === 'trace' ? !selectedTraceId : !manualText}
          >
            <FormattedMessage defaultMessage="Run test" description="Run test button" />
          </Button>
        </div>
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {/* Info about which hook this guardrail uses */}
        <Alert
          componentId="mlflow.gateway.guardrails.test-info"
          type="info"
          closable={false}
          message={intl.formatMessage(
            {
              defaultMessage:
                'This is a {hook}-invocation {operation} guardrail. It will test against {dataSource}.',
              description: 'Info about guardrail test behavior',
            },
            {
              hook: guardrail.hook === 'PRE' ? 'pre' : 'post',
              operation: guardrail.operation.toLowerCase(),
              dataSource: guardrail.hook === 'PRE' ? 'request input' : 'response output',
            },
          )}
        />

        {/* Input mode tabs */}
        {experimentId && (
          <div css={{ display: 'flex', gap: theme.spacing.xs }}>
            <Button
              componentId="mlflow.gateway.guardrails.test-mode-trace"
              type={inputMode === 'trace' ? 'primary' : undefined}
              size="small"
              onClick={() => setInputMode('trace')}
            >
              <FormattedMessage defaultMessage="Pick from traces" description="Tab to pick a trace" />
            </Button>
            <Button
              componentId="mlflow.gateway.guardrails.test-mode-manual"
              type={inputMode === 'manual' ? 'primary' : undefined}
              size="small"
              onClick={() => setInputMode('manual')}
            >
              <FormattedMessage defaultMessage="Manual input" description="Tab for manual text input" />
            </Button>
          </div>
        )}

        {/* Trace picker */}
        {inputMode === 'trace' && experimentId && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <Typography.Text bold>
              <FormattedMessage
                defaultMessage="Select a trace"
                description="Label for trace selection in test modal"
              />
            </Typography.Text>

            {tracesLoading && (
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                <Spinner size="small" />
                <FormattedMessage defaultMessage="Loading traces..." description="Loading traces" />
              </div>
            )}

            {tracesError && (
              <Alert
                componentId="mlflow.gateway.guardrails.test-traces-error"
                type="error"
                closable={false}
                message={tracesError}
              />
            )}

            {!tracesLoading && traces.length === 0 && !tracesError && (
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="No traces found for this endpoint. Try sending some requests first, or use manual input."
                  description="No traces found message"
                />
              </Typography.Text>
            )}

            {!tracesLoading && traces.length > 0 && (
              <div
                css={{
                  maxHeight: 280,
                  overflowY: 'auto',
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                }}
              >
                {traces.map((trace) => {
                  const isSelected = selectedTraceId === trace.trace_id;
                  const preview =
                    guardrail.hook === 'PRE'
                      ? trace.request_preview
                      : trace.response_preview;

                  return (
                    <div
                      key={trace.trace_id}
                      css={{
                        padding: theme.spacing.sm,
                        borderBottom: `1px solid ${theme.colors.border}`,
                        cursor: 'pointer',
                        backgroundColor: isSelected ? `${theme.colors.actionPrimaryBackgroundDefault}12` : 'transparent',
                        border: isSelected ? `2px solid ${theme.colors.actionPrimaryBackgroundDefault}` : '2px solid transparent',
                        borderRadius: theme.borders.borderRadiusMd,
                        '&:hover': { backgroundColor: theme.colors.tableRowHover },
                        '&:last-child': { borderBottom: 'none' },
                      }}
                      onClick={() => setSelectedTraceId(trace.trace_id)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          setSelectedTraceId(trace.trace_id);
                        }
                      }}
                    >
                      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                        <Typography.Text
                          css={{ fontSize: theme.typography.fontSizeSm, fontFamily: 'monospace' }}
                        >
                          {trace.trace_id.substring(0, 12)}...
                        </Typography.Text>
                        {trace.state && (
                          <span
                            css={{
                              fontSize: 11,
                              padding: '1px 6px',
                              borderRadius: 4,
                              backgroundColor: trace.state === 'OK' ? '#52c41a20' : '#ff4d4f20',
                              color: trace.state === 'OK' ? '#52c41a' : '#ff4d4f',
                              fontWeight: 600,
                            }}
                          >
                            {trace.state}
                          </span>
                        )}
                      </div>
                      <Typography.Text
                        color="secondary"
                        css={{
                          fontSize: theme.typography.fontSizeSm,
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          overflow: 'hidden',
                          wordBreak: 'break-word',
                        }}
                      >
                        {preview
                          ? preview.length > 150
                            ? `${preview.substring(0, 150)}...`
                            : preview
                          : '(no preview)'}
                      </Typography.Text>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Manual input */}
        {inputMode === 'manual' && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <Typography.Text bold>
              <FormattedMessage
                defaultMessage="Enter text to test"
                description="Label for manual text input in test modal"
              />
            </Typography.Text>
            <Input.TextArea
              componentId="mlflow.guardrails.test-modal.manual-input"
              value={manualText}
              onChange={(e) => setManualText(e.target.value)}
              rows={4}
              placeholder={intl.formatMessage({
                defaultMessage: 'Enter the text you want to test the guardrail against...',
                description: 'Placeholder for manual test input',
              })}
            />
          </div>
        )}

        {/* Test result */}
        {testError && (
          <Alert
            componentId="mlflow.gateway.guardrails.test-error"
            type="error"
            closable={false}
            message={testError}
          />
        )}

        {hasResult && (
          <div
            css={{
              border: `2px solid ${isPassed ? '#52c41a' : '#ff4d4f'}`,
              borderRadius: theme.borders.borderRadiusMd,
              overflow: 'hidden',
            }}
          >
            {/* Result header */}
            <div
              css={{
                padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                backgroundColor: isPassed ? '#52c41a12' : '#ff4d4f12',
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                borderBottom: `1px solid ${isPassed ? '#52c41a40' : '#ff4d4f40'}`,
              }}
            >
              <span css={{ fontSize: 20 }}>{isPassed ? '\u2705' : '\u274C'}</span>
              <Typography.Title level={4} css={{ margin: '0 !important' }}>
                {isPassed ? (
                  <FormattedMessage defaultMessage="Passed" description="Test passed label" />
                ) : (
                  <FormattedMessage defaultMessage="Failed" description="Test failed label" />
                )}
              </Typography.Title>
              {resultScore && (
                <span
                  css={{
                    marginLeft: 'auto',
                    fontSize: theme.typography.fontSizeSm,
                    padding: '2px 8px',
                    borderRadius: 4,
                    backgroundColor: isPassed ? '#52c41a20' : '#ff4d4f20',
                    color: isPassed ? '#52c41a' : '#ff4d4f',
                    fontWeight: 600,
                  }}
                >
                  score: {resultScore}
                </span>
              )}
            </div>

            {/* Rationale */}
            <div css={{ padding: theme.spacing.md }}>
              {resultRationale && (
                <>
                  <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                    <FormattedMessage defaultMessage="Rationale" description="Rationale label in test result" />
                  </Typography.Text>
                  <Typography.Text css={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {resultRationale}
                  </Typography.Text>
                </>
              )}

              {/* Input text used (only from /gateway/guardrails/test) */}
              {testResult?.input_text && (
                <>
                  <Typography.Text bold css={{ display: 'block', marginTop: theme.spacing.md, marginBottom: theme.spacing.xs }}>
                    <FormattedMessage defaultMessage="Tested against" description="Input text label in test result" />
                  </Typography.Text>
                  <div
                    css={{
                      padding: theme.spacing.sm,
                      backgroundColor: theme.colors.backgroundSecondary,
                      borderRadius: theme.borders.borderRadiusMd,
                      border: `1px solid ${theme.colors.border}`,
                      maxHeight: 120,
                      overflowY: 'auto',
                      fontSize: theme.typography.fontSizeSm,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      fontFamily: 'monospace',
                    }}
                  >
                    {testResult.input_text.length > 500
                      ? `${testResult.input_text.substring(0, 500)}...`
                      : testResult.input_text}
                  </div>
                </>
              )}

              {/* Modified text for mutation guardrails */}
              {testResult?.result?.modified_text && (
                <>
                  <Typography.Text bold css={{ display: 'block', marginTop: theme.spacing.md, marginBottom: theme.spacing.xs }}>
                    <FormattedMessage defaultMessage="Modified output" description="Modified text label in test result" />
                  </Typography.Text>
                  <div
                    css={{
                      padding: theme.spacing.sm,
                      backgroundColor: '#722ed108',
                      borderRadius: theme.borders.borderRadiusMd,
                      border: `1px solid #722ed140`,
                      maxHeight: 120,
                      overflowY: 'auto',
                      fontSize: theme.typography.fontSizeSm,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      fontFamily: 'monospace',
                    }}
                  >
                    {testResult.result.modified_text}
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
};

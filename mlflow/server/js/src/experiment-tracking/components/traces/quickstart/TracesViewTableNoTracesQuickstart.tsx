import { useState, type ReactNode } from 'react';
import {
  SegmentedControlButton,
  SegmentedControlGroup,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import {
  QUICKSTART_CONTENT,
  PYTHON_FRAMEWORK_OPTIONS,
  TS_FRAMEWORK_OPTIONS,
  getPythonConnectCode,
  TS_INSTALL_CODE,
  getTsConnectCode,
  getTsFrameworkCode,
  OTEL_INSTALL_CODE,
  OTEL_INSTRUMENT_CODE,
  getOtelEnvCode,
  type QUICKSTART_FLAVOR,
} from './TraceTableQuickstart.utils';
import { CodeBlock } from './components/CodeBlock';
import { StepSection } from './components/StepSection';
import { LanguageTab, type Language } from './components/LanguageTab';

const TRACING_VIDEO_START_SEC = 24;
const TRACING_VIDEO_URL = `https://mlflow.org/docs/latest/images/llms/tracing/tracing-top.mp4#t=${TRACING_VIDEO_START_SEC}`;

export const TracesViewTableNoTracesQuickstart = ({
  baseComponentId,
  experimentName,
  experimentId,
}: {
  baseComponentId: string;
  runUuid?: string;
  experimentName?: string;
  experimentId?: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const [language, setLanguage] = useState<Language>('python');
  const [selectedPythonFramework, setSelectedPythonFramework] = useState<QUICKSTART_FLAVOR>('openai');
  const [selectedTsFramework, setSelectedTsFramework] = useState<string>('openai');
  const [videoFailed, setVideoFailed] = useState(false);

  const hostname = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
  const trackingUri = `http://${hostname}:<port>`;
  const pythonConnectCode = getPythonConnectCode(trackingUri, experimentName || 'my-experiment');
  const tsConnectCode = getTsConnectCode(trackingUri, experimentId || '<experiment-id>');
  const otelEnvCode = getOtelEnvCode(trackingUri, experimentId || '<experiment-id>');

  const pythonCode = QUICKSTART_CONTENT[selectedPythonFramework].getCodeSource();
  const tsFrameworkCode = getTsFrameworkCode(trackingUri, experimentId || '<experiment-id>');
  const tsFramework = tsFrameworkCode[selectedTsFramework as keyof typeof tsFrameworkCode];
  const frameworkOptions = language === 'python' ? PYTHON_FRAMEWORK_OPTIONS : TS_FRAMEWORK_OPTIONS;
  const selectedFramework = language === 'python' ? selectedPythonFramework : selectedTsFramework;

  const handleFrameworkSelect = (key: string) => {
    if (language === 'python') {
      setSelectedPythonFramework(key as QUICKSTART_FLAVOR);
    } else {
      setSelectedTsFramework(key);
    }
  };

  return (
    <div
      css={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: `${theme.spacing.lg * 2}px ${theme.spacing.lg}px ${theme.spacing.lg * 3}px`,
        maxWidth: 860,
        margin: '0 auto',
      }}
    >
      {/* Hero section */}
      <Typography.Title level={2} css={{ textAlign: 'center', marginBottom: theme.spacing.xs }}>
        <FormattedMessage
          defaultMessage="Start tracing your LLM application"
          description="Title for the empty state when no traces are recorded"
        />
      </Typography.Title>
      <Typography.Paragraph
        color="secondary"
        css={{ textAlign: 'center', maxWidth: 520, marginBottom: theme.spacing.lg * 2 }}
      >
        <FormattedMessage
          defaultMessage="Traces show you how your LLM calls behave: what they cost, how they perform, and where things go wrong. Get started in a few minutes."
          description="Subtitle for the empty state when no traces are recorded"
        />
      </Typography.Paragraph>

      {/* Video preview (hidden if the remote video fails to load, e.g. offline) */}
      {!videoFailed && (
        <div
          css={{
            width: '100%',
            marginBottom: theme.spacing.lg * 2,
            borderRadius: theme.borders.borderRadiusMd,
            overflow: 'hidden',
            border: `1px solid ${theme.colors.border}`,
            boxShadow: '0 4px 24px rgba(0, 0, 0, 0.08)',
          }}
        >
          <video
            src={TRACING_VIDEO_URL}
            autoPlay
            muted
            playsInline
            onError={() => setVideoFailed(true)}
            onEnded={(event) => {
              const video = event.currentTarget;
              video.currentTime = TRACING_VIDEO_START_SEC;
              video.play().catch(() => {
                // Autoplay restrictions can reject; ignore since the video is muted.
              });
            }}
            css={{ width: '100%', display: 'block' }}
          />
        </div>
      )}

      {/* Language selector */}
      <div css={{ width: '100%', marginBottom: theme.spacing.lg }}>
        <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="Select your development language"
            description="Language selector heading for traces onboarding"
          />
        </Typography.Title>
        <LanguageTab language={language} setLanguage={setLanguage} />
      </div>

      {/* Steps */}
      <div css={{ width: '100%', display: 'flex', flexDirection: 'column', gap: theme.spacing.lg * 1.5 }}>
        <ConnectStep
          theme={theme}
          language={language}
          pythonConnectCode={pythonConnectCode}
          tsConnectCode={tsConnectCode}
          otelEnvCode={otelEnvCode}
        />
        <InstrumentStep
          theme={theme}
          language={language}
          frameworkOptions={frameworkOptions}
          selectedFramework={selectedFramework}
          onSelectFramework={handleFrameworkSelect}
          pythonCode={pythonCode}
          tsFramework={tsFramework}
        />
        <VerifyStep theme={theme} />
      </div>
    </div>
  );
};

/**
 * Step 1: Connect / Install
 */
const ConnectStep = ({
  theme,
  language,
  pythonConnectCode,
  tsConnectCode,
  otelEnvCode,
}: {
  theme: ReturnType<typeof useDesignSystemTheme>['theme'];
  language: Language;
  pythonConnectCode: string;
  tsConnectCode: string;
  otelEnvCode: string;
}) => {
  const title = (() => {
    if (language === 'python') {
      return (
        <FormattedMessage
          defaultMessage="Connect to the tracking server"
          description="Step 1 title for Python traces onboarding"
        />
      );
    }
    if (language === 'typescript') {
      return (
        <FormattedMessage
          defaultMessage="Install and initialize the MLflow TypeScript SDK"
          description="Step 1 title for TypeScript traces onboarding"
        />
      );
    }
    return (
      <FormattedMessage
        defaultMessage="Configure OpenTelemetry export"
        description="Step 1 title for OpenTelemetry traces onboarding"
      />
    );
  })();

  const description = (() => {
    if (language === 'python') {
      return (
        <FormattedMessage
          defaultMessage="Configure your Python environment to send traces to this MLflow server."
          description="Step 1 description for Python traces onboarding"
        />
      );
    }
    if (language === 'typescript') {
      return (
        <FormattedMessage
          defaultMessage="Install the <a>MLflow tracing SDK</a> for TypeScript using npm, then initialize it in your application."
          description="Step 1 description for TypeScript traces onboarding"
          values={{
            a: (text: string) => (
              <Typography.Link
                componentId="mlflow.traces.onboarding.npm_link"
                href="https://www.npmjs.com/package/@mlflow/core"
                openInNewTab
              >
                {text}
              </Typography.Link>
            ),
          }}
        />
      );
    }
    return (
      <FormattedMessage
        defaultMessage="Point the OTLP exporter at this MLflow server by setting these environment variables. See the <a>OpenTelemetry export docs</a> for the full reference."
        description="Step 1 description for OpenTelemetry traces onboarding"
        values={{
          a: (text: string) => (
            <Typography.Link
              componentId="mlflow.traces.onboarding.otel_docs_link"
              href="https://mlflow.org/docs/latest/genai/tracing/opentelemetry/export"
              openInNewTab
            >
              {text}
            </Typography.Link>
          ),
        }}
      />
    );
  })();

  return (
    <StepSection theme={theme} stepNumber={1} title={title} description={description}>
      {language === 'python' && (
        <CodeBlock
          theme={theme}
          code={pythonConnectCode}
          language="python"
          componentId="mlflow.traces.onboarding.step1.copy"
        />
      )}
      {language === 'typescript' && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <CodeBlock
            theme={theme}
            code={TS_INSTALL_CODE}
            language="bash"
            componentId="mlflow.traces.onboarding.step1.install.copy"
          />
          <CodeBlock
            theme={theme}
            code={tsConnectCode}
            language="typescript"
            componentId="mlflow.traces.onboarding.step1.connect.copy"
          />
        </div>
      )}
      {language === 'opentelemetry' && (
        <CodeBlock
          theme={theme}
          code={otelEnvCode}
          language="bash"
          componentId="mlflow.traces.onboarding.step1.otel_env.copy"
        />
      )}
    </StepSection>
  );
};

/**
 * Step 2: Add tracing
 */
const InstrumentStep = ({
  theme,
  language,
  frameworkOptions,
  selectedFramework,
  onSelectFramework,
  pythonCode,
  tsFramework,
}: {
  theme: ReturnType<typeof useDesignSystemTheme>['theme'];
  language: Language;
  frameworkOptions: { key: string; label: string }[];
  selectedFramework: string;
  onSelectFramework: (key: string) => void;
  pythonCode: string;
  tsFramework: { install: string; code: string };
}) => {
  const codeStyle = {
    fontSize: 12,
    backgroundColor: theme.colors.backgroundSecondary,
    padding: '1px 4px',
    borderRadius: 3,
  } as const;

  let description: ReactNode;
  if (language === 'python') {
    description = (
      <FormattedMessage
        defaultMessage="Choose your framework and add one line to enable automatic tracing, or use the {code} decorator for custom functions."
        description="Step 2 description for Python traces onboarding"
        values={{ code: <code css={codeStyle}>@mlflow.trace</code> }}
      />
    );
  } else if (language === 'typescript') {
    description = (
      <FormattedMessage
        defaultMessage="Choose your integration. Install the {code} package for automatic tracing, or use {trace} for custom functions."
        description="Step 2 description for TypeScript traces onboarding"
        values={{
          code: <code css={codeStyle}>@mlflow/openai</code>,
          trace: <code css={codeStyle}>mlflow.trace()</code>,
        }}
      />
    );
  } else {
    description = (
      <FormattedMessage
        defaultMessage="Install the OpenTelemetry SDK and instrument your application. Spans will be exported to MLflow via OTLP using the env vars set above."
        description="Step 2 description for OpenTelemetry traces onboarding"
      />
    );
  }

  return (
    <StepSection
      theme={theme}
      stepNumber={2}
      title={
        <FormattedMessage
          defaultMessage="Add tracing to your application"
          description="Step 2 title for traces onboarding"
        />
      }
      description={description}
    >
      {language === 'opentelemetry' ? (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <CodeBlock
            theme={theme}
            code={OTEL_INSTALL_CODE}
            language="bash"
            componentId="mlflow.traces.onboarding.step2.otel_install.copy"
          />
          <CodeBlock
            theme={theme}
            code={OTEL_INSTRUMENT_CODE}
            language="python"
            componentId="mlflow.traces.onboarding.step2.otel_code.copy"
          />
        </div>
      ) : (
        <>
          <Typography.Paragraph color="secondary" css={{ fontSize: 12, marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Don't see your framework? <a>Browse all integrations</a>."
              description="Link to MLflow tracing integrations doc page above the framework selector"
              values={{
                a: (text: string) => (
                  <Typography.Link
                    componentId="mlflow.traces.onboarding.integrations_link"
                    href="https://mlflow.org/docs/latest/genai/tracing/integrations"
                    openInNewTab
                  >
                    {text}
                  </Typography.Link>
                ),
              }}
            />
          </Typography.Paragraph>
          {/* Framework selector */}
          <div css={{ marginBottom: theme.spacing.sm }}>
            <SegmentedControlGroup
              spaced
              name="mlflow.traces.onboarding.framework-selector"
              componentId="mlflow.traces.onboarding.framework_selector"
              value={selectedFramework}
              onChange={(event) => onSelectFramework(event.target.value)}
            >
              {frameworkOptions.map(({ key, label }) => (
                <SegmentedControlButton key={key} value={key}>
                  {label}
                </SegmentedControlButton>
              ))}
            </SegmentedControlGroup>
          </div>

          {language === 'python' ? (
            <CodeBlock
              theme={theme}
              code={pythonCode}
              language="python"
              componentId="mlflow.traces.onboarding.step2.copy"
            />
          ) : (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {tsFramework.install && (
                <CodeBlock
                  theme={theme}
                  code={tsFramework.install}
                  language="bash"
                  componentId="mlflow.traces.onboarding.step2.install.copy"
                />
              )}
              <CodeBlock
                theme={theme}
                code={tsFramework.code}
                language="typescript"
                componentId="mlflow.traces.onboarding.step2.code.copy"
              />
            </div>
          )}
        </>
      )}
    </StepSection>
  );
};

/**
 * Step 3: View traces
 */
const VerifyStep = ({ theme }: { theme: ReturnType<typeof useDesignSystemTheme>['theme'] }) => {
  return (
    <StepSection
      theme={theme}
      stepNumber={3}
      title={<FormattedMessage defaultMessage="View your traces" description="Step 3 title for traces onboarding" />}
      description={
        <FormattedMessage
          defaultMessage="Run your application and traces will appear here automatically. For more information, visit the <a>MLflow Tracing documentation</a>."
          description="Step 3 description for traces onboarding"
          values={{
            a: (text: string) => (
              <Typography.Link
                componentId="mlflow.traces.onboarding.docs_link"
                href="https://mlflow.org/docs/latest/genai/tracing/"
                openInNewTab
              >
                {text}
              </Typography.Link>
            ),
          }}
        />
      }
      isPending
    />
  );
};

import { useState } from 'react';
import {
  Spacer,
  Typography,
  useDesignSystemTheme,
  CopyIcon,
  NewWindowIcon,
  BeakerIcon,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { Link } from '../../common/utils/RoutingUtils';
import Routes from '../../experiment-tracking/routes';
import OpenAiLogo from '../../common/static/logos/openai.svg';
import OpenAiLogoDark from '../../common/static/logos/openai-dark.svg';
import ScikitLearnLogo from '../../common/static/logos/scikit-learn.svg';
import { useHomePageViewState } from '../HomePageViewStateContext';
import { HomeQuickstartDrawer } from './HomeQuickstartDrawer';

type EvaluationFrameworkId = 'openai' | 'scikit_learn';

type EvaluationFrameworkDefinition = {
  id: EvaluationFrameworkId;
  message: string;
  snippet: string;
  logo?: string;
  selectedLogo?: string;
};

const EVALUATION_DOCS_URL = 'https://mlflow.org/docs/latest/llms/llm-evaluate/index.html';
const CONFIGURE_EVALUATION_EXPERIMENT_SNIPPET = `import mlflow

# Specify the tracking server URI, e.g. http://localhost:5000
mlflow.set_tracking_uri("http://<tracking-server-host>:<port>")
# If the experiment with the name "evaluations-quickstart" doesn't exist, MLflow will create it
mlflow.set_experiment("evaluations-quickstart")`;

const evaluationFrameworks: EvaluationFrameworkDefinition[] = [
  {
    id: 'openai',
    message: 'OpenAI',
    logo: OpenAiLogo,
    selectedLogo: OpenAiLogoDark,
    snippet: `import os
import openai
import mlflow
from mlflow.genai.scorers import Correctness, Guidelines

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define a simple QA dataset
dataset = [
    {
        "inputs": {"question": "Can MLflow manage prompts?"},
        "expectations": {"expected_response": "Yes!"},
    },
    {
        "inputs": {"question": "Can MLflow create a taco for my lunch?"},
        "expectations": {
            "expected_response": "No, unfortunately, MLflow is not a taco maker."
        },
    },
]


# 2. Define a prediction function to generate responses
def predict_fn(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content


# 3. Run the evaluation
results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=[
        # Built-in LLM judge
        Correctness(),
        # Custom criteria using LLM judge
        Guidelines(name="is_english", guidelines="The answer must be in English"),
    ],
)
`,
  },
  {
    id: 'scikit_learn',
    message: 'Scikit-learn',
    logo: ScikitLearnLogo,
    selectedLogo: ScikitLearnLogo,
    snippet: `import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Load and prepare data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create evaluation dataset
eval_df = pd.DataFrame(X_test, columns=wine.feature_names)
eval_df["target"] = y_test

with mlflow.start_run():
    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Comprehensive evaluation with one line
    result = mlflow.models.evaluate(
        model="models:/my-model/1",
        data=eval_df,
        targets="target",
        model_type="classifier",
        evaluators=["default"],
    )
`,
  },
];

export const RunEvaluationDrawer = () => {
  const { theme } = useDesignSystemTheme();
  const { isRunEvaluationDrawerOpen, closeRunEvaluationDrawer } = useHomePageViewState();
  const [selectedFramework, setSelectedFramework] = useState<EvaluationFrameworkId>('openai');

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      setSelectedFramework('openai');
      closeRunEvaluationDrawer();
    }
  };

  const selectedDefinition = evaluationFrameworks.find((framework) => framework.id === selectedFramework);
  const snippet = selectedDefinition?.snippet ?? '';

  return (
    <HomeQuickstartDrawer
      componentId="mlflow.home.run_evaluation.drawer"
      icon={<BeakerIcon css={{ width: 20, height: 20 }} />}
      title={
        <FormattedMessage
          defaultMessage="Run evaluation"
          description="Title for the run evaluation drawer on the Home page"
        />
      }
      intro={
        <FormattedMessage
          defaultMessage="Select an example and follow the steps to evaluate your models with MLflow."
          description="Helper text shown at the top of the run evaluation drawer"
        />
      }
      isOpen={isRunEvaluationDrawerOpen}
      onOpenChange={handleOpenChange}
      frameworks={evaluationFrameworks.map((framework) => ({
        id: framework.id,
        label: framework.message,
        logo: framework.logo,
        selectedLogo: framework.selectedLogo,
        buttonComponentId: `mlflow.home.run_evaluation.drawer.framework.${framework.id}`,
      }))}
      selectedFramework={selectedFramework}
      onSelectFramework={setSelectedFramework}
      contentCss={{ maxWidth: 860 }}
      leftFooter={
        <a
          href={EVALUATION_DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
          css={{
            marginTop: theme.spacing.sm,
            display: 'inline-flex',
            gap: theme.spacing.xs,
            alignItems: 'center',
            fontWeight: theme.typography.typographyBoldFontWeight,
          }}
        >
          <FormattedMessage
            defaultMessage="Learn more about evaluations"
            description="Link text directing users to evaluation documentation"
          />
          <NewWindowIcon css={{ fontSize: 14 }} />
        </a>
      }
    >
      <section
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
        }}
      >
        <Typography.Title level={4} css={{ margin: 0 }}>
          <FormattedMessage
            defaultMessage="1. Configure experiment and tracking URI"
            description="Section title for configuring experiment and tracking URI before running evaluations"
          />
        </Typography.Title>
        <div css={{ position: 'relative', width: 'min(100%, 800px)' }}>
          <CopyButton
            componentId="mlflow.home.run_evaluation.drawer.configure.copy"
            css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
            showLabel={false}
            copyText={CONFIGURE_EVALUATION_EXPERIMENT_SNIPPET}
            icon={<CopyIcon />}
          />
          <CodeSnippet
            showLineNumbers
            language="python"
            theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
            style={{
              padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
              marginTop: theme.spacing.xs,
            }}
          >
            {CONFIGURE_EVALUATION_EXPERIMENT_SNIPPET}
          </CodeSnippet>
        </div>
      </section>

      <section
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
        }}
      >
        <Typography.Title level={4} css={{ margin: 0 }}>
          <FormattedMessage
            defaultMessage="2. Run the evaluation"
            description="Section title introducing evaluation example snippet"
          />
        </Typography.Title>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Use the template below as a starting point for your evaluation script."
            description="Description for the evaluation code snippet"
          />
        </Typography.Text>
        <div css={{ position: 'relative', width: 'min(100%, 800px)' }}>
          <CopyButton
            componentId={`mlflow.home.run_evaluation.drawer.copy.${selectedFramework}`}
            css={{ zIndex: 1, position: 'absolute', top: theme.spacing.xs, right: theme.spacing.xs }}
            showLabel={false}
            copyText={snippet}
            icon={<CopyIcon />}
          />
          <CodeSnippet
            showLineNumbers
            language="python"
            theme={theme.isDarkMode ? 'duotoneDark' : 'light'}
            style={{
              padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
              marginTop: theme.spacing.xs,
            }}
          >
            {snippet}
          </CodeSnippet>
        </div>
      </section>

      <section
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
        }}
      >
        <Typography.Title level={4} css={{ margin: 0 }}>
          <FormattedMessage
            defaultMessage="3. Review evaluation results"
            description="Section title for viewing evaluation results"
          />
        </Typography.Title>
        <Typography.Text color="secondary" css={{ margin: 0 }}>
          <FormattedMessage
            defaultMessage="Open the MLflow UI to explore runs and compare evaluation outcomes."
            description="Introductory text for reviewing evaluation results"
          />
          <Spacer size="sm" />
          <ol
            css={{
              margin: 0,
              paddingLeft: theme.spacing.lg,
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.xs,
            }}
          >
            <li>
              <FormattedMessage
                defaultMessage="Open the {experimentsLink} page."
                description="Instruction to open experiments page from evaluation drawer"
                values={{
                  experimentsLink: (
                    <Link
                      to={Routes.experimentsObservatoryRoute}
                      target="_blank"
                      rel="noopener noreferrer"
                      css={{ display: 'inline-flex', gap: theme.spacing.xs, alignItems: 'center' }}
                    >
                      <FormattedMessage
                        defaultMessage="Experiments"
                        description="Link label for experiments page from evaluation drawer"
                      />
                      <NewWindowIcon css={{ fontSize: 14 }} />
                    </Link>
                  ),
                }}
              />
            </li>
            <li>
              <FormattedMessage
                defaultMessage="Select the run that logged your evaluation metrics."
                description="Instruction to select evaluation run"
              />
            </li>
            <li>
              <FormattedMessage
                defaultMessage="Compare metrics across runs to identify performance improvements."
                description="Instruction to compare evaluation results"
              />
            </li>
          </ol>
        </Typography.Text>
      </section>
      <section>
        <Typography.Text color="secondary" css={{ margin: 0 }}>
          <FormattedMessage
            defaultMessage="Learn more about evaluations in the {evaluationDocs}."
            description="Instruction to learn more about evaluations in the documentation"
            values={{
              evaluationDocs: (
                <a
                  href={EVALUATION_DOCS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  css={{ display: 'inline-flex', gap: theme.spacing.xs, alignItems: 'center' }}
                >
                  <FormattedMessage
                    defaultMessage="MLflow documentation"
                    description="Link to evaluation documentation"
                  />
                  <NewWindowIcon css={{ fontSize: 14 }} />
                </a>
              ),
            }}
          />
        </Typography.Text>
      </section>
    </HomeQuickstartDrawer>
  );
};

export default RunEvaluationDrawer;

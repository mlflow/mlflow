import type { getOctokit } from "@actions/github";
import type { context as ContextType } from "@actions/github";
import OpenAI from "openai";

type GitHub = ReturnType<typeof getOctokit>;
type Context = typeof ContextType;

export async function generateAutoLabelPrompt({
  github,
  context,
}: {
  github: GitHub;
  context: Context;
}) {
  const { owner, repo } = context.repo;

  const allLabels = await github.paginate(github.rest.issues.listLabelsForRepo, {
    owner,
    repo,
    per_page: 100,
  });

  const filteredLabels = allLabels
    .map((label) => label.name)
    .filter((name) => name.startsWith("area/") || name.startsWith("domain/"))
    .join("\n");

  let issueNumber;
  if (context.eventName === "pull_request") {
    const latestIssues = await github.rest.issues.listForRepo({
      owner,
      repo,
      state: "open",
      sort: "created",
      direction: "desc",
      per_page: 1,
    });
    issueNumber = latestIssues.data[0].number;
  } else {
    issueNumber = context.issue.number;
  }

  const issue = await github.rest.issues.get({
    owner,
    repo,
    issue_number: issueNumber,
  });

  const issueData = {
    title: issue.data.title,
    body: issue.data.body || "",
    labels: issue.data.labels
      .map((label) => (typeof label === "string" ? label : label.name))
      .filter((name): name is string => name !== undefined),
  };

  const currentLabels = issueData.labels.join(", ");

  const prompt = `You're an issue triage assistant for GitHub issues. Your task is to analyze the issue and list appropriate labels from the provided list.

CURRENT ISSUE CONTENT:
Title: ${issueData.title}

Body:
${issueData.body}

CURRENT LABELS ON ISSUE:
${currentLabels}

AVAILABLE AREA AND DOMAIN LABELS:
${filteredLabels}

TASK OVERVIEW:

Analyze the issue content above and list appropriate area labels (e.g., \`area/tracing\`) and domain labels (e.g., \`domain/genai\`) that should be added to this issue.

IMPORTANT GUIDELINES:

- Be thorough in your analysis
- Only suggest labels from the AVAILABLE AREA AND DOMAIN LABELS list above
- Don't suggest labels that are already on the issue
- It's okay to not add any labels if none are clearly applicable
- Only suggest area/ and domain/ labels`;

  return prompt;
}

export async function getAutoLabelsFromOpenAI({
  apiKey,
  baseUrl,
  prompt,
}: {
  apiKey: string;
  baseUrl: string;
  prompt: string;
}): Promise<string> {
  const client = new OpenAI({
    apiKey,
    baseURL: baseUrl,
  });

  const response = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: prompt }],
    max_tokens: 1000,
    temperature: 0.0,
  });

  return response.choices[0].message.content || "";
}

export async function autoLabel({
  github,
  context,
}: {
  github: GitHub;
  context: Context;
}): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY;
  const baseUrl = process.env.OPENAI_API_BASE;

  if (!apiKey) {
    throw new Error("OPENAI_API_KEY environment variable is not set");
  }

  if (!baseUrl) {
    throw new Error("OPENAI_API_BASE environment variable is not set");
  }

  const prompt = await generateAutoLabelPrompt({ github, context });
  return getAutoLabelsFromOpenAI({ apiKey, baseUrl, prompt });
}

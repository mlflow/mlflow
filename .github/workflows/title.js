function extractStackedPrBaseSha(prBody, headRef) {
  if (!prBody || !prBody.includes("Stacked PR")) {
    return null;
  }

  const marker = `[**${headRef}**]`;
  const lines = prBody.split("\n");

  for (const line of lines) {
    if (line.includes(marker)) {
      const match = line.match(/\/files\/(?<base>[a-f0-9]{7,40})\.\.(?<head>[a-f0-9]{7,40})/);
      if (match) {
        return match.groups.base;
      }
    }
  }

  return null;
}

async function getPrInfo(github, owner, repo, prNumber) {
  console.error("Fetching PR information...");

  const { data } = await github.rest.pulls.get({
    owner,
    repo,
    pull_number: prNumber,
  });

  return {
    title: data.title,
    body: data.body || "",
    headSha: data.head.sha,
    headRef: data.head.ref,
  };
}

async function getPrDiff(github, owner, repo, prNumber, body, headSha, headRef) {
  console.error("Fetching PR diff...");

  let diff;
  const baseSha = extractStackedPrBaseSha(body, headRef);

  if (baseSha) {
    console.error(
      `Detected stacked PR, fetching incremental diff: ${baseSha.substring(
        0,
        7
      )}..${headSha.substring(0, 7)}`
    );
    const response = await github.request("GET /repos/{owner}/{repo}/compare/{basehead}", {
      owner,
      repo,
      basehead: `${baseSha}...${headSha}`,
      headers: {
        accept: "application/vnd.github.v3.diff",
      },
    });
    diff = response.data;
  } else {
    const response = await github.rest.pulls.get({
      owner,
      repo,
      pull_number: prNumber,
      mediaType: {
        format: "diff",
      },
    });
    diff = response.data;
  }

  const maxLength = 50000;
  if (diff.length > maxLength) {
    diff = diff.substring(0, maxLength) + "\n\n... [diff truncated due to length] ...";
  }

  return diff;
}

async function getClosingIssues(github, owner, repo, prNumber) {
  console.error("Fetching closing issues...");

  const query = `
    query($owner: String!, $repo: String!, $prNumber: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $prNumber) {
          closingIssuesReferences(first: 10) {
            nodes {
              number
              title
            }
          }
        }
      }
    }
  `;

  try {
    const result = await github.graphql(query, {
      owner,
      repo,
      prNumber,
    });

    const nodes = result?.repository?.pullRequest?.closingIssuesReferences?.nodes;
    if (nodes && Array.isArray(nodes)) {
      return nodes.map((node) => ({
        number: node.number,
        title: node.title,
      }));
    }
    return [];
  } catch (error) {
    console.error(`Warning: Failed to fetch closing issues: ${error.message}`);
    return [];
  }
}

function extractDescription(body) {
  const pattern = /### What changes are proposed in this pull request\?\s*(.+?)(?=###|$)/is;
  const match = body.match(pattern);

  if (match) {
    return match[1].trim();
  }

  return body;
}

function buildPrompt(title, body, diff, linkedIssues = null) {
  const description = extractDescription(body).trim() || "(No description provided)";

  let linkedIssuesSection = "";
  if (linkedIssues && linkedIssues.length > 0) {
    linkedIssuesSection = "\n## Linked Issues\n";
    for (const { number, title } of linkedIssues) {
      linkedIssuesSection += `- #${number}: ${title}\n`;
    }
    linkedIssuesSection += "\n";
  }

  return `Rewrite the PR title to be more descriptive and follow the guidelines below.

## Current PR Title
${title}

## PR Description
${description}
${linkedIssuesSection}
## Code Changes (Diff)
\`\`\`diff
${diff}
\`\`\`

## Guidelines for a good PR title:
1. Start with a verb in imperative mood (e.g., "Add", "Fix", "Update", "Remove", "Refactor")
2. Be specific about what changed and where
3. Keep it concise (aim for 72 characters or less, 100 characters maximum)
4. Do not include issue numbers in the title (they belong in the PR body)
5. Focus on the "what" and "why", not the "how"
6. Use proper capitalization (capitalize first letter, no period at end)
7. Use backticks for code/file references (e.g., \`ClassName\`, \`function_name\`, \`module.path\`)

Rewrite the PR title following these guidelines.`;
}

function buildIssuePrompt(title, body) {
  const description = body.trim() || "(No description provided)";

  return `Rewrite the issue title to be more descriptive and follow the guidelines below.

## Current Issue Title
${title}

## Issue Description
${description}

## Guidelines for a good issue title:
1. Clearly describe the problem or feature request (e.g., "\`load_model\` fails with \`KeyError\`
   when model has nested flavors", "Support custom metric types in autologging")
2. Be specific about what the issue is about
3. Keep it concise (aim for 72 characters or less, 100 characters maximum)
4. Do not include issue numbers in the title
5. Focus on the problem or feature request
6. Use proper capitalization (capitalize first letter, no period at end)
7. Use backticks for code/file references (e.g., \`ClassName\`, \`function_name\`, \`module.path\`)

Rewrite the issue title following these guidelines.`;
}

async function callAnthropicApi(prompt, apiKey) {
  console.error("Calling Claude API...");

  const requestBody = {
    model: "claude-haiku-4-5-20251001",
    max_tokens: 256,
    messages: [{ role: "user", content: prompt }],
    output_config: {
      format: {
        type: "json_schema",
        schema: {
          type: "object",
          properties: {
            title: {
              type: "string",
              description: "The rewritten PR title.",
            },
          },
          required: ["title"],
          additionalProperties: false,
        },
      },
    },
  };

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    throw new Error(`Anthropic API error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  console.error("API Response:");
  console.error(JSON.stringify(data, null, 2));

  const content = JSON.parse(data.content[0].text);
  return content.title;
}

async function getIssueInfo(github, owner, repo, issueNumber) {
  console.error("Fetching issue information...");

  const { data } = await github.rest.issues.get({
    owner,
    repo,
    issue_number: issueNumber,
  });

  return {
    title: data.title,
    body: data.body || "",
  };
}

module.exports = async ({ github, context, core }) => {
  // Parse inputs from environment
  const repo = process.env.REPO;
  const number = parseInt(process.env.NUMBER, 10);
  const isPr = process.env.IS_PR === "true";
  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (!repo || !number || !apiKey) {
    throw new Error("Missing required environment variables: REPO, NUMBER, ANTHROPIC_API_KEY");
  }

  const [owner, repoName] = repo.split("/");

  let newTitle;

  if (isPr) {
    // Generate PR title
    const { title, body, headSha, headRef } = await getPrInfo(github, owner, repoName, number);
    console.error(`Original title: ${title}`);

    const diff = await getPrDiff(github, owner, repoName, number, body, headSha, headRef);
    const linkedIssues = await getClosingIssues(github, owner, repoName, number);

    if (linkedIssues.length > 0) {
      console.error(`Found ${linkedIssues.length} linked issue(s)`);
    }

    const prompt = buildPrompt(title, body, diff, linkedIssues);
    newTitle = await callAnthropicApi(prompt, apiKey);
  } else {
    // Generate issue title
    const { title, body } = await getIssueInfo(github, owner, repoName, number);
    console.error(`Original title: ${title}`);

    const prompt = buildIssuePrompt(title, body);
    newTitle = await callAnthropicApi(prompt, apiKey);
  }

  console.log(newTitle);
  core.setOutput("new_title", newTitle);
};

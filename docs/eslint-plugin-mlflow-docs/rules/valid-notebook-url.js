const fs = require("fs");
const path = require("path");

/**
 * ESLint rule to validate NotebookDownloadButton URLs.
 *
 * This rule ensures that NotebookDownloadButton components have valid MLflow repository URLs
 * that point to existing files in the repository.
 *
 * @type {import('eslint').Rule.RuleModule}
 */
module.exports = {
  meta: {
    type: "problem",
    docs: {
      description: "Detect NotebookDownloadButton with invalid MLflow repository URLs",
      category: "Possible Errors",
    },
    fixable: null,
    schema: [],
    messages: {
      missingHref: "NotebookDownloadButton is missing href attribute",
      emptyHref: "NotebookDownloadButton href is empty",
      invalidFormat:
        'NotebookDownloadButton href must start with "https://raw.githubusercontent.com/mlflow/mlflow/master/"',
      fileNotFound: 'NotebookDownloadButton href points to non-existent file: "{{path}}"',
    },
  },

  /**
   * Creates the rule implementation.
   *
   * @param {import('eslint').Rule.RuleContext} context - The ESLint rule context
   * @returns {import('eslint').Rule.RuleListener} The rule visitor methods
   */
  create(context) {
    return {
      /**
       * Validates JSX opening elements for NotebookDownloadButton components.
       *
       * @param {import('estree').Node} node - The JSX opening element node
       */
      JSXOpeningElement(node) {
        // Check if it's <NotebookDownloadButton >
        if (
          node.type !== "JSXOpeningElement" ||
          node.name.type !== "JSXIdentifier" ||
          node.name.name !== "NotebookDownloadButton"
        ) {
          return;
        }

        // Find href attribute
        const hrefAttr = node.attributes.find(
          (attr) => attr.type === "JSXAttribute" && attr.name.name === "href"
        );

        if (!hrefAttr) {
          context.report({
            node,
            messageId: "missingHref",
          });
          return;
        }

        // Get href value
        const hrefValue = getHrefValue(hrefAttr);

        if (!hrefValue) {
          context.report({
            node: hrefAttr,
            messageId: "emptyHref",
          });
          return;
        }

        // Validate MLflow repository URL
        validateMlflowUrl(context, hrefAttr, hrefValue);
      },
    };
  },
};

/**
 * Extracts the href value from a JSX attribute.
 *
 * @param {import('estree-jsx').JSXAttribute} attr - The JSX attribute node
 * @returns {string|null} The href value or null if it can't be determined
 */
function getHrefValue(attr) {
  // Handle string literals: href="value"
  if (attr.value.type === "Literal") {
    return attr.value.value;
  }

  // Handle JSX expressions with literals: href={"value"}
  if (attr.value.type === "JSXExpressionContainer" && attr.value.expression.type === "Literal") {
    return attr.value.expression.value;
  }

  // Handle template literals with no expressions: href={`value`}
  if (
    attr.value.type === "JSXExpressionContainer" &&
    attr.value.expression.type === "TemplateLiteral" &&
    attr.value.expression.expressions.length === 0
  ) {
    return attr.value.expression.quasis[0].value.raw;
  }

  // Can't determine value for dynamic expressions
  return null;
}

/**
 * Validates that an href points to a valid MLflow repository URL.
 *
 * Checks two things:
 * 1. URL format: Must start with https://raw.githubusercontent.com/mlflow/mlflow/master/
 * 2. File existence: The referenced file must exist in the local repository
 *
 * @param {import('eslint').Rule.RuleContext} context - The ESLint rule context
 * @param {import('estree-jsx').JSXAttribute} hrefAttr - The href attribute node
 * @param {string} href - The href value to validate
 */
function validateMlflowUrl(context, hrefAttr, href) {
  const expectedPrefix = "https://raw.githubusercontent.com/mlflow/mlflow/master/";

  // Check if URL starts with the expected prefix
  if (!href.startsWith(expectedPrefix)) {
    context.report({
      node: hrefAttr,
      messageId: "invalidFormat",
    });
    return;
  }

  // Extract the path after the prefix
  const filePath = href.substring(expectedPrefix.length);

  // Check if the file exists in the repository
  const repoRoot = findRepoRoot(context.filename);
  if (repoRoot) {
    const fullPath = path.join(repoRoot, filePath);

    if (!fs.existsSync(fullPath)) {
      context.report({
        node: hrefAttr,
        messageId: "fileNotFound",
        data: { path: filePath },
      });
    }
  }
}

/**
 * Finds the root directory of the git repository.
 *
 * Walks up the directory tree from the given file path until it finds
 * a .git directory, indicating the repository root.
 *
 * @param {string} startPath - The file path to start searching from
 * @returns {string|null} The repository root path or null if not found
 */
function findRepoRoot(startPath) {
  let currentDir = path.dirname(startPath);

  // Look for .git directory by walking up the directory tree
  while (currentDir !== path.dirname(currentDir)) {
    if (fs.existsSync(path.join(currentDir, ".git"))) {
      return currentDir;
    }
    currentDir = path.dirname(currentDir);
  }

  return null;
}

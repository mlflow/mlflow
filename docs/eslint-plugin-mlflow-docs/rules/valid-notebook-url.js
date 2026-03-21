const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const repoRootCache = new Map();

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
    type: 'problem',
    docs: {
      description: 'Detect NotebookDownloadButton with invalid MLflow repository URLs',
      category: 'Possible Errors',
    },
    fixable: null,
    schema: [],
    messages: {
      missingHref: 'NotebookDownloadButton is missing href attribute',
      emptyHref: 'NotebookDownloadButton href is empty',
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
        if (
          node.type !== 'JSXOpeningElement' ||
          node.name.type !== 'JSXIdentifier' ||
          node.name.name !== 'NotebookDownloadButton'
        ) {
          return;
        }
        const hrefAttr = node.attributes.find((attr) => attr.type === 'JSXAttribute' && attr.name.name === 'href');

        if (!hrefAttr) {
          context.report({
            node,
            messageId: 'missingHref',
          });
          return;
        }

        const hrefValue = getHrefValue(hrefAttr);

        if (!hrefValue) {
          context.report({
            node: hrefAttr,
            messageId: 'emptyHref',
          });
          return;
        }

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
  if (!attr.value) return null;

  if (attr.value.type === 'Literal') {
    return attr.value.value;
  }

  if (attr.value.type === 'JSXExpressionContainer' && attr.value.expression.type === 'Literal') {
    return attr.value.expression.value;
  }

  if (
    attr.value.type === 'JSXExpressionContainer' &&
    attr.value.expression.type === 'TemplateLiteral' &&
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
  const expectedPrefix = 'https://raw.githubusercontent.com/mlflow/mlflow/master/';

  if (!href.startsWith(expectedPrefix)) {
    context.report({
      node: hrefAttr,
      messageId: 'invalidFormat',
    });
    return;
  }

  const filePath = href.substring(expectedPrefix.length);
  const repoRoot = findRepoRoot();
  if (repoRoot) {
    const fullPath = path.join(repoRoot, filePath);

    if (!fs.existsSync(fullPath)) {
      context.report({
        node: hrefAttr,
        messageId: 'fileNotFound',
        data: { path: filePath },
      });
    }
  }
}

/**
 * Finds the root directory of the git repository using git rev-parse.
 *
 * @returns {string|null} The repository root path or null if not found
 */
function findRepoRoot() {
  const cacheKey = 'repoRoot';

  if (repoRootCache.has(cacheKey)) {
    return repoRootCache.get(cacheKey);
  }

  try {
    const repoRoot = execSync('git rev-parse --show-toplevel', {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();

    repoRootCache.set(cacheKey, repoRoot);
    return repoRoot;
  } catch {
    repoRootCache.set(cacheKey, null);
    return null;
  }
}

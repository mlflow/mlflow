/**
 * ESLint rule to enforce that import statements are at the top of MDX files.
 *
 * The MDX parser strips markdown content from the AST, so standard rules like
 * `import/first` cannot detect imports placed after markdown content. This rule
 * checks the raw source text to find non-ESM content between imports.
 *
 * @type {import('eslint').Rule.RuleModule}
 */
module.exports = {
  meta: {
    type: 'problem',
    docs: {
      description: 'Enforce that import statements are at the top of MDX files',
      category: 'Best Practices',
    },
    fixable: null,
    schema: [],
    messages: {
      importNotAtTop: 'Import "{{source}}" should be at the top of the file. Move all imports before any content.',
    },
  },

  create(context) {
    return {
      Program(node) {
        const sourceCode = context.sourceCode || context.getSourceCode();
        const text = sourceCode.getText();
        const lines = text.split('\n');

        // Collect line numbers occupied by ESM nodes (imports/exports)
        const esmLines = new Set();
        for (const child of node.body) {
          for (let line = child.loc.start.line; line <= child.loc.end.line; line++) {
            esmLines.add(line);
          }
        }

        // Skip frontmatter (---...---) lines
        const frontmatterLines = new Set();
        if (lines[0]?.trim() === '---') {
          frontmatterLines.add(1);
          for (let i = 1; i < lines.length; i++) {
            frontmatterLines.add(i + 1);
            if (lines[i].trim() === '---') break;
          }
        }

        // Find the first line with non-whitespace content that is not ESM or frontmatter
        let firstContentLine = null;
        for (let i = 0; i < lines.length; i++) {
          const lineNum = i + 1; // 1-indexed
          if (esmLines.has(lineNum)) continue;
          if (frontmatterLines.has(lineNum)) continue;
          if (lines[i].trim() === '') continue;
          firstContentLine = lineNum;
          break;
        }

        if (firstContentLine === null) return;

        // Report any imports that appear after the first content line
        for (const child of node.body) {
          if (child.type === 'ImportDeclaration' && child.loc.start.line > firstContentLine) {
            context.report({
              node: child,
              messageId: 'importNotAtTop',
              data: { source: child.source.value },
            });
          }
        }
      },
    };
  },
};

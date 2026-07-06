module.exports = {
  meta: {
    docs: {
      description: 'Disallow importing from restricted paths with a regexp',
      category: 'Possible Errors',
    },
    schema: [
      {
        type: 'object',
        properties: {
          patterns: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                pattern: { type: 'string', minLength: 1 },
                message: { type: 'string' },
                allowTypeImports: { type: 'boolean' },
              },
              additionalProperties: false,
              required: ['pattern'],
            },
          },
        },
        additionalProperties: false,
        required: ['patterns'],
      },
    ],
  },
  create: (context) => {
    const patterns = context.options[0].patterns.map((pattern) => {
      return {
        ...pattern,
        pattern: new RegExp(pattern.pattern),
      };
    });

    return {
      ImportDeclaration(node) {
        const path = node.source.value;
        const isImportType = node.importKind === 'type';

        const invalid = patterns.find((p) => {
          const isPathDisallowed = p.pattern.test(path);

          if (isPathDisallowed) {
            if (p.allowTypeImports && isImportType) {
              // This is a type import and allowTypeImports is on, it's fine
              return false;
            }
            // This is an invalid import
            return true;
          }

          return false;
        });
        if (invalid) {
          const data = path.match(invalid.pattern).groups;

          context.report({
            message: `Import path '${path}' is not allowed${invalid.message ? `:\n${invalid.message}` : '.'}`,
            node,
            data,
          });
        }
      },
    };
  },
};

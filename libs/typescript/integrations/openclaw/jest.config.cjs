const path = require('path');

/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  testMatch: ['**/*.test.ts'],
  moduleFileExtensions: ['ts', 'js', 'json', 'node'],
  modulePaths: [path.resolve(__dirname, '../../node_modules')],
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        tsconfig: {
          target: 'ES2022',
          module: 'CommonJS',
          moduleResolution: 'Node',
          esModuleInterop: true,
          strict: true,
          skipLibCheck: true,
          types: ['jest', 'node'],
          baseUrl: '.',
          paths: {
            'openclaw/plugin-sdk/plugin-entry': [
              './tests/__mocks__/openclaw/plugin-sdk/plugin-entry.ts',
            ],
            'openclaw/plugin-sdk/diagnostics-otel': [
              './tests/__mocks__/openclaw/plugin-sdk/diagnostics-otel.ts',
            ],
            '@mlflow/core': ['../../core/src/index.ts'],
            '@mlflow/core/*': ['../../core/src/*'],
          },
        },
      },
    ],
  },
  moduleNameMapper: {
    '^openclaw/plugin-sdk/plugin-entry$':
      '<rootDir>/tests/__mocks__/openclaw/plugin-sdk/plugin-entry.ts',
    '^openclaw/plugin-sdk/diagnostics-otel$':
      '<rootDir>/tests/__mocks__/openclaw/plugin-sdk/diagnostics-otel.ts',
    '^@mlflow/core$': '<rootDir>/../../core/src',
    '^@mlflow/core/(.*)$': '<rootDir>/../../core/src/$1',
    // Remap .js extension imports to source (for ESM-style imports in service.ts)
    '^\\./(.*)\\.js$': './$1',
    '^\\.\\./(.*)\\./js$': '../$1',
  },
  testTimeout: 30000,
  forceExit: true,
  detectOpenHandles: true,
};

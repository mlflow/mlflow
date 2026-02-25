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
            '@opencode-ai/plugin': ['./tests/__mocks__/@opencode-ai/plugin.ts'],
            'mlflow-tracing': ['../../core/src/index.ts'],
            'mlflow-tracing/*': ['../../core/src/*'],
          },
        },
      },
    ],
  },
  moduleNameMapper: {
    '^@opencode-ai/plugin$': '<rootDir>/tests/__mocks__/@opencode-ai/plugin.ts',
    '^mlflow-tracing$': '<rootDir>/../../core/src',
    '^mlflow-tracing/(.*)$': '<rootDir>/../../core/src/$1',
  },
  testTimeout: 30000,
  forceExit: true,
  detectOpenHandles: true,
};

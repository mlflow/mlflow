/** @type {import('ts-jest').JestConfigWithTsJest} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  testMatch: ['**/*.test.ts'],
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
        },
      },
    ],
  },
  // Only service.ts helpers are tested — no external deps needed
  moduleNameMapper: {
    '^@mlflow/core$': '<rootDir>/tests/__mocks__/@mlflow/core.ts',
    '^openclaw/plugin-sdk/(.*)$': '<rootDir>/tests/__mocks__/openclaw/plugin-sdk/$1.ts',
  },
};

const OverrideFiles = {
  TS: ['*.ts', '*.tsx', '*.mts', '*.cts'],
  JEST_ONLY: ['*.jest.{js,jsx,ts,tsx}'],
  TEST: ['*.test.{js,jsx,ts,tsx}'],
  JSX: ['*.jsx'],
  JS_JSX: ['*.js', '*.jsx', '*.mjs', '*.cjs'],
  TSX: ['*.tsx'],
  JSX_TSX: ['*.{jsx,tsx}'],
  CYPRESS: ['**/cypress/**', '*_spec.{js,jsx,ts,tsx}'],
  GRAPHQL: ['*.graphql'],
  STORYBOOK: ['**/storybook/**', '*.stories.{js,jsx,ts,tsx}', '**/stories/**'],
  PLAYWRIGHT: ['**/playwright/**'],
};

module.exports = {
  OverrideFiles,
};

module.exports = {
  extends: ['plugin:cypress/recommended'],
  plugins: ['cypress', 'chai-friendly'],
  env: {
    'cypress/globals': true,
  },
  rules: {
    'no-unused-expressions': 0,
    'chai-friendly/no-unused-expressions': 2,
    '@typescript-eslint/no-namespace': 0,
  },
};

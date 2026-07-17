module.exports = {
  extends: [require.resolve('./node-lib'), 'plugin:playwright/recommended'],
  rules: {
    'playwright/prefer-native-locators': 'error',
    'playwright/no-conditional-in-test': 'off',

    'react-hooks/rules-of-hooks': 'off', // this rule mistakes Playwright's use() function for the React function of the same name
    'react-hooks/exhaustive-deps': 'off', // this rule mistakes Playwright's use() function for the React function of the same name
  },
};

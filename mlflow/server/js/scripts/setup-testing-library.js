require('@testing-library/jest-dom/jest-globals');
const { configure } = require('@testing-library/react');

configure({
  asyncUtilTimeout: 10000,
});

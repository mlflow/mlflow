/**
 * To be used in an eslint config as:
 *
 * '@databricks/no-restricted-imports-regexp': [
 *   'error',
 *   {
 *     patterns: [
 *       ...require('@databricks/config-eslint/shared/no-restricted-imports-regexp-base'),
 *
 *       // any additional patterns here
 *     ],
 *   },
 * ],
 */

module.exports = [
  // antd
  {
    pattern: '^antd\\/(?:lib|es)((?:\\/[\\w-]+)(?!\\/style$))*$',
    message:
      "Do not import components from 'antd/lib/*'. Import directly from 'antd': `import { Button } from 'antd';`\n" +
      "For type imports, use `import type { ButtonProps } from 'antd/lib/button';`",
    allowTypeImports: true,
  },
  // @ant-design/icons
  {
    pattern: '^@ant-design\\/icons\\/.+$',
    message:
      "Do not import icons from '@ant-design/icons/*'. Import directly from '@ant-design/icons': `import { MoreOutlined } from '@ant-design/icons';`\n" +
      "For type imports, use `import type { CustomIconComponentProps } from '@ant-design/icons/lib/components/Icon';`",
    allowTypeImports: true,
  },
  // rc-* packages (antd components)
  {
    pattern: '^rc-[\\w-]+\\/lib\\/.+$',
    message:
      "Do not import icons from 'rc-*/lib/*'. Import directly from 'rc-*': `import { Foo } from 'rc-foo';`\n" +
      "For type imports, use `import type { SomeType } from 'rc-foo/lib/foo';`",
    allowTypeImports: true,
  },

  // Disallow relative imports from package's own `dist` folder
  // E.g. `import { Foo } from './dist/foo'`
  {
    pattern: '(?<relativePath>\\.\\.?\\/)dist($|\\/.+)',
    message:
      'Do not import from {{relativePath}}dist folder. Import from normal source files.\n' +
      'This may have been a result of an auto import',
  },

  // Disallow relative imports from 1st party packages' src folder, e.g. '../<pkg>/src/foo'.
  // This prevents scenarios like js/packages/foo/index.ts importing from js/packages/bar/src/index.ts with a relative import.
  // All imports should use the package name, e.g. import { Bar } from '@mlflow/bar'.
  {
    // Exclude @cypress-tests/ imports
    pattern: '(?<!@cypress-tests\\/)(?<relativePath>\\.\\.\\/)(?<pkgFolder>[\\w-]+?\\/)(?<folder>src|dist)($|\\/.+)',
    message:
      "Do not import from {{relativePath}}{{pkgFolder}}{{folder}} folder. If you try to import from a package, use the package's public API.\n" +
      "If you're importing from a file in the current package, use a relative path without `/{{folder}}`.\n" +
      'This may have been a result of an auto import.',
  },

  // Disallow imports from Enzyme as is has been deprecated in favor of Testing Library
  {
    pattern: '^enzyme$',
    message: 'Use React Testing Library (@testing-library/react) to render React Components in tests.',
  },

  // Disallow imports to stop the formik usage in favor of react-hook form
  {
    pattern: '^formik$',
    message: 'General recommendation is to use react-hook-form and we will coalesce the codebase around that.',
  },

  // Disallow imports to stop the redux-form usage in favor of react-hook form
  {
    pattern: '^redux-form(/immutable)?$',
    message: 'General recommendation is to use react-hook-form and we will coalesce the codebase around that.',
  },

  // Disallow imports from `src/`. These are normally allowed due to tsconfig.json's baseUrl, but we don't want them.
  {
    pattern: '^src(/|$)',
    message:
      'Do not import from `src/`. Use relative imports or include the package name, e.g. `import { Foo } from "@mlflow/mlflow/src/foo";`',
  },

  {
    pattern: '^jquery$',
    message: 'jQuery is a legacy dependency and should be avoided. Use native DOM apis or lodash instead.',
  },
];

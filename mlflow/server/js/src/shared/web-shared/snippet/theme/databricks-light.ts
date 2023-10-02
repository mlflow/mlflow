/**
 * Adapted from `material-light`
 * Ref: https://github.com/react-syntax-highlighter/react-syntax-highlighter/blob/b2457268891948f7005ccf539a70c000f0695bde/src/styles/prism/material-light.js#L1
 *
 * This theme overwrites colors to be similiar to the `@databricks/editor` theme.
 */

const databricksLightTheme = {
  'code[class*="language-"]': {
    textAlign: 'left',
    whiteSpace: 'pre',
    wordSpacing: 'normal',
    wordBreak: 'normal',
    wordWrap: 'normal',
    color: 'rgb(77, 77, 76)', // D
    background: '#fafafa',
    fontFamily: 'Monaco, Menlo, Ubuntu Mono, Consolas, source-code-pro, monospace',
    fontSize: '12px', // D
    lineHeight: '1.5em',
    MozTabSize: '4',
    OTabSize: '4',
    tabSize: '4',
    WebkitHyphens: 'none',
    MozHyphens: 'none',
    msHyphens: 'none',
    hyphens: 'none',
  },
  'pre[class*="language-"]': {
    textAlign: 'left',
    whiteSpace: 'pre',
    wordSpacing: 'normal',
    wordBreak: 'normal',
    wordWrap: 'normal',
    color: 'rgb(77, 77, 76)', // D
    background: '#fafafa',
    fontFamily: 'Monaco, Menlo, Ubuntu Mono, Consolas, source-code-pro, monospace',
    fontSize: '12px', // D
    lineHeight: '1.5em',
    MozTabSize: '4',
    OTabSize: '4',
    tabSize: '4',
    WebkitHyphens: 'none',
    MozHyphens: 'none',
    msHyphens: 'none',
    hyphens: 'none',
    overflow: 'auto',
    position: 'relative',
    margin: '0.5em 0',
    padding: '1.25em 1em',
  },
  'code[class*="language-"]::-moz-selection': {
    background: '#cceae7',
    color: '#263238',
  },
  'pre[class*="language-"]::-moz-selection': {
    background: '#cceae7',
    color: '#263238',
  },
  'code[class*="language-"] ::-moz-selection': {
    background: '#cceae7',
    color: '#263238',
  },
  'pre[class*="language-"] ::-moz-selection': {
    background: '#cceae7',
    color: '#263238',
  },
  'code[class*="language-"]::selection': {
    background: '#cceae7',
    color: '#263238',
  },
  'pre[class*="language-"]::selection': {
    background: '#cceae7',
    color: '#263238',
  },
  'code[class*="language-"] ::selection': {
    background: '#cceae7',
    color: '#263238',
  },
  'pre[class*="language-"] ::selection': {
    background: '#cceae7',
    color: '#263238',
  },
  ':not(pre) > code[class*="language-"]': {
    whiteSpace: 'normal',
    borderRadius: '0.2em',
    padding: '0.1em',
  },
  '.language-css > code': {
    color: '#f5871f', // D
  },
  '.language-sass > code': {
    color: '#f5871f', // D
  },
  '.language-scss > code': {
    color: '#f5871f', // D
  },
  '[class*="language-"] .namespace': {
    Opacity: '0.7',
  },
  atrule: {
    color: '#7c4dff',
  },
  'attr-name': {
    color: '#39adb5',
  },
  'attr-value': {
    color: '#f6a434',
  },
  attribute: {
    color: '#f6a434',
  },
  boolean: {
    color: '#7c4dff', // D
  },
  builtin: {
    color: '#39adb5',
  },
  cdata: {
    color: '#39adb5',
  },
  char: {
    color: '#39adb5',
  },
  class: {
    color: '#39adb5',
  },
  'class-name': {
    color: '#6182b8',
  },
  comment: {
    color: '#8e908c', // D
  },
  constant: {
    color: '#7c4dff', // D
  },
  deleted: {
    color: '#e53935',
  },
  doctype: {
    color: '#aabfc9',
  },
  entity: {
    color: '#e53935',
  },
  function: {
    color: '#4271ae', // D
  },
  hexcode: {
    color: '#f5871f', // D
  },
  id: {
    color: '#7c4dff',
    fontWeight: 'bold',
  },
  important: {
    color: '#7c4dff',
    fontWeight: 'bold',
  },
  inserted: {
    color: '#39adb5',
  },
  keyword: {
    color: '#8959a8', // D
  },
  number: {
    color: '#f5871f', // D
  },
  operator: {
    color: '#3e999f', // D
  },
  prolog: {
    color: '#aabfc9',
  },
  property: {
    color: '#39adb5',
  },
  'pseudo-class': {
    color: '#f6a434',
  },
  'pseudo-element': {
    color: '#f6a434',
  },
  punctuation: {
    color: 'rgb(77, 77, 76)', // D
  },
  regex: {
    color: '#6182b8',
  },
  selector: {
    color: '#e53935',
  },
  string: {
    color: '#3ba85f', // D
  },
  symbol: {
    color: '#7c4dff',
  },
  tag: {
    color: '#e53935',
  },
  unit: {
    color: '#f5871f', // D
  },
  url: {
    color: '#e53935',
  },
  variable: {
    color: '#c72d4c', // D
  },
};

export default databricksLightTheme;

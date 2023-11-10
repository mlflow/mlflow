/**
 * Adapted from `duotone-dark`
 * Ref: https://github.com/react-syntax-highlighter/react-syntax-highlighter/blob/b2457268891948f7005ccf539a70c000f0695bde/src/styles/prism/duotone-dark.js
 */

const databricksDuotoneDarkTheme = {
  'code[class*="language-"]': {
    fontFamily:
      'Consolas, Menlo, Monaco, "Andale Mono WT", "Andale Mono", "Lucida Console", "Lucida Sans Typewriter", "DejaVu Sans Mono", "Bitstream Vera Sans Mono", "Liberation Mono", "Nimbus Mono L", "Courier New", Courier, monospace',
    fontSize: '14px',
    lineHeight: '1.375',
    direction: 'ltr',
    textAlign: 'left',
    whiteSpace: 'pre',
    wordSpacing: 'normal',
    wordBreak: 'normal',
    MozTabSize: '4',
    OTabSize: '4',
    tabSize: '4',
    WebkitHyphens: 'none',
    MozHyphens: 'none',
    msHyphens: 'none',
    hyphens: 'none',
    background: '#2a2734',
    color: '#5DFAFC', // D
  },
  'pre[class*="language-"]': {
    fontFamily:
      'Consolas, Menlo, Monaco, "Andale Mono WT", "Andale Mono", "Lucida Console", "Lucida Sans Typewriter", "DejaVu Sans Mono", "Bitstream Vera Sans Mono", "Liberation Mono", "Nimbus Mono L", "Courier New", Courier, monospace',
    fontSize: '14px',
    lineHeight: '1.375',
    direction: 'ltr',
    textAlign: 'left',
    whiteSpace: 'pre',
    wordSpacing: 'normal',
    wordBreak: 'normal',
    MozTabSize: '4',
    OTabSize: '4',
    tabSize: '4',
    WebkitHyphens: 'none',
    MozHyphens: 'none',
    msHyphens: 'none',
    hyphens: 'none',
    background: '#2a2734',
    color: '#5DFAFC', // D
    padding: '1em',
    margin: '.5em 0',
    overflow: 'auto',
  },
  'pre > code[class*="language-"]': {
    fontSize: '1em',
  },
  'pre[class*="language-"]::-moz-selection': {
    textShadow: 'none',
    background: '#6a51e6',
  },
  'pre[class*="language-"] ::-moz-selection': {
    textShadow: 'none',
    background: '#6a51e6',
  },
  'code[class*="language-"]::-moz-selection': {
    textShadow: 'none',
    background: '#6a51e6',
  },
  'code[class*="language-"] ::-moz-selection': {
    textShadow: 'none',
    background: '#6a51e6',
  },
  'pre[class*="language-"]::selection': {
    textShadow: 'none',
    background: '#6a51e6',
  },
  'pre[class*="language-"] ::selection': {
    textShadow: 'none',
    background: '#6a51e6',
  },
  'code[class*="language-"]::selection': {
    textShadow: 'none',
    background: '#6a51e6',
  },
  'code[class*="language-"] ::selection': {
    textShadow: 'none',
    background: '#6a51e6',
  },
  ':not(pre) > code[class*="language-"]': {
    padding: '.1em',
    borderRadius: '.3em',
  },
  comment: {
    color: '#6c6783',
  },
  prolog: {
    color: '#6c6783',
  },
  doctype: {
    color: '#6c6783',
  },
  cdata: {
    color: '#6c6783',
  },
  punctuation: {
    color: '#6c6783',
  },
  namespace: {
    Opacity: '.7',
  },
  tag: {
    color: '#3AACE2', // D
  },
  operator: {
    color: '#3AACE2', // D
  },
  number: {
    color: '#3AACE2', // D
  },
  property: {
    color: '#5DFAFC', // D
  },
  function: {
    color: '#5DFAFC', // D
  },
  'tag-id': {
    color: '#eeebff',
  },
  selector: {
    color: '#eeebff',
  },
  'atrule-id': {
    color: '#eeebff',
  },
  'code.language-javascript': {
    color: '#c4b9fe',
  },
  'attr-name': {
    color: '#c4b9fe',
  },
  'code.language-css': {
    color: '#ffffff', // D
  },
  'code.language-scss': {
    color: '#ffffff', // D
  },
  boolean: {
    color: '#ffffff', // D
  },
  string: {
    color: '#ffffff', // D
  },
  entity: {
    color: '#ffffff', // D
    cursor: 'help',
  },
  url: {
    color: '#ffffff', // D
  },
  '.language-css .token.string': {
    color: '#ffffff', // D
  },
  '.language-scss .token.string': {
    color: '#ffffff', // D
  },
  '.style .token.string': {
    color: '#ffffff', // D
  },
  'attr-value': {
    color: '#ffffff', // D
  },
  keyword: {
    color: '#ffffff', // D
  },
  control: {
    color: '#ffffff', // D
  },
  directive: {
    color: '#ffffff', // D
  },
  unit: {
    color: '#ffffff', // D
  },
  statement: {
    color: '#ffffff', // D
  },
  regex: {
    color: '#ffffff', // D
  },
  atrule: {
    color: '#ffffff', // D
  },
  placeholder: {
    color: '#ffffff', // D
  },
  variable: {
    color: '#ffffff', // D
  },
  deleted: {
    textDecoration: 'line-through',
  },
  inserted: {
    borderBottom: '1px dotted #eeebff',
    textDecoration: 'none',
  },
  italic: {
    fontStyle: 'italic',
  },
  important: {
    fontWeight: 'bold',
    color: '#c4b9fe',
  },
  bold: {
    fontWeight: 'bold',
  },
  'pre > code.highlight': {
    Outline: '.4em solid #8a75f5',
    OutlineOffset: '.4em',
  },
  '.line-numbers.line-numbers .line-numbers-rows': {
    borderRightColor: '#2c2937',
  },
  '.line-numbers .line-numbers-rows > span:before': {
    color: '#3c3949',
  },
  '.line-highlight.line-highlight': {
    background: 'linear-gradient(to right, rgba(224, 145, 66, 0.2) 70%, rgba(224, 145, 66, 0))',
  },
};

export default databricksDuotoneDarkTheme;

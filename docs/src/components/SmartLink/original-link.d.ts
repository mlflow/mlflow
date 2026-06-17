// The webpack alias `@docusaurus-original/Link` resolves to the real
// `@docusaurus/Link` module, letting the SmartLink wrapper import the original
// component without a circular reference through the aliased `@docusaurus/Link`.
declare module '@docusaurus-original/Link' {
  import type Link from '@docusaurus/Link';

  const OriginalLink: typeof Link;
  export default OriginalLink;
}

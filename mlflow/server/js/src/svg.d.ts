declare module '*.svg' {
  import { SVGProps } from 'react';

  declare const ReactComponent: (props: SVGProps<SVGSVGElement>) => JSX.Element;

  const defaultExport: string;
  export { ReactComponent };

  // eslint-disable-next-line import/no-default-export -- SVGs are always default exported
  export default defaultExport;
}

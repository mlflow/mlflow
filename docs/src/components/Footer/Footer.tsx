import { cva, type VariantProps, cx } from "class-variance-authority";

import Logo from "@site/static/images/mlflow-logo-white.svg";
import useBaseUrl from '@docusaurus/useBaseUrl';

import { FooterMenuItem } from "../FooterMenuItem/FooterMenuItem";

const footerVariants = cva(
  "pb-150 flex flex-col pt-37 bg-linear-to-b from-brand-black to-brand-black bg-bottom bg-no-repeat bg-cover w-full bg-(--background-color-dark)",
  {
    variants: {
      variant: {
        blue: ` bg-[url('/images/footer-blue-bg.png')] bg-size-[100%_340px]`,
        red: `bg-[url('/images/footer-red-bg.png')] bg-size-[100%_340px]`,
        colorful: `bg-[url('/images/footer-colorful-bg.png')] bg-size-[100%_340px]`,
      },
    },
  },
);

export const Footer = ({ variant }: VariantProps<typeof footerVariants>) => {
  return (
    <footer className={cx(footerVariants({ variant }))}>
      <div className="flex flex-row justify-between items-start md:items-center px-6 lg:px-20 gap-10 xs:gap-0 max-w-container">
        <Logo className="h-[36px] shrink-0" />

        <div className="flex flex-col md:flex-row gap-10">
          {/* these routes are on the main mlflow.org site, which is hosted in a different repo */}
          <FooterMenuItem href="/" data-noBrokenLinkCheck>Product</FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org/releases" data-noBrokenLinkCheck>Releases</FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org/blog" data-noBrokenLinkCheck>Blog</FooterMenuItem>
          <FooterMenuItem href={useBaseUrl("/")}>
            Docs
          </FooterMenuItem>
        </div>
      </div>
    </footer>
  );
};

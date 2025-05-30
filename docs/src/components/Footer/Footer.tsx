import { cva, type VariantProps, cx } from "class-variance-authority";

import Logo from "@site/static/images/mlflow-logo-white.svg";

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
          <FooterMenuItem href="/">Product</FooterMenuItem>
          <FooterMenuItem href="/releases">Releases</FooterMenuItem>
          <FooterMenuItem href="/blog">Blog</FooterMenuItem>
          <FooterMenuItem href="/docs/latest" data-noBrokenLinkCheck>
            Docs
          </FooterMenuItem>
        </div>
      </div>
    </footer>
  );
};

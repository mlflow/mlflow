import { cva, type VariantProps, cx } from "class-variance-authority";

import Logo from "@site/static/images/mlflow-logo-white.svg";
import BlueBg from "@site/static/images/footer-blue-bg.png";
import RedBg from "@site/static/images/footer-red-bg.png";
import ColorfulBg from "@site/static/images/footer-colorful-bg.png";
import useBaseUrl from '@docusaurus/useBaseUrl';

import { FooterMenuItem } from "../FooterMenuItem/FooterMenuItem";

const footerVariants = cva(
  "pb-150 flex flex-col pt-37 bg-linear-to-b from-brand-black to-brand-black bg-bottom bg-no-repeat bg-cover w-full bg-(--background-color-dark) bg-size-[100%_340px]",
);

const getBackgroundImage = (variant: "blue" | "red" | "colorful") => {
  switch (variant) {
    case "blue":
      return BlueBg;
    case "red":
      return RedBg;
    case "colorful":
      return ColorfulBg;
  }
};

export const Footer = ({ variant }: { variant: "blue" | "red" | "colorful" }) => {
  return (
    <footer className={cx(footerVariants())} style={{ backgroundImage: `url(${getBackgroundImage(variant)})` }}>
      <div className="flex flex-row justify-between items-start md:items-center px-6 lg:px-20 gap-10 xs:gap-0 max-w-container">
        <Logo className="h-[36px] shrink-0" />

        <div className="flex flex-col md:flex-row gap-10">
          {/* these routes are on the main mlflow.org site, which is hosted in a different repo */}
          <FooterMenuItem href="https://mllfow.org">Product</FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org/releases">Releases</FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org/blog">Blog</FooterMenuItem>
          <FooterMenuItem href={useBaseUrl("/")}>
            Docs
          </FooterMenuItem>
        </div>
      </div>
    </footer>
  );
};

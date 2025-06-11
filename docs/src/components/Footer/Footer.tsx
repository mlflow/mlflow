import { cva, type VariantProps, cx } from "class-variance-authority";
import { useColorMode } from '@docusaurus/theme-common';

import LogoDark from "@site/static/images/mlflow-logo-white.svg";
import LogoLight from "@site/static/images/mlflow-logo-black.svg";
import BlueBg from "@site/static/images/footer-blue-bg.png";
import RedBg from "@site/static/images/footer-red-bg.png";
import ColorfulBg from "@site/static/images/footer-colorful-bg.png";
import BlueBgLight from "@site/static/images/footer-blue-bg-light.png";
import RedBgLight from "@site/static/images/footer-red-bg-light.png";
import ColorfulBgLight from "@site/static/images/footer-colorful-bg-light.png";
import useBaseUrl from '@docusaurus/useBaseUrl';

import { FooterMenuItem } from "../FooterMenuItem/FooterMenuItem";

const footerVariants = cva(
  "pb-150 flex flex-col pt-37 bg-bottom bg-no-repeat bg-cover w-full bg-size-[100%_340px] relative",
  {
    variants: {
      theme: {
        dark: "bg-brand-black",
        light: "bg-white",
      },
    },
  }
);

const getBackgroundImage = (variant: "blue" | "red" | "colorful", isDarkMode: boolean) => {
  if (isDarkMode) {
    switch (variant) {
      case "blue":
        return BlueBg;
      case "red":
        return RedBg;
      case "colorful":
        return ColorfulBg;
    }
  } else {
    switch (variant) {
      case "blue":
        return BlueBgLight;
      case "red":
        return RedBgLight;
      case "colorful":
        return ColorfulBgLight;
    }
  }
};

const getLogo = (isDarkMode: boolean) => {
  return isDarkMode ? LogoDark : LogoLight;
};

export const Footer = ({ variant }: { variant: "blue" | "red" | "colorful" }) => {
  const { colorMode } = useColorMode();
  const isDarkMode = colorMode === 'dark';
  const Logo = getLogo(isDarkMode);

  return (
    <footer 
      className={cx(footerVariants({ theme: isDarkMode ? 'dark' : 'light' }))} 
      style={{ backgroundImage: `url(${getBackgroundImage(variant, isDarkMode)})` }}
    >
      <div className="flex flex-row justify-between items-start md:items-center px-6 lg:px-20 gap-10 xs:gap-0 max-w-container relative z-20">
        <Logo className="h-[36px] shrink-0" />

        <div className="flex flex-col md:flex-row gap-10">
          {/* these routes are on the main mlflow.org site, which is hosted in a different repo */}
          <FooterMenuItem href="https://mlflow.org" isDarkMode={isDarkMode}>Product</FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org/releases" isDarkMode={isDarkMode}>Releases</FooterMenuItem>
          <FooterMenuItem href="https://mlflow.org/blog" isDarkMode={isDarkMode}>Blog</FooterMenuItem>
          <FooterMenuItem href={useBaseUrl("/")} isDarkMode={isDarkMode}>
            Docs
          </FooterMenuItem>
        </div>
      </div>
    </footer>
  );
};
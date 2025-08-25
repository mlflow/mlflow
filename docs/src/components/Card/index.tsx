import clsx from 'clsx';
import styles from './card.module.css';
import Link from '@docusaurus/Link';

export const CardGroup = ({ children, isSmall, cols, noGap }): JSX.Element => (
  <div
    className={clsx(
      styles.CardGroup,
      isSmall ? styles.AutofillColumns : cols ? styles[`Cols${cols}`] : styles.MaxThreeColumns,
      noGap && styles.NoGap,
    )}
  >
    {children}
  </div>
);

export const Card = ({ children, link = '' }): JSX.Element => {
  if (!link) {
    return <div className={clsx(styles.Card, styles.CardBordered)}>{children}</div>;
  }

  return (
    <Link className={clsx(styles.Link, styles.Card, styles.CardBordered)} to={link}>
      {children}
    </Link>
  );
};

export const PageCard = ({ headerText, link, text }): JSX.Element => (
  <Card link={link}>
    <span>
      <div className={clsx(styles.CardTitle, styles.BoxRoot, styles.PaddingBottom4)} style={{ pointerEvents: 'none' }}>
        <div
          className={clsx(
            styles.BoxRoot,
            styles.FlexFlex,
            styles.FlexAlignItemsCenter,
            styles.FlexDirectionRow,
            styles.FlexJustifyContentFlexStart,
            styles.FlexWrapNowrap,
          )}
          style={{ marginLeft: '-4px', marginTop: '-4px' }}
        >
          <div
            className={clsx(styles.BoxRoot, styles.BoxHideIfEmpty, styles.MarginTop4, styles.MarginLeft4)}
            style={{ pointerEvents: 'auto' }}
          >
            <span className="">{headerText}</span>
          </div>
        </div>
      </div>
      <span className={clsx(styles.TextColor, styles.CardBody)}>
        <p>{text}</p>
      </span>
    </span>
  </Card>
);

export const LogoCard = ({ description, children, link }): JSX.Element => (
  <Card link={link}>
    <div className={styles.LogoCardContent}>
      <div className={styles.LogoCardImage}>{children}</div>
      <p className={styles.TextColor}>{description}</p>
    </div>
  </Card>
);

export const SmallLogoCard = ({ children, link }) => (
  <div className={clsx(styles.Card, styles.CardBordered, styles.SmallLogoCardRounded)}>
    {link ? (
      <Link className={clsx(styles.Link)} to={link}>
        <div className={styles.SmallLogoCardContent}>
          <div className={clsx('max-height-img-container', styles.SmallLogoCardImage)}>{children}</div>
        </div>
      </Link>
    ) : (
      <div className={styles.SmallLogoCardContent}>
        <div className={clsx('max-height-img-container', styles.SmallLogoCardImage)}>{children}</div>
      </div>
    )}
  </div>
);

const RELEASE_URL = 'https://github.com/mlflow/mlflow/releases/tag/v';

export const NewFeatureCard = ({ children, description, name, releaseVersion, learnMoreLink = '' }) => (
  <Card>
    <div className={styles.NewFeatureCardWrapper}>
      <div className={styles.NewFeatureCardContent}>
        <div className={styles.NewFeatureCardHeading}>
          {name}
          <br />
          <hr className={styles.NewFeatureCardHeadingSeparator} />
        </div>
        <div className={styles.LogoCardImage}>{children}</div>
        <br />
        <p>{description}</p>
        <br />
      </div>

      <div className={styles.NewFeatureCardTags}>
        <div>
          {learnMoreLink && (
            <Link className="button button--outline button--sm button--primary" to={learnMoreLink}>
              Learn more
            </Link>
          )}
        </div>
        <Link className="button button--outline button--sm button--primary" to={`${RELEASE_URL}${releaseVersion}`}>
          released in {releaseVersion}
        </Link>
      </div>
    </div>
  </Card>
);

export const TitleCard = ({ title, description, link = '' }): JSX.Element => (
  <Card link={link}>
    <div className={styles.TitleCardContent}>
      <div className={clsx(styles.TitleCardTitle)} style={{ textAlign: 'left', fontWeight: 'bold' }}>
        {title}
      </div>
      <hr className={clsx(styles.TitleCardSeparator)} style={{ margin: '12px 0' }} />
      <p className={clsx(styles.TextColor)}>{description}</p>
    </div>
  </Card>
);

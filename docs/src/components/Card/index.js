import clsx from 'clsx';
import styles from './card.module.css';
import Link from '@docusaurus/Link';
export var CardGroup = function (_a) {
    var children = _a.children, isSmall = _a.isSmall, cols = _a.cols, noGap = _a.noGap;
    return (<div className={clsx(styles.CardGroup, isSmall ? styles.AutofillColumns : cols ? styles["Cols".concat(cols)] : styles.MaxThreeColumns, noGap && styles.NoGap)}>
    {children}
  </div>);
};
export var Card = function (_a) {
    var children = _a.children, _b = _a.link, link = _b === void 0 ? '' : _b;
    if (!link) {
        return <div className={clsx(styles.Card, styles.CardBordered)}>{children}</div>;
    }
    return (<Link className={clsx(styles.Link, styles.Card, styles.CardBordered)} to={link}>
      {children}
    </Link>);
};
export var PageCard = function (_a) {
    var headerText = _a.headerText, link = _a.link, text = _a.text;
    return (<Card link={link}>
    <span>
      <div className={clsx(styles.CardTitle, styles.BoxRoot, styles.PaddingBottom4)} style={{ pointerEvents: 'none' }}>
        <div className={clsx(styles.BoxRoot, styles.FlexFlex, styles.FlexAlignItemsCenter, styles.FlexDirectionRow, styles.FlexJustifyContentFlexStart, styles.FlexWrapNowrap)} style={{ marginLeft: '-4px', marginTop: '-4px' }}>
          <div className={clsx(styles.BoxRoot, styles.BoxHideIfEmpty, styles.MarginTop4, styles.MarginLeft4)} style={{ pointerEvents: 'auto' }}>
            <span className="">{headerText}</span>
          </div>
        </div>
      </div>
      <span className={clsx(styles.TextColor, styles.CardBody)}>
        <p>{text}</p>
      </span>
    </span>
  </Card>);
};
export var LogoCard = function (_a) {
    var description = _a.description, children = _a.children, link = _a.link;
    return (<Card link={link}>
    <div className={styles.LogoCardContent}>
      <div className={styles.LogoCardImage}>{children}</div>
      <p className={styles.TextColor}>{description}</p>
    </div>
  </Card>);
};
export var SmallLogoCard = function (_a) {
    var children = _a.children, link = _a.link;
    return (<div className={clsx(styles.Card, styles.CardBordered, styles.SmallLogoCardRounded)}>
    {link ? (<Link className={clsx(styles.Link)} to={link}>
        <div className={styles.SmallLogoCardContent}>
          <div className={clsx('max-height-img-container', styles.SmallLogoCardImage)}>{children}</div>
        </div>
      </Link>) : (<div className={styles.SmallLogoCardContent}>
        <div className={clsx('max-height-img-container', styles.SmallLogoCardImage)}>{children}</div>
      </div>)}
  </div>);
};
var RELEASE_URL = 'https://github.com/mlflow/mlflow/releases/tag/v';
export var NewFeatureCard = function (_a) {
    var children = _a.children, description = _a.description, name = _a.name, releaseVersion = _a.releaseVersion, _b = _a.learnMoreLink, learnMoreLink = _b === void 0 ? '' : _b;
    return (<Card>
    <div className={styles.NewFeatureCardWrapper}>
      <div className={styles.NewFeatureCardContent}>
        <div className={styles.NewFeatureCardHeading}>
          {name}
          <br />
          <hr className={styles.NewFeatureCardHeadingSeparator}/>
        </div>
        <div className={styles.LogoCardImage}>{children}</div>
        <br />
        <p>{description}</p>
        <br />
      </div>

      <div className={styles.NewFeatureCardTags}>
        <div>
          {learnMoreLink && (<Link className="button button--outline button--sm button--primary" to={learnMoreLink}>
              Learn more
            </Link>)}
        </div>
        <Link className="button button--outline button--sm button--primary" to={"".concat(RELEASE_URL).concat(releaseVersion)}>
          released in {releaseVersion}
        </Link>
      </div>
    </div>
  </Card>);
};
export var TitleCard = function (_a) {
    var title = _a.title, description = _a.description, _b = _a.link, link = _b === void 0 ? '' : _b, _c = _a.headerRight, headerRight = _c === void 0 ? undefined : _c, _d = _a.children, children = _d === void 0 ? undefined : _d;
    return (<Card link={link}>
    <div className={styles.TitleCardContent}>
      <div className={clsx(styles.TitleCardHeader)}>
        <div className={clsx(styles.TitleCardTitle)} style={{ textAlign: 'left', fontWeight: 'bold' }}>
          {title}
        </div>
        <div className={styles.TitleCardHeaderRight}>{headerRight}</div>
      </div>
      <hr className={clsx(styles.TitleCardSeparator)} style={{ margin: '12px 0' }}/>
      {children ? (<div className={clsx(styles.TextColor)}>{children}</div>) : (<p className={clsx(styles.TextColor)} dangerouslySetInnerHTML={{ __html: description }}/>)}
    </div>
  </Card>);
};

import React from 'react';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import { LucideIcon } from 'lucide-react';
import styles from './styles.module.css';

export interface TileCardProps {
  /**
   * The icon component to display at the top of the card
   */
  icon: LucideIcon;
  /**
   * The size of the icon (default: 32)
   */
  iconSize?: number;
  /**
   * The title of the card
   */
  title: string;
  /**
   * The description text
   */
  description: string;
  /**
   * The href for the link
   */
  href: string;
  /**
   * The link text (default: "Learn more →")
   */
  linkText?: string;
  /**
   * Additional CSS classes for the card
   */
  className?: string;
}

/**
 * A reusable tile card component for displaying feature cards with icon, title, description and link
 */
export default function TileCard({
  icon: Icon,
  iconSize = 32,
  title,
  description,
  href,
  linkText = 'Learn more →',
  className,
}: TileCardProps): JSX.Element {
  return (
    <Link href={href} className={clsx(styles.tileCard, className)}>
      <div className={styles.tileIcon}>
        <Icon size={iconSize} />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      <div className={styles.tileLink}>{linkText}</div>
    </Link>
  );
}

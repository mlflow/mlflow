import React from 'react';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import { LucideIcon } from 'lucide-react';
import styles from './styles.module.css';

export interface TileCardProps {
  /**
   * The icon component to display at the top of the card (optional if image is provided)
   */
  icon?: LucideIcon;
  /**
   * The image source to display at the top of the card (optional if icon is provided)
   */
  image?: string;
  /**
   * The size of the icon (default: 32) - only used when icon is provided
   */
  iconSize?: number;
  /**
   * The height of the icon/image container in pixels (optional)
   */
  containerHeight?: number;
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
  image,
  iconSize = 32,
  containerHeight,
  title,
  description,
  href,
  linkText = 'Learn more →',
  className,
}: TileCardProps): JSX.Element {
  // Ensure either icon or image is provided
  if (!Icon && !image) {
    throw new Error('TileCard requires either an icon or image prop');
  }

  const containerStyle = containerHeight ? { height: `${containerHeight}px` } : {};

  return (
    <Link href={href} className={clsx(styles.tileCard, className)}>
      <div className={styles.tileIcon} style={containerStyle}>
        {Icon ? <Icon size={iconSize} /> : <img src={image} alt={title} className={styles.tileImage} />}
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      <div className={styles.tileLink}>{linkText}</div>
    </Link>
  );
}

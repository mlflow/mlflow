import styles from "./download.btn.module.css";

export default function DownloadBtn({ link, text }): JSX.Element {
  return (
    <a href={link} className={styles.downloadBtn}>
      {text}
    </a>
  );
}

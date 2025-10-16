import React, { useEffect, useState, useRef } from 'react';
import styles from './styles.module.css';

interface GitHubIssue {
  number: number;
  title: string;
  html_url: string;
  user: {
    login: string;
    avatar_url: string;
  };
  created_at: string;
  comments: number;
  reactions: {
    '+1': number;
  };
  labels: Array<{
    name: string;
    color: string;
  }>;
}

interface GitHubIssuesProps {
  repo: string;
  label?: string;
  maxIssues?: number;
}

export default function GitHubIssues({
  repo = 'mlflow/mlflow',
  label = 'domain/genai',
  maxIssues = 10,
}: GitHubIssuesProps): JSX.Element {
  const [issues, setIssues] = useState<GitHubIssue[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'reactions' | 'created'>('reactions');
  const [showSortMenu, setShowSortMenu] = useState(false);
  const sortMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchIssues = async () => {
      setLoading(true);
      setError(null);

      try {
        // Build query with optional search
        let query = `repo:${repo} is:issue state:open label:${label}`;
        if (searchQuery.trim()) {
          query += ` ${searchQuery.trim()}`;
        }

        const sortParam = sortBy === 'reactions' ? 'reactions' : 'created';
        const url = `https://api.github.com/search/issues?q=${encodeURIComponent(query)}&sort=${sortParam}&order=desc&per_page=${maxIssues}`;

        const response = await fetch(url);
        if (!response.ok) {
          throw new Error('Failed to fetch issues');
        }

        const data = await response.json();
        setIssues(data.items || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    // Debounce search
    const timeoutId = setTimeout(() => {
      fetchIssues();
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [repo, label, maxIssues, searchQuery, sortBy]);

  // Close sort menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (sortMenuRef.current && !sortMenuRef.current.contains(event.target as Node)) {
        setShowSortMenu(false);
      }
    };

    if (showSortMenu) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showSortMenu]);

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'numeric',
      day: 'numeric',
      year: 'numeric',
    });
  };

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.header}>
          <h3 className={styles.title}>Feature requests</h3>
        </div>
        <div className={styles.loading}>Loading feature requests...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.container}>
        <div className={styles.header}>
          <h3 className={styles.title}>Feature requests</h3>
        </div>
        <div className={styles.error}>
          <p>Unable to load issues. Please visit GitHub directly:</p>
          <a
            href={`https://github.com/${repo}/issues?q=is%3Aissue+state%3Aopen+label%3A${label}+sort%3Areactions-%2B1-desc`}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.errorLink}
          >
            View on GitHub →
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h3 className={styles.title}>Feature requests</h3>
        <div className={styles.headerActions}>
          <div className={styles.searchBox}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" className={styles.searchIcon}>
              <path d="M10.68 11.74a6 6 0 01-7.922-8.982 6 6 0 018.982 7.922l3.04 3.04a.749.749 0 01-.326 1.275.749.749 0 01-.734-.215l-3.04-3.04zm-1.423-1.423a4.5 4.5 0 10-6.364-6.364 4.5 4.5 0 006.364 6.364z" />
            </svg>
            <input
              type="text"
              placeholder="Search..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={styles.searchInput}
            />
          </div>
          <div className={styles.sortDropdown} ref={sortMenuRef}>
            <button className={styles.sortButton} onClick={() => setShowSortMenu(!showSortMenu)}>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M3 4.5h10M3 8h6M3 11.5h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
              {sortBy === 'reactions' ? 'Upvotes' : 'Recent'}
            </button>
            {showSortMenu && (
              <div className={styles.sortMenu}>
                <button
                  className={sortBy === 'reactions' ? styles.sortMenuItemActive : styles.sortMenuItem}
                  onClick={() => {
                    setSortBy('reactions');
                    setShowSortMenu(false);
                  }}
                >
                  Upvotes
                </button>
                <button
                  className={sortBy === 'created' ? styles.sortMenuItemActive : styles.sortMenuItem}
                  onClick={() => {
                    setSortBy('created');
                    setShowSortMenu(false);
                  }}
                >
                  Recent
                </button>
              </div>
            )}
          </div>
          <a
            href={`https://github.com/${repo}/issues/new?template=feature_request_template.yaml`}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.newButton}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 2.5a.75.75 0 01.75.75v4h4a.75.75 0 010 1.5h-4v4a.75.75 0 01-1.5 0v-4h-4a.75.75 0 010-1.5h4v-4A.75.75 0 018 2.5z" />
            </svg>
            New
          </a>
        </div>
      </div>

      <div className={styles.issuesList}>
        {issues.map((issue) => (
          <a
            key={issue.number}
            href={issue.html_url}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.issueRow}
          >
            <div className={styles.voteColumn}>
              <div className={styles.voteCount}>{issue.reactions['+1']}</div>
              <div className={styles.voteLabel}>votes</div>
            </div>

            <div className={styles.issueContent}>
              <div className={styles.issueTitle}>{issue.title}</div>
              <div className={styles.issueMeta}>
                <span className={styles.author}>{issue.user.login}</span>
                <span className={styles.separator}>•</span>
                <span className={styles.date}>{formatDate(issue.created_at)}</span>
                {issue.comments > 0 && (
                  <>
                    <span className={styles.separator}>•</span>
                    <span className={styles.comments}>
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M1.75 1A1.75 1.75 0 000 2.75v8.5C0 12.216.784 13 1.75 13H3v1.543a1.457 1.457 0 002.487 1.03L8.061 13h6.189A1.75 1.75 0 0016 11.25v-8.5A1.75 1.75 0 0014.25 1H1.75z" />
                      </svg>
                      {issue.comments}
                    </span>
                  </>
                )}
              </div>
            </div>
          </a>
        ))}
      </div>

      <div className={styles.footer}>
        <a
          href={`https://github.com/${repo}/issues?q=is%3Aissue+state%3Aopen+label%3A${label}+sort%3Areactions-%2B1-desc`}
          target="_blank"
          rel="noopener noreferrer"
          className={styles.viewAllLink}
        >
          View all on GitHub →
        </a>
      </div>
    </div>
  );
}

var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
import React, { useEffect, useState, useRef } from 'react';
import { Search, ArrowUpDown, Plus, MessageSquare } from 'lucide-react';
import styles from './styles.module.css';
export default function GitHubIssues(_a) {
    var _this = this;
    var _b = _a.repo, repo = _b === void 0 ? 'mlflow/mlflow' : _b, _c = _a.label, label = _c === void 0 ? 'domain/genai' : _c, _d = _a.maxIssues, maxIssues = _d === void 0 ? 10 : _d;
    var _e = useState([]), issues = _e[0], setIssues = _e[1];
    var _f = useState(true), loading = _f[0], setLoading = _f[1];
    var _g = useState(null), error = _g[0], setError = _g[1];
    var _h = useState(''), searchQuery = _h[0], setSearchQuery = _h[1];
    var _j = useState('reactions'), sortBy = _j[0], setSortBy = _j[1];
    var _k = useState(false), showSortMenu = _k[0], setShowSortMenu = _k[1];
    var sortMenuRef = useRef(null);
    useEffect(function () {
        var fetchIssues = function () { return __awaiter(_this, void 0, void 0, function () {
            var query, sortParam, url, response, data, err_1;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        setLoading(true);
                        setError(null);
                        _a.label = 1;
                    case 1:
                        _a.trys.push([1, 4, 5, 6]);
                        query = "repo:".concat(repo, " is:issue state:open label:").concat(label);
                        if (searchQuery.trim()) {
                            query += " ".concat(searchQuery.trim());
                        }
                        sortParam = sortBy === 'reactions' ? 'reactions' : 'created';
                        url = "https://api.github.com/search/issues?q=".concat(encodeURIComponent(query), "&sort=").concat(sortParam, "&order=desc&per_page=").concat(maxIssues);
                        return [4 /*yield*/, fetch(url)];
                    case 2:
                        response = _a.sent();
                        if (!response.ok) {
                            throw new Error('Failed to fetch issues');
                        }
                        return [4 /*yield*/, response.json()];
                    case 3:
                        data = _a.sent();
                        setIssues(data.items || []);
                        return [3 /*break*/, 6];
                    case 4:
                        err_1 = _a.sent();
                        setError(err_1 instanceof Error ? err_1.message : 'An error occurred');
                        return [3 /*break*/, 6];
                    case 5:
                        setLoading(false);
                        return [7 /*endfinally*/];
                    case 6: return [2 /*return*/];
                }
            });
        }); };
        // Debounce search
        var timeoutId = setTimeout(function () {
            fetchIssues();
        }, 300);
        return function () { return clearTimeout(timeoutId); };
    }, [repo, label, maxIssues, searchQuery, sortBy]);
    // Close sort menu when clicking outside
    useEffect(function () {
        var handleClickOutside = function (event) {
            if (sortMenuRef.current && !sortMenuRef.current.contains(event.target)) {
                setShowSortMenu(false);
            }
        };
        if (showSortMenu) {
            document.addEventListener('mousedown', handleClickOutside);
        }
        return function () {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [showSortMenu]);
    var formatDate = function (dateString) {
        var date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            month: 'numeric',
            day: 'numeric',
            year: 'numeric',
        });
    };
    if (loading) {
        return (<div className={styles.container}>
        <div className={styles.header}>
          <h3 className={styles.title}>Feature requests</h3>
        </div>
        <div className={styles.loading}>Loading feature requests...</div>
      </div>);
    }
    if (error) {
        return (<div className={styles.container}>
        <div className={styles.header}>
          <h3 className={styles.title}>Feature requests</h3>
        </div>
        <div className={styles.error}>
          <p>Unable to load issues. Please visit GitHub directly:</p>
          <a href={"https://github.com/".concat(repo, "/issues?q=is%3Aissue+state%3Aopen+label%3A").concat(label, "+sort%3Areactions-%2B1-desc")} target="_blank" rel="noopener noreferrer" className={styles.errorLink}>
            View on GitHub →
          </a>
        </div>
      </div>);
    }
    return (<div className={styles.container}>
      <div className={styles.header}>
        <h3 className={styles.title}>Feature requests</h3>
        <div className={styles.headerActions}>
          <div className={styles.searchBox}>
            <Search size={16} className={styles.searchIcon}/>
            <input type="text" placeholder="Search..." value={searchQuery} onChange={function (e) { return setSearchQuery(e.target.value); }} className={styles.searchInput}/>
          </div>
          <div className={styles.sortDropdown} ref={sortMenuRef}>
            <button className={styles.sortButton} onClick={function () { return setShowSortMenu(!showSortMenu); }}>
              <ArrowUpDown size={16}/>
              {sortBy === 'reactions' ? 'Upvotes' : 'Recent'}
            </button>
            {showSortMenu && (<div className={styles.sortMenu}>
                <button className={sortBy === 'reactions' ? styles.sortMenuItemActive : styles.sortMenuItem} onClick={function () {
                setSortBy('reactions');
                setShowSortMenu(false);
            }}>
                  Upvotes
                </button>
                <button className={sortBy === 'created' ? styles.sortMenuItemActive : styles.sortMenuItem} onClick={function () {
                setSortBy('created');
                setShowSortMenu(false);
            }}>
                  Recent
                </button>
              </div>)}
          </div>
          <a href={"https://github.com/".concat(repo, "/issues/new?template=feature_request_template.yaml")} target="_blank" rel="noopener noreferrer" className={styles.newButton}>
            <Plus size={16}/>
            New
          </a>
        </div>
      </div>

      <div className={styles.issuesList}>
        {issues.map(function (issue) { return (<a key={issue.number} href={issue.html_url} target="_blank" rel="noopener noreferrer" className={styles.issueRow}>
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
                {issue.comments > 0 && (<>
                    <span className={styles.separator}>•</span>
                    <span className={styles.comments}>
                      <MessageSquare size={14}/>
                      {issue.comments}
                    </span>
                  </>)}
              </div>
            </div>
          </a>); })}
      </div>

      <div className={styles.footer}>
        <a href={"https://github.com/".concat(repo, "/issues?q=is%3Aissue+state%3Aopen+label%3A").concat(label, "+sort%3Areactions-%2B1-desc")} target="_blank" rel="noopener noreferrer" className={styles.viewAllLink}>
          View all on GitHub →
        </a>
      </div>
    </div>);
}

import os
import argparse, git, pickle, requests


def main():
    parser = argparse.ArgumentParser(
        description="Generate a changelog for the repo from the given previous branch. "
                    "Example usage:\n"
                    " $ python generate_changelog.py --prev-branch branch-0.9.1 > commits-parsed-0.9.1-master.txt")
    parser.add_argument("--prev-branch", nargs='?', required=True,
                        help="Previous release branch to compare to, e.g. branch-0.8")
    parser.add_argument("--curr-branch", nargs='?', default="master",
                        help="Current release (candidate) branch to compare to, e.g. branch-0.9. "
                             "Defaults to 'master'.")
    parser.add_argument("--skip-github", nargs='?', default=False,
                        help="Skip querying github and use the cached commits file.")

    parsed = parser.parse_args()
    prev_branch = parsed.prev_branch
    curr_branch = parsed.curr_branch
    skip_github = parsed.skip_github

    # get the list
    g = git.Git(os.getcwd())
    # TODO: automate `git fetch upstream`
    loginfo = g.log('--left-right', '--graph', '--cherry-pick', '--pretty=format:\'%an\t%s\'',
                    'upstream/'+prev_branch+'...'+curr_branch)

    newlogs = [str(l)[3:-1] for l in loginfo.split("\n") if str.startswith(str(l), '>')]
    #print(len(newlogs))

    github_cache_file = "commits-%s-%s.pkl" % (prev_branch, curr_branch)
    raw_github_cache_file = "raw-commits-%s-%s.pkl" % (prev_branch, curr_branch)  # for dev
    if skip_github:
        with open(github_cache_file, "rb") as f:
            commits = pickle.load(f)
        with open(raw_github_cache_file, "rb") as f:
            raw_commits = pickle.load(f)
    else:
        raw_commits = []
        commits = []
        for log in newlogs:
            [author_name, title] = log.split("\t")
            pr_num_ind = title.rfind('(#')
            if pr_num_ind < 0:
                continue
            pr_num = int(title[pr_num_ind+2:-1])

            # get the github handle from github
            user_name = os.environ.get("GITHUB_USERNAME")
            api_token = os.environ.get("GITHUB_API_TOKEN")
            print("...Sending github request for PR %d" % pr_num)
            r = requests.get('https://api.github.com/repos/mlflow/mlflow/pulls/%d' % pr_num,
                             auth=(user_name, api_token))
            rjson = r.json()
            raw_commits.append(rjson)

            github_handle = rjson['user']['login']
            commits.append({
                'author_name': author_name,
                'title': title[:pr_num_ind-1],
                'pr': pr_num,
                'author_github_handle': github_handle,
            })
        with open(github_cache_file, "wb") as f:
            pickle.dump(commits, f)
        with open(raw_github_cache_file, "wb") as f:
            pickle.dump(raw_commits, f)

    # TODO: do more stuff with labels etc

    # TODO: maybe make links to github PRs
    print("\n".join(['%s (#%d, @%s)' % (c['title'], c['pr'], c['author_github_handle']) for c in commits]))

if __name__ == '__main__':
    main()

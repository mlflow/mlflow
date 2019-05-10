import os
import argparse, git, requests

def main():
    parser = argparse.ArgumentParser(description='Generate a changelog for the repo from the given previous branch')
    parser.add_argument("--prev-branch", nargs='?', required=True, help='Previous release branch to compare to, e.g. branch-0.8')
    parser.add_argument("--curr-branch", nargs='?', default="master", help='Current release (candidate) branch to compare to, e.g. branch-0.9. Defaults to master.')

    parsed = parser.parse_args()
    prev_branch = parsed.prev_branch
    curr_branch = parsed.curr_branch

    # get the list
    g = git.Git(os.getcwd())
    loginfo = g.log('--left-right', '--graph', '--cherry-pick', '--pretty=format:\'%an\t%s\'', 'upstream/'+prev_branch+'...'+curr_branch)

    newlogs = [l[3:-1] for l in loginfo.split("\n") if str.startswith(l, '>')]
    print(len(newlogs))

    commits = []
    for log in newlogs:
        [author_name, title] = log.split("\t")
        pr_num_ind = title.rfind('(#')
        pr_num = int(title[pr_num_ind+2:-1])

        # get the github handle from github
        # TODO: env var for token & username
        r = requests.get('https://api.github.com/repos/mlflow/mlflow/pulls/%d' % pr_num, auth=(user_name, api_token))
        rjson = r.json()
        github_handle = rjson['user']['login']

        commits.append({
            'author_name': author_name,
            'title': title[:pr_num_ind-1],
            'pr': pr_num,
            'author_github_handle': github_handle,
        })
    # TODO: save commits to a file as a cache so we don't have to query github if we already have the info for a PR
    # TODO: maybe make links to github PRs
    print("\n".join(['%s (#%d, @%s)' % (c['title'], c['pr'], c['author_github_handle']) for c in commits]))

if __name__ == '__main__':
    main()

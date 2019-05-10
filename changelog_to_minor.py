import os
import argparse, git, requests
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Generate the small bug fixes and doc paragraph from a list of PRs')
    parser.add_argument("--small-prs-file", nargs='?', required=True, help='File listing small fixes and docs PRs')
    parser.add_argument("--author-handle-file", nargs='?', default=None, help='File listing PRs, with author github handles')  # shouldn't be needed after this release

    parsed = parser.parse_args()
    pr_file = parsed.small_prs_file
    handle_file = parsed.author_handle_file

    pr_to_handle = {}
    if handle_file is not None:
        for line in open(handle_file, "r"):
            pr_num_ind = line.rfind('(#')
            pr_num = line[pr_num_ind+1:pr_num_ind+5]   # SUPER HACKY WILL NOT WORK FOR NEXT RELEASE
            handle = line[pr_num_ind+7:-2]   # SUPER HACKY WILL NOT WORK FOR NEXT RELEASE
            pr_to_handle[pr_num] = handle

    handle_to_prs = defaultdict(list)
    for line in open(pr_file, "r"):
        pr_num_ind = line.rfind('(#')
        pr_num = line[pr_num_ind+1:pr_num_ind+5]   # SUPER HACKY WILL NOT WORK FOR NEXT RELEASE
        if pr_num in pr_to_handle:
            handle = pr_to_handle[pr_num]
        else:
            handle = line[pr_num_ind+7:-2]   # SUPER HACKY WILL NOT WORK FOR NEXT RELEASE
        handle_to_prs[handle].append(pr_num)

    text = "(" + "; ".join(
        [", ".join(pr_nums + [handle]) for handle, pr_nums in handle_to_prs.items()]
    ) + ")"
    print(text)

if __name__ == '__main__':
    main()

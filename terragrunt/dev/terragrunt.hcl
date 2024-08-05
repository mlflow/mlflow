include "root" {
  path = find_in_parent_folders("terragrunt-root.hcl")
}

terraform {
  source = "${path_relative_from_include()}/..//terraform"
}
locals {
  vars = read_terragrunt_config("params.hcl")
  account_id   = local.vars.locals.aws_account_id
  aws_region   = local.vars.locals.aws_region
  environment  = local.vars.locals.environment
  app_name     = local.vars.locals.app_name
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
    provider "aws" {
      region = "${local.aws_region}"
      # Only these AWS Account IDs may be operated on by this template
      allowed_account_ids = ["${local.account_id}"]
    }
  EOF
}

remote_state {
  backend = "s3"
  config = {
    encrypt        = true
    bucket         = "nuodata-terraform-remote-state-${local.environment}"
    key            = "${path_relative_to_include()}/${local.app_name}/terraform.tfstate"
    region         = local.aws_region
    dynamodb_table = "terraform-locks-${local.app_name}"
  }
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }
}

inputs = local.vars.locals
def build_for(){

  branch_name = "${env.GIT_BRANCH}"
  branch_name_split  = branch_name.split("-")
  buildfor = branch_name_split[0] + '-'+  branch_name_split[1]
  return buildfor
}
pipeline {
  agent {
    kubernetes {
      label 'jenkins-agent'
      defaultContainer 'jnlp'
    }
  }
  stages {
    stage('Test') {
      steps {
        container('jnlp') {
          sh 'echo Hello from Kubernetes pod!'
        }
      }
    }
  }
}

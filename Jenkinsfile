#!groovy

node {
    try {
        stage('Checkout') {
            // Ensure GitHub is trusted to avoid SSH authenticity prompt
            sh '''
                mkdir -p ~/.ssh
                ssh-keyscan github.com >> ~/.ssh/known_hosts
            '''
            checkout scm
        }

        stage('Deploy') {
            sh 'whoami'
            switch (env.BRANCH_NAME) {
                case 'dev':
                    sh 'bash -i ./.deployment/dev_deploy.sh'
                    break
                case 'faiss':
                    //sh 'bash -i ./.deployment/faiss_deploy.sh'
                    sh 'bash ./.deployment/faiss_deploy.sh'

                    break
                case 'main':
                    sh 'bash -i ./.deployment/main_deploy.sh'
                    break
                default:
                    echo "Deploy scripts only on dev, main & uat"
                    break
            }
        }
    }

    catch (err) {
        throw err
    }
}

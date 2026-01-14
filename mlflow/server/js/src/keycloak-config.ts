import Keycloak from 'keycloak-js';
// Keycloak configuration
const keycloakConfig = {
    url: 'https://identity.dev.nuodata.io/',
    realm: 'nuodata',
    clientId: 'nuodata-aiml-and-jupyterhub',
};
// Initialize Keycloak instance
const keycloak:any = new Keycloak(keycloakConfig);
export default keycloak;
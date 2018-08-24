package com.databricks.mlflow.client;

import java.util.List;
import org.apache.log4j.Logger;
import org.testng.Assert;
import org.testng.annotations.*;

public class BaseTest {
    private static final Logger logger = Logger.getLogger(BaseTest.class);
    static String apiUriDefault = "http://localhost:5001";
    static ApiClient client ;

    @BeforeSuite
    public static void beforeSuite() throws Exception {
        logger.info("apiUriDefault="+apiUriDefault);
        String apiUriProp = System.getenv("MLFLOW_TRACKING_URI");
        logger.info("apiUriProp="+apiUriProp);
        String apiUri = apiUriProp == null || apiUriProp.length()==0 ? apiUriDefault : apiUriProp;
        logger.info("apiUri="+apiUri);
        client = new ApiClient(apiUri);
        client.setVerbose(true);
    }
}

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import mlflow
from azureml.core import Workspace
from mlflow.server import _run_server
from mlflow.server.handlers import initialize_backend_stores
import threading
import random

# before testing, need to open mlflow ui on localhost 5000 (default port)
ws = Workspace.from_config()

def aml_ui(backend_store_uri, default_artifact_root, port, host):

    try:
        initialize_backend_stores(backend_store_uri, default_artifact_root)
    except Exception as e:  # pylint: disable=broad-except
        print(e)
        sys.exit(1)

    # TODO: We eventually want to disable the write path in this version of the server.
    try:
        _run_server(backend_store_uri, default_artifact_root, host, port, None, 1)
    except Exception as e:
        print("Running the mlflow server failed. Please see the logs above for details.")
        print(e)
        sys.exit(1)

class uiThread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      aml_ui(ws.get_mlflow_tracking_uri(), ws.get_mlflow_tracking_uri(), 5000, "0.0.0.0")

uiThread(1, "ui", 1).start()

# open MLflow UI
driver = webdriver.Chrome(r'C:\Users\t-vijia\node_modules\selenium\chromedriver.exe')
driver.maximize_window()
driver.get("localhost:5000")
while(not "MLflow" in driver.title):
    driver.get("localhost:5000")

# create an experiment
WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".experiment-list-create-btn"))).click()
inputExperimentName = driver.find_element_by_id("experimentName")
experimentName = "test" + str(random.randint(0, 1000000))
inputExperimentName.send_keys(experimentName)
driver.find_element_by_xpath("/html/body/div[4]/div/div[2]/div/div[2]/div[3]/div/button[2]").click()

# log a run to test experiment
file_name = 'myfile.txt'
with open(file_name, "w+") as f:
    f.write('This is an output file that will be uploaded.\n')

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experimentName)
with mlflow.start_run() as run:
    mlflow.log_param('alpha', 0.03)
    mlflow.log_metric('mse', 5)
    mlflow.set_tag('key', 'value')
    mlflow.log_artifact(file_name)

# refresh page, navigate back to experiment
driver.refresh()
WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//div[contains(@title,'{0}')]".format(experimentName)))).click()

# search for a run
inputSearchQuery = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".ExperimentView-search-controls input")))
inputSearchQuery.send_keys("params.alpha='0.03'")
driver.find_element_by_css_selector(".search-button").click()

# navigate to a run
WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div/div/div[2]/div/div/div[3]/div[2]/div/div/div[1]/div/div[3]/div[1]/div/div[2]/div/div"))).click()

# view artifacts
WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div/div/div/div[3]/div[5]/div/div[2]/div/div/div/div[1]/ul/li/div[1]")))
assert (file_name in driver.page_source) 

# delete tag
WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div/div/div/div[3]/div[4]/div/div[2]/div/div/div[1]/div/div/div/div/div/table/tbody/tr/td[3]/span/button[2]"))).click()
WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[4]/div/div/div/div[2]/div/div/div[2]/button[2]"))).click()
driver.refresh()

# return to experiment
WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div/div/div/div[1]/h1/a"))).click()

# delete run

# switch viewtype to deleted

# restore run

# switch viewtype to active

# model registry stuff

# delete experiment
driver.find_element_by_xpath("/div[@title='{0}']/button[last()]".format(experimentName)).click()
assert (experimentName not in driver.page_source) 

driver.close()
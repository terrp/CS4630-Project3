# CS 4630 Project 3: Supervised ML on HIGGS Dataset

**Team:** Tommy Sotherland, Andrew Terpinski, Michael Fattizzo, Britton Helman, Jonah Warren

This repository contains the Phase 3 implementation for evaluating classic supervised machine learning models on the UCI HIGGS dataset. The pipeline handles data loading (200k subsample), feature scaling, model training, and performance evaluation (Accuracy, F1, ROC-AUC, PR-AUC).

### Setup & Installation
* **Ensure Python 3.8+ is installed on your system.**
* **Create and activate a virtual environment (recommended):**
  * Windows: `python -m venv .venv` then `.venv\Scripts\activate`
  * Mac/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
  * *Alternatively, you can create a project in PyCharm using the same directory as `HiggsP3.py`*
* **Install the required libraries:**
  * Run `pip install -r requirements.txt`
* **Verify Data Placement:**
  * Ensure the raw dataset is located in the correct folder: `data/raw/HIGGS.csv.gz`
  * *Note: Do not unzip the file. Pandas will read the `.gz` archive directly.*

### Running the Models
The primary execution scripts are located in the scripts folder. They utilize command-line arguments to allow individual team members to train and evaluate specific models without altering the code.

**Phase 3A: Classic Supervised Models**
Execute a specific model using the `--model` flag:
* Linear SVM: `python HiggsP3.py --model linear_svm` *(1-3 min)*
* RBF SVM: `python HiggsP3.py --model rbf_svm` *(1-4 hr!)*
* K-Nearest Neighbors: `python HiggsP3.py --model knn` *(2-5 min)*
* Decision Tree: `python HiggsP3.py --model dt` *(5-15 sec)*
* Random Forest: `python HiggsP3.py --model rf` *(30-90 sec)*
* Gradient Boosting: `python HiggsP3.py --model xgb` *(2-5 min)*

Execute all Phase 3A models sequentially:
* `python HiggsP3.py --model all`

**Phase 3B: Integrated Models**
Execute all of the models in Part B:
* `python3 HiggsP3b.py --part all`

You can also run the parts separately with:
* `python3 HiggsP3b.py --part a`
* `python3 HiggsP3b.py --part b`

### Outputs
* The scripts automatically create an `outputs/` directory in your current working path if one does not exist.
* Upon completion of a model's inference phase, the runtime metrics and evaluation scores are printed to the console.
* These metrics are simultaneously appended to `outputs/phase3a_metrics.csv` to ensure all team members have a centralized log of the experimental results for the final report.

# **AspectMind: Vietnamese Aspect‑Based Sentiment Analysis**

## Description
AspectMind is a project that builds an aspect‑based sentiment analysis (ABSA) system for Vietnamese text. ABSA is a fine‑grained opinion‑mining technique that determines the sentiment of a sentence with respect to a specific aspect; unlike traditional sentiment analysis that treats a whole document as a single unit, ABSA can reveal that a review is overall positive yet expresses a negative feeling about a particular feature.
The project implements ABSA on the UIT‑ViSFD dataset, which consists of 11 122 manually annotated Vietnamese smartphone comments. The dataset was collected from a large Vietnamese e‑commerce website and annotated with ten aspects and three sentiment polarities (positive, negative and neutral). It is split into 7 786 training comments, 1 112 development comments and 2 224 test comments. The code uses PhoBERT, the first BERT‑based language model pre‑trained specifically for Vietnamese; the PhoBERT‑base and PhoBERT‑large variants are large monolingual models that outperform multilingual models on multiple Vietnamese NLP tasks.

## Features
+ PhoBERT‑based predictors – Implements single‑aspect and multi‑aspect classifiers based on PhoBERT. Using a monolingual pre‑trained model improves accuracy.
+ Modular architecture – Components are clearly organised under ```src/aspectmind```, separating data processing, model training, inference and threshold tuning.
+ Training & evaluation scripts – The ```scripts/``` directory includes tools for training, evaluation, threshold tuning and probability calibration. Scripts support plotting F1‑score against thresholds and evaluating models in multiple modes.
+ Streamlit demo application – ```demo/app_streamlit.py``` provides a lightweight web interface that lets you enter Vietnamese text and see predicted aspects and sentiments in real time.
+ Comprehensive tests – The ```tests/``` directory contains unit tests to ensure processing and inference pipelines work correctly.
+ Reproducible environment – ```requirements.txt```, ```pyproject.toml``` and ```Makefile``` facilitate easy setup and ensure consistent environments across systems.

## Project structure
```
aspectmind/
├── src/aspectmind/          # Core library: data handling, models and inference
│   ├── data/                # Data loading and preprocessing modules
│   ├── models/              # Model architectures and training
│   └── inference/           # PhoBERT‑based predictor classes
├── scripts/                 # Scripts for training, evaluation, threshold tuning and calibration
├── demo/app_streamlit.py    # Streamlit demo application
├── tests/                   # Unit test suite
├── pyproject.toml           # Python package & dependency declaration
├── requirements.txt         # Python dependencies
└── README.md                # This documentation (Vietnamese)
```

## Installation
1. Create a Python environment: Use Python ≥ 3.10. Creating a virtual environment is recommended:
```
python -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Download the data: Obtain the UIT‑ViSFD dataset from its project page. This dataset comprises 11 122 smartphone feedback comments annotated with 10 aspects and 3 sentiment polarities. The splits are 7 786 / 1 112 / 2 224 for train/dev/test. Place the raw files (CSV/JSON) into a ```data/raw/``` directory. Modules in ```src/aspectmind/data``` will load and convert the data into the required format.

4. Train the model: Use the ```scripts/evaluate_models.py``` script to train and evaluate. Example:
```
python scripts/evaluate_models.py \
    --data_dir data/raw \
    --model_type phobert \
    --output_dir outputs/phobert_single
```
Parameters such as --batch_size and --learning_rate can be adjusted as needed.

5. Tune thresholds & calibration: Use ```scripts/tune_threshold_phobert_single.py``` and ```scripts/plot_f1_vs_threshold.py``` to find the optimal decision thresholds based on F1‑score. The ```scripts/plot_calibration_curve.py``` helps plot probability calibration curves.

6. Run the Streamlit demo:
```
streamlit run demo/app_streamlit.py
```
Then open a browser at ```http://localhost:8501``` and enter Vietnamese text to see the aspects and sentiments predicted by the model.

## Usage example
Suppose you want to analyse the review:
"Điện thoại chạy mượt, pin tốt nhưng camera hơi kém." – “The phone runs smoothly, the battery is good but the camera is a bit weak.”
After running the predictor, the system might return a table like this:
| Aspect | Sentiment |
|------|------|
| Battery | Positive |
| Camera | Negative |

The above table is illustrative only; actual results depend on the model and threshold you choose.

## Results & evaluation
By fine‑tuning PhoBERT on the UIT‑ViSFD dataset, the model achieves high F1‑scores on both aspect detection and sentiment classification tasks. The script ```scripts/eval_test_4modes_phobert_single.py``` produces detailed reports and plots F1‑score against threshold. Owing to PhoBERT – a large monolingual language model dedicated to Vietnamese – the project’s model surpasses multilingual baselines and performs well on ABSA.

## Contributing
Contributions are welcome! If you find a bug or would like to add a feature:
1. Fork the repository and create a new branch.
2. Commit your changes with a clear message.
3. Open a Pull Request describing your changes and why they are needed.
4. Ensure all tests (pytest) pass before submitting.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## References
+ UIT‑ViSFD dataset: A smartphone feedback dataset containing 11 122 comments annotated with 10 aspects and 3 sentiment polarities, split into train/dev/test sets of 7 786 / 1 112 / 2 224 respectively.\
+ PhoBERT: A large monolingual language model dedicated to Vietnamese; PhoBERT‑base and PhoBERT‑large are the first models of their kind and outperform multilingual models on various Vietnamese NLP tasks.
+ Aspect‑Based Sentiment Analysis: A technique that determines sentiment with respect to individual aspects, addressing limitations of traditional sentiment analysis by pinpointing emotions about specific components within text.

# Report Classification using BERT

This repository contains the code for training and evaluating a BERT-based text classification model for accident reports and maintenance work orders classification. 

## Dataset

Here are a few examples of the types of reports this model can classify:

### Traffic Signal Work Order Examples

You can refer to [Traffic Signal Work Orders Data](https://datahub.austintexas.gov/Transportation-and-Mobility/Traffic-Signal-Work-Orders/4qmj-36nv/about_data).

### Aviation Maintenance Report Examples

For aviation maintenance reports, visit [Aviation Maintenance Reports Data](https://asrs.arc.nasa.gov/search/database.htmls).

## Usage

You can run the code in different modes: **training** or **testing**. In addition, you can choose to use a **pretrained** BERT model or a **custom** BERT model. Below are examples of how to run the code from the terminal in various cases.

```bash 
python main.py --mode train --model pretrained 

python main.py --mode train --model custom 

python main.py --mode test --model pretrained --model_path experiments/traffic_signal_classifier_pretrained

python main.py --mode test --model custom --model_path experiments/traffic_signal_classifier --vocab_path experiments/traffic_signal
```

This project is licensed under the MIT License. See the LICENSE file for more details.

# Visual Question Answering

### Data Prerequisites
Required data must be placed in

```data/processed/```

This folder contains 7 .gzip files and 4 .pkl files, containing all the train, validation and test data, together with all dictionaries.

Pretrained embeddings must be found in 
```data/embedding/```
under names `embedding_x.pkl`, where x denotes dimension.

### Running the models
File `main.py` contains the runnable endpoint of the project.
The model configuration is done within this file in `main`, and is passed as namespace object to the Trainer class.
The following model parameters can be configured:
- model_type: ['lstm', 'bow']
- use_pretrained_embeddings: [True, False]
- embedding_size: [50, 100, 200, 300]
- hidden_units: integer
- number_stacked_lstms: integer
- visual_model: [True, False]
- visual_features_location: list, with possible values being ['lstm_context', 'lstm_output', 'lstm_input']
- full_size_visual_features: [True, False] - (ignores 'lstm_context' from list above if set to False and sets the hidden_units to img_features_len)
- lstm_dropout: float in range [0, 1]

The trainer namespace configures parameters of the Trainer class:
- save: [True, False] - saves the final model
- verbose: [True, False] - print data on validate
- epochs: integer
- lr: float

#### Model files
The BOW and LSTM models can be found in files `bow.py` and `lstm.py`, respectively. 

Current drug discovery efforts revolve around a time-intensive timeline that can take 10-15 years, oftentimes longer for complex neurological diseases like Alzheimer’s disease. This is a direct result of the large chemical space (approximately 1060 possible drug-like molecules) to be explored. In current drug discovery procedures, candidate compounds are oftentimes empirically selected from the general chemical space, an inefficient approach. 

With the development of deep learning as an efficient method to process large amounts of data, recent research has proposed the use of deep learning approaches to accelerate the drug discovery process. Specifically, new research has demonstrated the potential for generative deep learning approaches in novel chemical generation using natural language processing (NLP) methodologies

All graphs and modules can be found in respective folders

## Baseline
Code for our baseline model can be found at baseline_lstm.py

[Baseline Training Code](baseline_lstm.py)

## Improved Baseline
Code for our improved LSTM model can be found at improved_lstm.py.

[Improved Training Code](improved_lstm.py)

## Hybrid
Code for our Hybrd GRU/LSTM model can be found at hybrid_lstm.py.

[Improved Training Code](hybrid_lstm.py)

## Transfer Learning
These files are responsible for using the alzheimers data to transfer learn to generate more specific chemical structures

[Baseline Transfer Learning Code](TL_baseline_lstm.py)<br>
[Improved Transfer Learning Code](TL_improved_lstm.py)<br>
[Hybrid Training Code](TL_hybrid.py)

## Evaluation
These two files are responsible for using the test data to evaluate both our general models and the ones we transfered learned<br>
[Evaluate General](evaluateModelGeneralDataset.py)<br>
[Evaluate Transfer learn](evaluateModelsSpecificDataset.py)

## Molecule Generation
To generate molecules, we can run this file to generate 100 and filter those that are valid

[Molecule Generation](molecule_generator.py)

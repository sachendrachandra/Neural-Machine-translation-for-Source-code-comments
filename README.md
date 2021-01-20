# TSE_project_2

This is a part of the Topics in Software Engineering course Project offered at IISc Bangalore by Associate Professor Aditya Kanade and Professor Shirish Shevade.

The project involves implementaion of the initial Transformer model for Machine Translation Task. It aims at generating Natural language comments for JAVA source code. It is an alternative approach to generate comments for JAVA source code inspired from the paper "https://xin-xia.github.io/publication/emse192.pdf". The paper used LSTM encoder-decoder architecture with attention mechanism to do the comment generation task. 

The dataset for the project is same as used by the authors of the paper mentioned above. The dataset can be accessed using the link https://drive.google.com/drive/folders/10g1vKYDjtDW3zTzWZSfn2kN8lkalqy7q?usp=sharing. Their are 4 different sets of data: RQ1, RQ2, RQ3 and RQ4. I have used RQ1.
The Dataset details are:

Source Code Vocabulary size = 30000

Natural language vocabulary size = 30000

Training Size = 445812

Test Size = 20000

Validation size = 20000

The details of the files are:

The project can be broken down into 5 parts:

First, the data processing needs to be done. The file Dataprocessing.py includes for doing the same. It involves converting the Training, Testing, Validation source code and target natural language comments into tensors and then generating iterators for converting them into batches of size 256. To generate iterators if batch size 256.

It can be executed as follows:

python3 Dataprcessing.py "Path to source vocabulary" "Path to natural Language vocabulary" "Path to training source codes" "Path to training natural language comments" "Path to testing source codes" "Path to Testing natural Language comments" "Path to validation source codes" "Path to Validation natural language comments"

Second, The Transformer model is to be created. The model is almost similar as is proposed in the paper "Attention is all You Need". The properties of the architecture are as follows:

Embedding Size =256
Number of heads = 8
NUmber of Layers = 2

The code for Transformer is included in the file Transformer.py

Third, is the the training , evaluation of teh model and then saving it. The model is trained for 6 epochs with Adam optimizer and Cross entropy loss. the test set loss was "0.00021304885184909386". The code for training, evaluating and saving the model can be seen in file Training_saving.py. The trained model can be downloaded using the link>

Fourth comes the part to load the model and and use it to generate comments for the code.The file loading_model.py contains the code for the same. The file "predicted_output_f" contains output for the source codes in the test file.

To execute the above file use command as below:

python3 loading_model.py "Path to saved model" "Path to vocabulary file for natural language comments" "Path to file where the ouput is to be saved"

The file Bleu_score_calculator.py is used to generate bleu score for the results obtained using above trained model. It takes two inputs.
the "predictions" file and the "reference" file. The predictions file contains the output generated by our model and reference file contained the test file for target natural language comments. The Bleu score for the translation task obtained was 0.8648

it can be executed as follow:

python3 Bleu_score_calculator.py "Path to prediction file" "Path to reference file"

At last, in order to execute the entire code right from data processing to saving the model, we can execute the file run.py as follows:

python3 run.py "Path to source vocabulary" "Path to natural Language vocabulary" "Path to training source codes" "Path to training natural language comments" "Path to testing source codes" "Path to Testing natural Language comments" "Path to validation source codes" "Path to Validation natural language comments" "Path where the model is to be saved" 






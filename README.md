# deep-learning

This project uses the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by a venture capital firm.

Inside the notebook is a CSV containing more than 34,000 organizations that have received funding from the venture capital firm over the span of several years. 

# Step 1: Preprocess the Data
Using knowledge of Pandas and scikit-learn’s StandardScaler(), I preprocessed this dataset. This step prepared me for Step 2, where I compiled, trained, and evaluated the neural network model.
<img width="1198" alt="Screen Shot 2023-06-05 at 1 49 09 PM" src="https://github.com/Phil-Mart/deep-learning/assets/120279988/23bc5933-41d2-48d7-97dc-6a24b7b851dd">
<img width="1198" alt="Screen Shot 2023-06-05 at 1 50 18 PM" src="https://github.com/Phil-Mart/deep-learning/assets/120279988/da6f7b74-092a-40c1-96dd-b58c0d4c200f">

# Step 2: Compile, Train, and Evaluate the Model
Using my knowledge of TensorFlow, I designed a neural network, or deep learning model, to create a binary classification model that can predict if an angel investment funded organization will be successful based on the features in the dataset. I considered how many inputs there were before determining the number of neurons and layers in this model. Once I completed that step, I compiled, trained, and evaluated my binary classification model to calculate the model’s loss and accuracy.
<img width="1198" alt="Screen Shot 2023-06-05 at 1 50 49 PM" src="https://github.com/Phil-Mart/deep-learning/assets/120279988/94b55ab7-c459-45b1-885c-075979369c6d">

# Step 3: Optimize the Model
Using my knowledge of TensorFlow, I optimize your model to achieve a target predictive accuracy higher than 75%. I did so by readding the 'Names' column, becuase this lead to a more rich data set with hidden nodes to make the predictive model better optimized. 
<img width="1198" alt="Screen Shot 2023-06-05 at 1 51 06 PM" src="https://github.com/Phil-Mart/deep-learning/assets/120279988/2da666b4-d294-445e-b3fb-f1b936d3f183">

# Step 4: Written Report on Neural Network Model

# Report
 
 ## Overview: the purpose of this analysis is to find the best set of parameters in order to create a model that can predict is a venture capital's ROI with an accuracy of 75% or higher. 
 Results:
 * Data Processing:
 * The target variable: column 'IS_SUCCESSFUL' 
 * The Features variables: after using pd.get_dummies, the data was transformed from categories to numeric types; all values were kept as features except for EIN numbers, names, and outcome columns.
 * Arguably, the only column that really should be eliminated is the EIN because it is sensitive data. All other parameters are pertinent to discovering which varibles could help determine if an investment is successful or not. To counter this, multiple analyses would have to be tested where one or more columns are deleted from the binary set in order to discover which parameters can be taken out of the machine learning application. 
 
 ## Compiling, Training, and Evaluating the Model:
 * I tested 3 hidden layers of 8 of Neurons using various activation functions, including Relu, Tanh, and Linear algorithms. I do so because adding layers or Neurons made no difference without changing the data. 
 * Each model's test peaked around 20 epochs, with a loss of ~50% and an accuracy of ~71%. 
 * <img width="635" alt="Screen Shot 2023-06-05 at 3 50 22 PM" src="https://github.com/Phil-Mart/deep-learning/assets/120279988/e6289b84-1dd7-4398-87a0-610b48559f99">

 * Without adding the 'NAMES' column to the model, I was not able to initially raise the model's accuracy above 75%. 
 * To increase accuracy, I first changed variable in the neural network itself to increase its computing agility. I added two hidden layers, gave each layer 8 nodes, and used a combination of different active functions; the results, however, remained the same. So, this was an indication that there may still have been noise in the dataset, or that the NAMES column needed to be added in order to bin loans together by company. 

## Summary: 
To find the best results, I would do exploratory permutation analysis to test every combination of columns added/excluded along with the most optimal neural network in order to find the highest level of accuracy possible. Also, if results became less fruitful after such analysis, I could also use other machine learning tools to see if a different supervised learning algorithm could yield better results. This could include Linear Regression, PCA compression, tree-decision making, or other methods. Then, I would need to visualize the entire datset to visualy tease out potential indicator to then be tested by the model. Lastly, I would look at the bin sizes to make sure that the model is not overfit or under-fit.

## LinkedIn: https://www.linkedin.com/in/phil-mart/

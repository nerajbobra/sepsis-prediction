# sepsis-prediction
Sepsis Prediction using Clinical Data (PhysioNet Computing in Cardiology Challenge 2019)

This project implements a sepsis prediction model using various clinical data sources. Specifically, the model takes 10 hours of input data and predicts the probability of sepsis within the next hour. On the test set, the model has an AUC of 0.76.

The data used for this project is from the 2019 PhysioNet Computing in Cardiology Challenge. The following link provides more information about the data and a link to download: https://physionet.org/content/challenge-2019/1.0.0/

The dataset is a series of PSV files, where each row represents a single hour of data. 

To run the code in this project, run the following notebooks:
1. `psv_to_df.ipynb`: This notebook loads the PhysioNet data PSV files and saves them into a Pandas DataFrame for ease of downstream analysis
2. `feature_engineering.ipynb`: This notebook generates 10 hour-windowed features and corresponding labels
3. `feature_selection.ipynb`: This notebook inspects feature correlations and removes any features that are highly correlated
4. `train_model.ipynb`: This notebook defines the model, trains it, and evaluates its performance on validation and test sets

The remainder of this readme will cover the different steps in the analysis pipeline.

## 1. Redefining Output Labels
According to the PhysioNet Challenge details, the labels for the provided data are as follows:
<br>For sepsis patients, SepsisLabel is 1 if `t≥tsepsis−6` and 0 if `t<tsepsis−6`
<br>For non-sepsis patients, SepsisLabel is 0

In other words, the SepsisLabel is set to 1 six hours before the onset of sepsis. However, for the purposes of this project, sepsis only needs to be predicted one hour in advance. So the labels are redefined such that:
<br>For sepsis patients, SepsisLabel is 1 if `t≥tsepsis` and 0 if `t<tsepsis`
<br>For non-sepsis patients, SepsisLabel is 0

To actually realize this change, the first six values of SepsisLabel equals 1 are set to 0 for each patient’s data.

## 2. Window the Data
For each patient, the data is windowed into ten hour windows with an output label corresponding to the sepsis state in the eleventh hour. The window is then slid forward by one hour, until there is no more data for that subject. Note that there is no overlap of two different patients in any given window.

## 3. Backfill Missing Data for Non-Sparse Variables; Calculate Median for Sparse Variables
Many of the variables in the dataset are sparse, as is expected with clinical data. However, HR, MAP, O2Sat, SBP, Resp are relatively continuous (less than 15% missing). For these variables, any missing data is replaced with backfilling the most recent non-NaN value. 

For the remainder of the variables, summarize the window of ten hours with the median of the values in that window. If all the values in that window are NaN, then just report the median as NaN.

A summary of the percentage of missing data per variable:
|   Variable              |   Percent Missing  |
|-------------------------|--------------------|
|   Age                   |   0                |
|   Gender                |   0                |
|   ICULOS                |   0                |
|   SepsisLabel           |   0                |
|   HospAdmTime           |   0                |
|   HR                    |   10               |
|   MAP                   |   12               |
|   O2Sat                 |   13               |
|   SBP                   |   15               |
|   Resp                  |   15               |
|   DBP                   |   31               |
|   Unit1                 |   39               |
|   Unit2                 |   39               |
|   Temp                  |   66               |
|   Glucose               |   83               |
|   Potassium             |   91               |
|   Hct                   |   91               |
|   FiO2                  |   92               |
|   Hgb                   |   93               |
|   pH                    |   93               |
|   BUN                   |   93               |
|   WBC                   |   94               |
|   Magnesium             |   94               |
|   Creatinine            |   94               |
|   Platelets             |   94               |
|   Calcium               |   94               |
|   PaCO2                 |   94               |
|   BaseExcess            |   95               |
|   Chloride              |   95               |
|   HCO3                  |   96               |
|   Phosphate             |   96               |
|   EtCO2                 |   96               |
|   SaO2                  |   97               |
|   PTT                   |   97               |
|   Lactate               |   97               |
|   AST                   |   98               |
|   Alkalinephos          |   98               |
|   Bilirubin_total       |   99               |
|   TroponinI             |   99               |
|   Fibrinogen            |   99               |
|   Bilirubin_direct      |   100              

## 4. Feature Standardization
Each of the variables is standardized by subtracting the mean and dividing by the standard deviation. Note that the mean and standard deviation are calculated from the training set, and the same scaling factors are applied to both the training and testing sets. The test set consists of 6000 randomly sampled patients from the original 40000 patients.

## 5. Feature Correlation Analysis
Any features with high correlation are redundant and unnecessarily increase model complexity. The correlations are visualized with a heat map as shown below:
![Heatmap](https://github.com/nerajbobra/sepsis-prediction/blob/master/figures/heatmap.png)

As the correlation values are not very high, none of the features were removed.

## 6. Define the Model
There are two categories of data in each window: time series data (with a sequence length of ten) and single measurements. The natural structure for time series data is a recurrent neural network, while the single measurement data would naturally be modeled by a simple shallow network. Therefore, two different models are trained and then merged into a single output, followed by a softmax layer. The model architecture is described below:
![Model Architecture](https://github.com/nerajbobra/sepsis-prediction/blob/master/figures/model_diagram.jpg)

Note that a mask layer is included in the second model. This is a natural approach to handle NaN values in the data. The mask layer requires all NaN values to be replaced with a constant, and then ignores any values equal that constant during training/evaluation of the model. A unique constant is pi, and therefore that value is used to replace all NaN values. Any constant would work here.

The implementation of the model using the Keras Functional API:
```
input1 = Input(shape=(INPUT_SEQ_LEN_MODEL1, INPUT_NUM_CH_MODEL1))
model1 = Bidirectional(LSTM(100, kernel_regularizer=l2(0.001), return_sequences=True))(input1)
model1 = Bidirectional(LSTM(75, kernel_regularizer=l2(0.001)))(model1)
model1 = Dense(35, kernel_regularizer=l2(0.001), activation='relu')(model1)
model1 = BatchNormalization()(model1)
model1 = Dense(15, kernel_regularizer=l2(0.001), activation='relu')(model1)
model1 = BatchNormalization()(model1)

input2 = Input(shape=(INPUT_FEATS_MODEL2,))
model2 = Masking(mask_value=np.pi)(input2)
model2 = Dense(30, kernel_regularizer=l2(0.001), activation='relu')(model2)
model2 = BatchNormalization()(model2)
model2 = Dense(15, kernel_regularizer=l2(0.001), activation='relu')(model2)
model2 = BatchNormalization()(model2)

model_add = Add()([model1, model2])
output = Dense(2, kernel_regularizer=l2(0.001), activation='softmax')(model_add)

model = Model(inputs=[input1, input2], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam'
```

## 7. Train the Model
Note that the ratio of sepsis to non-sepsis labels is very imbalanced, at a ratio of approximately 1:53 for the training set. There are many strategies to handle imbalanced data, but the simplest approach is to weight the loss function to penalize the under-represented class higher.

The validation set is defined as a random 20% subset of the training set.

```
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
history = model.fit([X_train_cont, X_train_cat],
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=50,
                    validation_data=([X_val_cont, X_val_cat], y_val),
                    callbacks=[earlystop, checkpoint],
                    class_weight=class_weights,
                    verbose=1)
```

## 8. Evaluate the Model
The ROC of the validation set:
![Validation ROC](https://github.com/nerajbobra/sepsis-prediction/blob/master/figures/ROC_val.png)

The ROC of the test (holdout) set:
![Test ROC](https://github.com/nerajbobra/sepsis-prediction/blob/master/figures/ROC_test.png)

As can be seen from the ROC plots, the performance on the training and test sets is very similar. Therefore, it would be expected that the model performance would be comparable to new data collected in the field.

## Next Steps
Beyond experimenting with different model architectures, there are many potential improvements that can be made to this project. Changing the window size will have a large effect on the performance of the model, as has been demonstrated by other researchers. Additionally, trying a different strategy for handling missing data, such as interpolation, may also yield improvements in performance. 

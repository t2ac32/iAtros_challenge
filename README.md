# Hypertension Alert System

The repository contains the solution for the iAtros AI & Backend Engineer challenge.

The solution contains two parts.
1. First a prototyping notebook for function testing and explaining reasoning process.
2. A cli tool that queries for a patient and predicts it mesurments returning an alert summary.

## Prototyping
Theres a notebook (iatros_challenge.ipynb) contains the step by step implementation of the solution incluiding:
1. Implementation of requests between HAPI-fhir server and app
2. Pagination handeling
3. Filling the Dataframe
4. Running the main query
5. Preprocess and clead the Dataframe

6. Binary classification
  - Model setup
  - Model definition
7. Running the model
8. Prediction test on a row of dataframe

## Installation

### Install Conda environment
1. `conda env create -f environment.yml`
2. Activate the environment:
  `conda activate env`
3. Verify the environment was installed correctly:
  `conda env list`

### Install PIP Requirements
1. `pip install -r requirements.txt` 

## How to Run the prediction App

From terminal:

### Example: Arterial Pressure Prediction for a subject on database
   
- Args:
    - observ: Code for observation defaul to 85354-9 LOINC code  
      'Blood pressure panel with all children optional'
    - subject: Subject code in HAPI-FHIR test server
    - path to pretained model, default to models/hypertension_model
   
` python app.py  --observ 85354-9 --subject 1598464 --model ./models/hypertension_model.pkl `


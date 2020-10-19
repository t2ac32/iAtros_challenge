# Hypertension Alert System

## Installation

## How to

From terminal:

**Example: Arterial Pressure Prediction for a subject on database**
   
- Args:
    - observ: Code for observation defaul to 85354-9 LOINC code  
      'Blood pressure panel with all children optional'
    - subject: Subject code in HAPI-FHIR test server
    - path to pretained model, default to models/hypertension_model
   
` python app.py  --observ 85354-9 --subject 1598464 --model ./models/hypertension_model.pkl `


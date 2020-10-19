
import math
import click
from typing import Union
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlsplit

import pandas as pd
import seaborn as sb
import json
import requests
import fhirclient

from tabulate import tabulate
from IPython.display import display, HTML
#fhir server clients and configuration
from fhirclient import client
import fhirclient.models.patient as p
import fhirclient.models.observation as o
import fhirclient.models.bundle as b

from pathlib import Path
from fastai.basics import *
from fastai.callback.all import *
from fastai.tabular.all import *
import torch
import torch.nn as nn


HAPI_URL = 'http://hapi.fhir.org/baseR4'

settings = {
    'app_id': 'fhir',
    'api_base': 'http://hapi.fhir.org/baseR4/'
}
SMART = client.FHIRClient(settings=settings) 

def search_observation(obs_code: str, server, subject= None ) -> pd.DataFrame:
    """ 
        Recibes a code conforming to SNOMED-CT 
        e.j. http://bioportal.bioontology.org/ontologies/SNOMEDCT/?p=classes&conceptid=http%3A%2F%2Fpurl.bioontology.org%2Fontology%2FSNOMEDCT%2F38341003&jump_to_nav=true

        Args:
            obs_code [str] -- SNOMED-CT or LOINC conforming code
            server   [   ] -- instance of fhirclient server
            subject  [   ] -- id of subject in fhir database
        
        Returns:
            iatros_df [pandas.DataFrame]
    """
    # Create search query
    payload = {'status' : 'final',
                'code': {'$and': [obs_code]},    
            }
    if subject is not None:
        payload['subject'] = subject
    fs = o.Observation.where(struct = payload)
    # Perfom query to receive a Bundle resourceType since it contains pagination link.
    bundle = fs.perform(server)
    #print(json.dumps(bundle.as_json(), indent=2))
    
    # Pass bundle to handel pagination and save entries in Dataframe
    iatros_df = handle_pagination(bundle)
    
    return iatros_df

def handle_pagination(bundle: fhirclient.models.bundle) -> pd.DataFrame:
    """
        Handles a resourceType: "Bundle" entries, if pagination
        link avalable iterates looking for next page entries and
        queries the server for the next page.

        Arguments:
            bundle -- fhirclient.models.bundle

        Returns:
            pandas.DataFrame -- A Data frame containing inputs from
                                 the paginated  requests.
    """
    frames = []
    #Keep requesting while pagination link exists
    while True:
        entries = [be.resource for be in bundle.entry] if bundle is not None and bundle.entry is not None else None
        print('Retrieved {}/{} entries...'.format(len(bundle.entry) if bundle.entry else 0, bundle.total if bundle.total else ' '))
        
        #Get a temp dataframe from current bundle entries
        temp_df =  append_entries_to_dataset(bundle)
        frames.append(temp_df)
        #Look for a pagination link
        if entries is not None and len(entries) > 0:
            next_link = get_next_link_in(bundle)  
            url_params = get_url_params(next_link)
            if next_link is not None:
                if len(url_params) > 1:
                    #Query for next page
                    response = requests.get(HAPI_URL, params=url_params)
                    if response.status_code == 200:
                        #Initialize a bundle object from request response as json
                        try: 
                            bundle = b.Bundle(response.json())
                        except Exception as e:
                            print('An error ocurred while creating Bundle object')
                            print(e)
                            print('Error query: ', next_link)
                            return pd.concat(frames)
            else:
                return pd.concat(frames)
        else:
             return pd.concat(frames)

def get_url_params(url:str):
    o = urlparse(url)
    query = parse_qsl(o.query)
    
    params = dict(parse_qsl(urlsplit(url).query))
    
    return params

def get_next_link_in(bundle):
    if bundle.link is not None:
        for link in bundle.link:
            if link.relation == 'next':
                return link.url
    else:
         return None

def append_entries_to_dataset(bundle)-> pd.DataFrame:
    columns = ['Patient_Ref','Dia','Sys','Units']
    d= []
    for entry in bundle.entry:
        row = {}
        resource = entry.resource
        subject_ref = resource.subject.reference.replace('Patient/','')
        for comp in resource.component:
            if comp.valueQuantity is not None and comp.valueQuantity is not float('nan'):
                bp_val = comp.valueQuantity.value
                bp_val_unit = comp.valueQuantity.unit
                if bp_val is not None:
                    if comp.code.text == "Diastolic Blood Pressure":
                        row['Dia'] = bp_val
                    elif comp.code.text == "Systolic Blood Pressure":
                        row['Sys'] = bp_val
            else:
                print('Non value quantity found')
                continue
            row['Units'] = bp_val_unit
            row['Patient_Ref'] = subject_ref 
                
        if ('Sys' in row.keys()) and ('Dia' in row.keys()):
            sistolic = row['Sys'] 
            diastolic = row['Dia'] 
            if 130 <= sistolic <= 139 and 80 <= diastolic <= 89:
                row['Hypertension'] = True
            elif sistolic >= 140 and diastolic >= 90 :
                row['Hypertension'] = True
            else:
                row['Hypertension'] = False  
           
        d.append(row)
    
    return pd.DataFrame(d)

def train(data: pd.DataFrame):
    try: 
        data = data.drop(columns=['Units','Patient_Ref'])
    except:
        print('Column not in dataframe')
        print('Available columns: ', iatros_df.columns)

    #Config Tabular DataLoader:
    cont_names = ['Dia','Sys']
    y_name = 'Hypertension' #Target
    procs = [Categorify, Normalize] #pre-processes to apply on data 
    dls = TabularDataLoaders.from_df(iatros_df , procs=procs,cont_names=cont_names,y_names= y_name,y_block = CategoryBlock, bs=64)

    #DATA split
    splits = RandomSplitter(valid_pct= 0.2)(range_of(iatros_df))
    dls.show_batch()

    #setup model configurations
    learn = setup_model(model_name= 'hypertension_model')

    #Train the model
    learn.fit_one_cycle(50)
    #save the model 
    saved_model_path = learn.save(file = model_name)
    print('\n'*5)
    learn.show_results()

    print('\n'*5)
    print(tabulate([],[['Model was saved to: ', save_model_path],
                       ['The validation accuracy is:',acc * 10]],
                       tablefmt="grid"))
    

def setup_model(model_name: str):
    models_path = Path.cwd()
    callbacks = [] 

    if not models_path.is_dir():
        models_path.mkdir(exist_ok=True)
    
    
    callbacks = [SaveModelCallback(every='improvement',
                                        monitor='val_loss', 
                                        name= model_name)]

    learn = tabular_learner(dls,path=models_path, metrics=accuracy,callbacks=callbacks)
    return learn

def predict(data: pd.DataFrame, model_path:Path):
    try: 
        data = data.drop(columns=['Units','Patient_Ref'])
    except:
        print('Column not in dataframe')
        print('Available columns: ', data.columns)
    
    #Load pre existing model 
    learn = load_learner(model_path, cpu=True)

    #Iterate subject Dataframe 
    headers = ['Norm-Diastolic', 'Norm-Systolic', 'Alert Status']
    table = []
    for index, row in data.iterrows():
        row, clas, probs = learn.predict(row)
        alert = "Hypertension Alert" if clas == 1 else 'Normal vitals'
        dia = row['Dia']
        sys = row['Sys']

        tab_row = [dia, sys, alert]
        table.append(tab_row)
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

'''
@click.command()
@click.option('--dia', default=120, help='Diastolic pressure value in mmHg.')
@click.option('--sys', default=80, help='Sistolic pressure value in mmHg.')
def predict_vals(dia, sys):
'''


@click.command()
@click.option('--observ', default='85354-9', help='Observation code string defaul to blood pressure')
@click.option('--subject', help='request link to query bundle.')
@click.option('--model', default= Path('models/hypertension_model.pth'),
                         help='Path to model including model filename, e.j models/mymodel.pth')
def predict_bundle(observ, subject, model):
    """Receives a query for pressure values and predict on each entry"""
    #Reuse SMART server instance
    #ej. subject: 1598464
    iatros_df = search_observation(observ, SMART.server, subject=subject)
    '''Remove nan values'''
    iatros_df = iatros_df.dropna()
    '''Reset index count and remove it'''
    iatros_df.reset_index(drop=True,inplace=True)
    
    #Predict
    print('Running prediction on model: ', model)
    predict(data=iatros_df, model_path= model)

if __name__ == '__main__':
    predict_bundle()
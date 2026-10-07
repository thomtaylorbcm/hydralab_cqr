import numpy as np
import pandas as pd
from typing import List, Optional
from itertools import accumulate

def cum_concat_to_list(x):
    return list(accumulate(x))

def structured_to_unstructured(df: pd.DataFrame, 
                              domain_metadata: str, 
                              stu_id, 
                              stu_datetime, 
                              cols_to_concat: list,
                              concat_syntax_separator_list: Optional[List[str]],
                              prefix_prompt: "", 
                              end_prompt = ". ", 
                              date_time_errors_handling = "ignore",
                              dt_time_format = None) -> pd.DataFrame: 
    """ 
    Transformation of Structured To Unstructered (STU) data for later tokenization and vector embeddings in neural networks 
    relying on unstructured text inputs (e.g., Large Language Models). This requires function requires 3 key inputs from 
    source data: 
        1. a unique identifier of the sample (e.g., a patient Medical Record Number)
        2. a unique date, time, or datestamp to specify the longitudinal 
            time sequence in logical order of observations.
        3. at least 1 column of content to be considered as unstructured text for later vector embeddings. 

    Args:
        df: pandas data frame input (source data from EHR or survey data).
        
        domain_metadata: domain of data or source of data (e.g., Encounters, Medications, progress notes, etc.)
        
        stu_id:  unique identifier of the sample observation (e.g., a patient MRN)
        
        stu_datetime: the datetime stamp. 
        
        cols_to_concat: a list of column names in the pandas DataFrame that hould be concatenated to unstructured text.
        
        concat_syntax_separater_list: list of separating charachter strings to create more 
            pseudo language. must be of length cols_to_concat-1 in length. Note: Setting to None is only appropriate 
            for creating a dataframe from a single column. Alternatively, pass an empty list [] to the argument. 
        
        prefix_prompt: a string to being the string of column elements to be concatenated. potentially 
            useful to create more natural language inputs in vector emebeddings. Not specifically necessary however. 
        
        dt_time_format:  specify the datatime format rather than allow Pandas library to infer. 
            For example, use of '%Y-%m-%d'

    Returns:
        a Pandas dataframe with the Structured-To-Unstructured (STU) data for each stu_id and by datatime input.
    
    References:
        None. 
    """
    #no separation specified, assumed a space is needed to align as much as possible with structured data architecture
    if concat_syntax_separator_list==None:
        suffix_space = [""]
    else:
        concat_syntax_separator_list
        assert len(cols_to_concat)-1==len(concat_syntax_separator_list), "Specify k-1 syntax separators to separate columns included in cols_to_concat list"
        suffix_space = concat_syntax_separator_list + [""] #to lengths make equal with cols_to_concat for looping purposes
     
    ##[old]suffix_space = concat_syntax_separator_list + [""] #to lengths make equal with cols_to_concat for looping purposes
    suffix_append_list = []
    for i in range(len(cols_to_concat)): 
        tempcol = df[cols_to_concat[i]].astype(str) + suffix_space[i] #convert to str since need to tokenize anyway, so losing any numeric or date context as numpy/pandas recognize it
        tempcol = tempcol.fillna("")
        suffix_append_list.append(tempcol)
    catbase = pd.concat(suffix_append_list, axis=1)
    baselish = catbase.apply("".join, axis=1)
    ehrlish_col = prefix_prompt + baselish + end_prompt

    # infer datetime format
    if dt_time_format is None:
        df[stu_datetime] = pd.to_datetime(df[stu_datetime], errors = date_time_errors_handling) 
    else:
        df[stu_datetime] = pd.to_datetime(df[stu_datetime], format=dt_time_format, errors = date_time_errors_handling)

    #concatenate id, datetime col, and unstructured text 
    catdf = pd.concat([df[[stu_id, stu_datetime]], ehrlish_col], axis=1)
    catdf['stu_domain'] = np.repeat(domain_metadata, len(catdf))
    catdf.columns = ['stu_id', 'stu_datetime', 'stu_text', 'stu_domain']
    #check to make sure indexing issue did not create misalignment between timecol and unstructured ehrlish feature
    shape_equality_assertion = "concatenated dataframe is of shape: " + str(catdf.shape[0]) + " and the input dataframe is of shape: " + str(df.shape[0])
    assert catdf.shape[0]==df.shape[0], shape_equality_assertion
    return catdf


def timeunit_concatenator(df: pd.DataFrame, 
                          id_column: str, 
                          datetime_column: str, 
                          text_column: str, 
                          join_string = '; ',
                          structured_text_column_return_name = 'stucurrent_text',
                          ascend=False) -> pd.DataFrame:
    """ 
    Cumulative transformation of Structured To Unstructered (STU) data for later tokenization and vector embeddings in neural networks 
    relying on unstructured text inputs (e.g., Large Language Models). In the order of the time column, text will be concatenated. 
        1. a unique identifier of the sample (e.g., a patient Medical Record Number)
        2. a unique date, time, or datestamp to specify the longitudinal 
            time sequence in logical order of observations.
        3. at least 1 column of content to be considered as unstructured text for later vector embeddings. 

    Args:
        df: pandas data frame input (source data from EHR or survey data).
               
        stu_id:  unique identifier of the sample observation (e.g., a patient MRN).
        
        stu_datetime: the datetime stamp. 
        
        stu_text: a string of information--possibly duplicated.
        
        join_string:  string to separate out structured data element rows to be concatenated.
        
        structured_text_column_return_name: rename the column returned in case joins to other 
            datasets are needed. Default is 'stu_current_text' to reflect that the text is from
            the current timedate stamp returned. 
        
        ascend:  order of timedate in ascending or descending. Default is False (for time descending from most recent time to least recent.)

    Returns:
        a Pandas dataframe with the Structured-To-Unstructured (STU) data for each stu_id and by datatime input 
        with concatenatued cumulative text within the lookback interval specified. 
    
    References:
        None. 
    """
    df.sort_values([id_column, datetime_column], ascending=[ascend, ascend], inplace=True)
#     df.reset_index(drop=True)
#     #get the first observation of each text string based on sorting of above to keep size of text string for later tokenization. 
#     df['text_obs_rank'] = df.groupby([id_column, text_column])[datetime_column].rank(method='first', na_option='bottom', ascending=True)
#     seldf = df.loc[df['text_obs_rank']==1,:].copy()
#     del df
#     seldf.drop(columns=['text_obs_rank'], inplace=True)
#     seldf.reset_index(drop=True)
#     #make sure properly sorted again in direction
#    seldf.sort_values([id_column, datetime_column], ascending=[ascend, ascend], inplace=True)
    ##need to reconcatenate everything by date
    seldf = df.copy()
    cumdf = seldf.groupby([id_column, datetime_column], as_index=False)[text_column].apply(join_string.join)
    cumdf.columns = [id_column, datetime_column, structured_text_column_return_name]
    return cumdf

def cumulative_concatenator(df: pd.DataFrame, 
                            id_column: str, 
                            datetime_column: str, 
                            text_column: str, 
                            join_string = '; ',
                            lookback_datetime_units = 365, #lookback a year
                            datetime_max_time_lag = 0, 
                            ascend=False) -> pd.DataFrame:
    """ 
    Cumulative transformation of Structured To Unstructered (STU) data for later tokenization and vector embeddings in neural networks 
    relying on unstructured text inputs (e.g., Large Language Models). In the order of the time column, text will be concatenated. 
        1. a unique identifier of the sample (e.g., a patient Medical Record Number)
        2. a unique date, time, or datestamp to specify the longitudinal 
            time sequence in logical order of observations.
        3. at least 1 column of content to be considered as unstructured text for later vector embeddings. 

    Args:
        df: pandas data frame input (source data from EHR or survey data).
               
        stu_id:  unique identifier of the sample observation (e.g., a patient MRN).
        
        stu_datetime: the datetime stamp. 
        
        stu_text: a string of information--possibly duplicated.
        
        join_string:  string to separate out structured data element rows to be concatenated.
        
        lookback_datetime_units: lookback datetime units. This typically is days for EHR, buy may 
            need to change in the future. This defines the interval from which the lookback of information is made.
            This date may not exist, but all datetimes will be included if the date of an 
            event/observation occurs after that date, but before 'lookback_datetime_units' 
            argument (see 'lookback_datetime_units'). This argument is primarily developed to
            address size of input space (for limited computational environments, as well as input embeddings sizes
            of smaller attention models. 
            Default is 365 (for 365 days, or 1 year lookback).
        
        datetime_max_time_lag: this defines a washout period to prevent endogeneity or collider bias (e.g., 
            events that are logged before potential outcomes/targets, but may have actually occurred for the 
            sample [patient] before the feature is logged). An example is a lab test that is logged potentially
            7 days after it actually was present for the patient. May be particularly relevent for 
            acute care outcomes. 
            Default is 0. 

    Returns:
        a Pandas dataframe with the Structured-To-Unstructured (STU) data for each stu_id and by datatime input 
        with concatenatued cumulative text within the lookback interval specified. 
    
    References:
        None. 
    """
    df.sort_values([id_column, datetime_column], ascending=[ascend, ascend], inplace=True)
    df.reset_index(drop=True)
    #get the first observation of each text string based on sorting of above to keep size of text string for later tokenization. 
    df['text_obs_rank'] = df.groupby([id_column, text_column])[datetime_column].rank(method='first', na_option='bottom', ascending=True)
    seldf = df.loc[df['text_obs_rank']==1,:].copy()
    del df
    seldf.drop(columns=['text_obs_rank'], inplace=True)
    seldf.reset_index(drop=True)
    #make sure properly sorted again in direction
    seldf.sort_values([id_column, datetime_column], ascending=[ascend, ascend], inplace=True)
    ##need to reconcatenate everything by date
    cumdf = seldf.groupby([id_column, datetime_column], as_index=False)[text_column].apply('; '.join)
    cumdf.columns = [id_column, datetime_column, 'stucurrent_text']
    #make sure returns desired sorting order
    cumdf.sort_values([id_column, datetime_column], ascending=[False, False], inplace=True)
    maxdatedf = cumdf.groupby([id_column]).agg(max_date=(datetime_column, np.max))
    maxdatedf.reset_index(drop=False)
    maxcumdf = pd.merge(cumdf, maxdatedf, on=[id_column], how='left')
    maxcumdf['lookback_date'] = maxcumdf['max_date'] - pd.DateOffset(days=lookback_datetime_units)
    maxcumdf['datetime_max_time_lag'] = maxcumdf['max_date'] - pd.DateOffset(days=datetime_max_time_lag)
    #print(maxcumdf.shape)
    selcumdf = maxcumdf.loc[(maxcumdf[datetime_column]>=maxcumdf['lookback_date']) & (maxcumdf[datetime_column]<=maxcumdf['datetime_max_time_lag']),:].copy()
    #print(selcumdf.shape)
    #cumsum() blows out memory, so just collapsing 
    #idcumdf = selcumdf.groupby([id_column, datetime_column], as_index=False)['stucurrent_text'].apply(lambda x: '; '.join(x)).cumsum()
    idcumdf = selcumdf.groupby([id_column], as_index=False)['stucurrent_text'].apply(lambda x: '; '.join(x))
    ##get intervals by person 
    lookback_range_df = selcumdf.groupby([id_column]).agg(last_date_observed = (datetime_column, np.max),
                                                          lookback_start=('lookback_date', np.min), 
                                                          lookback_end=('datetime_max_time_lag', np.max),
                                                          encounters = (datetime_column, lambda x: x.nunique()))
    mdf = pd.merge(idcumdf, lookback_range_df, on=[id_column], how='left')
    return mdf

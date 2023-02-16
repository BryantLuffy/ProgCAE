def survive_select(survive_data, data, p_thresh):
    """
    Select features related to survival data based on Cox proportional hazards regression.

    Args:
    - survive_data (pandas DataFrame): A DataFrame containing survival data (OS and OS.time).
    - data (pandas DataFrame): A DataFrame containing the features to be evaluated.
    - p_thresh (float): The threshold of the p-value for a feature to be considered related to survival data.

    Returns:
    - pandas DataFrame: A DataFrame containing the selected features related to survival data.

    """
    from lifelines import CoxPHFitter
    # Select only the OS and OS.time columns from the survival data
    survive_data = survive_data.iloc[:, 0:2]
    # Get the column names of the survival data
    columns_names = survive_data.T.columns.values
    # Transpose the data DataFrame and set the column names to the survival data column names
    dat_trans = data.T
    dat_trans.columns = columns_names
    # Transpose the data back to its original format
    dat_trans = dat_trans.T
    selected_features = []
    # Loop through each column in the data DataFrame
    for i in range(dat_trans.shape[1]):
        # Add the current feature to the survival data DataFrame
        survive_data['feature'] = dat_trans.iloc[:, i].values
        # Fit a Cox proportional hazards model
        cpf = CoxPHFitter()
        cpf.fit(survive_data, 'OS.time', 'OS')
        # Check if the p-value for the feature is below the threshold
        if cpf.summary['p'].values <= p_thresh:
            selected_features.append(i)
    # Return the selected features as a DataFrame
    return dat_trans.iloc[:, selected_features]
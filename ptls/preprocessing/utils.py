def extract_predefined_mappings_from_feature_config(feature_config):
    """Extract predefined mappings from feature_config for categorical fields
    
    Parameters
    ----------
    feature_config : dict
        Feature configuration dictionary containing field definitions
        
    Returns
    -------
    dict
        Dictionary mapping column names to their predefined idx_map
    """
    predefined_mappings = {}
    
    for field_name, field_config in feature_config.items():
        if field_config.get('type') == 'categorical' and 'idx_map' in field_config:
            predefined_mappings[field_name] = field_config['idx_map']
    
    return predefined_mappings


def get_categorical_columns_from_feature_config(feature_config):
    """Get list of categorical column names from feature_config
    
    Parameters
    ----------
    feature_config : dict
        Feature configuration dictionary containing field definitions
        
    Returns
    -------
    list
        List of categorical column names
    """
    categorical_columns = []
    
    for field_name, field_config in feature_config.items():
        if field_config.get('type') == 'categorical':
            categorical_columns.append(field_name)
    
    return categorical_columns
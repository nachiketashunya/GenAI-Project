import pandas as pd
from typing import Dict, List, Callable

# Predefined mappings
model_capabilities = {
    "model_1": ["color", "pattern", "print_or_pattern_type"],
    # Add other models and their capabilities
}

category_attributes = {
    "Men Tshirts": ["color", "pattern", "print_or_pattern_type", "neck", "sleeve_length"],
    "Sarees": ["color", "pattern", "print_or_pattern_type", "fabric", "border"],
    # Add other categories and their attributes
}

# Mapping of attributes to column names, specific to each category
attribute_to_column = {
    "Men Tshirts": {
        "color": "attr_1",
        "pattern": "attr_2",
        "print_or_pattern_type": "attr_3",
        "neck": "attr_4",
        "sleeve_length": "attr_5"
    },
    "Sarees": {
        "color": "attr_1",
        "pattern": "attr_2",
        "print_or_pattern_type": "attr_3",
        "fabric": "attr_4",
        "border": "attr_5"
    },
    # Add mappings for all categories
}

# Mock function to represent model prediction
def predict_attributes(model: str, data: pd.DataFrame) -> pd.DataFrame:
    # In a real scenario, this function would use the actual model to make predictions
    # Here, we're just adding a prefix to show it's a prediction
    predictions = data.copy()
    for _, row in predictions.iterrows():
        category = row['Category']
        for attr in model_capabilities[model]:
            col = attribute_to_column[category][attr]
            predictions.at[_, col] = f"predicted_{attr}_" + str(row[col])
    return predictions

def process_csv(input_file: str, output_file: str):
    # Read the global CSV file
    df = pd.read_csv(input_file)
    
    # Process for each model
    for model, attributes in model_capabilities.items():
        print(f"Processing with {model}...")
        
        # Create a subset of the DataFrame with relevant attributes and categories
        relevant_categories = [cat for cat, attrs in category_attributes.items() if all(attr in attrs for attr in attributes)]
        
        # Create a mask for relevant rows and columns
        category_mask = df['Category'].isin(relevant_categories)
        column_mask = ['id', 'Category'] + [
            attribute_to_column[cat][attr]
            for cat in relevant_categories
            for attr in attributes
            if attr in category_attributes[cat]
        ]
        
        subset = df[category_mask][column_mask].copy()
        
        # Identify rows with missing values
        rows_with_missing = subset[subset.isnull().any(axis=1)]
        
        if not rows_with_missing.empty:
            # Make predictions
            predictions = predict_attributes(model, rows_with_missing)
            
            # Update only the missing values in the original DataFrame
            for index, row in predictions.iterrows():
                category = row['Category']
                for attr in attributes:
                    if attr in category_attributes[category]:
                        col = attribute_to_column[category][attr]
                        if pd.isna(df.at[index, col]):
                            df.at[index, col] = row[col]
        
        print(f"Completed processing with {model}")
    
    # Save the updated DataFrame
    df.to_csv(output_file, index=False)
    print(f"Updated CSV saved to {output_file}")

# Usage
input_file = "input.csv"
output_file = "output_updated.csv"
process_csv(input_file, output_file)
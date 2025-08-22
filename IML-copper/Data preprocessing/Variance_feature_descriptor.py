import pandas as pd

# Read data
composition_df = pd.read_excel('data.xlsx', sheet_name='composition-properties', index_col=0)
properties_df = pd.read_excel('data.xlsx', sheet_name='element properties', index_col=0)
alloyfactor_df = pd.read_excel('calculated_properties_corrected.xlsx', index_col=0)
# Preprocess element properties table: ensure index (element names) is correct
properties_df.index = properties_df.index.str.strip()
alloyfactor_df.index = alloyfactor_df.index.str.strip()

# Create result DataFrame
result_df = pd.DataFrame(index=composition_df.index, columns=properties_df.columns)

# Iterate through each alloy
for alloy in composition_df.index:
    # Iterate through each property
    for prop in properties_df.columns:
        element_factor_value = alloyfactor_df.loc[alloy, prop]
        total = 0
        # Iterate through each element column (extract element names)
        for element_col in composition_df.columns:
            # Uniformly handle bracket formats (support full-width "（" and half-width "(")
            element = element_col.split('（')[0].split('(')[0].strip()
            if element in properties_df.index:
                # Get percentage and convert to decimal
                percentage = composition_df.loc[alloy, element_col] / 100
                # Get the property value corresponding to the element
                element_prop_value = properties_df.loc[element, prop]
                
                # Accumulate calculation results
                total += percentage * ((element_prop_value - element_factor_value)**2)
        result_df.loc[alloy, prop] = total


# Save results (column range B1-O21)
result_df.to_excel('calculated_properties_Summary.xlsx')

print("Calculation completed, results saved to calculated_properties_Summary.xlsx")

df = pd.read_excel('calculated_properties_Summary.xlsx')
df.columns = df.columns+ '_1'
df = df.drop(df.columns[0], axis=1)

df1 = pd.read_excel('calculated_properties_corrected.xlsx')
df1 = df1.rename(columns={df1.columns[0]: 'alloy'})
df = pd.concat([df1, df], axis=1)
df.to_excel('calculated_properties_S.xlsx')

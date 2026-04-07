# Fix for the Altair chart error in 34_S-ARC_ext.ipynb
# Error: Objects with 'config' attribute cannot be used within VConcatChart

'''
To fix the error in your notebook, replace the line:

chart1 & chart2

with:

# Extract the config from individual charts
config = {}
if hasattr(chart1, 'config') and chart1.config is not None:
    config.update(chart1.config)
if hasattr(chart2, 'config') and chart2.config is not None:
    config.update(chart2.config)

# Create a new combined chart with the config applied at the top level
combined_chart = alt.vconcat(chart1, chart2)

# Apply the config to the combined chart
if config:
    combined_chart = combined_chart.configure(**config)
else:
    # If you specifically need the strokeOpacity=0 setting
    combined_chart = combined_chart.configure_view(strokeOpacity=0)

# Display the combined chart
combined_chart
'''

# Instructions:
# 1. Run this file to see the fix instructions
# 2. Apply the fix to your notebook
# 3. Re-run the cell that was causing the error

print("Fix instructions for Altair chart error:")
print("\nError: Objects with 'config' attribute cannot be used within VConcatChart")
print("\nThe issue is that you're trying to concatenate charts that have individual config settings.")
print("When using the '&' operator (which calls vconcat()), you can't have config attributes on the individual charts.")
print("\nSolution:")
print("1. Remove the config from individual charts")
print("2. Apply the config to the combined chart instead")
print("\nReplace:\n  chart1 & chart2")
print("\nWith:\n  alt.vconcat(chart1, chart2).configure_view(strokeOpacity=0)")
print("\nThis moves the configuration to the combined chart rather than having it on the individual charts.")
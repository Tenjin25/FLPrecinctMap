import json
from collections import OrderedDict

# Path to the aggregated JSON file
json_path = 'data/fl_elections_aggregated.json'

# Load the JSON data
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Check if 'results_by_year' exists
if 'results_by_year' in data:
    # Sort the years numerically (as strings)
    sorted_years = sorted(data['results_by_year'].keys(), key=int)
    ordered_results = OrderedDict()
    for year in sorted_years:
        ordered_results[year] = data['results_by_year'][year]
    data['results_by_year'] = ordered_results

    # Write back to the file with pretty formatting
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print('results_by_year reordered by year.')
else:
    print('results_by_year not found in the JSON file.')



# # 1. Print the hyperparameters neatly to the console
# print("--- Best Hyperparameters ---")
# best_params_dict = best_hps.values

# for param_name, param_value in best_params_dict.items():
#     print(f"{param_name}: {param_value}")
# print("----------------------------")

# # 2. Save the hyperparameters to a JSON file for future reference
# with open('best_hyperparameters.json', 'w') as f:
#     json.dump(best_params_dict, f, indent=4)
    
# print("Hyperparameters successfully saved to 'best_hyperparameters.json'")
import pandas as pd
import numpy as np
import math
df = pd.read_excel("E:\VScode\Capstone\datasets\Quiz1-Student Version_Questions2,4,5,6_Main_Dataset.xlsx") 

# # Select the 'Corruption level Rating Score (0-100)' column
# sample_scores = df['Corruption level Rating Score (0-100)']

# # Compute the sample mean
# sample_mean = sample_scores.mean()
# print(f"Mean of the selected corruption scores: {sample_mean:.2f}")

# # Compute the sample standard deviation (s)
# sample_std_dev = sample_scores.std(ddof=1)  # Use ddof=1 for sample standard deviation, not for population)

# # Compute the sample size (n)
# sample_size = len(sample_scores)

# # Compute the standard error (SE)
# standard_error = sample_std_dev / np.sqrt(sample_size)
# print(f"Standard error: {standard_error:.2f}")

# # t-value for 95% confidence interval (given as 2.04)
# t_value = 2.04

# # Compute the margin of error
# margin_of_error = t_value * standard_error

# # Compute the confidence interval
# lower_bound = sample_mean - margin_of_error
# upper_bound = sample_mean + margin_of_error
# print(f"95% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")

#---

# # Calculate the total population size
# N = len(df)

# # Group by Stratum
# strata_group = df.groupby('Stratum')

# # Compute the mean and size for each stratum
# stratum_means = strata_group['Corruption level Rating Score (0-100)'].mean()
# stratum_sizes = strata_group.size()

# # Compute the weight for each stratum (Wh)
# stratum_weights = stratum_sizes / N

# # Output results
# print(f"Stratum Weights (W_h):\n{stratum_weights}")

# #Compute a mean
# mean= df['Corruption level Rating Score (0-100)'].mean().round(2)

# print("Compute a mean:", mean)

# df['Y'] = (df['Corruption level Rating Score (0-100)']-mean)**2  

# stratums = df['Stratum'].unique()
# sh_values = {}

# for stratum in stratums:
#     sh_values[stratum] = df[df['Stratum'] == stratum]['Y'].sum() / float(len(df[df['Stratum'] == stratum]['Y']) - 1)

# #Standart error for Stratified
# import math
# standart_strat = math.sqrt(((0.25**2*sh_values['North America']/8)+
#                             (0.25**2*sh_values['South America']/8)+
#                             (0.25**2*sh_values['Europe']/8)+
#                             (0.25**2*sh_values['Central Asia']/8)))

# print("Compute a standard error for Stratified part:", round(standart_strat, 2))

# # d value
# standart_error_srs = df['Corruption level Rating Score (0-100)'].sem()
# d = round(standart_strat / standart_error_srs, 2)
# print("Compute d-value:", d)

# # d value squared
# d_squared = round(d*d, 2)
# print("Compute d-squared:", d_squared)

# #neff = sample size / d^2
# neff = round(len(df)/(d_squared), 2)
# print("Compute Neff:", neff)

#---

#Compute a mean
mean= df['Corruption level Rating Score (0-100)'].mean().round(2)
print("Compute a mean:", mean)

#SE
df['Y'] = (df['Corruption level Rating Score (0-100)']-mean)**2
clusters = ['Canada', 'USA', 'Columbia', 'Brazil', 'Spain', 'France', 'Uzbekistan', 'Kazakhstan']
shc_values = []

for cluster in clusters:
    shc = df[df['Cluster'] == cluster]['Y'].sum() / 3
    shc_values.append(shc)


SE_with_clustering = sum((0.125**2 * shc / 4) for shc in shc_values)

SE_with_clustering = math.sqrt(SE_with_clustering)
print("Compute a standard error for Clusteting Random Sampling:", round(SE_with_clustering, 2))


#d value
standart_error_srs = df['Corruption level Rating Score (0-100)'].sem()
d_value = round(SE_with_clustering / standart_error_srs, 2)
print("Compute d-value:", d_value)

#d squared
d_squared = round(d_value**2, 2)
print("Compute d-squared:", d_squared)

# Compute roh
roh = ((d_squared - 1)/(4-1)) # divide by mean in clusters
print("Compute roh:", roh.round(2))


#asdasd

#neff
neff = round(len(df)/d_squared, 2)

print("Neff:", neff)
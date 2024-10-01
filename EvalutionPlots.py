
import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelConfusionMatrix
from pandas import DataFrame, concat
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from itertools import combinations
from sklearn.metrics import multilabel_confusion_matrix
DATASET_DIR = "./"
label_dict = {'No Pesticide':0,'carbophenothion':1,'coumaphos':2,'oxamyl':3,'phosmet':4,'thiabenzadole':5}
Num_classes = len(label_dict)
reversed_dict = {value: key for key, value in label_dict.items()} 
data1 = torch.load(DATASET_DIR+"/SERSFormer2_0_multireg_multiclass_test.pt")
#dataset= list(data1)[0]
print(len(data1))
dataset_outputs = torch.load(DATASET_DIR+'/Predictions_ofnormaltest.pt')





class_preds = torch.cat([x[f'preds_class'] for x in dataset_outputs])
class_targets = torch.cat([x[f'targets_class'] for x in dataset_outputs])
#class_preds = self.test_step_class_pred
print(len(class_preds))
#class_targets = self.test_step_class_tar
conf_mat = MultilabelConfusionMatrix(num_labels=Num_classes)
conf_vals = conf_mat(class_preds, class_targets)
fig, ax = plt.subplots()
# Plot confusion matrices using Seaborn
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, cm in enumerate(conf_vals.cpu()):
    # Set the global font scale
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(reversed_dict[i])
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('True')
#sns.heatmap(conf_vals.cpu() , annot=True, cmap="Blues", fmt="d")
#plt.title("Confusion Matrix")
#plt.tight_layout()
#plt.savefig(DATASET_DIR+"/Training2/SERSFOrmer2_0_multireg_multiclass/ConfusionMatrix.png")

# Find unique combinations of labels present in the data
# Compute multilabel confusion matrix
# Threshold for binarization (adjust according to your needs)
threshold = 0.5

# Convert probabilistic predictions to binary predictions
class_pred = np.where(class_preds >= threshold, 1, 0)
mcm = multilabel_confusion_matrix(class_targets,class_pred)

label_combinations = set()
for i in range(len(class_targets)):
    unique_labels_true = np.where(class_targets[i] == 1)[0]
    unique_combinations = list(combinations(unique_labels_true, 2))  # Get all pairs of unique labels
    label_combinations.update(unique_combinations)
#print(label_combinations)
#plt.figure(figsize=(12, 10))
#print(mcm.shape)

num_plots = len(label_combinations)
num_cols = 3  # Adjust number of columns as needed
num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows needed

plt.figure(figsize=(25, 8 * num_rows))

for idx, (l1, l2) in enumerate(label_combinations):
    # Get the confusion matrix for label l1 and l2
    '''cm_l1 = conf_vals[l1]
    cm_l2 = conf_vals[l2]
    print(len(cm_l1))
    # Combine confusion matrices for l1 and l2
    #combined_cm = cm_l1 + cm_l2
    # Create an empty confusion matrix
    # Create an empty co-occurrence matrix (2x2) for binary values
    combined_cm = np.zeros((2, 2), dtype=int)

    # Extract predictions and targets for label l1 and l2
    preds_l1 = class_preds[:, l1].cpu().numpy()  # Get predictions for label l1
    preds_l2 = class_preds[:, l2].cpu().numpy()  # Get predictions for label l2

    # Iterate through each pair of predictions for l1 and l2 and update the co-occurrence matrix
    for pred_l1, pred_l2 in zip(preds_l1, preds_l2):
        if 0 <= pred_l1 <= 1 and 0 <= pred_l2 <= 1:  # Ensure binary values (0 or 1)
            combined_cm[int(pred_l1), int(pred_l2)] += 1
        #else:
        #    print(f"Warning: Skipping out-of-bounds label (pred_l1: {pred_l1}, pred_l2: {pred_l2})")
    '''
    co_occurrence_matrix = np.zeros((2, 2), dtype=int)

    # Extract predictions and targets for label l1 and l2
    preds_l1 = (class_preds[:, l1]>0.5).float().numpy()  # Get predictions for label l1
    #print(preds_l1)
    preds_l2 = (class_preds[:, l2]>0.5).float().numpy()  # Get predictions for label l2
    targets_l1 = class_targets[:, l1].cpu().numpy()  # Get true targets for label l1
    targets_l2 = class_targets[:, l2].cpu().numpy()  # Get true targets for label l2

    # Iterate through each pair of predictions and true labels for l1 and l2
    for pred_l1, pred_l2, target_l1, target_l2 in zip(preds_l1, preds_l2, targets_l1, targets_l2):
        # True Positive (TP): Both predicted and actual are 1 for l1 and l2
        if pred_l1 == 1 and pred_l2 == 1 and target_l1 == 1 and target_l2 == 1:
            co_occurrence_matrix[1, 1] += 1

        # True Negative (TN): Both predicted and actual are 0 for l1 and l2
        elif pred_l1 == 0 and pred_l2 == 0 and target_l1 == 0 and target_l2 == 0:
            co_occurrence_matrix[0, 0] += 1

        # False Negative (FN): True label has both l1 and l2 as 1, but predicted as 0 or one of them is 0
        elif target_l1 == 1 and target_l2 == 1 and (pred_l1 == 0 or pred_l2 == 0):
            co_occurrence_matrix[1, 0] += 1

        # False Positive (FP): True label has both l1 and l2 as 0, but at least one predicted as 1
        elif target_l1 == 0 and target_l2 == 0 and (pred_l1 == 1 or pred_l2 == 1):
            co_occurrence_matrix[0, 1] += 1
    combined_cm=co_occurrence_matrix
    # Plot the combined confusion matrix
    plt.subplot(num_rows, num_cols, idx + 1)
    #plt.imshow(combined_cm, interpolation='nearest', cmap=plt.cm.Blues)
    sns.heatmap(combined_cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f'{reversed_dict[l1]} & {reversed_dict[l2]}')
    
    #plt.colorbar()
    #tick_marks = np.arange(2)
    #plt.xticks(tick_marks, ['Predicted 0', 'Predicted 1'])
    #plt.yticks(tick_marks, ['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Add text annotations
    thresh = combined_cm.max() / 2.
    #for k in range(2):
    #    for l in range(2):
    #        plt.text(l, k, format(combined_cm[k, l], 'd'),
    #                 horizontalalignment="center",
    #                 color="white" if combined_cm[k, l] > thresh else "black")
#plt.tight_layout()
#plt.savefig(DATASET_DIR+"/Training2/SERSFOrmer2_0_multireg_multiclass/ConfusionMatrix_mixture.png")
# Concatenate regression predictions and targets
test_reg_pred = torch.cat([x['preds_reg'] for x in dataset_outputs])
test_reg_tar = torch.cat([x['targets_reg'] for x in dataset_outputs])

# Convert to numpy
reg_preds_np = test_reg_pred.numpy()
reg_targets_np = test_reg_tar.numpy()

# Create DataFrames for predictions and targets
output_dim = reg_preds_np.shape[1]
pred_data = DataFrame(reg_preds_np, columns=[f'Output_{i+1}' for i in range(output_dim)])
true_data = DataFrame(reg_targets_np, columns=[f'Output_{i+1}' for i in range(output_dim)])

# Add a column to distinguish between true and predicted
pred_data['Type'] = 'Predicted'
true_data['Type'] = 'True'

# Combine the DataFrames
combined_data = concat([pred_data, true_data])

# Melt the combined DataFrame for seaborn plotting
combined_data_melted = combined_data.melt(id_vars='Type', var_name='Output', value_name='Value')
# Plot Violin Plots for each label combination and output

# Plot Violin Plots for each label combination
sns.set_theme(font_scale=2.0)
plt.figure(figsize=(18, 6 * num_rows))

for idx, (l1, l2) in enumerate(label_combinations):
    # Get indices of data points corresponding to each label combination
    indices = np.where((class_targets[:, l1] == 1) & (class_targets[:, l2] == 1))[0]
    
    # Extract relevant regression predictions and targets
    reg_preds_comb = reg_preds_np[indices]
    reg_targets_comb = reg_targets_np[indices]

    # Create DataFrames for the combined data
    pred_data_comb = DataFrame(reg_preds_comb, columns=[f'{reversed_dict[i]}' for i in range(output_dim)])
    true_data_comb = DataFrame(reg_targets_comb, columns=[f'{reversed_dict[i]}' for i in range(output_dim)])
    pred_data_comb['Type'] = 'Predicted'
    true_data_comb['Type'] = 'True'
    combined_data_comb = concat([pred_data_comb, true_data_comb])
    combined_data_melted_comb = combined_data_comb.melt(id_vars='Type', var_name='Output', value_name='Value')

    # Plot violin plot
    plt.subplot(num_rows, num_cols, idx + 1)
    sns.violinplot(x='Output', y='Value', hue='Type', data=combined_data_melted_comb, split=True, inner="box", cut=0, scale='area',legend=None)
    plt.title(f'{reversed_dict[l1]} & {reversed_dict[l2]}')
    plt.xlabel('Pesticides')
    plt.ylabel('Concentrations')
    plt.legend().remove()
    # Adjust x-ticks to prevent overlapping
    plt.xticks(ticks=range(output_dim), labels=[f'{reversed_dict[i]}' for i in range(output_dim)], rotation=45, ha='right')
    plt.yticks() 

    # Extract unique target values from the true data
    #unique_targets = sorted(true_data_comb.columns)
    #unique_targets.pop(1)
    #print(unique_targets)
    # Set up subplots for each unique target value
    #fig2, axes = plt.subplots(nrows=1, ncols=len(unique_targets), figsize=(15, 5), sharey=True)
    
    # Create violin plots for each target value
    #for i, target_value in enumerate(unique_targets):
     #   target_df = DataFrame({
     #       'True': true_data_comb[target_value],
     #       'Pred': pred_data_comb[target_value]
     #   })
     #   print(target_df)
     #   sns.violinplot(x='True', y='Pred', data=target_df, inner="sticks", color="lightgreen", cut=0, ax=axes[i])
     #   axes[i].set_title(f'Target = {target_value}')
    
    #plt.tight_layout()
    #plt.show()
plt.subplot(num_rows, num_cols, idx + 1)

plt.tight_layout()
# Add a single legend after all subplots are drawn
handles, labels = plt.gca().get_legend_handles_labels()  # Get the handles and labels from the last plot
plt.legend(bbox_to_anchor=(2.04, 0.5), loc="center left", borderaxespad=0)
plt.show()
plt.savefig(DATASET_DIR+"/Training2/SERSFOrmer2_0_multireg_multiclass/Violinplot_mixture_conc.png")
# Set up the overall figure to accommodate all subplots

sns.set_theme(font_scale=1.5)
plt.figure(figsize=(23, 10 * len(label_combinations)),constrained_layout=True)

for idx, (l1, l2) in enumerate(label_combinations):
    # Get indices of data points corresponding to each label combination
    indices = np.where((class_targets[:, l1] == 1) & (class_targets[:, l2] == 1))[0]
    
    # Extract relevant regression predictions and targets
    reg_preds_comb = reg_preds_np[indices]
    reg_targets_comb = reg_targets_np[indices]

    # Create DataFrames for the combined data
    pred_data_comb = DataFrame(reg_preds_comb, columns=[f'{reversed_dict[i]}' for i in range(output_dim)])
    true_data_comb = DataFrame(reg_targets_comb, columns=[f'{reversed_dict[i]}' for i in range(output_dim)])
    pred_data_comb['Type'] = 'Predicted'
    true_data_comb['Type'] = 'True'
    combined_data_comb = concat([pred_data_comb, true_data_comb])
    combined_data_melted_comb = combined_data_comb.melt(id_vars='Type', var_name='Output', value_name='Value')

    # Create a new figure for each combination
    fig, axes = plt.subplots(nrows=1, ncols=len(sorted(true_data_comb.columns)), figsize=(25, 5), sharey=True)

    # Main violin plot for the label combination
    sns.violinplot(x='Output', y='Value', hue='Type', data=combined_data_melted_comb, split=True, inner="box", cut=0, scale='area', ax=axes[0],legend=False)
    axes[0].set_title(f'{reversed_dict[l1]} & {reversed_dict[l2]}')
    axes[0].legend().remove()
    axes[0].set_xlabel('Pesticides')
    axes[0].set_ylabel('Concentrations')
    fig.legend(loc='outside lower center')
    axes[0].tick_params(axis='y', color='black')
# Put a legend below current axis
    
    # Adjust x-ticks to prevent overlapping
    axes[0].set_xticks(range(output_dim))
    axes[0].set_xticklabels([f'{reversed_dict[i]}' for i in range(output_dim)], rotation=45, ha='right')
    
    # Extract unique target values from the true data
    unique_targets = sorted(true_data_comb.columns)
    unique_targets.pop(1)  # Modify as needed, here it removes the second element

    # Subplots for unique targets within the same row
    for i, target_value in enumerate(unique_targets):
        target_df = DataFrame({
            'True': true_data_comb[target_value],
            'Pred': pred_data_comb[target_value]
        })
        sns.violinplot(x='True', y='Pred', data=target_df, inner="sticks", color="lightblue", cut=0, ax=axes[i + 1])
        axes[i + 1].set_title(f'Target = {target_value}')
        

    plt.tight_layout()
    
    # Save the figure for each label combination
    
    plt.savefig(f'{DATASET_DIR}/Training2/SERSFOrmer2_0_multireg_multiclass/violin_plot_combination_{idx + 1}_{reversed_dict[l1]}_{reversed_dict[l2]}.png')
    
    # Close the figure to avoid memory issues and overlapping plots
    plt.close(fig)


"""
test_reg_pred = torch.cat([x[f'preds_reg'] for x in dataset_outputs])
test_reg_tar = torch.cat([x[f'targets_reg'] for x in dataset_outputs])
print(test_reg_pred.shape)
#for i in range(test_reg_pred.shape[0]):
reg_preds = test_reg_pred[:, :]
reg_targets = test_reg_tar[:, :]
print(reg_preds.shape,reg_targets.shape)
data = [[x, y] for (x, y) in zip(reg_targets, reg_preds)]
reg_preds_np = reg_preds.squeeze().numpy()
reg_targets_np = reg_targets.squeeze().numpy()
df = DataFrame({'True': reg_targets_np, 'Pred': reg_preds_np})
    # Scatter plot
# Calculate the point density
xy = torch.vstack([reg_targets.T,reg_preds.T]).cpu().detach().numpy()
print(df.shape)
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = reg_targets[idx], reg_preds[idx], z[idx]
fig1, ax = plt.subplots(figsize=(12, 6))
plt.scatter(x,y,c=z,s=100)
  
# Get unique target values
unique_targets = df['True'].unique()
unique_targets = sorted(unique_targets)
print(unique_targets)
# Set up subplots
num_targets = len(unique_targets)
fig2, axes = plt.subplots(nrows=1, ncols=num_targets, figsize=(15, 5),sharey=True)
# Create violin plots for each target value
for i, target_value in enumerate(unique_targets):
    target_df = df[df['True'] == target_value]
    sns.violinplot(x='True', y='Pred', data=target_df, inner="sticks", color="lightgreen",cut=0 ,ax=axes[i])
    axes[i].set_title(f'Target = {target_value}')
plt.tight_layout()
plt.xlabel("True Concentration")
plt.ylabel("Predicted Concentration")
plt.title("True and Predicted Concentration")
plt.tight_layout()
"""

concentration_columns = [f'Output_{i+1}' for i in range(output_dim)]
# Plot overall violin plots for each target value
# Plot overall violin plots for concentration values
# Plot overall violin plots for concentration values
plt.figure(figsize=(15, 8))

# For each target value, plot a separate violin plot for concentration data
for target_idx in range(Num_classes):
    # Get indices of data points where the current target is present
    indices = np.where(class_targets[:, target_idx] == 1)[0]
    
    # Extract relevant regression predictions and targets for the current target
    reg_preds_comb = reg_preds_np[indices]
    reg_targets_comb = reg_targets_np[indices]

    # Create DataFrames for the combined data
    pred_data_comb = DataFrame(reg_preds_comb, columns=[f'Output_{i+1}' for i in range(output_dim)])
    true_data_comb = DataFrame(reg_targets_comb, columns=[f'Output_{i+1}' for i in range(output_dim)])
    pred_data_comb['Type'] = 'Predicted'
    true_data_comb['Type'] = 'True'
    combined_data_comb = concat([pred_data_comb, true_data_comb])
    combined_data_comb = combined_data_comb[concentration_columns + ['Type']]
    combined_data_melted_comb = combined_data_comb.melt(id_vars='Type', var_name='Output', value_name='Value')

    # Plot violin plot for the current target value
    plt.subplot((Num_classes + 2) // 3, 3, target_idx + 1)
    sns.violinplot(x='Type', y='Value', data=combined_data_melted_comb, inner="box", cut=0, scale='area')
    plt.legend().remove()
    plt.title(f'{reversed_dict[target_idx]}')
    plt.xlabel('Type')
    plt.ylabel('Values')

plt.tight_layout()
plt.show()
plt.savefig(DATASET_DIR+"/Training2/SERSFOrmer2_0_multireg_multiclass/Violinplot.png")

'''
num_plots = len(label_combinations)
num_cols = 2
num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows needed
plt.figure(figsize=(12, 5 * num_rows))
# Loop through each present combination of labels
plot_idx=1

for l1 in range(len(reversed_dict)):
    for l2 in range(l1 + 1, len(reversed_dict)):  # Ensure only unique combinations are considered
        if (l1, l2) in label_combinations or (l2, l1) in label_combinations:
            # Get the confusion matrix for label l1 and l2
            cm_l1 = conf_vals[l1]
            cm_l2 = conf_vals[l2]
            combined_cm = cm_l1 + cm_l2 
            
            # Plot the confusion matrix
            plt.subplot(num_rows, num_cols, plot_idx)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(reversed_dict[l1] + ' & ' + reversed_dict[l2])
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Predicted 0', 'Predicted 1'])
            plt.yticks(tick_marks, ['Actual 0', 'Actual 1'])

            # Add text annotations
            thresh = cm.max() / 2.
            for k in range(2):
                for l in range(2):
                    plt.text(l, k, format(cm[k, l], 'd'),
                             horizontalalignment="center",
                             color="white" if cm[k, l] > thresh else "black")
            plot_idx+=1
# Adjust layout and show plot
'''

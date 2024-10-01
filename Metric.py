import torch
import torchmetrics
from torchmetrics.regression import MeanSquaredError,R2Score
from copy import deepcopy
from typing import Optional
class MultiLabelUnifiedMetric(torchmetrics.Metric):
    def __init__(self ,num_labels=1,dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_labels=num_labels
        # Add state variables for storing intermediate results
        self.add_state("multiplied_outputs", default=torch.empty(0, num_labels), dist_reduce_fx="cat")
        self.add_state("targets", default=torch.empty(0, num_labels), dist_reduce_fx="cat")

        # Initialize MSE and R2 score metrics
        self.mse = torchmetrics.regression.MeanSquaredError(num_outputs=num_labels)#.cuda()
        self.r2score = torchmetrics.regression.R2Score(num_outputs=num_labels)#.cuda()

    def update(self, preds: torch.Tensor, target: torch.Tensor, target2: torch.Tensor):
        assert preds.shape == target.shape == target2.shape, "All tensors must have the same shape"
        preds = preds#.cuda()
        target = target#.cuda()
        target2 = target2#.cuda()
        # Element-wise multiplication
        multiplied_output = preds * target2
        multiplied_output = multiplied_output#.cuda()
        # Update state
        #if self.multiplied_outputs.numel() == 0:
        #self.multiplied_outputs = multiplied_output
        #self.targets = target
        
        #else:
        self.multiplied_outputs = torch.cat([self.multiplied_outputs, multiplied_output],dim=0)#.cuda()
        self.targets = torch.cat([self.targets, target],dim=0)#.cuda()
        #print(self.multiplied_outputs.shape,self.targets.shape)
    def compute(self):
        # Calculate MSE and R2 scores
        mse_value = self.mse(self.multiplied_outputs, self.targets)#.cuda()
        r2_value = self.r2score(self.multiplied_outputs, self.targets)#.cuda()

        return {"mse": mse_value, "r2": r2_value}

    def reset(self):
        # Reset state variables
        self.multiplied_outputs = torch.empty(0, self.num_labels)#.cuda()
        self.targets = torch.empty(0, self.num_labels)#.cuda()

    
class CustomMetricCollection(torchmetrics.Metric):
    def __init__(self, metrics):
        super().__init__()

        # Store the metrics in the collection
        self.metrics = torchmetrics.MetricCollection(metrics)

    def update(self, multi_label, preds, target):
        # Unpack the multi_label inputs
        preds1, preds2, target2 = multi_label

        # Update the multi-label metric
        self.metrics['unified_metric'].update(preds1, preds2, target2)

        # Update other metrics
        self.metrics['MSE'].update(preds, target)
        self.metrics['R2Score'].update(preds, target)
        

    def compute(self):
        # Compute all metrics
        return self.metrics.compute()

    def reset(self):
        # Reset all metrics
        self.metrics.reset()

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "CustomMetricCollection":
        """Make a copy of the metric collection

        Args:
            prefix: a string to append in front of the metric keys
            postfix: a string to append after the keys of the output dict

        """
        mc = deepcopy(self)
        cloned_metrics = {}
        for name, metric in mc.metrics.items():
            new_name = name
            if prefix:
                new_name = f"{prefix}_{new_name}"
            if postfix:
                new_name = f"{new_name}_{postfix}"
            cloned_metrics[new_name] = metric
        mc.metrics = torchmetrics.MetricCollection(cloned_metrics)
        return mc

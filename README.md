The Weak OT version of the algorithm, everything else is the same.

To find all changes, search for "TODO: change cosine similarity to Weak OT"

In total two changes, adding a function "_sinkhorn_plan_1d_hist" and a function "_weak_ot_cost_1d_hist", and replace the old "align_loss" function

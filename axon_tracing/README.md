# Preparation

Find a good center point and reasonably sized bounding box (from MoBIE)

# Workflow

- `1_prepare_table` Make MoBIE table for all boutons in the bounding box, go through them with annotation tool to keep the real ones
- `2_predict_boundaries` Predict boundaries for the bounding box
- `3_problem` compute superpixels, graph and edge costs
- `4_lifted_problem_from_boutons` add repulsive costs between different boutons (could also include attractive costs at some point)
- `5_lmc` solve lmc, map boutons to segments and discard everything that is not mapped
- `6_to_mobie` export the bouton seg


# axon dendrite classification???

# Preparation

- Find a good center point and reasonably sized bounding box
- Make MoBIE table for all boutons in the bounding box, go through them with annotation tool to keep the real ones

# Segmentation Workflow

- Predict boundaries for the bounding box
- compute superpixels, graph and edge costs
- add repulsive costs between different boutons
- solve lmc, map boutons to segments and discard everything that is not mapped


# axon dendrite classification???
